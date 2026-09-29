#!/usr/bin/env python3
# coding=utf-8
# Adapted from
# https://github.com/deepseek-ai/DeepSpec/blob/main/scripts/data/prepare_target_cache.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract intermediate features from DeepSeek-V4 using existing infer.sh infrastructure.

This script leverages the existing OfflineInference class and ExecutionEngine
without modifying any code in models/ or executor/. It registers hooks on the
model to capture intermediate hidden states and writes them to disk in DeepSpec
format.

Usage:
    python extract_features.py \
        --config config/extract_deepseek_v4.yaml \
        --input-file train_data.jsonl \
        --output-dir /path/to/cache \
        --target-layer-ids 40 41 42
"""

import argparse
import json
import logging
import os
import sys
import time
import yaml

import torch
import torch.distributed as dist
import torch_npu

# Add recipes root to path
_RECIPES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _RECIPES_ROOT not in sys.path:
    sys.path.insert(0, _RECIPES_ROOT)

from executor.core.config import InferenceConfig
from executor.offline.offline_inference import OfflineInference
from executor.utils.logging_config import setup_logging
from executor.utils.forward_metadata import (
    get_forward_metadata,
    reset_forward_metadata,
    set_forward_metadata,
)

# Import our custom modules
from module.dspark_data_preparation.feature_hooks import FeatureHookManager
from module.dspark_data_preparation.data_utils import (
    load_jsonl_dataset,
    tokenize_conversation,
)
from module.dspark_data_preparation.cache_writer import (
    AsyncTargetCacheWriter,
    LocalCacheWriteSummary,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    cleanup_target_cache_tmp_dir,
    finalize_target_cache_index,
    load_local_cache_write_summary,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DeepSeek-V4 intermediate features"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (InferenceConfig format)",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Input JSONL file with conversations",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for target cache",
    )
    parser.add_argument(
        "--target-layer-ids",
        nargs="+",
        type=int,
        default=[40, 41, 42],
        help="Decoder layer indices to capture (default: 40 41 42)",
    )
    parser.add_argument(
        "--min-loss-tokens",
        type=int,
        default=14,
        help="Minimum assistant tokens required (default: 14)",
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=64 * 1024**3,
        help="Maximum bytes per shard file (default: 64 GiB)",
    )
    parser.add_argument(
        "--chat-template",
        default="deepseek",
        help="Chat template name (default: deepseek)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Process only first N samples (for testing)",
    )
    hc_group = parser.add_mutually_exclusive_group()
    hc_group.add_argument(
        "--mean-hc-mult",
        dest="mean_hc_mult",
        action="store_true",
        help="Average the hc_mult dimension before saving features",
    )
    hc_group.add_argument(
        "--no-mean-hc-mult",
        dest="mean_hc_mult",
        action="store_false",
        help="Preserve the hc_mult dimension when saving features",
    )
    parser.set_defaults(mean_hc_mult=None)
    return parser.parse_args()


def process_hc_mult(
    tensor: torch.Tensor,
    *,
    hc_mult: int,
    mean_hc_mult: bool,
    name: str,
) -> torch.Tensor:
    """Optionally reduce the hyper-connection dimension for persistence."""
    if tensor.ndim == 3:
        if int(tensor.shape[1]) != int(hc_mult):
            raise ValueError(
                f"{name} hc_mult dimension {tensor.shape[1]} != model hc_mult {hc_mult}"
            )
        if mean_hc_mult:
            tensor = tensor.mean(dim=1)
    elif tensor.ndim != 2:
        raise ValueError(
            f"{name} must have shape (tokens, hidden) or "
            f"(tokens, hc_mult, hidden), got {tuple(tensor.shape)}"
        )
    return tensor.contiguous()


def init_distributed():
    """Initialize distributed environment using environment variables.

    Returns:
        Tuple of (local_rank, global_rank, world_size, distributed)
    """
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank_offset = int(os.getenv("RANK_OFFSET", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    global_rank = local_rank + rank_offset
    distributed = world_size > 1

    logger.info(
        f"[extract_features] rank={global_rank}/{world_size} "
        f"(local={local_rank}, distributed={distributed})"
    )

    return local_rank, global_rank, world_size, distributed


def main():
    setup_logging()
    args = parse_args()

    # Initialize distributed
    local_rank, global_rank, world_size, distributed = init_distributed()

    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    feature_cfg = yaml_dict.get("feature_extraction", {}) or {}
    mean_hc_mult = (
        args.mean_hc_mult
        if args.mean_hc_mult is not None
        else bool(feature_cfg.get("mean_hc_mult", False))
    )

    # CRITICAL: Force eager mode for hooks to work
    yaml_dict.setdefault("model_config", {})["exe_mode"] = "eager"
    yaml_dict["model_config"]["enable_cache_compile"] = False

    # Create InferenceConfig
    config = InferenceConfig.from_dict(
        yaml_dict,
        global_rank=global_rank,
        local_rank=local_rank,
    )

    logger.info("=" * 80)
    logger.info("Feature Extraction Configuration")
    logger.info(f"  Input file: {args.input_file}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Target layers: {args.target_layer_ids}")
    logger.info(f"  Chat template: {args.chat_template}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Min loss tokens: {args.min_loss_tokens}")
    logger.info(f"  Mean hc_mult: {mean_hc_mult}")
    logger.info(f"  Rank: {global_rank}/{world_size}")
    logger.info("=" * 80)

    # Prepare output directory (rank 0 only)
    output_dir = os.path.abspath(args.output_dir)

    # Initialize distributed BEFORE any dist.barrier() calls
    if distributed:
        # Set device first
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        logger.info(f"[rank {global_rank}] Set device: {device}")

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend="hccl",
                rank=global_rank,
                world_size=world_size,
            )
            logger.info(f"[rank {global_rank}] HCCL initialized")

    if global_rank == 0:
        try:
            prepare_target_cache_output_dir(output_dir)
            logger.info(f"Prepared output directory: {output_dir}")
        except FileExistsError:
            logger.error(
                f"Output directory already contains a completed cache: {output_dir}"
            )

    if distributed:
        dist.barrier()

    # Create rank-specific temp directory
    rank_dir = os.path.join(output_dir, "_tmp", f"rank_{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {args.input_file}")
    all_samples = load_jsonl_dataset([args.input_file])

    if args.num_samples is not None:
        all_samples = all_samples[: args.num_samples]
        logger.info(f"Limited to first {args.num_samples} samples")

    total_samples = len(all_samples)
    logger.info(f"Total samples: {total_samples}")

    # Initialize OfflineInference (this loads the model)
    logger.info("Initializing OfflineInference...")
    llm = OfflineInference(config)
    logger.info("OfflineInference initialized successfully")

    # Get tokenizer
    tokenizer = llm.engine.tokenizer

    # Access the model
    engine = llm.engine
    model = llm.engine.main_worker.model
    device = engine.device
    logger.info(f"Model type: {type(model).__name__}")

    # Pre-tokenize all samples and keep only valid ones, so that every rank
    # ends up with the same count of forward-able samples (no in-loop `continue`
    # skips that would desynchronize the MoE all-to-all collective).
    #
    # This is CRITICAL for MoE models with expert parallelism: if ranks process
    # different numbers of samples, the all-to-all collectives in dispatch_double_routing
    # will hang (HCCL error). Pre-validation ensures deterministic sample counts.
    logger.info("Pre-tokenizing dataset to filter invalid samples...")
    tokenized_samples = []
    skipped = 0
    for idx, sample in enumerate(all_samples):
        conversation = sample.get("conversations")
        if not conversation:
            skipped += 1
            continue

        # Tokenize and validate conversation format
        tokenized = tokenize_conversation(
            conversation=conversation,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            min_loss_tokens=args.min_loss_tokens,
        )
        if tokenized is None:
            skipped += 1
            continue

        # Check for empty sequences after padding/truncation
        seq_len = int(tokenized["attention_mask"].sum().item())
        if seq_len == 0:
            skipped += 1
            continue

        # Store (original_index, tokenized_data) tuple for later reference
        tokenized_samples.append((idx, tokenized))

    logger.info(
        f"Pre-tokenization done: {len(tokenized_samples)} valid, {skipped} skipped"
    )

    # Compute local sample range for this rank.
    # CRITICAL: every rank MUST process the same number of samples because each
    # forward pass contains MoE all-to-all collectives (dispatch_double_routing)
    # on moe_ep_group (all ranks). Drop the remainder so every rank gets exactly
    # base = total_valid // world_size samples. All samples here are already
    # pre-validated, so the forward loop never skips -> no collective mismatch.
    #
    # Example: 101 valid samples, 8 ranks -> each rank gets 101//8=12 samples,
    # dropping 101%8=5 remainder samples to prevent HCCL deadlock.
    total_valid = len(tokenized_samples)
    per_rank_count = total_valid // world_size
    local_start = global_rank * per_rank_count
    local_end = local_start + per_rank_count
    local_samples = tokenized_samples[local_start:local_end]
    local_count = len(local_samples)
    dropped = total_valid - per_rank_count * world_size

    # Warn about dropped samples (only on rank 0 to avoid duplicate logs)
    if dropped > 0 and global_rank == 0:
        logger.warning(
            f"Dropping {dropped} remainder valid sample(s) to keep all {world_size} "
            f"ranks at equal count ({per_rank_count}); this avoids MoE all-to-all "
            f"deadlock. Run a separate single-card pass to recover them if needed."
        )

    logger.info(
        f"Rank {global_rank}: processing samples [{local_start}:{local_end}) "
        f"= {local_count} samples"
    )

    # Get hidden size and hyper-connection multiplicity
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(getattr(model.config, "hc_mult", 1))
    output_hc_mult = 1 if mean_hc_mult else hc_mult
    logger.info(
        f"Hidden size: {hidden_size}, hc_mult: {hc_mult}, "
        f"saved_hc_mult: {output_hc_mult}"
    )

    # Register hooks
    logger.info("Registering feature extraction hooks...")
    hook_manager = FeatureHookManager(
        model=model,
        target_layer_ids=args.target_layer_ids,
    )
    hook_manager.register_hooks()
    logger.info("Hooks registered successfully")

    # Create cache writer
    writer = AsyncTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=args.max_shard_bytes,
        max_queue_size=16,  # Adjust based on memory
    )

    # Process samples
    num_processed = 0
    num_valid_samples = 0
    t_start = time.time()

    logger.info("Starting feature extraction...")

    try:
        # Process each pre-validated sample
        # No in-loop skips since all samples are already validated,
        # ensuring synchronized collective operations across all ranks
        for local_idx, (orig_idx, tokenized) in enumerate(local_samples):

            # Move tokenized data to device
            full_input_ids = tokenized["input_ids"].to(device)
            full_attention_mask = tokenized["attention_mask"].to(device)
            seq_len = int(full_attention_mask.sum().item())

            # Prepare contiguous input tensors for inference
            input_ids = full_input_ids[:seq_len].contiguous()
            seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

            # Clear previous captures to avoid stale data
            hook_manager.clear_captured()

            # Build model inputs with proper metadata for prefill phase
            reset_forward_metadata()
            model_inputs = engine._build_model_inputs(
                input_ids,
                is_prefill=True,
                seq_lens=seq_lens,
            )
            set_forward_metadata(is_prefill=True)
            model_inputs["forward_metadata"] = get_forward_metadata()

            # Run forward pass (registered hooks capture intermediate features automatically)
            # This triggers MoE all-to-all collectives, which is why equal sample counts are critical
            with torch.no_grad():
                output, _ = engine.main_worker.inference(
                    model_inputs,
                    is_prefill=True,
                )

            # Get captured features
            try:
                features = hook_manager.get_captured_features()
            except RuntimeError as e:
                logger.warning(f"Failed to capture features for sample {orig_idx}: {e}")
                num_processed += 1
                continue

            # Extract sequence (remove batch dim, move to CPU)
            seq_len = int(tokenized["attention_mask"].sum().item())

            # Write to cache
            target_hidden = process_hc_mult(
                features["target_hidden"][:seq_len],
                hc_mult=hc_mult,
                mean_hc_mult=mean_hc_mult,
                name="target_hidden",
            )
            last_hidden = process_hc_mult(
                features["last_hidden"][:seq_len],
                hc_mult=hc_mult,
                mean_hc_mult=mean_hc_mult,
                name="last_hidden",
            )
            if last_hidden.ndim != 2:
                raise ValueError(
                    "last_hidden contains hc_mult dimension; enable --mean-hc-mult "
                    "because target_last_hidden_states has no hc_mult storage dimension"
                )
            if target_hidden.shape[-1] != len(args.target_layer_ids) * hidden_size:
                raise ValueError(
                    "target hidden dimension does not match consumer contract: "
                    f"got {target_hidden.shape[-1]}, expected "
                    f"{len(args.target_layer_ids) * hidden_size}"
                )
            if output_hc_mult > 1 and target_hidden.ndim != 3:
                raise ValueError(
                    f"target hidden must preserve hc_mult={output_hc_mult} as a "
                    f"3-D tensor when averaging is disabled, got {tuple(target_hidden.shape)}"
                )
            if target_hidden.ndim == 3 and target_hidden.shape[1] != output_hc_mult:
                raise ValueError(
                    f"target hidden output hc_mult {target_hidden.shape[1]} != "
                    f"expected {output_hc_mult}"
                )
            writer.write_sample(
                input_ids=tokenized["input_ids"][:seq_len].cpu(),
                attention_mask=tokenized["attention_mask"][:seq_len].cpu(),
                loss_mask=tokenized["loss_mask"][:seq_len].cpu(),
                target_hidden_states=target_hidden.cpu(),
                target_last_hidden_states=last_hidden.cpu(),
            )

            num_valid_samples += 1
            num_processed += 1

            # Log progress
            if num_processed % 10 == 0 or num_processed == local_count:
                elapsed = time.time() - t_start
                rate = num_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[rank {global_rank}] Processed {num_processed}/{local_count} "
                    f"samples ({num_valid_samples} valid, {rate:.2f} samples/s)"
                )

    finally:
        # Clean up
        hook_manager.remove_hooks()
        writer.close()

    logger.info(
        f"[rank {global_rank}] Completed: {num_valid_samples} valid samples out of "
        f"{num_processed} processed"
    )

    # Write local summary
    summary = LocalCacheWriteSummary(
        global_rank=global_rank,
        source_sample_start=local_start,
        source_sample_end=local_end,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))

    # Distributed coordination
    if distributed:
        dist.barrier()

        # Build global shard map (rank 0)
        shard_map = None
        summaries = None
        if global_rank == 0:
            summaries = [
                load_local_cache_write_summary(
                    os.path.join(output_dir, "_tmp", f"rank_{rank}")
                )
                for rank in range(world_size)
            ]
            shard_map, shards = build_global_target_cache_shard_map(summaries)

        # Broadcast shard map
        broadcast_payload = [shard_map]
        dist.broadcast_object_list(broadcast_payload, src=0)
        shard_map = broadcast_payload[0]

        # Rename local shards
        local_summary = load_local_cache_write_summary(rank_dir)
        rename_local_target_cache_shards(
            output_dir=output_dir,
            rank_dir=rank_dir,
            summary=local_summary,
            shard_map=shard_map,
        )

        dist.barrier()

        # Finalize (rank 0)
        if global_rank == 0:
            num_valid_samples = finalize_target_cache_index(
                output_dir=output_dir,
                summaries=summaries,
                shard_map=shard_map,
            )

            # Write manifest
            manifest = {
                "version": 2,
                "num_samples": int(num_valid_samples),
                "num_shards": len(shards),
                "target_layer_ids": args.target_layer_ids,
                "hidden_dtype": "bfloat16",
                "token_dtype": "int32",
                "mask_dtype": "uint8",
                "index_record_size": 56,
                "hidden_size": int(hidden_size),
                "hc_mult": int(output_hc_mult),
                "mean_hc_mult": bool(mean_hc_mult),
                "shards": shards,
                "target_model_name_or_path": str(config.model_config.model_path),
                "source_jsonl_paths": [args.input_file],
                "chat_template": args.chat_template,
                "max_length": args.max_length,
                "min_loss_tokens": args.min_loss_tokens,
            }
            write_target_cache_manifest(output_dir=output_dir, manifest=manifest)

            cleanup_target_cache_tmp_dir(output_dir)

            elapsed = time.time() - t_start
            logger.info("=" * 80)
            logger.info("Feature extraction completed successfully!")
            logger.info(f"  Valid samples: {num_valid_samples}/{total_samples}")
            logger.info(f"  Time elapsed: {elapsed:.1f}s")
            logger.info(f"  Output: {output_dir}")
            logger.info("=" * 80)

        dist.barrier()
        dist.destroy_process_group()

    else:
        # Single-rank: no distributed coordination needed
        logger.info(f"Feature extraction completed: {num_valid_samples} samples → {output_dir}")


if __name__ == "__main__":
    main()
