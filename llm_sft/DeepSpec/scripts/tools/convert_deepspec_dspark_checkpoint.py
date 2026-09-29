#!/usr/bin/env python3
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

"""Convert a DeepSpec DeepSeek-V4 DSpark training checkpoint for CANN inference.

The DeepSpec training checkpoint used by this project stores proposal blocks as
``layers.<stage>.*`` and stores routed-expert weights in their grouped-matmul
runtime layout. It also trains one dense attention output bottleneck, while the
CANN runtime represents the same bottleneck as ``o_groups`` independent input
groups followed by a shared output projection. The CANN DSpark loader expects
proposal blocks under ``mtp.<stage>.*`` and standard per-expert
``w1``/``w2``/``w3`` tensors so that its weight loader can apply
expert/tensor-parallel sharding.

This script performs BF16 layout/name conversion without quantization. It
deliberately drops the training-only ``embed_tokens.weight`` and
``lm_head.weight`` tensors because the CANN proposal model reuses the target
model embedding and LM head. CANN's DSpark Confidence Head is bias-free, so a
legacy training checkpoint's ``confidence_head.proj.bias`` is also dropped.
This compatibility conversion is not numerically lossless for that head.

The output is an independent proposal checkpoint intended for
``speculative_config.draft_model_path``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
_OUTPUT_LAYER_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")
_CONFIDENCE_HEAD_BIAS_RE = re.compile(
    r"^layers\.\d+\.confidence_head\.proj\.bias$"
)
_SHARED_MODEL_WEIGHT_NAMES = (
    "embed.weight",
    "head.weight",
    "embed_tokens.weight",
    "lm_head.weight",
)
_FUSED_MOE_SUFFIXES = {
    "w13": ("mlp.w13_weight", "ffn.w13_weight"),
    "w2": ("mlp.w2_weight", "ffn.w2_weight"),
}
_SHARED_EXPERT_GATE_UP_SUFFIXES = (
    "mlp.shared_experts.gate_up_proj.weight",
    "ffn.shared_experts.gate_up_proj.weight",
)

# These are the tokenizer assets understood by the Hugging Face loaders used
# by CANN.  Keep this list explicit so model weights and unrelated metadata are
# never copied into the converted checkpoint.
_TOKENIZER_FILE_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

_DS_PARK_ALIASES = {
    "dspark_block_size": ("dspark_block_size", "block_size"),
    "dspark_target_layer_ids": (
        "dspark_target_layer_ids",
        "target_layer_ids",
    ),
    "dspark_noise_token_id": (
        "dspark_noise_token_id",
        "mask_token_id",
    ),
    "dspark_markov_rank": ("dspark_markov_rank", "markov_rank"),
}

_SHARED_TARGET_FIELDS = (
    "vocab_size",
    "hidden_size",
    "bos_token_id",
    "eos_token_id",
)

_TARGET_ROPE_FIELDS = (
    "rope_scaling",
    "rope_parameters",
    "rope_theta",
    "compress_rope_theta",
    "max_position_embeddings",
    "max_seq_len",
)

_REQUIRED_LAST_STAGE_SUFFIXES = (
    "norm.weight",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "confidence_head.proj.weight",
)


class ConversionError(RuntimeError):
    """Raised when a checkpoint cannot be converted safely."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"Expected a JSON object in {path}")
    return value


def _resolve_config_path(path: Path) -> Path:
    return path / "config.json" if path.is_dir() else path


def copy_target_tokenizer(
    target_root: Optional[Path], output_dir: Path
) -> list[str]:
    """Copy tokenizer assets beside the converted checkpoint.

    The target root is also where ``config.json`` is loaded from.  A target
    model is expected to provide at least one tokenizer asset; failing here
    prevents a conversion that cannot be loaded by the downstream inference
    stack.
    """
    if target_root is None:
        return []

    copied_files = []
    for file_name in _TOKENIZER_FILE_NAMES:
        source_path = target_root / file_name
        if not source_path.is_file():
            continue
        shutil.copy2(source_path, output_dir / file_name)
        copied_files.append(file_name)

    if not copied_files:
        raise ConversionError(
            "No tokenizer files found in target root: "
            f"{target_root}. Expected one of: {', '.join(_TOKENIZER_FILE_NAMES)}"
        )
    return copied_files


def _first_present(config: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = config.get(name)
        if value is not None:
            return value
    return None


def _as_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConversionError(f"{field} must be a positive integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConversionError(
            f"{field} must be a positive integer, got {value!r}"
        ) from exc
    if result <= 0:
        raise ConversionError(f"{field} must be greater than zero, got {result}")
    return result


def _as_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConversionError(f"{field} must be a non-negative integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConversionError(
            f"{field} must be a non-negative integer, got {value!r}"
        ) from exc
    if result < 0:
        raise ConversionError(f"{field} must be non-negative, got {result}")
    return result


def _check_target_compatibility(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> None:
    mismatches = []
    # The proposal is allowed to use a different number/shape of experts from
    # the target. Only tensors actually shared at runtime (embedding and LM
    # head) impose common vocabulary/hidden dimensions and token semantics.
    for field in _SHARED_TARGET_FIELDS:
        source_value = source.get(field)
        target_value = target.get(field)
        if (
            source_value is not None
            and target_value is not None
            and source_value != target_value
        ):
            mismatches.append(
                f"{field}: training={source_value!r}, target={target_value!r}"
            )
    if mismatches:
        details = "\n  ".join(mismatches)
        raise ConversionError(
            "The training and target configs are incompatible:\n  " + details
        )


def build_output_config(
    source: Mapping[str, Any],
    target: Optional[Mapping[str, Any]],
    num_stages: Optional[int],
    target_num_layers: Optional[int],
) -> tuple[dict[str, Any], int, int]:
    """Build the independent BF16 proposal config and return layer counts."""
    if target is not None:
        _check_target_compatibility(source, target)
        output = dict(target)
        output.update(source)
    else:
        output = dict(source)

    inferred_stages = _first_present(
        source, ("dspark_num_layers", "n_mtp_layers", "num_hidden_layers")
    )
    resolved_stages = _as_positive_int(
        num_stages if num_stages is not None else inferred_stages,
        "n_mtp_layers",
    )

    inferred_target_layers = source.get("num_target_layers")
    if inferred_target_layers is None and target is not None:
        inferred_target_layers = target.get("num_hidden_layers")
    resolved_target_layers = _as_positive_int(
        target_num_layers
        if target_num_layers is not None
        else inferred_target_layers,
        "target num_hidden_layers",
    )

    if target is not None and target.get("num_hidden_layers") is not None:
        target_layers_from_config = _as_positive_int(
            target["num_hidden_layers"], "target config num_hidden_layers"
        )
        if target_layers_from_config != resolved_target_layers:
            raise ConversionError(
                "Target layer count mismatch: "
                f"resolved {resolved_target_layers}, but target config has "
                f"{target_layers_from_config}"
            )

    for output_name, aliases in _DS_PARK_ALIASES.items():
        value = _first_present(source, aliases)
        if value is None and target is not None:
            value = _first_present(target, aliases)
        if value is None:
            raise ConversionError(
                f"Cannot resolve {output_name}; expected one of {aliases}"
            )
        output[output_name] = value

    target_layer_ids = output["dspark_target_layer_ids"]
    if not isinstance(target_layer_ids, list) or not target_layer_ids:
        raise ConversionError("dspark_target_layer_ids must be a non-empty list")
    if target_layer_ids != sorted(set(target_layer_ids)):
        raise ConversionError(
            "dspark_target_layer_ids must be unique and strictly increasing"
        )
    if any(
        isinstance(layer_id, bool)
        or not isinstance(layer_id, int)
        or layer_id < 0
        or layer_id >= resolved_target_layers
        for layer_id in target_layer_ids
    ):
        raise ConversionError(
            "dspark_target_layer_ids must refer to valid target-model layers"
        )

    output["architectures"] = ["DeepseekV4ForCausalLM"]
    output["model_type"] = "deepseek_v4"
    output["num_hidden_layers"] = resolved_target_layers
    output["n_mtp_layers"] = resolved_stages
    output["torch_dtype"] = "bfloat16"
    output["quantization_config"] = None
    output.pop("compression_config", None)
    output.pop("expert_dtype", None)

    # Proposal and target attention must use the same positional encoding.
    # Training configs can use a newer Transformers schema (for example,
    # ``rope_type`` instead of this runtime's ``type``), so the target CANN
    # config is authoritative for all shared RoPE fields.
    if target is not None:
        for field in _TARGET_ROPE_FIELDS:
            if field in target:
                output[field] = target[field]
        for optional_field in (
            "rope_scaling",
            "rope_parameters",
            "compress_rope_theta",
        ):
            if optional_field not in target:
                output.pop(optional_field, None)

    # ``num_hidden_layers`` describes the target-model depth because proposal
    # layers are addressed at ``num_hidden_layers + stage_idx``. Training
    # checkpoints can carry one ``layer_types``/``mlp_layer_types`` entry per
    # proposal stage (and may use training-only values such as ``hash_moe``).
    # Transformers validates both optional lists before the DSpark model can
    # adjust its config, so keep only a complete schedule copied from the
    # target config and drop all incompatible training schedules.
    for schedule_name in ("layer_types", "mlp_layer_types"):
        target_schedule = target.get(schedule_name) if target is not None else None
        if (
            isinstance(target_schedule, list)
            and len(target_schedule) == resolved_target_layers
        ):
            output[schedule_name] = list(target_schedule)
        else:
            output.pop(schedule_name, None)

    # CANN DSpark currently supports only a bias-free Confidence Head. Legacy
    # training bias tensors are dropped during weight conversion below.
    output["dspark_confidence_head_bias"] = False

    # Remove training-only aliases whose meanings conflict with CANN fields.
    output.pop("dspark_num_layers", None)
    output.pop("num_target_layers", None)

    required_ratio_count = resolved_target_layers + resolved_stages
    source_ratios = source.get("compress_ratios")
    target_ratios = target.get("compress_ratios") if target is not None else None
    if isinstance(source_ratios, list) and len(source_ratios) >= required_ratio_count:
        compress_ratios = list(source_ratios[:required_ratio_count])
    elif isinstance(target_ratios, list) and len(target_ratios) >= required_ratio_count:
        compress_ratios = list(target_ratios[:required_ratio_count])
        if isinstance(source_ratios, list) and len(source_ratios) == resolved_stages:
            compress_ratios[-resolved_stages:] = source_ratios
    else:
        # Only entries at num_hidden_layers + stage_idx are consumed by the
        # proposal blocks. DSpark uses the uncompressed sliding-window path, so
        # one is the correct fallback for these entries.
        prefix = (
            list(target_ratios[:resolved_target_layers])
            if isinstance(target_ratios, list)
            and len(target_ratios) >= resolved_target_layers
            else [1] * resolved_target_layers
        )
        proposal_ratios = (
            list(source_ratios)
            if isinstance(source_ratios, list)
            and len(source_ratios) == resolved_stages
            else [1] * resolved_stages
        )
        compress_ratios = prefix + proposal_ratios
    output["compress_ratios"] = compress_ratios

    return output, resolved_stages, resolved_target_layers


def discover_safetensor_files(input_path: Path) -> list[Path]:
    """Discover source shards in stable order."""
    if input_path.is_file():
        if input_path.suffix != ".safetensors":
            raise ConversionError(f"Input file is not a safetensors file: {input_path}")
        return [input_path]

    index_path = input_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ConversionError(f"Invalid weight_map in {index_path}")
        file_names = list(dict.fromkeys(weight_map.values()))
        files = [input_path / str(name) for name in file_names]
    else:
        single_file = input_path / "model.safetensors"
        files = [single_file] if single_file.is_file() else sorted(input_path.glob("*.safetensors"))

    if not files:
        raise ConversionError(f"No safetensors files found under {input_path}")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ConversionError("Missing safetensors shard(s): " + ", ".join(missing))
    return files


def inspect_source_keys(files: Iterable[Path]) -> set[str]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ConversionError(
            "safetensors is required; install it with: pip install safetensors"
        ) from exc

    keys: set[str] = set()
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in keys:
                    raise ConversionError(f"Duplicate tensor key in input shards: {key}")
                keys.add(key)
    return keys


def transform_key(name: str, num_stages: int) -> Optional[str]:
    """Return the output name for a non-MoE source key."""
    if name in _SHARED_MODEL_WEIGHT_NAMES:
        return None

    match = _LAYER_RE.match(name)
    if match is None:
        raise ConversionError(
            f"Unsupported training tensor name {name!r}; expected layers.<stage>.*"
        )

    stage_idx = int(match.group(1))
    suffix = match.group(2)
    if stage_idx >= num_stages:
        raise ConversionError(
            f"Tensor {name!r} uses stage {stage_idx}, but n_mtp_layers={num_stages}"
        )

    if suffix == "confidence_head.proj.bias":
        return None

    if suffix in (*_FUSED_MOE_SUFFIXES["w13"], *_FUSED_MOE_SUFFIXES["w2"]):
        raise ConversionError(
            f"Internal error: fused MoE tensor {name!r} must be expanded first"
        )
    if suffix in _SHARED_EXPERT_GATE_UP_SUFFIXES:
        raise ConversionError(
            f"Internal error: fused shared-expert tensor {name!r} "
            "must be expanded first"
        )

    if suffix == "mlp.gate.e_score_correction_bias":
        suffix = "ffn.gate.bias"
    elif suffix == "ffn.gate.e_score_correction_bias":
        suffix = "ffn.gate.bias"
    elif suffix in (
        "mlp.shared_experts.down_proj.weight",
        "ffn.shared_experts.down_proj.weight",
    ):
        suffix = "ffn.shared_experts.w2.weight"
    elif suffix.startswith("mlp."):
        suffix = "ffn." + suffix.removeprefix("mlp.")

    return f"mtp.{stage_idx}.{suffix}"


def _validate_moe_shape(
    source_name: str,
    shape: tuple[int, ...],
    config: Mapping[str, Any],
) -> None:
    if len(shape) != 3:
        raise ConversionError(
            f"{source_name} must be rank 3, got shape {list(shape)}"
        )
    experts = _as_positive_int(config.get("n_routed_experts"), "n_routed_experts")
    hidden = _as_positive_int(config.get("hidden_size"), "hidden_size")
    intermediate = _as_positive_int(
        config.get("moe_intermediate_size"), "moe_intermediate_size"
    )
    if source_name.endswith(_FUSED_MOE_SUFFIXES["w13"]):
        expected = (experts, hidden, 2 * intermediate)
    else:
        expected = (experts, intermediate, hidden)
    if shape != expected:
        raise ConversionError(
            f"Unexpected training layout for {source_name}: got {list(shape)}, "
            f"expected {list(expected)}"
        )


def _fused_moe_kind(name: str, num_stages: int) -> Optional[tuple[int, str]]:
    match = _LAYER_RE.match(name)
    if match is None:
        return None
    stage_idx = int(match.group(1))
    if stage_idx >= num_stages:
        raise ConversionError(
            f"Tensor {name!r} uses stage {stage_idx}, but n_mtp_layers={num_stages}"
        )
    suffix = match.group(2)
    if suffix in _FUSED_MOE_SUFFIXES["w13"]:
        return stage_idx, "w13"
    if suffix in _FUSED_MOE_SUFFIXES["w2"]:
        return stage_idx, "w2"
    return None


def _shared_expert_gate_up_stage(name: str, num_stages: int) -> Optional[int]:
    match = _LAYER_RE.match(name)
    if match is None:
        return None
    stage_idx = int(match.group(1))
    if stage_idx >= num_stages:
        raise ConversionError(
            f"Tensor {name!r} uses stage {stage_idx}, but n_mtp_layers={num_stages}"
        )
    return stage_idx if match.group(2) in _SHARED_EXPERT_GATE_UP_SUFFIXES else None


def expand_fused_moe_tensor(
    source_name: str,
    tensor: Any,
    config: Mapping[str, Any],
    num_stages: int,
) -> Iterable[tuple[str, Any]]:
    """Split one DeepSpec fused MoE tensor into CANN checkpoint tensors.

    DeepSpec stores w13 as [expert, hidden, gate+up] and w2 as
    [expert, intermediate, hidden].  PyTorch checkpoint-side Linear weights are
    [out, in], so CANN expects w1/w3 [intermediate, hidden] and
    w2 [hidden, intermediate].
    """
    kind_info = _fused_moe_kind(source_name, num_stages)
    if kind_info is None:
        raise ConversionError(f"Not a fused MoE tensor: {source_name}")
    stage_idx, kind = kind_info
    _validate_moe_shape(source_name, tuple(tensor.shape), config)

    experts = _as_positive_int(config.get("n_routed_experts"), "n_routed_experts")
    intermediate = _as_positive_int(
        config.get("moe_intermediate_size"), "moe_intermediate_size"
    )
    for expert_idx in range(experts):
        prefix = f"mtp.{stage_idx}.ffn.experts.{expert_idx}"
        if kind == "w13":
            expert_weight = tensor[expert_idx]
            gate = expert_weight[:, :intermediate]
            up = expert_weight[:, intermediate:]
            yield f"{prefix}.w1.weight", gate.transpose(0, 1).contiguous()
            yield f"{prefix}.w3.weight", up.transpose(0, 1).contiguous()
        else:
            yield (
                f"{prefix}.w2.weight",
                tensor[expert_idx].transpose(0, 1).contiguous(),
            )


def expand_shared_expert_gate_up_tensor(
    source_name: str,
    tensor: Any,
    config: Mapping[str, Any],
    num_stages: int,
) -> Iterable[tuple[str, Any]]:
    """Split training's fused shared SwiGLU projection into CANN w1/w3."""
    stage_idx = _shared_expert_gate_up_stage(source_name, num_stages)
    if stage_idx is None:
        raise ConversionError(f"Not a fused shared-expert tensor: {source_name}")
    hidden = _as_positive_int(config.get("hidden_size"), "hidden_size")
    intermediate = _as_positive_int(
        config.get("moe_intermediate_size"), "moe_intermediate_size"
    )
    expected = (2 * intermediate, hidden)
    if tuple(tensor.shape) != expected:
        raise ConversionError(
            f"Unexpected training layout for {source_name}: got {list(tensor.shape)}, "
            f"expected {list(expected)}"
        )

    prefix = f"mtp.{stage_idx}.ffn.shared_experts"
    yield f"{prefix}.w1.weight", tensor[:intermediate].contiguous()
    yield f"{prefix}.w3.weight", tensor[intermediate:].contiguous()


def transform_attention_output_projection(
    source_name: str,
    tensor: Any,
    config: Mapping[str, Any],
) -> Optional[Any]:
    """Convert DeepSpec's dense output bottleneck to CANN's grouped layout.

    DeepSpec computes::

        y = W_b @ (W_a @ concat(x_0, ..., x_{G-1}))

    CANN computes each ``W_a`` input slice independently, concatenates the
    results, and applies one larger ``W_b``. Splitting ``W_a`` by input columns
    and repeating ``W_b`` by input columns is therefore algebraically exact::

        y = [W_b ... W_b] @ concat(W_a_0 @ x_0, ..., W_a_G @ x_G)

    Return ``None`` for tensors that are not attention ``wo_a``/``wo_b``.
    """
    match = _LAYER_RE.match(source_name)
    if match is None:
        return None
    suffix = match.group(2)
    if suffix not in ("attn.wo_a.weight", "attn.wo_b.weight"):
        return None

    num_heads = _as_positive_int(
        config.get("num_attention_heads"), "num_attention_heads"
    )
    head_dim = _as_positive_int(config.get("head_dim"), "head_dim")
    o_lora_rank = _as_positive_int(config.get("o_lora_rank"), "o_lora_rank")
    o_groups = _as_positive_int(config.get("o_groups"), "o_groups")
    hidden_size = _as_positive_int(config.get("hidden_size"), "hidden_size")
    dense_input_size = num_heads * head_dim
    if dense_input_size % o_groups != 0:
        raise ConversionError(
            "num_attention_heads * head_dim must be divisible by o_groups: "
            f"{dense_input_size} vs {o_groups}"
        )
    group_input_size = dense_input_size // o_groups

    if suffix == "attn.wo_a.weight":
        expected = (o_lora_rank, dense_input_size)
        if tuple(tensor.shape) != expected:
            raise ConversionError(
                f"Unexpected DeepSpec wo_a shape for {source_name}: "
                f"got {list(tensor.shape)}, expected {list(expected)}"
            )
        return (
            tensor.reshape(o_lora_rank, o_groups, group_input_size)
            .permute(1, 0, 2)
            .contiguous()
            .reshape(o_groups * o_lora_rank, group_input_size)
        )

    expected = (hidden_size, o_lora_rank)
    if tuple(tensor.shape) != expected:
        raise ConversionError(
            f"Unexpected DeepSpec wo_b shape for {source_name}: "
            f"got {list(tensor.shape)}, expected {list(expected)}"
        )
    return tensor.repeat(1, o_groups).contiguous()


def validate_output_keys(
    keys: set[str],
    num_stages: int,
    num_experts: int,
    num_shared_experts: int = 1,
) -> None:
    stages = {
        int(match.group(1))
        for name in keys
        if (match := _OUTPUT_LAYER_RE.match(name)) is not None
    }
    expected_stages = set(range(num_stages))
    if stages != expected_stages:
        raise ConversionError(
            f"Output stages are {sorted(stages)}, expected {sorted(expected_stages)}"
        )

    required = {
        "mtp.0.main_proj.weight",
        "mtp.0.main_norm.weight",
    }
    last_stage = num_stages - 1
    required.update(
        f"mtp.{last_stage}.{suffix}" for suffix in _REQUIRED_LAST_STAGE_SUFFIXES
    )
    missing = sorted(required - keys)
    if missing:
        raise ConversionError(
            "Required CANN DSpark tensor(s) missing after conversion: "
            + ", ".join(missing)
        )

    confidence_biases = sorted(
        name for name in keys if name.endswith(".confidence_head.proj.bias")
    )
    if confidence_biases:
        raise ConversionError(
            "CANN DSpark requires a bias-free Confidence Head, but the output "
            "contains: "
            + ", ".join(confidence_biases)
        )

    if num_shared_experts < 0:
        raise ConversionError(
            f"n_shared_experts must be non-negative, got {num_shared_experts}"
        )

    missing_expert_keys = []
    for stage_idx in range(num_stages):
        stage_prefix = f"mtp.{stage_idx}.ffn"
        for key in (f"{stage_prefix}.gate.weight", f"{stage_prefix}.gate.bias"):
            if key not in keys:
                missing_expert_keys.append(key)
        if num_shared_experts > 0:
            for projection in ("w1", "w2", "w3"):
                key = f"{stage_prefix}.shared_experts.{projection}.weight"
                if key not in keys:
                    missing_expert_keys.append(key)
        for expert_idx in range(num_experts):
            for projection in ("w1", "w2", "w3"):
                key = (
                    f"mtp.{stage_idx}.ffn.experts.{expert_idx}."
                    f"{projection}.weight"
                )
                if key not in keys:
                    missing_expert_keys.append(key)
    if missing_expert_keys:
        preview = ", ".join(missing_expert_keys[:10])
        suffix = " ..." if len(missing_expert_keys) > 10 else ""
        raise ConversionError(
            f"Missing {len(missing_expert_keys)} CANN MoE tensor(s): "
            + preview
            + suffix
        )


class ShardWriter:
    """Write tensors incrementally so the whole checkpoint is never resident."""

    def __init__(self, directory: Path, max_shard_bytes: int):
        self.directory = directory
        self.max_shard_bytes = max_shard_bytes
        self.buffer: dict[str, Any] = {}
        self.buffer_bytes = 0
        self.parts: list[tuple[Path, list[str]]] = []
        self.total_size = 0

    def ensure_capacity(self, tensor_bytes: int) -> None:
        if self.buffer and self.buffer_bytes + tensor_bytes > self.max_shard_bytes:
            self.flush()

    def add(self, name: str, tensor: Any) -> None:
        tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
        self.ensure_capacity(tensor_bytes)
        self.buffer[name] = tensor
        self.buffer_bytes += tensor_bytes
        self.total_size += tensor_bytes
        if self.buffer_bytes >= self.max_shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ConversionError(
                "safetensors is required; install it with: pip install safetensors"
            ) from exc

        part_number = len(self.parts) + 1
        path = self.directory / f"part-{part_number:05d}.safetensors"
        keys = list(self.buffer)
        save_file(self.buffer, str(path), metadata={"format": "pt"})
        self.parts.append((path, keys))
        self.buffer = {}
        self.buffer_bytes = 0

    def finalize(self) -> dict[str, str]:
        self.flush()
        if not self.parts:
            raise ConversionError("No tensors were written")

        weight_map: dict[str, str] = {}
        part_count = len(self.parts)
        for part_idx, (old_path, keys) in enumerate(self.parts, start=1):
            if part_count == 1:
                new_name = "model.safetensors"
            else:
                new_name = f"model-{part_idx:05d}-of-{part_count:05d}.safetensors"
            new_path = self.directory / new_name
            old_path.rename(new_path)
            for key in keys:
                weight_map[key] = new_name

        if part_count > 1:
            index = {
                "metadata": {"total_size": self.total_size},
                "weight_map": weight_map,
            }
            with (self.directory / "model.safetensors.index.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(index, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        return weight_map


def convert_weights(
    files: Iterable[Path],
    output_dir: Path,
    config: Mapping[str, Any],
    num_stages: int,
    max_shard_bytes: int,
) -> tuple[dict[str, str], Counter[str], list[str], Optional[float]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ConversionError(
            "safetensors is required; install it with: pip install safetensors"
        ) from exc

    writer = ShardWriter(output_dir, max_shard_bytes)
    output_keys: set[str] = set()
    dtype_counts: Counter[str] = Counter()
    dropped: list[str] = []
    dropped_confidence_head_bias: Optional[float] = None

    for source_file in files:
        print(f"Reading {source_file}", flush=True)
        with safe_open(str(source_file), framework="pt", device="cpu") as handle:
            for source_name in handle.keys():
                moe_kind = _fused_moe_kind(source_name, num_stages)
                if moe_kind is not None:
                    tensor = handle.get_tensor(source_name)
                    for output_name, expert_tensor in expand_fused_moe_tensor(
                        source_name, tensor, config, num_stages
                    ):
                        if output_name in output_keys:
                            raise ConversionError(
                                f"Duplicate output tensor name: {output_name}"
                            )
                        writer.add(output_name, expert_tensor)
                        output_keys.add(output_name)
                        dtype_counts[str(expert_tensor.dtype)] += 1
                    print(
                        f"  {source_name} -> per-expert w1/w3"
                        if moe_kind[1] == "w13"
                        else f"  {source_name} -> per-expert w2",
                        flush=True,
                    )
                    continue

                if _shared_expert_gate_up_stage(source_name, num_stages) is not None:
                    tensor = handle.get_tensor(source_name)
                    for output_name, expert_tensor in expand_shared_expert_gate_up_tensor(
                        source_name, tensor, config, num_stages
                    ):
                        if output_name in output_keys:
                            raise ConversionError(
                                f"Duplicate output tensor name: {output_name}"
                            )
                        writer.add(output_name, expert_tensor)
                        output_keys.add(output_name)
                        dtype_counts[str(expert_tensor.dtype)] += 1
                    print(
                        f"  {source_name} -> shared-expert w1/w3",
                        flush=True,
                    )
                    continue

                output_name = transform_key(source_name, num_stages)
                if output_name is None:
                    if _CONFIDENCE_HEAD_BIAS_RE.fullmatch(source_name):
                        if dropped_confidence_head_bias is not None:
                            raise ConversionError(
                                "Expected one Confidence Head bias tensor, but found "
                                f"another one: {source_name}"
                            )
                        tensor = handle.get_tensor(source_name)
                        if tensor.numel() != 1:
                            raise ConversionError(
                                "confidence_head.proj.bias must contain one value, "
                                f"got shape {list(tensor.shape)} for {source_name}"
                            )
                        dropped_confidence_head_bias = float(
                            tensor.reshape(-1)[0].item()
                        )
                        print(
                            f"  {source_name} -> dropped "
                            "(CANN requires confidence_head bias=False)",
                            flush=True,
                        )
                    dropped.append(source_name)
                    continue
                if output_name in output_keys:
                    raise ConversionError(f"Duplicate output tensor name: {output_name}")

                tensor = handle.get_tensor(source_name)
                output_projection = transform_attention_output_projection(
                    source_name, tensor, config
                )
                if output_projection is not None:
                    tensor = output_projection
                dtype_counts[str(tensor.dtype)] += 1
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()

                writer.add(output_name, tensor)
                output_keys.add(output_name)
                print(
                    f"  {source_name} -> {output_name} "
                    f"shape={list(tensor.shape)} dtype={tensor.dtype}",
                    flush=True,
                )

    num_experts = _as_positive_int(config.get("n_routed_experts"), "n_routed_experts")
    configured_shared_experts = config.get("n_shared_experts")
    num_shared_experts = _as_nonnegative_int(
        1 if configured_shared_experts is None else configured_shared_experts,
        "n_shared_experts",
    )
    validate_output_keys(
        output_keys,
        num_stages,
        num_experts,
        num_shared_experts=num_shared_experts,
    )
    weight_map = writer.finalize()
    if set(weight_map) != output_keys:
        raise ConversionError("Internal error: output index does not match converted keys")
    return weight_map, dtype_counts, dropped, dropped_confidence_head_bias


def verify_written_checkpoint(output_dir: Path, expected_keys: set[str]) -> None:
    files = discover_safetensor_files(output_dir)
    actual_keys = inspect_source_keys(files)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ConversionError(
            f"Written checkpoint verification failed; missing={missing}, extra={extra}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a BF16 DeepSpec DeepSeek-V4 DSpark training checkpoint "
            "to an independent CANN DSpark proposal checkpoint."
        )
    )
    parser.add_argument(
        "--input-path",
        required=True,
        type=Path,
        help="Training checkpoint directory or model.safetensors file.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="New output directory. It must not already exist.",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        help="Training config.json. Defaults to <input directory>/config.json.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        help=(
            "Optional target-model root directory. Its config.json is loaded, "
            "shared dimensions are checked, and runtime fields/tokenizer files "
            "are copied from it."
        ),
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        help="Override proposal stage count (normally inferred as 3).",
    )
    parser.add_argument(
        "--target-num-layers",
        type=int,
        help="Override target depth (normally inferred from num_target_layers=43).",
    )
    parser.add_argument(
        "--max-shard-size-gib",
        type=float,
        default=2.0,
        help="Approximate output shard size in GiB (default: 2).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not input_path.exists():
        raise ConversionError(f"Input path does not exist: {input_path}")
    if output_path.exists():
        raise ConversionError(
            f"Output path already exists: {output_path}. Use a new directory."
        )
    if args.max_shard_size_gib <= 0 or not math.isfinite(args.max_shard_size_gib):
        raise ConversionError("--max-shard-size-gib must be a finite positive number")

    if args.source_config is not None:
        source_config_path = _resolve_config_path(args.source_config.expanduser().resolve())
    else:
        input_dir = input_path if input_path.is_dir() else input_path.parent
        source_config_path = input_dir / "config.json"
    source_config = _load_json(source_config_path)

    target_config = None
    target_root = None
    target_config_path = None
    if args.target_root is not None:
        target_root = args.target_root.expanduser().resolve()
        if not target_root.is_dir():
            raise ConversionError(f"Target root is not a directory: {target_root}")
        target_config_path = target_root / "config.json"
        target_config = _load_json(target_config_path)

    source_files = discover_safetensor_files(input_path)
    source_keys = inspect_source_keys(source_files)
    source_confidence_bias_keys = sorted(
        key for key in source_keys if _CONFIDENCE_HEAD_BIAS_RE.fullmatch(key)
    )
    if len(source_confidence_bias_keys) > 1:
        raise ConversionError(
            "Expected at most one Confidence Head bias tensor, found: "
            + ", ".join(source_confidence_bias_keys)
        )
    source_has_confidence_bias = bool(source_confidence_bias_keys)

    output_config, num_stages, target_layers = build_output_config(
        source=source_config,
        target=target_config,
        num_stages=args.num_stages,
        target_num_layers=args.target_num_layers,
    )

    max_shard_bytes = int(args.max_shard_size_gib * 1024**3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        (
            weight_map,
            output_dtype_counts,
            dropped,
            dropped_confidence_head_bias,
        ) = convert_weights(
            files=source_files,
            output_dir=temporary_path,
            config=output_config,
            num_stages=num_stages,
            max_shard_bytes=max_shard_bytes,
        )
        with (temporary_path / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(output_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        copied_tokenizer_files = copy_target_tokenizer(
            target_root=target_root,
            output_dir=temporary_path,
        )

        conversion_info = {
            "format": "cann-deepseek-v4-dspark-bf16-proposal",
            "source_path": str(input_path),
            "source_config": str(source_config_path),
            "target_root": str(target_root) if target_root else None,
            "target_config": str(target_config_path) if target_config_path else None,
            "target_tokenizer_files": copied_tokenizer_files,
            "n_mtp_layers": num_stages,
            "target_num_hidden_layers": target_layers,
            "converted_tensor_count": len(weight_map),
            "dropped_tensors": dropped,
            "output_tensor_dtype_counts": dict(output_dtype_counts),
            "moe_transform": (
                "split fused w13 into per-expert w1/w3 and transpose; "
                "split fused w2 into per-expert w2 and transpose"
            ),
            "attention_output_projection_transform": (
                "split dense wo_a input columns into o_groups and stack them "
                "on its output dimension; repeat shared wo_b across its input "
                "dimension"
            ),
            "source_confidence_head_bias": source_has_confidence_bias,
            "output_confidence_head_bias": False,
            "dropped_confidence_head_bias": dropped_confidence_head_bias,
        }
        with (temporary_path / "conversion_info.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(conversion_info, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        verify_written_checkpoint(temporary_path, set(weight_map))
        os.rename(temporary_path, output_path)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise

    print("\nConversion completed successfully.")
    print(f"Output: {output_path}")
    print(f"Proposal stages: {num_stages}")
    print(f"Target layers: {target_layers}")
    print(f"Converted tensors: {len(weight_map)}")
    print(f"Dropped tensors: {dropped}")
    print(f"Output tensor dtypes: {dict(output_dtype_counts)}")
    if source_has_confidence_bias:
        print(
            "WARNING: a legacy Confidence Head bias was dropped because CANN "
            "requires bias=False. See conversion_info.json for its value; "
            "confidence probabilities are not numerically equivalent.",
            file=sys.stderr,
        )
    print(
        "IMPORTANT: use this directory as speculative_config.draft_model_path, "
        "with dspark_confidence_head_bias=false."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
