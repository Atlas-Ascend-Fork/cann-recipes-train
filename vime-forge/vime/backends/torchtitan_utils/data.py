# Adapted from
# https://github.com/vllm-project/vime/blob/5ab8cc75/vime/backends/megatron_utils/data.py
# which is derived from THUDM's slime at:
# https://github.com/THUDM/slime/blob/v0.3.1/slime/backends/megatron_utils/data.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 Zhipu AI
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

import logging
import math
from argparse import Namespace
from typing import NamedTuple
from collections.abc import Sequence
from numbers import Number

import numpy as np
import torch
import torch.distributed as dist

from vime.observability import logging_utils, train_metric_utils
from vime.utils.distributed_utils import get_dp_gloo_group, get_dp_src_rank
from vime.utils.flops_utils import calculate_fwd_flops
from vime.observability.metric_utils import compute_pass_rate, compute_rollout_step
from vime.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


class CPResponseIndexSpan(NamedTuple):
    """Typed local-logit/global-target span for one sample's response."""

    local_logit_start: int
    global_target_start: int
    length: int


def build_cp_response_index_spans(response_offsets, seq_len: int, cp_degree: int, cp_rank: int):
    """Build response spans from the contiguous global token range owned by one CP rank."""
    seq_len, cp_degree, cp_rank = int(seq_len), int(cp_degree), int(cp_rank)
    if cp_degree < 1 or not 0 <= cp_rank < cp_degree:
        raise ValueError("invalid CP degree/rank")
    if seq_len % cp_degree:
        raise ValueError(f"sequence length {seq_len} is not divisible by CP degree {cp_degree}")
    width = seq_len // cp_degree
    owned_start, owned_end = cp_rank * width, (cp_rank + 1) * width
    result = []
    for start, end in response_offsets:
        overlap_start = max(owned_start, start - 1)
        overlap_end = min(owned_end, end - 1)
        if overlap_start >= overlap_end:
            result.append(
                CPResponseIndexSpan(
                    local_logit_start=0,
                    global_target_start=start,
                    length=0,
                )
            )
        else:
            result.append(
                CPResponseIndexSpan(
                    local_logit_start=overlap_start - owned_start,
                    global_target_start=overlap_start + 1,
                    length=overlap_end - overlap_start,
                )
            )
    return result


def slice_fields_by_global_targets(fields, response_offsets, cp_response_index_spans):
    """Slice response-shaped RL fields using each rank's contiguous target span."""
    out = []
    for value, (start, end), spec in zip(fields, response_offsets, cp_response_index_spans, strict=True):
        start_i, end_i = int(start), int(end)
        target_start = spec.global_target_start
        length = spec.length
        target_end = target_start + length
        if length and (target_start < start_i or target_end > end_i):
            raise ValueError("global target is outside the response interval")
        if value is None:
            out.append(None)
            continue
        relative_start = target_start - start_i
        relative_end = relative_start + length
        size = value.numel() if isinstance(value, torch.Tensor) else len(value)
        if size != end_i - start_i:
            raise ValueError("RL field length does not match the sample response length")
        out.append(value[relative_start:relative_end] if length else value[:0])
    return out


def restore_global_response_fields(fields, response_offsets, cp_response_index_spans, cp_group=None):
    """Restore CP-local response tensors with one collective for the whole micro-batch."""
    full_fields = []
    flat_parts = []
    split_lengths = []
    for value, (start, end), spec in zip(fields, response_offsets, cp_response_index_spans, strict=True):
        start_i, end_i = int(start), int(end)
        response_length = end_i - start_i
        if value is None:
            full_fields.append(None)
            # full_fields might have form [Tensor, None, Tensor,...]
            continue
        local_length = spec.length
        if not isinstance(value, torch.Tensor) or value.numel() != local_length:
            raise ValueError("CP response field length does not match local target count")
        target_start = spec.global_target_start
        target_end = target_start + local_length
        if local_length and (target_start < start_i or target_end > end_i):
            raise ValueError("global target is outside the response interval")
        full = value.new_zeros((response_length,))
        if local_length:
            relative_start = target_start - start_i
            full[relative_start : relative_start + local_length] = value
        full_fields.append(full)
        flat_parts.append(full)
        split_lengths.append(response_length)

    if flat_parts and cp_group is not None and dist.is_initialized() and dist.get_world_size(group=cp_group) > 1:
        flat = torch.cat(flat_parts)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=cp_group)
        restored_parts = list(flat.split(split_lengths))
        part_index = 0
        for index, value in enumerate(full_fields):
            if value is not None:
                full_fields[index] = restored_parts[part_index]
                part_index += 1
    return full_fields


# ---------------------------------------------------------------------------
# DP-reduce helpers, free of parallel-state calls.
# ---------------------------------------------------------------------------


def _gather_and_reduce_log_dict(
    log_dict: dict,
    *,
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    dp_src_rank: int,
    dp_group,
) -> dict | None:
    """Gather per-rank log_dicts across ``dp_group``, reduce on ``dp_src_rank``, and log.

    A ``(sum, count)`` value reduces to ``Σsum / Σcount``, a plain one to ``Σ / dp_size``. Returns
    the reduced and prefixed dict on ``dp_src_rank``, None elsewhere. ⚠ dp_group must span exactly
    the ranks that gather (see init_dp_gloo_group); the global group deadlocks at tp>1.
    """
    dp_size = dist.get_world_size(group=dp_group)
    if dist.get_rank() == dp_src_rank:
        gathered = [None] * dp_size
        dist.gather_object(log_dict, gathered, dst=dp_src_rank, group=dp_group)
        reduced: dict = {}
        for key in log_dict:
            values = [d[key] for d in gathered]
            first = values[0]
            if isinstance(first, tuple) and len(first) == 2:
                total_sum = sum(v[0] for v in values)
                total_count = sum(v[1] for v in values)
                reduced[key] = total_sum / total_count if total_count else 0.0
            else:
                reduced[key] = sum(values) / dp_size
        reduced_log_dict = {f"{metric_name}/{k}": v for k, v in reduced.items()}
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        logging_utils.log(args, reduced_log_dict, step_key="rollout/step")
        return reduced_log_dict
    dist.gather_object(log_dict, None, dst=dp_src_rank, group=dp_group)
    return None


def _rollout_log_metric_contribution(
    per_rank_reducer_sum: float,
    *,
    num_rollouts_in_rollout: int,
    dp_size: int,
) -> tuple[float, float]:
    """Return ``(sum, count)`` for the gather step of a per-rollout-mean metric.

    CP ranks already carry the same restored full-response fields and are members
    of the gather group.
    The gather reduces to sum_DP_full / num_rollouts, matching train_one_step.
    """
    return per_rank_reducer_sum, num_rollouts_in_rollout / dp_size


def _get_sum_of_sample_mean(
    total_lengths,
    response_lengths,
    loss_masks,
    sample_denoms=None,
    **_kwargs,
):
    """Per-rollout-mean reducer: returns ``f(x) -> Σ_i (x_i * mask_i).sum() /
    clamp(denom_i, 1)``."""
    if sample_denoms is None:
        sample_denoms = [m.sum() for m in loss_masks]

    def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
        return sum(
            (x_i * loss_mask_i).sum() / torch.clamp_min(denom, 1)
            for x_i, loss_mask_i, denom in zip(
                x.split(response_lengths, dim=0), loss_masks, sample_denoms, strict=False
            )
        )

    return sum_of_sample_mean


class DataIterator:
    """Iterator over a rollout dict following an explicit micro-batch index schedule."""

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_indices: list[list[int]],
    ) -> None:
        """Iterate ``rollout_data`` (per-sample fields for this DP rank) in the micro-batches
        ``micro_batch_indices`` selects by local sample index."""
        self.rollout_data = rollout_data
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch for the requested keys.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        indices = self.micro_batch_indices[self.offset]
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                batch[key] = [vals[i] for i in indices]
        self.offset += 1
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def get_data_iterator(rollout_data: RolloutBatch) -> list[DataIterator]:
    # torchtitan has no VPP; return a length-1 list preserving the outer list structure for indexed access.
    return [DataIterator(rollout_data, rollout_data["micro_batch_indices"])]


def build_packed_microbatch(
    tokens_list,
    device,
    response_lengths=None,
    pad_to_multiple=1,
):
    """Pack the sequences of one micro-batch into a single flat token stream (THD).
    Round the global stream up to pad_to_multiple.

    Returns dict:
      - packed_ids:   LongTensor[T]      -- all sequences concatenated, plus the optional tail pad
      - cu_seqlens:   IntTensor[n(+1)+1] -- cumulative boundaries with the leading 0; used to slice
                                            each sample's response segment out of the logits
      - position_ids: LongTensor[T]      -- position reset to 0..L-1 per sequence. get_attention_masks
                                            reads every reset to 0 as a document start, so the tail
                                            pad carries 0..pad-1 (one dummy document) rather than
                                            zeros, which would read as pad-many 1-token documents.
      - seqlens:      list[int]          -- per-sequence length, tail pad segment included
      - response_offsets: optional global half-open target intervals, one per sample
    """
    if pad_to_multiple is None or int(pad_to_multiple) < 1:
        raise ValueError("pad_to_multiple must be a positive integer")
    seqlens = [int(t.numel()) for t in tokens_list]
    ids = torch.cat([t.to(device=device, dtype=torch.long).view(-1) for t in tokens_list])
    positions = torch.cat([torch.arange(L, device=device, dtype=torch.long) for L in seqlens])
    T = int(ids.numel())
    pad = (-T) % int(pad_to_multiple)
    if pad:
        ids = torch.cat([ids, torch.zeros(pad, dtype=torch.long, device=device)])
        positions = torch.cat([positions, torch.arange(pad, device=device, dtype=torch.long)])
        seqlens = seqlens + [pad]  # tail pad is a dummy document, excluded from loss
    cu = [0]
    for L in seqlens:
        cu.append(cu[-1] + L)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    result = {
        "packed_ids": ids,
        "cu_seqlens": cu_seqlens,
        "position_ids": positions,
        "seqlens": seqlens,
    }
    if response_lengths is not None:
        offsets = []
        start = 0
        for length, response_length in zip(seqlens[: len(response_lengths)], response_lengths, strict=True):
            offsets.append((start + length - int(response_length), start + length))
            start += length
        result["response_offsets"] = offsets
    return result


def get_batch(
    args,
    data_iterator: "DataIterator",
    parallel_dims,
    device,
    keys: Sequence[str] | None = None,
    *,
    sequence_padding_multiple: int = 1,
) -> dict:
    """Fetch a micro-batch from the iterator then build a packed micro-batch and
    response metadata for contiguous CP shards.

    Required input fields are tokens, total_lengths, response_lengths and loss_masks.
    Optional requested fields are selected with the same sample indices. returns is
    processed when explicitly requested; it is not included in the default keys.

    Loss targets are selected from packed_ids using response offsets.

    Return the global packed tensors from build_packed_microbatch. With CP > 1,
    add cp_response_index_spans, local_response_lengths and cp_group.
    Slice response-shaped RL fields, including teacher_log_probs, by the targets owned
    by this rank's logits. Unknown response-shaped fields raise an error.

    total_lengths, response_lengths and sample denominators remain global per-sample
    metadata. packed_ids remains available here for global target-to-token lookup.
    This layout requires context_parallel_load_balancer=None.
    BSHD and multimodal processing are not supported.
    """
    if keys is None:
        keys = [
            "tokens", "loss_masks", "total_lengths", "response_lengths",
            "log_probs", "advantages", "rollout_mask_sums", "ref_log_probs",
            "rollout_log_probs",
            # Global sample indices name the per-sample quantities the precision guard
            # records. They are what makes those records comparable across backends and
            # parallel layouts, where rank-local positions differ. get_next returns None for
            # a missing key, so rollout_data without it is fine.
            "sample_indices",
        ]

    assert "tokens" in keys
    batch = data_iterator.get_next(keys)

    tokens_list = batch.pop("tokens")  # list[LongTensor[L_i]], one per sample
    # qkv_format='bshd' is a separate (unsupported) branch kept as a stub.
    if getattr(args, "qkv_format", "thd") == "bshd":
        # todo: bshd not supported
        raise ValueError("qkv_format='bshd' is not supported by the torchtitan backend")

    # Combine model-required alignment with the configured CP/TP alignment.
    # Each contiguous CP shard must satisfy the model's sequence padding multiple.
    # Tail padding remains a dummy document outside the sample response loss.
    packed_seq_alignment = int(getattr(args, "packed_seq_alignment", 1))
    response_lengths = batch.get("response_lengths")
    cp_size = max(int(parallel_dims.cp), 1)
    tp_size = max(int(parallel_dims.tp), 1)
    pad_to_multiple = cp_size * math.lcm(
        sequence_padding_multiple,
        packed_seq_alignment * tp_size,
    )
    packed_mb = build_packed_microbatch(
        tokens_list,
        device,
        response_lengths=response_lengths,
        pad_to_multiple=pad_to_multiple,
    )
    batch.update(packed_mb)  # adds packed_ids, cu_seqlens, position_ids, seqlens
    # Model inputs are sliced in the engine; response fields are sliced below
    if cp_size > 1:
        padded_length = packed_mb["packed_ids"].numel()
        cp_mesh = parallel_dims.get_mesh("cp")
        batch["cp_response_index_spans"] = build_cp_response_index_spans(
            packed_mb["response_offsets"], padded_length, cp_size, cp_mesh.get_local_rank()
        )
        batch["local_response_lengths"] = [x.length for x in batch["cp_response_index_spans"]]
        batch["cp_group"] = cp_mesh.get_group()
        response_field_keys = (
            "loss_masks",
            "advantages",
            "returns",
            "ref_log_probs",
            "rollout_log_probs",
            "log_probs",
            "teacher_log_probs",
        )
        for key in response_field_keys:
            values = batch.get(key)
            if values is not None:
                batch[key] = slice_fields_by_global_targets(
                    values, packed_mb["response_offsets"], batch["cp_response_index_spans"]
                )
        sample_count = len(response_lengths or ())
        # These tuples describe intervals/layouts, not per-token response values.
        metadata_keys = ("response_offsets", "cp_response_index_spans")
        for key, values in batch.items():
            if key in response_field_keys or key in metadata_keys or not isinstance(values, (list, tuple)):
                continue
            if len(values) != sample_count or not values:
                continue
            if all(
                (
                    isinstance(value, torch.Tensor)
                    and value.ndim == 1
                    and value.numel() == int(response_length)
                )
                or (
                    isinstance(value, (list, tuple))
                    and len(value) == int(response_length)
                )
                for value, response_length in zip(values, response_lengths, strict=True)
            ):
                raise ValueError(
                    f"response-shaped field {key!r} is missing from the CP slicing whitelist"
                )

    # Preserve full sample lengths and denominators alongside the local response
    # fields; local_response_lengths describes the latter when CP is enabled.

    # todo: multimodal not supported
    # multimodal_train_inputs = batch.get("multimodal_train_inputs", None)

    return batch


def _gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict,
    parallel_dims,
) -> dict | None:
    """Gather per-rank metrics across the DP group, reduce on the DP source rank, and log.

    Uses the DP-scoped gloo subgroup from init_dp_gloo_group: only TP rank 0 calls this, so the
    global group would deadlock at tp>1. parallel_dims is unused, kept for the call-site signature.
    """
    return _gather_and_reduce_log_dict(
        log_dict,
        metric_name=metric_name,
        args=args,
        rollout_id=rollout_id,
        dp_src_rank=get_dp_src_rank(),
        dp_group=get_dp_gloo_group(),
    )


def _is_tp_rank0(parallel_dims) -> bool:
    """Return True on TP rank 0 (replaces mpu.get_tensor_model_parallel_rank() == 0)."""
    if parallel_dims.tp > 1:
        tp_mesh = parallel_dims.get_optional_mesh(["tp"])
        if tp_mesh is not None:
            return tp_mesh.get_local_rank() == 0
    # tp==1 or no tp mesh: all ranks are TP rank 0.
    return True


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
    parallel_dims,
) -> None:
    """Summarize rollout fields and log reduced metrics on TP rank 0.

    Token-level metrics (log_probs / returns / advantages / values) are weighted by loss_masks and
    rollout_mask_sums; keys absent from rollout_data are skipped by the loop.
    """
    # PP is always last stage; gate on TP rank 0 only.
    if _is_tp_rank0(parallel_dims):
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        max_seq_lens = rollout_data.get("max_seq_lens", None)
        # Per-rollout denominators matching the training loss (rollout_mask_sums).
        rollout_mask_sums = rollout_data.get("rollout_mask_sums", None)
        dp_world = parallel_dims.dp_shard * parallel_dims.dp_replicate
        num_rollouts_in_rollout = sum(rollout_data["global_batch_sizes"])

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_routed_experts",
                "max_seq_lens",
                "global_batch_sizes",
                "num_microbatches",
                "micro_batch_indices",
            ]:
                continue
            # Emit (sum, count) so _gather_log_data can do a weighted average across DP ranks.
            if isinstance(val, (list, tuple)):
                if not val:
                    continue
                count = len(val)
                if isinstance(val[0], torch.Tensor):
                    # NOTE: clone().detach() prevents in-place modification for the next rollout.
                    if key in [
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                        "returns",
                        "advantages",
                        "values",
                        "teacher_log_probs",
                        "opd_reverse_kl",
                    ]:
                        tensor = torch.cat(val).clone().detach()
                        sum_of_sample_mean = _get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            rollout_mask_sums,
                            qkv_format=getattr(args, "qkv_format", "thd"),
                            max_seq_lens=max_seq_lens,
                        )
                        sum_value, count = _rollout_log_metric_contribution(
                            sum_of_sample_mean(tensor).item(),
                            num_rollouts_in_rollout=num_rollouts_in_rollout,
                            dp_size=dp_world,
                        )
                        log_dict[key] = (sum_value, count)
                        continue
                    tensor = torch.cat(val).clone().detach()
                    # CP ranks carry replicated full-response fields after forward_only restore;
                    # their copies are already accounted for by the gather count.
                    per_rank_sum = tensor.mean() * count
                    sum_value = per_rank_sum.item()
                else:
                    # Rollout batches may carry non-metric metadata (for
                    # example prompt strings).  Do not try to reduce those
                    # fields as numeric metrics.
                    if not isinstance(val[0], Number):
                        continue
                    sum_value = sum(val)
                log_dict[key] = (sum_value, count)
            elif isinstance(val, torch.Tensor):
                log_dict[key] = (val.float().mean().item(), 1)
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")

        reduced_log_dict = _gather_log_data("rollout", args, rollout_id, log_dict, parallel_dims)
        if args.ci_test and reduced_log_dict is not None:
            # Initial actor/ref zero-KL check.
            if (
                rollout_id == 0
                and not getattr(args, "ci_disable_kl_checker", False)
                # routing replay: skip this assert. The hook recomputes gate
                # (module.gate(x) + score_func) whose float32 path differs slightly
                # from the original router forward, causing tiny numerical differences
                # in logits and thus log_probs even though routing is consistent.
                and not getattr(args, "use_rollout_routing_replay", False)
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # TODO: figure out why there is a small numerical difference in log_probs and ref_log_probs in CI test, and whether it's expected or not.
                # assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
                assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
            if "rollout/log_probs" in reduced_log_dict:
                assert -1 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 1

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data, parallel_dims)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data, parallel_dims)

    if args.log_correct_samples:
        if _is_tp_rank0(parallel_dims):
            log_dict = {}
            response_lengths = rollout_data["response_lengths"]
            loss_masks = rollout_data["loss_masks"]
            total_lengths = rollout_data["total_lengths"]

            def quantile(total_value, n_quantiles, data) -> dict:
                assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."

                quantiles = [((i + 1) / n_quantiles) for i in range(n_quantiles)]
                cut_points = [total_value * q for q in quantiles]
                cut_points[-1] = total_value

                count = [0] * n_quantiles
                for d in data:
                    for i, point in enumerate(cut_points):
                        if d <= point:
                            count[i] += 1
                            break

                total = sum(count) + 1e-9
                percentile = [c / total for c in count]

                percentile = {f"p{min(math.ceil(q*100),100)}": p for q, p in zip(quantiles, percentile, strict=True)}
                return percentile

            raw_rewards = rollout_data["raw_reward"]
            # Additional metrics for correct cases are calculated separately below.
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_entropy = []
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward == 1:
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    correct_entropy.append(-rollout_data["log_probs"][i])
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                # NOTE: per-sample-mean over the correct subset, not per-rollout.
                sum_of_sample_mean = _get_sum_of_sample_mean(
                    correct_total_lengths, correct_response_lengths, correct_loss_masks, sample_denoms=None
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
    parallel_dims,
) -> None:
    """
    Log multi-turn auxiliary metrics (raw/observed response lengths, round numbers).

    Operates only on TP rank 0 (PP is always last stage).
    """
    if _is_tp_rank0(parallel_dims):
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:
                    device = val[0].device

                    # Vectorized length calculation using torch.
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU.
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        _gather_log_data("multi_turn", args, rollout_id, log_dict, parallel_dims)


def log_passrate(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
    parallel_dims,
) -> None:
    """Compute pass@k from raw_reward reshaped to [group_number, group_size], averaged over
    problems, and log it on TP rank 0."""
    if _is_tp_rank0(parallel_dims):
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue
            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )
        _gather_log_data("passrate", args, rollout_id, log_dict, parallel_dims)


def _get_fwd_flops_calculator(args: Namespace):
    """Return the generic FLOPs calculator only when its arch inputs exist."""
    required = {
        "hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "vocab_size",
        "ffn_hidden_size",
        "num_experts",
        "num_layers",
    }
    if getattr(args, "num_experts", None) is not None:
        required.update({"moe_ffn_hidden_size", "moe_router_topk"})
    if getattr(args, "multi_latent_attention", False):
        required.update(
            {
                "q_lora_rank",
                "kv_lora_rank",
                "qk_head_dim",
                "qk_pos_emb_head_dim",
                "v_head_dim",
            }
        )
    missing = sorted(name for name in required if getattr(args, name, None) is None)
    if missing:
        logger.warning(
            "Skipping generic FLOPs metrics because model architecture args are missing: %s",
            ", ".join(missing),
        )
        return None
    return lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args) / dist.get_world_size() / 1e12


def log_perf_data(rollout_id: int, args: Namespace, parallel_dims) -> None:
    """Log performance metrics (throughput, FLOPs) on the primary rank."""
    loss_mesh = parallel_dims.get_optional_mesh("loss")
    loss_rank = loss_mesh.get_local_rank() if loss_mesh is not None else 0
    train_metric_utils.log_perf_data(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            _is_tp_rank0(parallel_dims)
            # PP is always last stage (no pipeline parallelism).
            and loss_rank == 0
        ),
        compute_total_fwd_flops=_get_fwd_flops_calculator(args),
    )


def tensors_to_cpu(tensor_list):
    """Detach a list of device tensors to CPU for Ray object store transfer. None passes through."""
    if tensor_list is None:
        return None
    return [t.detach().cpu() for t in tensor_list]


def tensors_to_gpu(tensor_list, device=None):
    """Move a list of CPU tensors back to the device as float32. None passes through."""
    if tensor_list is None:
        return None
    if device is None:
        device = torch.cuda.current_device()
    return [t.to(device=device, dtype=torch.float32) for t in tensor_list]
