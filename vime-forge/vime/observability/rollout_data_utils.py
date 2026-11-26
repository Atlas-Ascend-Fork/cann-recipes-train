import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vime.utils.types import Sample

logger = logging.getLogger(__name__)

_ROLLOUT_DATA_TENSOR_DTYPES = {
    "tokens": torch.long,
    "loss_masks": torch.int,
    "rollout_log_probs": torch.float32,
    "rollout_top_p_token_ids": torch.int32,
    "rollout_top_p_token_offsets": torch.int32,
    "teacher_log_probs": torch.float32,
    "rollout_routed_experts": None,
}


def _cpu_tensor(value, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    tensor = torch.as_tensor(value, dtype=dtype) if dtype is not None else torch.as_tensor(value)
    return tensor.detach().cpu().contiguous()


def tensorize_rollout_data_for_training(rollout_data: dict[str, Any]) -> None:
    for key, dtype in _ROLLOUT_DATA_TENSOR_DTYPES.items():
        if key in rollout_data:
            rollout_data[key] = [_cpu_tensor(value, dtype=dtype) for value in rollout_data[key]]

    if "multimodal_train_inputs" in rollout_data:
        rollout_data["multimodal_train_inputs"] = [
            (
                {
                    key: _cpu_tensor(value) if isinstance(value, (np.ndarray, torch.Tensor)) else value
                    for key, value in mm_dict.items()
                }
                if mm_dict is not None
                else None
            )
            for mm_dict in rollout_data["multimodal_train_inputs"]
        ]

    if "rollout_mask_sums" in rollout_data:
        rollout_data["rollout_mask_sums"] = _cpu_tensor(
            rollout_data["rollout_mask_sums"],
            dtype=torch.float32,
        )


def validate_rollout_routed_experts_for_replay(
    routed_experts: list[torch.Tensor],
    args,
) -> None:
    """Reject incomplete PP routing captures before R3 consumes them."""
    if not routed_experts:
        raise ValueError("R3 is enabled but no rollout routed-experts tensors were returned.")

    num_layers = int(args.num_layers)
    topk = int(args.moe_router_topk)
    moe_layer_freq = getattr(args, "moe_layer_freq", None)
    if isinstance(moe_layer_freq, (list, tuple)):
        moe_layers = [layer_id for layer_id, freq in enumerate(moe_layer_freq[:num_layers]) if int(freq) != 0]
    else:
        moe_layers = list(range(num_layers))

    num_experts = getattr(args, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(getattr(args, "model_config", None), "num_experts", None)
    if num_experts is not None:
        num_experts = int(num_experts)

    for sample_idx, experts in enumerate(routed_experts):
        experts = torch.as_tensor(experts)
        if experts.ndim != 3 or tuple(experts.shape[1:]) != (num_layers, topk):
            raise ValueError(
                "Invalid rollout routed-experts shape for R3: "
                f"sample={sample_idx}, got={tuple(experts.shape)}, "
                f"expected=(*, {num_layers}, {topk})."
            )
        if experts.shape[0] == 0:
            raise ValueError(f"R3 sample {sample_idx} has no routed-experts rows.")
        if topk > 1:
            missing_layers = [layer_id for layer_id in moe_layers if not torch.count_nonzero(experts[:, layer_id, :])]
            if missing_layers:
                raise ValueError(
                    "R3 routed-experts capture is all zero for MoE layers "
                    f"{missing_layers} in sample {sample_idx}. This usually means "
                    "vLLM pipeline stages did not aggregate their disjoint routing "
                    "captures; refusing to replay expert 0 everywhere."
                )

        # Dense layers only carry placeholders, not routing decisions.
        if not moe_layers:
            continue
        moe_experts = experts[:, moe_layers, :]

        # Every token's top-k experts must be distinct: a token routed to the
        # same expert twice would make the replayed routing map ambiguous.
        _validate_unique_experts(sample_idx, moe_experts, topk)

        # ... and every expert id must be in [0, num_experts).
        _validate_expert_id_range(sample_idx, moe_experts, num_experts)



def _validate_unique_experts(
    sample_idx: int,
    moe_experts: torch.Tensor,
    topk: int,
) -> None:
    """Reject tokens whose top-k expert ids are not distinct."""
    sorted_topk = torch.sort(moe_experts, dim=-1).values
    duplicate = (sorted_topk[..., 1:] == sorted_topk[..., :-1]).any(dim=-1)
    if bool(duplicate.any()):
        duplicate_tokens = int(duplicate.any(dim=-1).sum())
        raise ValueError(
            "R3 rollout routed-experts contain duplicate expert ids within a "
            f"token: sample={sample_idx}, topk={topk}, "
            f"duplicate_tokens={duplicate_tokens}."
        )


def _validate_expert_id_range(
    sample_idx: int,
    moe_experts: torch.Tensor,
    num_experts: int | None,
) -> None:
    """Reject expert ids outside [0, num_experts)."""
    min_expert = int(moe_experts.min())
    max_expert = int(moe_experts.max())
    if min_expert < 0 or (num_experts is not None and max_expert >= num_experts):
        raise ValueError(
            "R3 rollout routed-experts ids out of range: "
            f"sample={sample_idx}, min={min_expert}, max={max_expert}, "
            f"num_experts={num_experts}."
        )


def validate_rollout_id_annotated(node, depth=0):
    """Walk the rollout function's nested output and validate ``rollout_id`` only
    when a compact / subagent pattern is detected.

    "Compact" = the rollout function wraps multiple training samples from one
    rollout execution into a ``list[Sample]``. In vime's convention the
    default rollout shape is ``list[list[Sample]]`` (depth-2: prompt × rollout)
    so its leaf ``list[Sample]`` lands at depth 1 and we skip validation,
    preserving backward compatibility. A compact rollout adds a third level:
    ``list[list[list[Sample]]]`` (prompt × rollout × samples-from-one-rollout),
    so the leaf ``list[Sample]`` lands at depth ≥ 2. At that point we require
    every sibling to carry a non-None ``rollout_id`` and to share the same
    value, so the loss reducer counts the rollout once instead of N times.
    """
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            rids = [sample.rollout_id for sample in node]
            missing = [i for i, rollout_id in enumerate(rids) if rollout_id is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but rollout_id is unset on "
                f"positions {missing}. Set Sample.rollout_id on every sibling so the loss "
                "reducer can aggregate them as one rollout instead of N."
            )
            assert len(set(rids)) == 1, f"Sibling samples from one compact rollout must share rollout_id; got {rids}."
        return
    for item in node:
        validate_rollout_id_annotated(item, depth + 1)


def load_debug_rollout_data(path_template, *, rollout_id: int, subsample_ratio=None) -> list[Sample]:
    data = torch.load(path_template.format(rollout_id=rollout_id), weights_only=False)["samples"]
    data = [Sample.from_dict(sample) for sample in data]
    if subsample_ratio is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * subsample_ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            "Subsample loaded debug rollout data using ratio=%s and change num rows %s -> %s",
            subsample_ratio,
            original_num_rows,
            len(data),
        )
    return data


def save_debug_rollout_data(path_template, data, *, rollout_id: int, evaluation: bool) -> None:
    if path_template is None:
        return

    path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
    logger.info(f"Save debug rollout data to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if evaluation:
        samples = [sample.to_dict() for info in data.values() for sample in info["samples"]]
    else:
        samples = [sample.to_dict() for sample in data]

    torch.save({"rollout_id": rollout_id, "samples": samples}, path)
