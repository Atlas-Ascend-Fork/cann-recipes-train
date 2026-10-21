# Adapted from
# https://github.com/volcengine/verl/blob/5b550b0c/verl/workers/engine/torchtitan/utils.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""torchtitan compatibility layer: a few helpers ported from verl `workers/engine/torchtitan/utils.py`.

These are utilities that torchtitan does not include but vime's implementation depends on:
- assert_model_matches_checkpoint: verifies the recipe's model config against the checkpoint's
  config.json, field by field.
- enable_fsdp_gradient_division: torchtitan disables FSDP gradient division by default; re-enable
  when vime's loss normalization requires it.

Note: NoOpDataLoader is intentionally not ported; TitanEngine passes dataloader=None directly.

This module imports torch at the module top level and **should only be loaded via the
torchtitan backend path (lazy import through engine.py)**.
"""

import logging

import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule

from ._arch_fields import hf_arch_signature, values_equal

logger = logging.getLogger(__name__)


# HF model_type -> torchtitan model name (ported from verl utils.py:68-77).
_HF_MODEL_TYPE_TO_TORCHTITAN_NAME = {
    "qwen2": "qwen3",
    "qwen3": "qwen3",
    "qwen2_moe": "qwen3",
    "qwen3_moe": "qwen3",
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
    "llama": "llama3",
    "llama4": "llama4",
    "deepseek_v3": "deepseek_v3",
    "deepseek_v4": "deepseek_v4",
    "gpt_oss": "gpt_oss",
}


def _model_signature(cfg) -> dict:
    """The {signature_field: value} arch signature of a model Config, the torchtitan-side twin of
    _arch_fields.hf_arch_signature.

    RoPE sits on each attention config, and the FFN hidden size is the up-projection's out_features
    (FeedForward.Config holds w1/w2/w3 Linear configs, not a scalar). A field that cannot be read
    (an MoE block has .moe, not .feed_forward) is omitted. Qwen3.5 validation below
    rejects checkpoint fields missing from this signature rather than skipping them.
    """
    layers = getattr(cfg, "layers", None) or []
    blk = layers[0] if layers else None
    # Qwen3.5 starts with linear attention: find a full-attention layer for RoPE
    # and standard head dimensions, and a GDN layer for the linear dimensions.
    attn = next((b.attention for b in layers if getattr(b, "attention", None) is not None), None)
    delta = next((b.delta_net for b in layers if getattr(b, "delta_net", None) is not None), None)
    ff = getattr(blk, "feed_forward", None) if blk is not None else None
    rope = getattr(attn, "rope", None)
    norm = getattr(cfg, "norm", None)

    n_heads = getattr(attn, "n_heads", None)
    n_kv = getattr(attn, "n_kv_heads", None)
    head_dim = getattr(attn, "head_dim", None)
    dim = getattr(cfg, "dim", None)
    # Dense FFN hidden size = out_features of the up-projection (w1). None for MoE blocks.
    ffn_hidden = getattr(getattr(ff, "w1", None), "out_features", None)

    candidates = {
        "dim": dim,
        "n_layers": len(layers) or None,
        "vocab_size": getattr(cfg, "vocab_size", None),
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim if head_dim is not None else ((dim // n_heads) if (dim and n_heads) else None),
        "ffn_hidden": ffn_hidden,
        "rope_theta": getattr(rope, "theta", None),
        "norm_eps": getattr(norm, "eps", None),
    }
    if delta is not None:
        def projection_heads(name, head_dim):
            width = getattr(getattr(delta, name, None), "out_features", None)
            return width // head_dim if width and head_dim and width % head_dim == 0 else None

        key_dim = getattr(delta, "key_head_dim", None)
        value_dim = getattr(delta, "value_head_dim", None)
        candidates.update(
            linear_conv_kernel_dim=getattr(delta, "conv_kernel_size", None),
            linear_key_head_dim=key_dim,
            linear_value_head_dim=value_dim,
            linear_num_key_heads=projection_heads("in_proj_q", key_dim),
            linear_num_value_heads=projection_heads("in_proj_v", value_dim),
            # Unknown or ambiguous blocks must not be mistaken for linear attention.
            layer_types=tuple(
                "full_attention" if getattr(b, "attention", None) is not None and getattr(b, "delta_net", None) is None
                else "linear_attention" if getattr(b, "delta_net", None) is not None and getattr(b, "attention", None) is None
                else None
                for b in layers
            ),
        )
        rotary_dim = getattr(attn, "rotary_dim", None)
        if rotary_dim is not None and head_dim:
            candidates["partial_rotary_factor"] = rotary_dim / head_dim
    return {k: v for k, v in candidates.items() if v is not None}


def assert_model_matches_checkpoint(model_spec, hf_config) -> None:
    """Verify the recipe's model against the checkpoint's config.json, field by field.

    Takes the parsed HF config, not a path: the actor already loaded it once, rank by rank.

    Every field comparable on both sides (see ``_arch_fields.SIGNATURE_FIELDS``) must be equal, and
    dim / n_layers / vocab_size must be comparable at all, so a config exposing none of them cannot
    pass by default. A mismatch raises, naming the fields that differ.
    """
    hf_config = getattr(hf_config, "text_config", hf_config)

    model_type = getattr(hf_config, "model_type", None)
    if model_type is None:
        raise ValueError("HuggingFace config is missing the 'model_type' field")
    expected_name = _HF_MODEL_TYPE_TO_TORCHTITAN_NAME.get(model_type)
    if expected_name is None:
        logger.warning(
            "model_type='%s' is not in the family table, so only the architecture signature is "
            "checked: a checkpoint from an unsupported family passes as long as the numbers match.",
            model_type,
        )
    if expected_name is not None and expected_name != model_spec.name:
        raise ValueError(
            f"Checkpoint model_type='{model_type}' maps to the torchtitan family "
            f"'{expected_name}', but the recipe builds '{model_spec.name}'. Select a recipe from "
            f"vime/backends/torchtitan_utils/models/{expected_name}/config_registry.py."
        )

    hf_sig = hf_arch_signature(hf_config)
    model_sig = _model_signature(model_spec.model)
    core = ("dim", "n_layers", "vocab_size")
    missing = [c for c in core if c not in hf_sig or c not in model_sig]
    if missing:
        raise ValueError(
            f"Cannot compare the checkpoint against flavor '{model_spec.flavor}': the identity "
            f"fields {missing} are unreadable on one side (hf={sorted(hf_sig)}, "
            f"model={sorted(model_sig)})."
        )

    comparable = [k for k in hf_sig if k in model_sig]
    if model_spec.name == "qwen3_5":
        unreadable = [k for k in hf_sig if k not in model_sig]
        if unreadable:
            raise ValueError(f"Qwen3.5 recipe cannot validate checkpoint fields: {unreadable}")
    mismatch = [
        (k, hf_sig[k], model_sig[k]) for k in comparable if not values_equal(k, hf_sig[k], model_sig[k])
    ]
    if mismatch:
        diff = "; ".join(f"{k}: checkpoint={hv} vs flavor={fv}" for k, hv, fv in mismatch)
        raise ValueError(
            f"Recipe flavor '{model_spec.name}/{model_spec.flavor}' does not match the checkpoint "
            f"arch: {diff}. Point --config at the right recipe, or add a recipe "
            f"that derives this architecture from the family preset (see the module docstring of "
            f"vime/backends/torchtitan_utils/models/{model_spec.name}/config_registry.py)."
        )
    logger.info(
        "Checkpoint arch verified against flavor '%s/%s' (model_type=%s); %d fields compared",
        model_spec.name, model_spec.flavor, model_type, len(comparable),
    )


def enable_fsdp_gradient_division(model: nn.Module, dp_size: int) -> None:
    """Re-enable FSDP automatic gradient division (disabled by torchtitan by default). Ported from verl utils.py:151-167.

    torchtitan defaults to disable_fsdp_gradient_division (gradient_divide_factor=1.0, paired with
    global token normalization). If vime's loss scales by dp_size, set the factor back to dp_size
    here so gradients are averaged over DP and summed over CP. The FSDP mesh can
    include CP ranks, but CP must not contribute to the divide factor.
    """
    assert dp_size >= 1, f"FSDP gradient divide factor must be >= 1, got {dp_size}"
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(float(dp_size))
