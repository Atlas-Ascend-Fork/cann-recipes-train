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

"""Shared HF-config arch field spec for the torchtitan backend (single source of truth).

Two consumers, both keyed off the HF config (config.json):
  * ``_titan_compat.assert_model_matches_checkpoint``: strict arch signature validation (fields
    with ``sig=True``), the recipe's model must equal the checkpoint's on all of them.
  * ``arguments.torchtitan_parse_args`` through ``populate_vime_arch_args``: fill and validate the
    megatron-named arch args that vime-shared code (``flops_utils`` / ``vllm_rollout``) reads.

Intentionally torchtitan-free (only reads a HF AutoConfig object) so it is safe to import
from ``arguments.py`` at parse time. The flavor-side (torchtitan Config) extractors live in
``_titan_compat.py``, keyed by the same field names.
"""

import math


def _rope_theta(hf):
    # Some configs nest rope_theta inside a rope_parameters/rope_scaling dict.
    for dict_attr in ("rope_parameters", "rope_scaling"):
        d = getattr(hf, dict_attr, None)
        if isinstance(d, dict) and d.get("rope_theta") is not None:
            return d["rope_theta"]
    return getattr(hf, "rope_theta", None)


def _head_dim(hf):
    hd = getattr(hf, "head_dim", None)
    if hd is not None:
        return hd
    h = getattr(hf, "hidden_size", None)
    n = getattr(hf, "num_attention_heads", None)
    return (h // n) if (h and n) else None


def _n_kv_heads(hf):
    v = getattr(hf, "num_key_value_heads", None)
    return v if v is not None else getattr(hf, "num_attention_heads", None)


def _num_experts(hf):
    for name in ("num_experts", "num_local_experts", "n_routed_experts"):
        v = getattr(hf, name, None)
        if v is not None:
            return v
    return None


# Each field: name, HF extractor (-> value or None), vime arg name (or None if not a vime
# arg), whether it participates in the flavor signature, whether it is a float (compared
# with a tolerance instead of ==).
def _f(name, hf, arg, sig, is_float=False):
    return {"name": name, "hf": hf, "arg": arg, "sig": sig, "float": is_float}


HF_ARCH_FIELDS = [
    _f("dim",         lambda h: getattr(h, "hidden_size", None),         "hidden_size",         True),
    _f("n_layers",    lambda h: getattr(h, "num_hidden_layers", None),   "num_layers",          True),
    _f("vocab_size",  lambda h: getattr(h, "vocab_size", None),          "vocab_size",          True),
    _f("n_heads",     lambda h: getattr(h, "num_attention_heads", None), "num_attention_heads", True),
    _f("n_kv_heads",  _n_kv_heads,                                       "num_query_groups",    True),
    _f("head_dim",    _head_dim,                                         "kv_channels",         True),
    _f("ffn_hidden",  lambda h: getattr(h, "intermediate_size", None),   "ffn_hidden_size",     True),
    _f("rope_theta",  _rope_theta,                                       None,                  True, True),
    _f("norm_eps",    lambda h: getattr(h, "rms_norm_eps", None),        None,                  True, True),
    # MoE: arg-backfill only (no flavor-signature extractor); skipped when absent (dense).
    _f("num_experts",     _num_experts,                                     "num_experts",         False),
    _f("moe_ffn",         lambda h: getattr(h, "moe_intermediate_size", None), "moe_ffn_hidden_size", False),
    _f("moe_router_topk", lambda h: getattr(h, "num_experts_per_tok", None),   "moe_router_topk",     False),
]

# Hybrid attention fields must match even when the dense model dimensions do.
# They participate in validation only, not the Megatron-style argument backfill.
for _name in (
    "linear_conv_kernel_dim", "linear_key_head_dim", "linear_value_head_dim",
    "linear_num_key_heads", "linear_num_value_heads",
):
    HF_ARCH_FIELDS.append(_f(_name, lambda h, n=_name: getattr(h, n, None), None, True))
HF_ARCH_FIELDS.append(_f(
    "layer_types", lambda h: tuple(h.layer_types) if getattr(h, "layer_types", None) else None,
    None, True,
))
HF_ARCH_FIELDS.append(_f(
    "partial_rotary_factor",
    lambda h: (getattr(h, "rope_parameters", None) or {}).get("partial_rotary_factor"),
    None, True, True,
))

SIGNATURE_FIELDS = [f["name"] for f in HF_ARCH_FIELDS if f["sig"]]
_FLOAT_FIELDS = {f["name"] for f in HF_ARCH_FIELDS if f["float"]}


def values_equal(name, a, b):
    """Compare two arch values by field name; float fields use a relative tolerance."""
    if a is None or b is None:
        return False
    if name in _FLOAT_FIELDS:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=0.0)
    return a == b


def _text_config(hf_config):
    """Multimodal HF configs keep language-model architecture fields in text_config."""
    return getattr(hf_config, "text_config", hf_config)


def hf_arch_signature(hf_config):
    """Return {sig_field: value} from a HF config, skipping fields absent on the config."""
    hf_config = _text_config(hf_config)
    sig = {}
    for f in HF_ARCH_FIELDS:
        if not f["sig"]:
            continue
        v = f["hf"](hf_config)
        if v is not None:
            sig[f["name"]] = v
    return sig


_MISSING = object()


def populate_vime_arch_args(args, hf_config):
    """Fill the megatron-named arch args that vime-shared code (flops_utils / vllm_rollout) reads
    from the HF config, which is their only source here.

    Every field ends up present (None when the config has no value, so ``args.num_experts`` on a
    dense model reads rather than raises), and a value already set that disagrees with the config
    is an error. Returns the ``(arg, value)`` pairs filled, for logging.
    """
    hf_config = _text_config(hf_config)
    filled, conflicts = [], []
    for f in HF_ARCH_FIELDS:
        arg = f["arg"]
        if arg is None:
            continue
        hf_val = f["hf"](hf_config)
        cur = getattr(args, arg, _MISSING)
        if hf_val is None:
            if cur is _MISSING:
                setattr(args, arg, None)  # ensure the attribute exists (e.g. dense: num_experts)
            continue
        if cur is _MISSING or cur is None:
            setattr(args, arg, hf_val)
            filled.append((arg, hf_val))
        elif not values_equal(f["name"], cur, hf_val):
            conflicts.append(f"--{arg.replace('_', '-')}={cur} != hf_config.{f['name']}={hf_val}")
    if conflicts:
        raise AssertionError(
            "torchtitan arch args disagree with the HF config (config.json is the source of "
            "truth): " + "; ".join(conflicts)
        )
    return filled
