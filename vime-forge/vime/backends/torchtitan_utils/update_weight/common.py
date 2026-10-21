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

"""torchtitan weight iteration and HF format conversion utilities."""
from collections.abc import Callable, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from .weight_quantization import dsv4_export_dtype, export_rollout_weights, validate_rollout_quantization_config
from .weight_quantization import get_mxfp4_export_config


def _state_dict_chunks(model):
    """Yield the model's state_dict a piece at a time, one transformer block per piece.

    Keys match the whole-model ones. Chunking keeps one layer's ``FusedQKVLinear`` save hook (a
    Replicate gather in fp32) alive at a time instead of the whole stack.
    """
    for name, child in model.named_children():
        if isinstance(child, (nn.ModuleDict, nn.ModuleList)):
            for sub_name, sub in child.named_children():
                yield sub.state_dict(prefix=f"{name}.{sub_name}.")
        else:
            yield child.state_dict(prefix=f"{name}.")

    # Anything held directly on the model rather than on a child.
    own = {n: p for n, p in model.named_parameters(recurse=False)}
    own.update(model.named_buffers(recurse=False))
    if own:
        yield own


def _hf_params_dict(model_parts, sd_adapter, raw_params=None):
    """Build the whole HF-format params dict at once: state_dict + sd_adapter.to_hf + weight tying.

    Only the EP>1 path uses this; it needs every expert tensor in hand to pair with the names
    gathered from the other EP ranks. Everything else streams through _state_dict_chunks.

    The state dict comes from the module, not from DCP's ``get_model_state_dict``: ``to_hf``
    expects the keys the state-dict hooks produce (``FusedQKVLinear`` presents ``wqkv`` as
    ``wq`` / ``wk`` / ``wv``), which DCP's attribute walk cannot resolve.

    ``raw_params`` replaces the read-from-model step, so gradients or optimizer state of the same
    shape convert through the same path. Its keys must match the model's ``state_dict``.
    """
    if raw_params is not None:
        params = dict(raw_params)
    else:
        params = {}
        for m in model_parts:
            params.update(m.state_dict())

    if sd_adapter is not None:
        params = sd_adapter.to_hf(params)

    if "model.embed_tokens.weight" in params and "lm_head.weight" not in params:
        params["lm_head.weight"] = params["model.embed_tokens.weight"]

    return params


def _full_tensor(param, device, dtype, *, unwrap_mxfp4=False):
    """Cast float tensors to ``dtype``, then gather a DTensor to full on device.

    The cast comes first so the collective moves the narrower dtype; the result is bitwise the same
    either way. Non-float buffers keep their dtype; ``dtype=None`` keeps native precision, which
    only the precision guard asks for.
    """
    if unwrap_mxfp4:
        local = param.to_local() if isinstance(param, DTensor) else param
        wrapper = local if callable(getattr(local, "to_tensor", None)) else None
        if wrapper is not None:
            raw = wrapper.to_tensor()
            if isinstance(param, DTensor):
                param = DTensor.from_local(
                    raw,
                    param.device_mesh,
                    param.placements,
                    run_check=False,
                    shape=param.shape,
                    stride=param.stride(),
                )
            else:
                param = raw
    if dtype is not None and param.is_floating_point() and param.dtype != dtype:
        param = param.to(dtype)
    t = param.full_tensor() if isinstance(param, DTensor) else param
    return t if t.device == device else t.to(device, non_blocking=True)


def _resolve_param_dtype(param_dtype, name, param):
    """Resolve the forward/export dtype for one HF parameter.

    The callable form is intentionally small and temporary: DSV4 has a few
    FP32 forward-policy overrides while the remaining parameters use the
    mixed-precision parameter dtype. Keeping the resolver here ensures the
    dtype decision happens before DTensor materialization and before any
    rollout quantization.
    """
    if callable(param_dtype):
        return param_dtype(name, param)
    return param_dtype


def get_hf_per_tensor(
    model_parts,
    sd_adapter,
    device,
    *,
    param_dtype: torch.dtype | Callable[[str, torch.Tensor], torch.dtype | None] | None,
    parallel_dims=None,
    raw_params=None,
    yield_experts_on_all_ranks=False,
    train_quantization=None,
    model_type=None,
) -> Callable[[], Iterator[tuple[str, torch.Tensor]]]:
    """Validate export config now and return a reusable weight-iterator factory.

    Each invocation reads current model state and starts fresh collectives. Creating
    the factory does not materialize weights; all ranks must consume each iterator.
    """
    validate_rollout_quantization_config(train_quantization)
    quantized_rollout = bool(getattr(train_quantization, "enable_quantized_training", False))
    rollout_quantization = getattr(train_quantization, "recipe", None)
    mxfp4_config = get_mxfp4_export_config(train_quantization) if quantized_rollout else None

    def effective_param_dtype(name, param):
        dtype = _resolve_param_dtype(param_dtype, name, param)
        # Keep None's native-precision contract for checkpoint/precision exports.
        # DSV4's FP32 state is independent of whether its linear weights are quantized.
        if model_type == "deepseek_v4" and dtype is not None:
            return dsv4_export_dtype(
                name,
                dtype,
                recipe=rollout_quantization,
                mxfp4_config=mxfp4_config,
            )
        return dtype

    def iter_weights():
        ep_group, ep_size = _get_ep_group(parallel_dims)
        if ep_group is None:
            chunks = [dict(raw_params)] if raw_params is not None else (
                chunk for m in model_parts for chunk in _state_dict_chunks(m)
            )
            tied_embedding = None
            saw_lm_head = False
            for chunk in chunks:
                hf_chunk = sd_adapter.to_hf(chunk) if sd_adapter is not None else chunk
                for name, param in hf_chunk.items():
                    tensor = _full_tensor(
                        param, device, effective_param_dtype(name, param),
                        unwrap_mxfp4=mxfp4_config is not None,
                    )
                    saw_lm_head = saw_lm_head or name == "lm_head.weight"
                    if name == "model.embed_tokens.weight":
                        tied_embedding = tensor
                    yield name, tensor
            if tied_embedding is not None and not saw_lm_head:
                yield "lm_head.weight", tied_embedding
            return

        params = _hf_params_dict(model_parts, sd_adapter, raw_params)
        expert_params = [(n, p) for n, p in params.items() if ".experts." in n]
        non_expert_params = [(n, p) for n, p in params.items() if ".experts." not in n]
        for name, param in non_expert_params:
            yield name, _full_tensor(
                param, device, effective_param_dtype(name, param),
                unwrap_mxfp4=mxfp4_config is not None,
            )

        is_src = (not dist.is_initialized()) or dist.get_rank() == 0
        ep_names = [None] * ep_size
        dist.all_gather_object(ep_names, [n for n, _ in expert_params], group=ep_group)
        assert all(len(names) == len(expert_params) for names in ep_names), (
            f"EP ranks hold different expert counts {[len(n) for n in ep_names]}; "
            "names and tensors would be paired wrongly"
        )
        for i, (name, raw_param) in enumerate(expert_params):
            tensor = _full_tensor(
                raw_param, device, effective_param_dtype(name, raw_param),
                unwrap_mxfp4=mxfp4_config is not None,
            ).contiguous()
            gathered = [torch.empty_like(tensor) for _ in range(ep_size)]
            dist.all_gather(gathered, tensor, group=ep_group)
            if not is_src and not yield_experts_on_all_ranks:
                continue
            for ep_rank in range(ep_size):
                yield ep_names[ep_rank][i], gathered[ep_rank]

    def rollout_weights_iter():
        weights = iter_weights()
        if quantized_rollout:
            return export_rollout_weights(
                weights,
                recipe=rollout_quantization,
                mxfp4_config=mxfp4_config,
            )
        return weights

    return rollout_weights_iter

def _get_ep_group(parallel_dims):
    """Return (ep_group, ep_size) or (None, 1) when EP is not enabled."""
    if parallel_dims is None or parallel_dims.ep <= 1:
        return None, 1
    ep_mesh = parallel_dims.get_optional_mesh(["ep"])
    ep_group = ep_mesh.get_group() if ep_mesh is not None else None
    return ep_group, parallel_dims.ep
