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
from collections import OrderedDict
from collections.abc import Callable, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from .weight_quantization import iter_named_tensors, quantize_and_gather_weight, view_as_uint8


def _state_dict_chunks(model, *, fsdp_param_dtypes=None):
    """Yield the model's state_dict a piece at a time, one transformer block per piece.

    Keys match the whole-model ones. Chunking keeps one layer's ``FusedQKVLinear`` save-hook
    tensors materialized at a time instead of the whole stack.
    """
    for name, child in model.named_children():
        if isinstance(child, (nn.ModuleDict, nn.ModuleList)):
            for sub_name, sub in child.named_children():
                prefix = f"{name}.{sub_name}."
                if fsdp_param_dtypes is None:
                    yield sub.state_dict(prefix=prefix)
                else:
                    yield sub.state_dict(
                        destination=_FSDPExportStateDict(fsdp_param_dtypes),
                        prefix=prefix,
                        keep_vars=True,
                    )
        else:
            if fsdp_param_dtypes is None:
                yield child.state_dict(prefix=f"{name}.")
            else:
                yield child.state_dict(
                    destination=_FSDPExportStateDict(fsdp_param_dtypes),
                    prefix=f"{name}.",
                    keep_vars=True,
                )

    # Anything held directly on the model rather than on a child.
    own = _FSDPExportStateDict(fsdp_param_dtypes) if fsdp_param_dtypes is not None else {}
    own.update(model.named_parameters(recurse=False))
    own.update(model.named_buffers(recurse=False))
    if own:
        yield own


class _FSDPExportStateDict(OrderedDict):
    """Apply FSDP forward dtypes before save hooks split or rename parameters."""

    def __init__(self, param_dtypes):
        super().__init__()
        self._param_dtypes = param_dtypes
        self._metadata = OrderedDict()

    @torch.no_grad()
    def __setitem__(self, name, tensor):
        dtype = (
            self._param_dtypes.get(id(tensor))
            if isinstance(self._param_dtypes, dict)
            else self._param_dtypes
        )
        if dtype is not None and tensor.is_floating_point():
            local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            if isinstance(local_tensor, nn.Parameter):
                local_tensor = local_tensor.data
            if not callable(getattr(local_tensor, "to_inference_weight", None)):
                tensor = tensor.to(dtype)
        super().__setitem__(name, tensor)


def _collect_fsdp_param_dtypes(model_parts):
    """Map current sharded parameters to their effective FSDP forward dtypes."""
    dtypes = {}
    # No public API exposes each parameter's effective FSDP dtype.
    for model in model_parts:
        state = fully_shard.state(model)
        state._lazy_init()
        for fsdp_state in state._state_ctx.all_states:
            for group in fsdp_state._fsdp_param_groups:
                for param in group.fsdp_params:
                    if param.param_dtype is not None:
                        dtypes[id(param.sharded_param)] = param.param_dtype
    return dtypes


def _hf_params_dict(model_parts, sd_adapter, raw_params=None, *, fsdp_param_dtypes=None):
    """Build the full HF state for EP export and precision guard.

    EP export needs every expert tensor in hand to pair with the names gathered from the
    other EP ranks. Non-EP export streams through _state_dict_chunks.

    The state dict comes from the module, not from DCP's ``get_model_state_dict``: ``to_hf``
    expects the keys the state-dict hooks produce (``FusedQKVLinear`` presents ``wqkv`` as
    ``wq`` / ``wk`` / ``wv``), which DCP's attribute walk cannot resolve.

    Optional ``raw_params`` must use the same state-dict keys, allowing gradients and optimizer
    state to share this path.
    """
    if raw_params is not None:
        params = _FSDPExportStateDict(fsdp_param_dtypes) if fsdp_param_dtypes is not None else {}
        params.update(raw_params)
    else:
        params = {}
        for model in model_parts:
            for chunk in _state_dict_chunks(model, fsdp_param_dtypes=fsdp_param_dtypes):
                params.update(chunk)

    if sd_adapter is not None:
        params = sd_adapter.to_hf(params)

    if "model.embed_tokens.weight" in params and "lm_head.weight" not in params:
        params["lm_head.weight"] = params["model.embed_tokens.weight"]

    return params


@torch.no_grad()
def _get_export_tensors(param, device):
    """Export either inference payload/scale or a full ordinary tensor."""
    local_weight = param.to_local() if isinstance(param, DTensor) else param

    # Wrapper tensors are Tensor subclasses; classify them before ordinary tensors.
    if isinstance(local_weight, nn.Parameter):
        local_weight = local_weight.data
    if callable(getattr(local_weight, "to_inference_weight", None)):
        # Quantized wrappers export packed data and scales.
        return quantize_and_gather_weight(param, device)
    else:
        # Ordinary parameters only need to be materialized from their FSDP/TP shards.
        return (param.full_tensor() if isinstance(param, DTensor) else param,)


def _ep_all_gather(tensor, ep_group, ep_size):
    """Gather one EP tensor using a HCCL-supported transport dtype."""
    # Make the byte view contiguous: NPU copy kernels do not support every MX dtype.
    transport_tensor = view_as_uint8(tensor).contiguous()
    gathered = [torch.empty_like(transport_tensor) for _ in range(ep_size)]
    dist.all_gather(gathered, transport_tensor, group=ep_group)
    return [item.view(tensor.dtype) for item in gathered]


def get_hf_per_tensor(
    model_parts,
    sd_adapter,
    device,
    *,
    param_dtype: torch.dtype | None = None,
    fsdp_param_dtypes=None,
    parallel_dims=None,
    raw_params=None,
    yield_experts_on_all_ranks=False,
) -> Callable[[], Iterator[tuple[str, torch.Tensor]]]:
    """Return a reusable HF weight factory; all ranks must consume each iterator.

    Use either a uniform ``param_dtype`` or collected per-parameter FSDP dtypes.
    """
    def _iter_weights():
        ep_group, ep_size = _get_ep_group(parallel_dims)
        param_dtypes = param_dtype if param_dtype is not None else fsdp_param_dtypes
        if ep_group is not None:
            chunks = [_hf_params_dict(model_parts, sd_adapter, raw_params, fsdp_param_dtypes=param_dtypes)]
        elif raw_params is not None:
            chunk = _FSDPExportStateDict(param_dtypes) if param_dtypes is not None else {}
            chunk.update(raw_params)
            chunks = [chunk]
        else:
            chunks = (
                chunk
                for model in model_parts
                for chunk in _state_dict_chunks(model, fsdp_param_dtypes=param_dtypes)
            )
        expert_params = []
        tied_embedding = None
        saw_lm_head = False
        for chunk in chunks:
            if ep_group is None and sd_adapter is not None:
                chunk = sd_adapter.to_hf(chunk)
            for name, param in chunk.items():
                if ep_group is not None and ".experts." in name:
                    expert_params.append((name, param))
                    continue
                export_tensors = _get_export_tensors(param, device)
                saw_lm_head = saw_lm_head or name == "lm_head.weight"
                if name == "model.embed_tokens.weight":
                    tied_embedding = export_tensors
                yield from iter_named_tensors(name, export_tensors)

        if tied_embedding is not None and not saw_lm_head:
            yield from iter_named_tensors("lm_head.weight", tied_embedding)
        if ep_group is not None:
            ep_names = [None] * ep_size
            dist.all_gather_object(ep_names, [name for name, _ in expert_params], group=ep_group)
            assert all(len(names) == len(expert_params) for names in ep_names), (
                f"EP ranks hold different expert counts {[len(names) for names in ep_names]}; "
                "names and tensors would be paired wrongly"
            )
            is_src = (not dist.is_initialized()) or dist.get_rank() == 0
            for index, (_, param) in enumerate(expert_params):
                # Restore TP/FSDP shards before gathering experts across EP ranks.
                export_tensors = _get_export_tensors(param, device)
                gathered = [_ep_all_gather(tensor, ep_group, ep_size) for tensor in export_tensors]
                if is_src or yield_experts_on_all_ranks:
                    for ep_rank, expert_tensors in enumerate(zip(*gathered)):
                        yield from iter_named_tensors(ep_names[ep_rank][index], expert_tensors)

    return _iter_weights


def _get_ep_group(parallel_dims):
    """Return (ep_group, ep_size) or (None, 1) when EP is not enabled."""
    if parallel_dims is None or parallel_dims.ep <= 1:
        return None, 1
    ep_mesh = parallel_dims.get_optional_mesh(["ep"])
    ep_group = ep_mesh.get_group() if ep_mesh is not None else None
    return ep_group, parallel_dims.ep
