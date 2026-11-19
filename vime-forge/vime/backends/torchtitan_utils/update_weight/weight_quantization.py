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

"""Encoding and transport helpers for wrapper-backed quantized weights."""

import torch
from torch.distributed.tensor import DTensor


_MX_TRANSPORT_DTYPES = frozenset((torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.float8_e8m0fnu))


def encode_inference_weight(weight):
    """Quantize a training wrapper into its packed HF weight and logical scales."""
    from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig

    weight_config = weight.weight_config
    if (
        weight.dtype == torch.float32
        and isinstance(weight_config, BlockMXQuantizeConfig)
        and weight_config.mxfp4_fake_quantize_config is None
    ):
        # npu_dynamic_block_mx_quant accepts only FP16/BF16 inputs.
        weight = weight.to(torch.bfloat16)
    inference_weight = weight.to_inference_weight()
    qweight = inference_weight.qdata
    scale = inference_weight.scale.reshape(*qweight.shape[:-1], -1)
    return qweight, scale


def _gather_qdata_or_scale(tensor, param):
    """Rebuild a qdata or scale tensor's global TP/FSDP layout."""
    global_shape = list(param.shape)
    global_shape[-1] = tensor.shape[-1]
    for mesh_dim, placement in enumerate(param.placements):
        # Expand the global width when a mesh placement shards the last logical dimension.
        if placement.is_shard() and placement.dim % param.ndim == param.ndim - 1:
            global_shape[-1] *= param.device_mesh.size(mesh_dim)
    global_stride = [1] * len(global_shape)
    for dim in range(len(global_shape) - 2, -1, -1):
        global_stride[dim] = global_stride[dim + 1] * global_shape[dim + 1]
    # HCCL needs contiguous byte buffers for MX dtypes.
    transport_tensor = view_as_uint8(tensor).contiguous()
    distributed_tensor = DTensor.from_local(
        transport_tensor, param.device_mesh, param.placements,
        run_check=False, shape=torch.Size(global_shape), stride=tuple(global_stride),
    )
    return distributed_tensor.full_tensor().view(tensor.dtype)


@torch.no_grad()
def quantize_and_gather_weight(param, device):
    """Quantize local shards, then gather weights and scales across TP/FSDP."""
    param = param.to(device)
    local_weight = param.to_local() if isinstance(param, DTensor) else param
    export_tensors = encode_inference_weight(local_weight)
    if isinstance(param, DTensor):
        return tuple(_gather_qdata_or_scale(tensor, param) for tensor in export_tensors)
    else:
        return export_tensors


def iter_named_tensors(name: str, tensors):
    """Yield the HF names for a quantized weight and its scale."""
    yield name, tensors[0]
    if len(tensors) == 2:
        yield name.removesuffix(".weight") + ".weight_scale", tensors[1]


def view_as_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """View MX dtypes as uint8 because HCCL does not support their collectives."""
    return tensor.view(torch.uint8) if tensor.dtype in _MX_TRANSPORT_DTYPES else tensor
