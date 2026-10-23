# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-ascend's W8A8 linear implementations; original notice:
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

# HiF8 linear for Ascend A5, following w8a8_mxfp8.py and w8a8_dynamic.py.
from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.logger import logger

from vllm_ascend.device.hif8_compat import HIFLOAT8_DTYPE, ensure_hif8_pts_linear_available

from .base import AscendLinearScheme
from .registry import register_scheme


@register_scheme("W8A8_HIF8_PTS", "linear")
class AscendW8A8HiF8PTSLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A8_HIF8_PTS (HiFloat8, per-tensor scaling).

    Weights arrive already quantized; activations are quantized in apply(),
    by dynamic per-tensor scaling ('pts') or a direct cast with unit scale ('cast').

    RL refit lifecycle, same contract as w8a8_mxfp8.py:
    restore_weights_for_rl_loading() -> model.load_weights() ->
    process_weights_after_loading(), once per policy sync.
    """

    model_dtype = None
    _act_mode_logged = None

    def __init__(self):
        ensure_hif8_pts_linear_available("W8A8_HIF8_PTS linear quantization")
        vllm_config = get_current_vllm_config()
        self.dst_type_max = float(vllm_config.quant_config.quant_description.get("hif8_dst_type_max", 16.0))
        # 'pts' dynamically scales each activation tensor; 'cast' uses unit scale.
        self.act_mode = str(vllm_config.quant_config.quant_description.get("hif8_act_mode", "pts"))
        if self.act_mode not in ("pts", "cast"):
            raise ValueError(
                f"[W8A8_HIF8_PTS] unknown hif8_act_mode {self.act_mode!r}; expected 'pts' or 'cast'"
            )
        # Allocated lazily and then retained, so its data_ptr survives ACL graph replay.
        self._cast_ones_scale = None
        if type(self)._act_mode_logged != self.act_mode:
            logger.info("[W8A8_HIF8_PTS] activation mode: %s", self.act_mode)
            type(self)._act_mode_logged = self.act_mode

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        # uint8 carrier holding HiF8 bytes, loader layout [N, K]
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.uint8)}
        return params_dict

    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        # One per-tensor scalar expanded to [N, 1] so the stock per-channel loader
        # concatenates fused q/k/v (gate/up) shards and slices TP for us.
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        # cast mode keeps every per-forward amax/scale computation out of the captured graph.
        if self.act_mode == "cast":
            quantized_x = torch_npu.npu_dtype_cast(x, dtype=HIFLOAT8_DTYPE)
            if self._cast_ones_scale is None or self._cast_ones_scale.device != x.device:
                self._cast_ones_scale = torch.ones(1, dtype=torch.float32, device=x.device)
            pertensor_scale = self._cast_ones_scale
        else:
            quantized_x, pertensor_scale = torch_npu.npu_dynamic_quant(
                x,
                dst_type=HIFLOAT8_DTYPE,
                quant_mode="pertensor",
                dst_type_max=self.dst_type_max,
            )

        if bias is not None and bias.dtype != torch.float32:
            # The HiF8 quantized matmul kernel requires float32 bias.
            bias = bias.to(torch.float32)

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=pertensor_scale,
            bias=bias,
            output_dtype=x.dtype,
            x1_dtype=HIFLOAT8_DTYPE,
            x2_dtype=HIFLOAT8_DTYPE,
        )
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)

        return output

    def process_weights_after_loading(self, layer):
        """Transform loader-layout params into kernel layout."""

        if getattr(layer, "_hif8_transformed", False):
            return

        if not hasattr(layer, "_hif8_original_shapes"):
            layer._hif8_original_shapes = {
                "weight": tuple(layer.weight.data.shape),
                "weight_scale": tuple(layer.weight_scale.data.shape),
            }

        if not hasattr(layer, "hif8_weight_buf"):
            layer.hif8_weight_buf = layer.weight.data.transpose(0, 1).contiguous()
        else:
            # RL reload path: copy in place so the ACL graph's captured data_ptr never changes.
            layer.hif8_weight_buf.copy_(layer.weight.data.transpose(0, 1).contiguous())
        layer.weight.data = layer.hif8_weight_buf

        # [N, 1] -> (N,): pure view, same storage, data_ptr unchanged
        layer.weight_scale.data = layer.weight_scale.data.reshape(-1)

        layer._hif8_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Restore loader-layout shapes before model.load_weights() in RL refit."""

        if not getattr(layer, "_hif8_transformed", False):
            return

        if not hasattr(layer, "_hif8_original_shapes"):
            err_msg = (
                "[vllm-ascend/W8A8_HIF8_PTS] Cannot restore weights: original "
                "shapes not recorded. "
                "This should not happen if process_weights_after_loading was called first."
            )
            raise RuntimeError(err_msg)

        orig_scale_shape = layer._hif8_original_shapes["weight_scale"]

        # [N, K] view over the SAME buffer (no .contiguous()), so the loader's
        # writes land in the storage the ACL graph captured.
        target_weight = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight.data.copy_(target_weight)

        layer.weight_scale.data = layer.weight_scale.data.reshape(orig_scale_shape)

        layer._hif8_transformed = False
