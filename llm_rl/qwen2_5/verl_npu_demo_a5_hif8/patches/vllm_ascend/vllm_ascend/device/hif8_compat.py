# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

# HiF8 capability checks, following vllm-ascend's mxfp_compat.py interface.
import torch_npu

# An integer dtype code, not a torch.dtype. None when this torch_npu build has no HiF8.
HIFLOAT8_DTYPE = getattr(torch_npu, "hifloat8", None)


def _get_missing_symbols(symbols: tuple[str, ...]) -> list[str]:
    return [symbol for symbol in symbols if not hasattr(torch_npu, symbol)]


def _ensure_symbols_available(feature: str, symbols: tuple[str, ...]) -> None:
    missing_symbols = _get_missing_symbols(symbols)
    if not missing_symbols:
        return
    missing_symbols_str = ", ".join(missing_symbols)
    raise RuntimeError(
        f"{feature} requires a torch_npu runtime with HiF8 support. "
        f"Missing symbols: {missing_symbols_str}. "
        "Please upgrade torch_npu or disable HiF8 quantization."
    )


def ensure_hif8_pts_linear_available(feature: str) -> None:
    """Check the dtype and operator symbols used for HiF8 per-tensor quantization."""
    _ensure_symbols_available(feature, ("hifloat8", "npu_dynamic_quant", "npu_quant_matmul", "npu_dtype_cast"))
