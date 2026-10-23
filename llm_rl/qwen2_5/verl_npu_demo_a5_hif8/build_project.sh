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

#!/bin/bash
# Install the A5 low-precision rollout stack. CANN 9.2.0 must already be sourced.
# vLLM and vLLM-Ascend are built from source at the pinned commits; the pip
# packages used by the A2/A3 sample are not interchangeable with them.

set -ex

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}
VERL_SRC_DIR=${VERL_SRC_DIR:-${WORKSPACE_DIR}/verl}
VLLM_SRC_DIR=${VLLM_SRC_DIR:-${WORKSPACE_DIR}/vllm}
VLLM_ASCEND_SRC_DIR=${VLLM_ASCEND_SRC_DIR:-${WORKSPACE_DIR}/vllm-ascend}

python3 --version

bash "${SCRIPT_DIR}/apply_all_patches.sh"

python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt"

# vLLM first, as an empty skeleton: vllm-ascend builds against it and provides the kernels.
cd "${VLLM_SRC_DIR}"
VLLM_TARGET_DEVICE=empty python3 -m pip install -e . --no-build-isolation

# SOC_VERSION must match the hardware or the kernels build but fail to load.
cd "${VLLM_ASCEND_SRC_DIR}"
export SOC_VERSION=${SOC_VERSION:-ascend950pr_9579}
export COMPILE_CUSTOM_KERNELS=${COMPILE_CUSTOM_KERNELS:-1}
python3 -m pip install -e . --no-deps --no-build-isolation

cd "${VERL_SRC_DIR}"
python3 -m pip install -e . --no-deps

# Registered by an import side effect; a silent failure surfaces much later as an unquantized run.
python3 -c "
from vllm_ascend.quantization.methods import AscendW8A8HiF8PTSLinearMethod
print('W8A8_HIF8_PTS scheme registered:', AscendW8A8HiF8PTSLinearMethod.__name__)
"

# verl 0.9 reaches the rollout through vLLM's OpenAI server entrypoint.
python3 -c "import vllm.entrypoints.cli.serve; print('vllm serve entrypoint ok')"

python3 -m pip list --format=freeze | grep -E '^(verl|vllm|vllm-ascend|torch|torch-npu|math-verify)=' || true
