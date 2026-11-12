#!/bin/bash
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

set -ex

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

VERL_SRC_DIR=${VERL_SRC_DIR:-/workspace/verl}
TORCHTITAN_SRC_DIR=${TORCHTITAN_SRC_DIR:-/workspace/torchtitan}
TORCHTITAN_NPU_SRC_DIR=${TORCHTITAN_NPU_SRC_DIR:-/workspace/torchtitan-npu}
VLLM_ASCEND_SRC_DIR=${VLLM_ASCEND_SRC_DIR:-/workspace/vllm-ascend}

cd "${SCRIPT_DIR}"
cp -r "${VERL_SRC_DIR}/verl" ./
cp -r "${TORCHTITAN_SRC_DIR}/torchtitan" ./
cp -r "${TORCHTITAN_NPU_SRC_DIR}/torchtitan_npu" ./
cp -r "${VLLM_ASCEND_SRC_DIR}" ./vllm-ascend

python --version
python3 --version
python3 -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org
python3 -m pip install torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org
python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt"

bash "${SCRIPT_DIR}/apply_all_patches.sh"

if python3 -m pip list --format=freeze | grep -qx 'vllm==0.15.0+empty'; then
    cd "${SCRIPT_DIR}/vllm-ascend"
    pip install -e . --no-deps --no-build-isolation

    cd "${SCRIPT_DIR}"
    cp -r ./vllm-ascend/vllm_ascend ./
else
    echo "Skip vllm-ascend build because vllm==0.15.0+empty is not installed."
fi

ls -l
