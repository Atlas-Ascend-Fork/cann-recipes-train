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
# Apply the A5 low-precision rollout changes to the fetched framework trees.
# Per repository convention: patches/<framework>/*.patch are git patches,
# patches/<framework>/<path>.py are whole files copied in.
# Every patch is dry-run first, so a tree at the wrong commit fails early.

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PATCH_ROOT="${SCRIPT_DIR}/patches"

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}
VERL_SRC_DIR=${VERL_SRC_DIR:-${WORKSPACE_DIR}/verl}
VLLM_ASCEND_SRC_DIR=${VLLM_ASCEND_SRC_DIR:-${WORKSPACE_DIR}/vllm-ascend}

apply_patch() {  # <patch file> <framework tree>
    echo -n "Applying $(basename "$1") to $2 ... "
    [[ -d "$2" ]] || { echo "[FAIL]: $2 not found" >&2; exit 1; }
    git -C "$2" apply --check "$1" 2>/dev/null \
        || { echo "[FAIL]: dry run failed, check the tree is at the pinned commit" >&2; exit 1; }
    git -C "$2" apply "$1"
    echo "[OK]"
}

copy_file() {  # <source file> <path inside the tree> <framework tree>
    echo -n "Copying $2 to $3 ... "
    mkdir -p "$3/$(dirname "$2")"
    cp "$1" "$3/$2"
    echo "[OK]"
}

# verl: route HiF8 weights through the quantized refit path.
apply_patch "${PATCH_ROOT}/verl/0001-verl-feature-hif8_rollout.patch" "${VERL_SRC_DIR}"

# vllm-ascend: whole files first, then the registration patch that imports them.
for rel in \
    vllm_ascend/device/hif8_compat.py \
    vllm_ascend/quantization/methods/w8a8_hif8.py
do
    copy_file "${PATCH_ROOT}/vllm_ascend/${rel}" "${rel}" "${VLLM_ASCEND_SRC_DIR}"
done

apply_patch "${PATCH_ROOT}/vllm_ascend/0001-vllm_ascend-feature-register_hif8.patch" "${VLLM_ASCEND_SRC_DIR}"

echo "All patches applied."
