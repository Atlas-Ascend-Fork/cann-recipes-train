#!/usr/bin/env bash
# coding=utf-8
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

# Feature Extraction Launch Script
#
# This script launches feature extraction using the same environment setup
# as executor/scripts/infer.sh, ensuring compatibility with the existing
# inference infrastructure.
#
# Usage:
#   bash extract_features.sh <config> <input_jsonl> <output_dir> [target_layers] [--mean-hc-mult|--no-mean-hc-mult]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MODULE_DIR="${SCRIPT_DIR}"
RECIPES_ROOT=$(cd "${MODULE_DIR}/../.." &>/dev/null && pwd)
MODEL_DIR="${RECIPES_ROOT}/models/deepseek_v4"

# ---- Parse arguments ----
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <config> <input_jsonl> <output_dir> [target_layers] [--mean-hc-mult|--no-mean-hc-mult]"
    echo ""
    echo "Example:"
    echo "  $0 config/extract_deepseek_v4.yaml \\"
    echo "     train_datasets/data.jsonl \\"
    echo "     ~/.cache/deepspec/dsv4_cache \\"
    echo "     \"40 41 42\""
    exit 1
fi

CONFIG="${1}"
INPUT_JSONL="${2}"
OUTPUT_DIR="${3}"
TARGET_LAYERS="${4:-40 41 42}"
MEAN_HC_MULT_ARG=()
case "${5:-}" in
    --mean-hc-mult|--no-mean-hc-mult)
        MEAN_HC_MULT_ARG=("${5}")
        ;;
esac

# Validate inputs
if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: Config file not found: ${CONFIG}"
    exit 1
fi

if [[ ! -f "${INPUT_JSONL}" ]]; then
    echo "Error: Input JSONL file not found: ${INPUT_JSONL}"
    exit 1
fi

# ---- Environment setup (same as infer.sh) ----
export PYTHONPATH="${RECIPES_ROOT}:${PYTHONPATH:-}"

# Source model-specific environment if available
MODEL_SET_ENV="${MODEL_DIR}/set_env.sh"
if [[ -f "${MODEL_SET_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${MODEL_SET_ENV}"
fi

# Detect platform version from NPU device name
detect_platform_version() {
    if [[ -n "${DETECTED_PLATFORM_VERSION:-}" ]]; then
        return
    fi
    local device_name
    device_name=$(python3 -c "import torch;import torch_npu; print(torch.npu.get_device_name())" 2>/dev/null || true)
    if [[ -z "${device_name}" ]]; then
        echo "[WARN] torch.npu.get_device_name() failed, defaulting platform to 950"
        export DETECTED_PLATFORM_VERSION="950"
        return
    fi
    if [[ "${device_name}" == *"950"* ]]; then
        export DETECTED_PLATFORM_VERSION="950"
    elif [[ "${device_name}" == *"910_93"* ]]; then
        export DETECTED_PLATFORM_VERSION="A3"
    elif [[ "${device_name}" == *"910B"* ]]; then
        export DETECTED_PLATFORM_VERSION="A2"
    else:
        echo "[WARN] Unknown device: ${device_name}, defaulting platform to 950"
        export DETECTED_PLATFORM_VERSION="950"
    fi
    echo "[INFO] Detected platform version: ${DETECTED_PLATFORM_VERSION} (device: ${device_name})"
}

detect_platform_version

# Count NPU devices
if python3 -c "import torch; torch.npu.is_available()" 2>/dev/null; then
    NPROCS=$(python3 -c "import torch; print(torch.npu.device_count())")
else
    NPROCS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
fi
NPROCS=${NPROCS:-1}
echo "[INFO] NPU count: ${NPROCS}"

# HCCL setup
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
LOCAL_HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
export HCCL_IF_IP=${HCCL_IF_IP:-${LOCAL_HOST}}
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

# Platform-specific HCCL configuration (matches executor/scripts/function.sh:set_hccl)
# Read micro_batch_mode from config
micro_batch_mode=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}')).get('model_config', {}).get('micro_batch_mode', 0))" 2>/dev/null || echo "0")

# 910c/A3 platforms need HCCL AIV mode (not 950)
if [[ "${DETECTED_PLATFORM_VERSION}" != "950" ]]; then
    export HCCL_OP_EXPANSION_MODE=AIV
    echo "[INFO] Set HCCL_OP_EXPANSION_MODE=AIV for platform ${DETECTED_PLATFORM_VERSION}"
fi

# Micro-batch mode requires unsetting HCCL_OP_EXPANSION_MODE
if [[ ${micro_batch_mode} -eq 1 ]]; then
    unset HCCL_OP_EXPANSION_MODE
    echo "[INFO] Unset HCCL_OP_EXPANSION_MODE (micro_batch_mode=1)"
fi

# ---- Launch setup (matches function.sh:launch_infer_task) ----
export MASTER_ADDR="${MASTER_ADDR:-${LOCAL_HOST}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-23456}"
export WORLD_SIZE="${NPROCS}"
export LOCAL_WORLD_SIZE="${NPROCS}"
export RANK_OFFSET="${RANK_OFFSET:-0}"

cores=$(nproc)
avg_core_per_rank=$(( cores / NPROCS ))
core_gap=$(( avg_core_per_rank - 1 ))
[[ $core_gap -lt 0 ]] && core_gap=0

# ---- Info ----
echo "========================================="
echo "DeepSeek-V4 Feature Extraction"
echo "  Config: ${CONFIG}"
echo "  Input:  ${INPUT_JSONL}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Target layers: ${TARGET_LAYERS}"
echo "  NPU count: ${NPROCS}"
echo "  Platform: ${DETECTED_PLATFORM_VERSION}"
echo "========================================="

mkdir -p "$(dirname "${OUTPUT_DIR}")" 2>/dev/null || true

# ---- Launch per-NPU processes (matches function.sh pattern) ----
echo "[INFO] Launching ${NPROCS} ranks..."

for ((i = 0; i < NPROCS; i++)); do
    export LOCAL_RANK=$i
    export RANK_ID=$(( i + RANK_OFFSET ))
    start=$(( i * avg_core_per_rank ))
    end=$(( start + core_gap ))
    cmdopt="${start}-${end}"

    taskset -c "${cmdopt}" python3 "${MODULE_DIR}/extract_features.py" \
        --config "${CONFIG}" \
        --input-file "${INPUT_JSONL}" \
        --output-dir "${OUTPUT_DIR}" \
        --target-layer-ids ${TARGET_LAYERS} \
        "${MEAN_HC_MULT_ARG[@]}" &
done

wait

echo "[INFO] Feature extraction completed."
echo "[INFO] Output written to: ${OUTPUT_DIR}"
