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

# DeepSeek-V4-Flash single-node HiF8 low-precision pretrain launcher for Ascend 950.
#
# Copy this script into torchtitan-npu/scripts/ and run it FROM the torchtitan-npu
# source root (same convention as run_train_dsv4_flash_A5_BF16_pretrain.sh):
#     cd /home/code/torchtitan-npu
#     bash scripts/run_train_dsv4_flash_A5_HiF8_single_node.sh
#
# It targets the HiF8 QAT config `debug_deepseek_v4_flash_single_node_hif8_qat`
# in the (tracked) benchmark file
#   torchtitan_npu/experiments/ao_npu/benchmarks/e2e/dsv4_flash_single_node/config_registry.py
# That directory has no __init__.py chain, so the config is only importable as the
# flat top-level module `config_registry` after cd-ing into it and putting the
# torchtitan-npu repo root on PYTHONPATH -- both handled below.
#
# Requirements:
#   * Ascend 950 (HiF8 has no python-level HW guard; non-A5 fails at op call)
#   * torchtitan-npu on branch `master` with its deps installed
#     (`pip install -r requirements.txt`, notably torchao==0.17.0)
#   * CANN env sourced BEFORE running (this script also sources the defaults below)
#
# Overridable env vars:
#   NGPU            NPUs for this node                     (default 8)
#   DATASET         torchtitan dataset registry key       (default c4_test, ships with torchtitan-npu)
#   DATASET_PATH    absolute path to the dataset folder   (default ${REPO_ROOT}/tests/assets/c4_test)
#   HF_ASSETS_PATH  tokenizer folder (tokenizer.json ...) (default /data/models/DeepSeek-V4-Flash)
#   OUTPUT_FOLDER   run output / profiling / log root     (default ${BENCH_DIR}/outputs/A5_hif8_<ts>)

# TODO change to your environment
source /usr/local/Ascend/cann/set_env.sh

set -euo pipefail

# Run from the torchtitan-npu source root (cwd), same as the sibling A5 scripts.
REPO_ROOT="$(pwd)"
BENCH_DIR="${REPO_ROOT}/torchtitan_npu/experiments/ao_npu/benchmarks/e2e/dsv4_flash_single_node"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export PYTHONUNBUFFERED=1

NGPU="${NGPU:-8}"
export LOG_RANK="${LOG_RANK:-0}"
MODULE="${MODULE:-config_registry}"
CONFIG="${CONFIG:-debug_deepseek_v4_flash_single_node_hif8_qat}"
TRAIN_FILE="${TRAIN_FILE:-torchtitan_npu.entry}"
COMM_MODE="${COMM_MODE:-}"

DATASET="${DATASET:-c4_test}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/tests/assets/c4_test}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/data/models/DeepSeek-V4-Flash}"

OUTPUT_FOLDER="${OUTPUT_FOLDER:-${BENCH_DIR}/outputs/A5_hif8_${TIMESTAMP}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_FOLDER}/A5_hif8_${TIMESTAMP}.log}"
mkdir -p "${OUTPUT_FOLDER}"

# Project-local compile caches only -- safe to wipe, scoped to the benchmark dir.
rm -rf "${BENCH_DIR}/torchinductor_root" "${BENCH_DIR}/torch_compile_debug" "${BENCH_DIR}/.npu_kernels_root"

EXTRA_ARGS=(
    --hf_assets_path "${HF_ASSETS_PATH}"
    --dump_folder "${OUTPUT_FOLDER}"
    --checkpoint.no_enable
    --profiling.enable_profiling
    --profiling.no_enable_online_parse
    --profiling.profile_ranks 0
    --profiling.profile_step_start 6
    --profiling.profile_step_end 7
    --profiling.profile_record_shapes
    --training.steps 10
    --training.global_batch_size 64
    --training.local_batch_size 1
    --training.num_mtp_modules 1
    "$@"
    dataloader:config
    --dataloader.dataset "${DATASET}"
    --dataloader.dataset_path "${DATASET_PATH}"
)

TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"

echo "==== Launch env ===="
echo "REPO_ROOT       = ${REPO_ROOT}"
echo "BENCH_DIR       = ${BENCH_DIR}"
echo "NGPU            = ${NGPU}"
echo "MODULE          = ${MODULE}"
echo "CONFIG          = ${CONFIG}"
echo "DATASET         = ${DATASET}"
echo "DATASET_PATH    = ${DATASET_PATH}"
echo "HF_ASSETS_PATH  = ${HF_ASSETS_PATH}"
echo "OUTPUT_FOLDER   = ${OUTPUT_FOLDER}"
echo "LOG_FILE        = ${LOG_FILE}"
echo "PYTHONPATH      = ${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
echo "===================="

# cd so the flat module name `config_registry` is importable as a top-level
# module (this benchmark dir has no __init__.py chain).
cd "${BENCH_DIR}"

if [ -n "${COMM_MODE}" ]; then
    echo "Running with comm_mode=${COMM_MODE}"
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m "${TRAIN_FILE}" \
        --module "${MODULE}" --config "${CONFIG}" \
        --comm.mode="${COMM_MODE}" --training.steps=1 "$@"
else
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    TASK_QUEUE_ENABLE=2 \
    HCCL_CONNECT_TIMEOUT=3600 \
    STREAMS_PER_DEVICE=32 \
    MULTI_STREAM_MEMORY_RESERVE=1 \
    TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
    torchrun --nproc_per_node="${NGPU}" --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
        --local-ranks-filter "${LOG_RANK}" --role rank --tee 3 \
        -m "${TRAIN_FILE}" --module "${MODULE}" --config "${CONFIG}" "${EXTRA_ARGS[@]}" \
        2>&1 | tee -a "${LOG_FILE}"
fi
