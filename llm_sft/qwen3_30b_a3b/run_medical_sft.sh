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
#
# Qwen3-30B-A3B

set -e

NGPU=${NGPU:-16}
MODULE=${MODULE:-torchtitan_npu.models.qwen3}
CONFIG=${CONFIG:-sft_qwen3_30ba3b_medical_tnd}
TRAIN_DATA=${TRAIN_DATA:-./assets/medical_r1/train.jsonl}
MODEL_DIR=${MODEL_DIR:-./assets/hf/Qwen3-30B-A3B}
LOG_RANK=${LOG_RANK:-0}

cd "$(dirname "$0")"

echo "=== Medical SFT ==="
echo "NGPU=${NGPU}  MODULE=${MODULE}  CONFIG=${CONFIG}"
echo "TRAIN_DATA=${TRAIN_DATA}"
echo "MODEL_DIR=${MODEL_DIR}"
echo ""

PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
CPU_AFFINITY_CONF=1 \
TASK_QUEUE_ENABLE=2 \
HCCL_CONNECT_TIMEOUT=3600 \
TRAIN_DATA="${TRAIN_DATA}" MODEL_DIR="${MODEL_DIR}" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
  -m torchtitan_npu.entry --module ${MODULE} --config ${CONFIG}
