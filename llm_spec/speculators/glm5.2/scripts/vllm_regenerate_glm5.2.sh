#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

# 启动 GLM-5.2-W4A8C8 response regeneration 服务（不启用 hidden-states 提取），
# 供 §6.3 中文数据重生成使用；英文用 §6.1 的公开 regen 数据集，不需要本服务。
# 用法: MODEL_PATH=/path/to/GLM-5.2-W4A8C8 bash vllm_regenerate_glm5.2.sh
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=400
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/lib
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTHONHASHSEED=0

MODEL_PATH=${MODEL_PATH:-/data/GLM-5.2-w4a8c8}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-4}

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-expert-parallel \
    --seed 42 \
    --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --max-num-seqs 32 \
    --quantization ascend \
    --gpu-memory-utilization 0.88 \
    --enforce-eager \
    --additional-config '{"recompute_scheduler_enable": false,"multistream_overlap_shared_expert": true, "enable_dsa_cp":false,"enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true,"c8_enable_reshape_optim":false}'
