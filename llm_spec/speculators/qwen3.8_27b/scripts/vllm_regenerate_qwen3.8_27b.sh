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

# 启动 Qwen3.8-27B response regeneration 服务（不启用 hidden-states 提取）。
# 用法: MODEL_PATH=/path/to/Qwen3.8-27B bash vllm_regenerate_qwen3.8_27b.sh
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1

MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3.8-27B}
PORT=${PORT:-8000}

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --max-num-seqs 256 \
    --max-model-len 33792 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill \
    --trust-remote-code \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}'
