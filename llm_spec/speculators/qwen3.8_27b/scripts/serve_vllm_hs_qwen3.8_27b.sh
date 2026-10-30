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

# 启动单个 vLLM hidden-states 提取服务（DSpark 草稿在线训练的 serving 侧，后台运行）。
# 适用 speculators >= e8f8879（launch_vllm.py 采用「自有参数 -- vLLM 透传参数」接口）。
#
# 用法（默认值写死在下方配置区，均可用环境变量覆盖；同机多实例时覆盖
#       PORT/DEVICES/MC_HANDSHAKE_PORT）：
#   PORT=8010 DEVICES=0,1 MODEL_PATH=/data/models/Qwen3.8-27B \
#   MOONCAKE_LOCAL_HOSTNAME=<本机IP> MC_HANDSHAKE_PORT=51010 \
#     bash scripts/serve_vllm_hs_qwen3.8_27b.sh
#
# 日志：./serve_vllm_hs_<port>.log；PID：./serve_vllm_hs_<port>.pid
# 停止：kill $(cat serve_vllm_hs_<port>.pid)
#
# 布局建议（单机 8 卡）：
#   实例 A：卡 0,1 -> 端口 8010；实例 B：卡 2,3 -> 端口 8011；卡 4~7 留给训练
#   （train_dspark_qwen3.8_27b_online.sh，4 rank）。多机部署时每节点照此起 1~2 个实例。
#
# hidden-states 传输 backend 由 HS_BACKEND 选择，默认 mooncake（官方 backend）：
#   - mooncake：vLLM 侧 connector 将 hidden-states 写入 mooncake 分布式 store，
#     训练侧按 req_id 经传输引擎直连拉取，无需共享文件系统。前置条件：
#       ① vLLM 与训练两侧环境均已安装 hs_connectors（仓库根 pip install -e
#          hs_connectors）与 mooncake-transfer-engine-npu；
#       ② 任一节点常驻 mooncake_master（地址见 MOONCAKE_MASTER，默认
#          127.0.0.1:50051），网络放通 各节点->master 与 master->各节点握手端口；
#       ③ MOONCAKE_LOCAL_HOSTNAME 必须显式设为本机可路由 IP——不设会注册成
#          127.0.0.1，master 侧出现幽灵 segment，跨机传输报 TRANSFER_FAIL(-800)；
#       ④ 同机多实例时 MC_HANDSHAKE_PORT 每实例必须唯一（如 51010/51011）。
#   - file：单机零依赖。写 {req_id}.safetensors 到 --hidden-states-path（HS_PATH），
#     训练侧与 vLLM 需共享文件系统。HS_PATH 建议 /dev/shm（tmpfs），容量账约
#     61KB/token，100G ≈ 1.6M token，训练侧不消费会涨满，服务重启前先清空目录。
#
# 调优：--enable-chunked-prefill + max-num-batched-tokens
#     32768 提取吞吐 +25~35%、TTFT max 约 14s -> 10s；要求 vLLM 的
#     ExampleHiddenStatesConnector 支持 chunked prefill（vllm#43805 及以后）。
#   - 不要设置 --served-model-name：训练客户端会校验 GET /v1/models 的 id 与
#     --verifier-name-or-path 一致，改名会被拒（"doesn't match found model_id"）。
set -euo pipefail

# 在 speculators 仓库根目录执行（launch_vllm.py 为仓库内相对路径）
cd "$(dirname "$0")/.."

# ============ 配置区（均可用环境变量覆盖）============
PORT=${PORT:-8010}
DEVICES=${DEVICES:-0,1}                            # 本实例占用的 NPU 卡
MODEL_PATH=${MODEL_PATH:-/data/models/Qwen3.8-27B}
LOG=${LOG:-./serve_vllm_hs_${PORT}.log}
PIDFILE=${PIDFILE:-./serve_vllm_hs_${PORT}.pid}

# hidden-states backend（mooncake | file）
HS_BACKEND=${HS_BACKEND:-mooncake}

# mooncake 参数（HS_BACKEND=mooncake 时生效；训练侧须指向同一 master）
MOONCAKE_MASTER=${MOONCAKE_MASTER:-127.0.0.1:50051}
MOONCAKE_METADATA_SERVER=${MOONCAKE_METADATA_SERVER:-P2PHANDSHAKE}   # 免 metadata server
MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}                          # tcp | rdma
MOONCAKE_GLOBAL_SEGMENT_GIB=${MOONCAKE_GLOBAL_SEGMENT_GIB:-8}        # 每客户端注册的全局可见内存
MOONCAKE_LOCAL_BUFFER_GIB=${MOONCAKE_LOCAL_BUFFER_GIB:-4}            # 每客户端本地暂存 buffer
MOONCAKE_WRITER_THREADS=${MOONCAKE_WRITER_THREADS:-8}                # vLLM 侧异步写线程数
MC_HANDSHAKE_PORT=${MC_HANDSHAKE_PORT:-}            # 引擎握手端口，同机多实例时每实例唯一

# file backend 参数（HS_BACKEND=file 时生效）
HS_PATH=${HS_PATH:-/dev/shm/hs_${PORT}/}

# Qwen3.8-27B（64 层）提取 5 层，末层 64 由 launch_vllm 自动追加；
# 换模型时须与训练侧 --target-layer-ids 严格一致
TARGET_LAYER_IDS=${TARGET_LAYER_IDS:-"1 16 31 46 61"}

# vLLM 实例参数
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
# 9216 = 8192 训练 + 1024 余量（模型 max_position_embeddings 无上限问题）
MAX_MODEL_LEN=${MAX_MODEL_LEN:-9216}
TP=${TP:-2}
# 量化 verifier（如 w8a8 ascend 量化模型）时设置: EXTRA_VLLM_ARGS="--quantization ascend"
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-}
# ====================================================

# CANN 环境（pyACL 的 acl 模块等；注意 ASCEND_HOME_PATH 已设不等于已 source，
# 这里无条件 source，幂等）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1

BACKEND_ARGS=(--hidden-states-backend "$HS_BACKEND")
if [ "$HS_BACKEND" = "mooncake" ]; then
    : "${MOONCAKE_LOCAL_HOSTNAME:?HS_BACKEND=mooncake 时必须设 MOONCAKE_LOCAL_HOSTNAME=<本机可路由 IP>}"
    export MOONCAKE_LOCAL_HOSTNAME
    [ -n "$MC_HANDSHAKE_PORT" ] && export MC_HANDSHAKE_PORT
    BACKEND_ARGS+=(
        --mooncake-master "$MOONCAKE_MASTER"
        --mooncake-metadata-server "$MOONCAKE_METADATA_SERVER"
        --mooncake-protocol "$MOONCAKE_PROTOCOL"
        --mooncake-global-segment-gib "$MOONCAKE_GLOBAL_SEGMENT_GIB"
        --mooncake-local-buffer-gib "$MOONCAKE_LOCAL_BUFFER_GIB"
        --mooncake-writer-threads "$MOONCAKE_WRITER_THREADS"
    )
else
    mkdir -p "$HS_PATH"
    BACKEND_ARGS+=(--hidden-states-path "$HS_PATH")
fi

echo "[hs] port=$PORT  devices=$DEVICES  backend=$HS_BACKEND  layers=[$TARGET_LAYER_IDS]  master=$MOONCAKE_MASTER"

CMD=(python scripts/launch_vllm.py train "$MODEL_PATH" \
    "${BACKEND_ARGS[@]}" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --trust-remote-code \
    -- \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill \
    --seed 42 \
    --distributed-executor-backend mp \
    --async-scheduling \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}' \
    ${EXTRA_VLLM_ARGS})

setsid nohup "${CMD[@]}" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "[hs] pid=$(cat "$PIDFILE")  log=$LOG  pidfile=$PIDFILE"
