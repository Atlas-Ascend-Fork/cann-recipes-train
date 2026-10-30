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

# GLM-5.2-W4A8C8 DSpark hidden-states 提取服务（每个引擎节点执行一次，后台运行）。
#
# 用法：
#   NODE_IP=10.0.0.1 DP_ADDRESS=10.0.0.1 DP_RANK_START=0 \
#   MOONCAKE_MASTER=10.0.0.3:50051 MODEL_PATH=/data/GLM-5.2-w4a8c8 \
#     bash scripts/serve_vllm_hs_glm5.2.sh
# Node1 设置 DP_RANK_START=4，并连接另一个 master（双 master 见 README §7.1）。
#
# 日志：./hs_engines.log；PID：./hs_engines.pid
# 停止：kill $(cat hs_engines.pid)
set -euo pipefail

# 在样例目录执行（日志/PID 落在样例目录，不进 scripts/）
cd "$(dirname "$0")/.."

# ============ 配置区（均可用环境变量覆盖）============
NODE_IP=${NODE_IP:-}                                          # 必填：本机对外 IP
DP_ADDRESS=${DP_ADDRESS:-}                                    # 必填：master 那台的 IP，两台填同一个值
DP_RANK_START=${DP_RANK_START:-}                              # 必填：Node0 填 0，Node1 填 4
MOONCAKE_MASTER=${MOONCAKE_MASTER:-}                          # 必填：本机引擎连接的 master；双 master 时两台不同（README §7.1）
MODEL_PATH=${MODEL_PATH:-/data/GLM-5.2-w4a8c8}                # 量化权重目录；须与训练侧 --verifier-name-or-path 一致
TARGET_LAYER_IDS=${TARGET_LAYER_IDS:-"5 23 42 60 78"}         # 取层，须与训练侧 --target-layer-ids 一致
EAGLE_LAYERS=${TARGET_LAYER_IDS// /,}                         # speculative-config 的 eagle_aux_hidden_state_layer_ids 用逗号形式
DP_SIZE=${DP_SIZE:-8}                                         # 全局 DP（跨两台）
DP_SIZE_LOCAL=${DP_SIZE_LOCAL:-4}                             # 本节点引擎实例数
TP_SIZE=${TP_SIZE:-4}                                         # 每实例卡数
DP_RPC_PORT=${DP_RPC_PORT:-10521}                             # 引擎间 DP RPC 端口，两台须填同一个值
VLLM_START_PORT=${VLLM_START_PORT:-6700}
MC_HANDSHAKE_PORT_BASE=${MC_HANDSHAKE_PORT_BASE:-51010}       # 每实例唯一，master 回连用
MOONCAKE_GLOBAL_SEGMENT_GIB=${MOONCAKE_GLOBAL_SEGMENT_GIB:-25}   # 每引擎注册内存；本机 4 引擎合计 100 GiB
MOONCAKE_LOCAL_BUFFER_GIB=${MOONCAKE_LOCAL_BUFFER_GIB:-4}
MOONCAKE_WRITER_THREADS=${MOONCAKE_WRITER_THREADS:-8}
nic_name=${nic_name:-bond0}                                   # 改成自己的网卡名
LOG=${LOG:-./hs_engines.log}
PIDFILE=${PIDFILE:-./hs_engines.pid}
# ====================================================

# 必填项检查（空串同样拦下：set -u 只拦未设置，拦不住空字符串）
: "${NODE_IP:?必填：本机对外 IP}" "${DP_ADDRESS:?必填：master 那台的 IP，两台填同一个值}" \
  "${DP_RANK_START:?必填：Node0 填 0，Node1 填 DP_SIZE_LOCAL（默认 4）}" \
  "${MOONCAKE_MASTER:?必填：本机引擎连接的 master}"

IP=$NODE_IP

echo "[hs] 本机 IP=$IP  layers=[$EAGLE_LAYERS]  dp-size=$DP_SIZE 本机 $DP_SIZE_LOCAL×tp$TP_SIZE" \
     "dp-rank-start=$DP_RANK_START 端口 $VLLM_START_PORT-$((VLLM_START_PORT+DP_SIZE_LOCAL-1))" \
     "mooncake master=$MOONCAKE_MASTER 握手端口 $MC_HANDSHAKE_PORT_BASE-$((MC_HANDSHAKE_PORT_BASE+DP_SIZE_LOCAL-1))"

# ---- 引擎环境 ----
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$IP
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=400
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/lib
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTHONHASHSEED=0
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export MOONCAKE_LOCAL_HOSTNAME=$IP

: > "$PIDFILE"
for i in $(seq 0 $((DP_SIZE_LOCAL - 1))); do
  port=$((VLLM_START_PORT + i))
  rank=$((DP_RANK_START + i))
  dev=$(seq -s, $((i * TP_SIZE)) $(((i + 1) * TP_SIZE - 1)))
  hs_port=$((MC_HANDSHAKE_PORT_BASE + i))

  CMD=(vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$port" \
      --data-parallel-size "$DP_SIZE" \
      --data-parallel-rank "$rank" \
      --data-parallel-address "$DP_ADDRESS" \
      --data-parallel-rpc-port "$DP_RPC_PORT" \
      --tensor-parallel-size "$TP_SIZE" \
      --enable-expert-parallel \
      --speculative-config "{\"method\":\"extract_hidden_states\",\"num_speculative_tokens\":1,\"draft_model_config\":{\"hf_config\":{\"eagle_aux_hidden_state_layer_ids\":[${EAGLE_LAYERS}]}}}" \
      --seed 42 \
      --max-model-len 65536 \
      --additional-config '{"recompute_scheduler_enable": false,"multistream_overlap_shared_expert": true, "enable_dsa_cp":false,"enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true,"c8_enable_reshape_optim":false}' \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --max-num-seqs 32 \
      --quantization ascend \
      --gpu-memory-utilization 0.88 \
      --enforce-eager \
      --kv-transfer-config \
      "{
          \"kv_connector\": \"MooncakeHiddenStatesConnector\",
          \"kv_role\": \"kv_producer\",
          \"kv_connector_module_path\": \"hs_connectors.mooncake_hidden_states_connector\",
          \"kv_connector_extra_config\": {
              \"mooncake\": {
                  \"local_hostname\": \"$IP\",
                  \"metadata_server\": \"P2PHANDSHAKE\",
                  \"master_server_address\": \"$MOONCAKE_MASTER\",
                  \"global_segment_size\": $((MOONCAKE_GLOBAL_SEGMENT_GIB * 1073741824)),
                  \"local_buffer_size\": $((MOONCAKE_LOCAL_BUFFER_GIB * 1073741824)),
                  \"protocol\": \"tcp\",
                  \"device_name\": \"\",
                  \"num_writer_threads\": $MOONCAKE_WRITER_THREADS
              }
          }
      }")

  echo "[hs] engine dp_rank=$rank port=$port devices=$dev handshake=$hs_port -> $LOG"
  MC_HANDSHAKE_PORT="$hs_port" ASCEND_RT_VISIBLE_DEVICES="$dev" setsid nohup "${CMD[@]}" >> "$LOG" 2>&1 &
  echo $! >> "$PIDFILE"
done

echo "[hs] $DP_SIZE_LOCAL 个引擎已后台拉起，PID 记在 $PIDFILE，日志 $LOG"
