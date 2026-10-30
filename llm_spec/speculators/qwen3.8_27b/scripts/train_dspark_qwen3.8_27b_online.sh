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

# DSpark 草稿模型在线训练启动脚本（后台运行；两个样例共用同一份内容，文件各自带模型后缀）。
# 模型差异全部放在同目录的 train_config_<模型>.yaml（平铺 KEY: VALUE）：
# 本脚本按自身文件名 train_dspark_<模型>_online.sh 推导并读取它，
# torchrun 目标同样按文件名推导为 route_rank_endpoints_<模型>.py（模型名中的 . 写成 _）
# （按 LOCAL_RANK 给每个 rank 注入单一 --vllm-endpoint）。
#
# 用法：
#   bash scripts/train_dspark_<模型>_online.sh               # 自动读同目录 config
#   CONFIG=/path/to/other.yaml bash scripts/train_dspark_<模型>_online.sh
#   同名环境变量优先于 yaml 文件值（如 TRAIN_IP=<IP> bash ...）
#
# 日志：$OUTPUT_DIR/train.log；PID：$OUTPUT_DIR/train.pid
# 停止：kill $(cat $OUTPUT_DIR/train.pid)
#
# 关键约束：
#   - ENGINES 空格分隔的端点表（host:port），rank r 使用第 r % 条数 个，
#     分摊各 rank 的生成请求负载；mooncake 下 hidden-states 数据腿走 store，
#     与 endpoint 绑定无关
#   - TARGET_LAYER_IDS 必须与提取服务侧的取层完全一致（见各自 config）
#   - HS_BACKEND=mooncake 时：训练节点须与 vLLM 节点同装 hs_connectors 与
#     mooncake-transfer-engine-npu，且设 TRAIN_IP=<本机可路由 IP>
set -euo pipefail

# 在 speculators 仓库根目录执行（SCRIPT_DIR 先固化成绝对路径，任意目录调用均可）
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."

# ---- 读取模型配置（平铺 yaml；同名环境变量优先）----
TAG=$(basename "$0")                 # train_dspark_<模型>_online.sh
TAG=${TAG#train_dspark_}
TAG=${TAG%_online.sh}                # <模型>
CONFIG=${CONFIG:-"$SCRIPT_DIR/train_config_${TAG}.yaml"}
if [ ! -f "$CONFIG" ]; then
    echo "[train] 找不到配置文件 $CONFIG（可用 CONFIG=<路径> 指定）" >&2
    exit 1
fi
eval "$(python3 - "$CONFIG" <<'PYEOF'
import shlex, sys
try:
    import yaml
except ImportError:
    sys.exit("读取 train_config 需要 PyYAML：pip install pyyaml")
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
for key, value in cfg.items():
    if value is None:
        value = ""
    print(f"{key}=${{{key}-{shlex.quote(str(value))}}}")
PYEOF
)"

# 必备项检查（缺项时给出明确指向）
for v in VERIFIER DATA_PATH OUTPUT_DIR NPROC TRAIN_DEVICES ENGINES NUM_WORKERS \
         HS_BACKEND MOONCAKE_MASTER MOONCAKE_METADATA_SERVER MOONCAKE_PROTOCOL \
         MOONCAKE_GLOBAL_SEGMENT_GIB MOONCAKE_LOCAL_BUFFER_GIB \
         TARGET_LAYER_IDS NUM_LAYERS BLOCK_SIZE MAX_ANCHORS LOSS_FN \
         OPTIMIZER EPOCHS LR SEQ_LEN CHECKPOINT_FREQ; do
    : "${!v:?配置文件 $CONFIG 缺少 $v}"
done

# 可选项兜底（config 未定义时按空处理）
HOST_MASTERS=${HOST_MASTERS:-}
TRAIN_IP=${TRAIN_IP:-}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}
LOG=${LOG:-$OUTPUT_DIR/train.log}

SAVE=$OUTPUT_DIR/checkpoints
LOGDIR=$OUTPUT_DIR/logs
mkdir -p "$SAVE" "$LOGDIR"

export ENGINES

# 双 master 拓扑（按需，见 GLM5.2 config）：与 ENGINES 等长同序的 master 列表，
# wrapper 按 rank 注入对应的 --mooncake-master；HOST_MASTERS 留空则单 master
MOONCAKE_MASTER_ROUTES=""
if [ -n "$HOST_MASTERS" ]; then
  for e in $ENGINES; do
    e=${e#*://}    # 端点允许带 http:// 前缀（wrapper 兼容），先剥掉再取 host
    host=${e%%:*}
    m=""
    for pair in $HOST_MASTERS; do
      case "$pair" in
        "$host="*) m=${pair#*=} ;;
      esac
    done
    if [ -z "$m" ]; then
      echo "[train] HOST_MASTERS='$HOST_MASTERS' 里没有引擎机器 $host 的 master" >&2
      exit 1
    fi
    MOONCAKE_MASTER_ROUTES="${MOONCAKE_MASTER_ROUTES}${m},"
  done
  MOONCAKE_MASTER_ROUTES=${MOONCAKE_MASTER_ROUTES%,}
fi
export MOONCAKE_MASTER_ROUTES

if [ -n "$TRAIN_IP" ]; then
  export MOONCAKE_LOCAL_HOSTNAME="$TRAIN_IP"
fi

N_ENDPOINTS=$(echo $ENGINES | wc -w | tr -d ' ')
echo "[train] model=$TAG  草稿主干 $NUM_LAYERS 层  epochs=$EPOCHS  lr=$LR  $OPTIMIZER  ckpt_freq=$CHECKPOINT_FREQ  workers=$NUM_WORKERS"
echo "[train] layers='$TARGET_LAYER_IDS'  backend=$HS_BACKEND  endpoints=$N_ENDPOINTS 条（rank r 取第 r % $N_ENDPOINTS 条）"
if [ -n "$MOONCAKE_MASTER_ROUTES" ]; then
  echo "[train] mooncake 双 master：$(printf '%s\n' ${MOONCAKE_MASTER_ROUTES//,/ } | uniq -c | tr '\n' ' ')"
else
  echo "[train] mooncake master=$MOONCAKE_MASTER（单 master）"
fi
echo "[train] local_hostname=${TRAIN_IP:-<未设，按主机名解析>}"

export ASCEND_RT_VISIBLE_DEVICES="$TRAIN_DEVICES"

BACKEND_ARGS=(--hidden-states-backend "$HS_BACKEND")
if [ "$HS_BACKEND" = "mooncake" ]; then
    : "${MOONCAKE_LOCAL_HOSTNAME:?HS_BACKEND=mooncake 时必须设 TRAIN_IP=<本机可路由 IP>（环境变量）}"
    BACKEND_ARGS+=(
        --mooncake-master "$MOONCAKE_MASTER"
        --mooncake-metadata-server "$MOONCAKE_METADATA_SERVER"
        --mooncake-protocol "$MOONCAKE_PROTOCOL"
        --mooncake-global-segment-gib "$MOONCAKE_GLOBAL_SEGMENT_GIB"
        --mooncake-local-buffer-gib "$MOONCAKE_LOCAL_BUFFER_GIB"
    )
fi

# torchrun 通过 wrapper 按 LOCAL_RANK 注入 --vllm-endpoint；
# 模型专用 flag 由 config 的 TRAIN_EXTRA_ARGS 注入，命令行追加参数原样透传。
CMD=(torchrun --standalone --nproc-per-node "$NPROC" \
    "scripts/route_rank_endpoints_${TAG//./_}.py" \
    --verifier-name-or-path "$VERIFIER" \
    --speculator-type dspark \
    --num-layers "$NUM_LAYERS" \
    --draft-arch qwen3 \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha 1.0 \
    --loss-fn "$LOSS_FN" \
    --draft-attn-impl sdpa \
    --total-seq-len "$SEQ_LEN" \
    --epochs "$EPOCHS" \
    --optimizer "$OPTIMIZER" \
    --lr "$LR" \
    --on-missing generate \
    --log-freq 1 \
    --full-attention-indices 0 1 2 3 4 \
    --checkpoint-freq "$CHECKPOINT_FREQ" \
    --logger tensorboard \
    --log-dir "$LOGDIR" \
    --save-path "$SAVE" \
    --data-path "$DATA_PATH" \
    "${BACKEND_ARGS[@]}" \
    --num-workers "$NUM_WORKERS" \
    ${TRAIN_EXTRA_ARGS} \
    "$@")

setsid nohup "${CMD[@]}" >> "$LOG" 2>&1 &
TPID=$!
echo "$TPID" > "$OUTPUT_DIR/train.pid"
echo "[train] torchrun pid=$TPID  log=$LOG  pidfile=$OUTPUT_DIR/train.pid"
# 监控提示：
#   - tail -f $LOG 跟踪训练日志
#   - tensorboard --logdir $LOGDIR
