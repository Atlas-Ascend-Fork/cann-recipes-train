#!/bin/bash
set -euo pipefail

# 根据实际情况修改 ascend-toolkit 路径；需包含 HiF8 量化融合算子支持
ASCEND_TOOLKIT_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"
if [ ! -f "${ASCEND_TOOLKIT_ENV}" ]; then
    echo "[ERROR] 未找到 CANN 环境脚本: ${ASCEND_TOOLKIT_ENV}，请按实际安装路径修改后重试。" >&2
    exit 1
fi
# set_env.sh 内部会引用可能尚未定义的环境变量(如 LD_LIBRARY_PATH)，source 期间临时关闭 -u
set +u
source "${ASCEND_TOOLKIT_ENV}"
set -u

# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=2 # 开启FSDP2时，不能置为1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export ASCEND_LAUNCH_BLOCKING=0
# FP8
# 量化 FlashAttention: 默认关闭。开启需要 CANN 提供 hifloat8 量化融合注意力算子
# (npu_quant_fusion_attention / npu_quant_fusion_attention_grad), 商发 CANN 9.2.0 暂不包含;
# 关闭后仍保留线性层 GEMM 的 HiF8 量化(9.2.0 商发包完整支持), 注意力回退到 BF16 融合算子。
export FP8_FA='false'
# export FP8_ALL2ALL='true'

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# 注意：目前开启FSDP2训练时，尚不支持TP/PP/VPP
TP=1
PP=1
VP=1
CP=8
MBS=1
GRAD_ACC_STEP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

MM_DATA="examples/wan2.1/14b/t2v_fp8/feature_data.json"
MM_MODEL="examples/wan2.1/14b/t2v_fp8/pretrain_model_fsdp2.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="./weights/Wan2.1-T2V-14B-Converted/"  # 转换后的 wandit 权重路径
SAVE_PATH="./save_dir"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --virtual-pipeline-model-parallel-size ${VP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-workers 0 \
    --lr 1e-5 \
    --min-lr 1e-5 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay-style constant \
    --weight-decay 0 \
    --lr-warmup-init 0 \
    --lr-warmup-iters 0 \
    --clip-grad 0.0 \
    --train-iters 2000 \
    --exit-interval 2000 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --distributed-timeout-minutes 20 \
    --use-torch-fsdp2 \
    --untie-embeddings-and-output-weights \
    --ckpt-format torch_dcp \
    --fsdp2-config-path examples/wan2.1/fsdp2_config.yaml \
    --optimizer-selection fused_torch_adamw \
    --use-cpu-initialization \
    --attention-mask-type general \
    --seed 42 \
    --transformer-impl transformer_engine \
    --fp8-format hif8 \
    --fp8-recipe delayed \
    --dual-delayed-scaling \
    --use-deter-comp \
"
# --fp8-format hif8 \
# --fp8-recipe delayed \
# --dual-delayed-scaling \
# --use-deter-comp \
MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 10000 \
    --eval-iters 10 \
    --load $LOAD_PATH \
    --save $SAVE_PATH \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
LOG_FILE="logs/train_${logfile}.log"
mkdir -p logs

# 管道中 tee 的退出码会掩盖 torchrun 的退出码，这里用 PIPESTATUS 取训练进程的真实退出码
set +e
torchrun $DISTRIBUTED_ARGS pretrain_sora.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee "${LOG_FILE}"
TRAIN_RC=${PIPESTATUS[0]}
set -e

chmod 440 "${LOG_FILE}"
if [ -d "${SAVE_PATH}" ]; then
    find "${SAVE_PATH}" -type d -exec chmod 750 {} \;
    find "${SAVE_PATH}" -type f -exec chmod 640 {} \;
fi

if [ "${TRAIN_RC}" -ne 0 ]; then
    echo "[ERROR] 训练进程异常退出(exit=${TRAIN_RC})，日志见 ${LOG_FILE}，跳过性能统计。" >&2
    exit "${TRAIN_RC}"
fi

# 性能统计：日志中无匹配行时 STEP_TIME 为空，直接跳过，避免 awk 语法错误与除零
STEP_TIME=$(grep "elapsed time per iteration" "${LOG_FILE}" | awk -F ':' '{print$5}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}' || true)
if [ -z "${STEP_TIME}" ] || [ "$(awk -v t="${STEP_TIME}" 'BEGIN{print (t + 0 > 0) ? 1 : 0}')" != "1" ]; then
    echo "[WARN] 未从 ${LOG_FILE} 中解析到有效的 'elapsed time per iteration'，跳过性能统计。" >&2
else
    SPS=$(awk -v gbs="${GBS}" -v st="${STEP_TIME}" 'BEGIN{printf "%.3f\n", gbs * 1000 / st}')
    echo "Elapsed Time Per iteration: ${STEP_TIME}, Average Samples per Second: ${SPS}"
fi
