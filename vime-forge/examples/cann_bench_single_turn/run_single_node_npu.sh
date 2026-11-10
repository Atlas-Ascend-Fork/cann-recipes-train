#!/usr/bin/env bash
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

# Shared TorchTitan Ascend CP launcher; model-specific entry points set defaults.
set -euo pipefail
DRY_RUN=0
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_NAME=''
EXTRA_ARGS=()
while (( $# )); do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --local)
            [[ $# -ge 2 && -n "${2:-}" && "${2:-}" != --* ]] || { echo '--local needs a file name' >&2; exit 2; }
            LOCAL_NAME="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
OVERRIDE_ARGS=()
if [[ -n "$LOCAL_NAME" ]]; then
    # Match the generic Titan launcher: bare/relative names live under common/.
    case "$LOCAL_NAME" in
        /*|[A-Za-z]:/*) LOCAL_SH="$LOCAL_NAME" ;;
        *) LOCAL_SH="$ROOT/scripts/torchtitan/common/$LOCAL_NAME" ;;
    esac
    [[ -f "$LOCAL_SH" ]] || { echo "--local names a file that does not exist: $LOCAL_SH" >&2; exit 2; }
    source "$LOCAL_SH"
fi
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
: "${MODEL_DIR:?Set MODEL_DIR to an HF checkpoint supported by main}"
: "${DATA_PATH:?Set DATA_PATH to an external online/reference JSONL}"
: "${CANN_EVAL_URL:?Set CANN_EVAL_URL to a v1 evaluation service}"
TITAN_MODULE="${TITAN_MODULE:-vime.backends.torchtitan_utils.models.qwen3_5}"
TITAN_CONFIG="${TITAN_CONFIG:-vime_qwen35_4b_cp}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/cann_single_turn/$(date +%Y%m%d_%H%M%S)}"
export CANN_ARCHIVE_DIR="${CANN_ARCHIVE_DIR:-$OUTPUT_DIR/rollout_jsonl}"
export CANN_EVAL_ALIAS="${CANN_EVAL_ALIAS:-vime-cann-single-turn-$(date +%Y%m%d_%H%M%S)}"
TRAIN_NPUS="${TRAIN_NPUS:-4}"
ROLLOUT_NPUS="${ROLLOUT_NPUS:-4}"
TP_SIZE="${TP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-2}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
RESPONSE_LEN="${RESPONSE_LEN:-36864}"
CONTEXT_LEN="${CONTEXT_LEN:-36864}"
VLLM_MODEL_LEN="${VLLM_MODEL_LEN:-65536}"
TEMPERATURE="${TEMPERATURE:-0.6}"
THINKING_TOKEN_BUDGET="${THINKING_TOKEN_BUDGET:-12288}"
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-train_async.py}"
if [[ -n "${CANN_ASYNC_EVALUATION+x}" ]]; then export CANN_ASYNC_EVALUATION; fi
case "${CANN_ASYNC_EVALUATION:-0}" in
    0) ;;
    1) [[ "$TRAIN_ENTRYPOINT" == train_async.py && "${RUN_MODE:-train}" != rollout ]] || {
        echo 'CANN_ASYNC_EVALUATION=1 requires train_async.py training' >&2; exit 2;
    } ;;
    *) echo 'CANN_ASYNC_EVALUATION must be 0 or 1' >&2; exit 2 ;;
esac
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
for name in TRAIN_NPUS ROLLOUT_NPUS TP_SIZE CP_SIZE ROLLOUT_TP_SIZE RESPONSE_LEN CONTEXT_LEN \
    VLLM_MODEL_LEN ROLLOUT_BATCH_SIZE SAMPLES_PER_PROMPT; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done
(( TP_SIZE == 1 && CP_SIZE > 1 )) || { echo 'Ascend Qwen CP requires TP_SIZE=1 and CP_SIZE>1' >&2; exit 2; }
(( TRAIN_NPUS % (TP_SIZE * CP_SIZE) == 0 )) || { echo 'TRAIN_NPUS must be divisible by TP_SIZE * CP_SIZE' >&2; exit 2; }
DP_SIZE="${DP_SIZE:-$((TRAIN_NPUS / (TP_SIZE * CP_SIZE)))}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((CONTEXT_LEN / CP_SIZE))}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * SAMPLES_PER_PROMPT))}"
for name in DP_SIZE MAX_TOKENS_PER_GPU GLOBAL_BATCH_SIZE; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done
(( TP_SIZE * CP_SIZE * DP_SIZE == TRAIN_NPUS )) || { echo 'TP_SIZE * CP_SIZE * DP_SIZE must equal TRAIN_NPUS' >&2; exit 2; }
(( ROLLOUT_NPUS % ROLLOUT_TP_SIZE == 0 )) || { echo 'ROLLOUT_NPUS must be divisible by ROLLOUT_TP_SIZE' >&2; exit 2; }
(( GLOBAL_BATCH_SIZE % DP_SIZE == 0 )) || { echo 'GLOBAL_BATCH_SIZE must be divisible by DP_SIZE' >&2; exit 2; }
(( CONTEXT_LEN <= VLLM_MODEL_LEN )) || { echo 'CONTEXT_LEN must not exceed VLLM_MODEL_LEN' >&2; exit 2; }
# Disjoint training/inference resources; users prepare the validated CP runtime.
# Device capability belongs only to the training actor, not the inference environment.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_ENABLE_NZ=0 VLLM_USE_AOT_COMPILE=0
export VLLM_ASCEND_NPUGRAPH_CAPTURE_LIMIT="${VLLM_ASCEND_NPUGRAPH_CAPTURE_LIMIT:-512}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$OUTPUT_DIR/tensorboard}"
LOAD_ARGS=()
# Swap supports model weights, but not restoring optimizer state.
if [[ -n "${TRAIN_LOAD:-}" ]]; then LOAD_ARGS=(--load "$TRAIN_LOAD" --load-mode weights); fi
ASYNC_ARGS=()
if [[ "$TRAIN_ENTRYPOINT" == train_async.py && "${RUN_MODE:-train}" != rollout ]]; then
    ASYNC_ARGS=(--use-tis --tis-clip 2.0 --tis-clip-low 0.0)
fi
THINKING_ARGS=()
if [[ "$THINKING_TOKEN_BUDGET" != none ]]; then
    [[ "$THINKING_TOKEN_BUDGET" =~ ^[0-9]+$ ]] || { echo 'THINKING_TOKEN_BUDGET must be nonnegative or none' >&2; exit 2; }
    THINKING_ARGS=(--rollout-thinking-token-budget "$THINKING_TOKEN_BUDGET"
        --vllm-reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}')
fi
EXTRA_MODE=()
if [[ "${RUN_MODE:-train}" == rollout ]]; then
    EXTRA_MODE=(--debug-rollout-only)
fi
EVAL_ARGS=()
if [[ -n "${EVAL_DATA_PATH:-}" ]]; then
    EVAL_ARGS=(--eval-prompt-data cann_bench "$EVAL_DATA_PATH" --eval-input-key input --eval-label-key label
        --eval-interval "${EVAL_INTERVAL:-16}" --n-samples-per-eval-prompt 5
        --eval-temperature "$TEMPERATURE"
        --eval-max-response-len "$RESPONSE_LEN" --eval-max-context-len "$CONTEXT_LEN")
fi
DURATION_ARGS=(--num-epoch "${NUM_EPOCH:-10}")
if [[ -n "${NUM_ROLLOUT:-}" ]]; then
    [[ "$NUM_ROLLOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_ROLLOUT must be a positive integer' >&2; exit 2; }
    DURATION_ARGS=(--num-rollout "$NUM_ROLLOUT")
else
    [[ "${NUM_EPOCH:-10}" =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_EPOCH must be a positive integer' >&2; exit 2; }
fi
TRAIN_CMD=("${PYTHON:-python3}" -u -m examples.cann_bench_single_turn.run
    --train-backend torchtitan --actor-num-nodes 1 --actor-num-gpus-per-node "$TRAIN_NPUS"
    --num-gpus-per-node "$((TRAIN_NPUS + ROLLOUT_NPUS))" --rollout-num-gpus "$ROLLOUT_NPUS"
    --module "$TITAN_MODULE" --config "$TITAN_CONFIG" --freeze-params-name-list vision_encoder
    --train-env-vars '{"TORCH_NPU_DEVICE_CAPABILITY":"0.0","MULTI_STREAM_MEMORY_REUSE":"2"}'
    --parallelism.data_parallel_replicate_degree 1 --parallelism.data_parallel_shard_degree "$DP_SIZE"
    --parallelism.tensor_parallel_degree "$TP_SIZE" --parallelism.pipeline_parallel_degree 1
    --parallelism.context_parallel_degree "$CP_SIZE" --parallelism.context_parallel_load_balancer None
    --hf-checkpoint "$MODEL_DIR" "${LOAD_ARGS[@]}"
    --save "$OUTPUT_DIR/checkpoints" --save-interval "${SAVE_INTERVAL:-16}" --checkpoint.interval 1
    # Keep periodic saves, exporting HF weights only (no optimizer checkpoint).
    --checkpoint.load-only --save-hf "$OUTPUT_DIR/checkpoints/rollout_{rollout_id}"
    --prompt-data "$DATA_PATH" --input-key input --label-key label --metadata-key metadata
    --apply-chat-template --rollout-shuffle "${DURATION_ARGS[@]}"
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$SAMPLES_PER_PROMPT"
    --global-batch-size "$GLOBAL_BATCH_SIZE" --micro-batch-size 1
    --rollout-max-response-len "$RESPONSE_LEN" --rollout-max-context-len "$CONTEXT_LEN"
    --rollout-skip-special-tokens
    --rollout-temperature "$TEMPERATURE" "${THINKING_ARGS[@]}"
    --calculate-per-token-loss --balance-data --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
    --log-probs-chunk-size 256 --recompute-loss-function
    # CLI imports replace the recipe list: retain both CP kernels alongside swap.
    --override.imports torchtitan_npu.override.qwen3_5.varlen_attention.asc_cp,torchtitan_npu.override.qwen3_5.gated_delta.context_parallel,torchtitan_npu.override.common.optimizer.swap_optimizer
    --optimizer.implementation fused
    --advantage-estimator grpo --optimizer.param-groups.0.optimizer-kwargs.lr "${LR:-1e-6}"
    --eps-clip 0.2 --eps-clip-high 0.28
    --kl-loss-coef 0.0 --kl-coef 0.0 --kl-loss-type low_var_kl --entropy-coef 0.0
    --rollout-num-gpus-per-engine "$ROLLOUT_TP_SIZE" --vllm-max-model-len "$VLLM_MODEL_LEN"
    --vllm-server-concurrency "${CONCURRENCY:-32}" --vllm-gpu-memory-utilization 0.7
    --vllm-weight-sync-mode native --no-vllm-weight-sync-packed
    --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --no-vllm-async-scheduling
    --update-weights-interval 1 --use-tensorboard
    --custom-config-path "$ROOT/examples/cann_bench_single_turn/config.yaml"
    --custom-generate-function-path examples.cann_bench_single_turn.generate.generate
    --rollout-sample-hook-path examples.cann_bench_single_turn.hooks.prepare_sample
    --custom-rm-path examples.cann_bench_single_turn.reward.reward_func
    --dynamic-sampling-filter-path examples.cann_bench_single_turn.hooks.keep_trainable_group
    --custom-rollout-log-function-path examples.cann_bench_single_turn.metrics.log_train
    --custom-eval-rollout-log-function-path examples.cann_bench_single_turn.metrics.log_eval
    "${EVAL_ARGS[@]}" "${EXTRA_MODE[@]}" "${ASYNC_ARGS[@]}" "${OVERRIDE_ARGS[@]}" "${EXTRA_ARGS[@]}")
if (( DRY_RUN )); then
    "${PYTHON:-python3}" -c 'import json, sys; print(json.dumps({"command": sys.argv[1:]}, indent=2))' "${TRAIN_CMD[@]}"
    exit 0
fi
# Reuse public configuration/cleanup without changing the public launcher.
export VIME_PREPARED_ENV=1
TRAIN_PYTHON="${PYTHON:-python3}"
export VIME_JSON_PYTHON="$TRAIN_PYTHON"
MODE=sync
if [[ "$TRAIN_ENTRYPOINT" == train_async.py && "${RUN_MODE:-train}" != rollout ]]; then MODE=async; fi
VIME_RAY_TEMP_DIR="${VIME_RAY_TEMP_DIR:-/tmp/vime-cann-ray-s${VIME_SLOT:-0}}"
source "$ROOT/scripts/torchtitan/common/launcher.sh"
configure_run --
ARCHIVE_DIR="$OUTPUT_DIR"
TMPDIR="${TMPDIR:-$VIME_WORKSPACE/tmp}"
export VIME_LAUNCHER_RUN_ID TMPDIR TRITON_CACHE_DIR VIME_VLLM_PYTHON VIME_VLLM_EXECUTABLE
export HCCL_HOST_SOCKET_PORT_RANGE HCCL_NPU_SOCKET_PORT_RANGE
RUNTIME_ENV_JSON="$("$TRAIN_PYTHON" -c '
import json
from examples.cann_bench_single_turn.run import worker_environment
print(json.dumps({"env_vars": worker_environment()}))
')"
# Task-specific submission stays here; public launcher functions are unchanged.
export MASTER_ADDR ASCEND_RT_VISIBLE_DEVICES HF_HOME ASCEND_CACHE_PATH
mkdir -p "$OUTPUT_DIR" "$TMPDIR"
reclaim_slot || die "could not reclaim slot ${VIME_SLOT}"
quoted_train_cmd
RAY_CMD=("$TRAIN_PYTHON" -m ray.scripts.scripts)
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf '%q ' "${RAY_CMD[@]}" start "${RAY_HEAD_ARGS[@]}"
    printf '\n'
    printf '%q ' "${RAY_CMD[@]}" job submit \
        "--address=http://127.0.0.1:${VIME_RAY_DASHBOARD_PORT}" \
        "--runtime-env-json=$RUNTIME_ENV_JSON" -- "${QUOTED_TRAIN_CMD[@]}"
    printf '\n'
} > "$OUTPUT_DIR/rerun.sh"
chmod +x "$OUTPUT_DIR/rerun.sh"
"${RAY_CMD[@]}" start "${RAY_HEAD_ARGS[@]}"
"${RAY_CMD[@]}" job submit \
    --address="http://127.0.0.1:${VIME_RAY_DASHBOARD_PORT}" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- "${QUOTED_TRAIN_CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
