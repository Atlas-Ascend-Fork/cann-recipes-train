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

# Qwen3.5-4B asynchronous GRPO parameters; execution belongs to the shared launcher.
set -euo pipefail

# AgenticRL entries configure the run before sourcing this model script.
if [[ "${RUN_CONFIGURED:-0}" != 1 ]]; then
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    source "${SCRIPT_DIR}/common/launcher.sh"
    MODE=${MODE:-async}
    configure_run "$@"
fi
export MODEL_PATH="${MODEL_PATH:-${VIME_MODEL_ROOT}/Qwen3.5-4B}"

SEQ_LEN=${SEQ_LEN:-10240}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
TRAIN_TP=${TRAIN_TP:-1}
TRAIN_DP=${TRAIN_DP:-4}
TRAIN_CP=${TRAIN_CP:-1}
for degree in TRAIN_TP TRAIN_DP TRAIN_CP; do
    [[ "${!degree}" =~ ^[1-9][0-9]*$ ]] || die "$degree must be a positive integer, got: ${!degree}"
done
train_world_size=$((TRAIN_TP * TRAIN_DP * TRAIN_CP))
TRAIN_GPUS=${TRAIN_GPUS:-$train_world_size}
[[ "$TRAIN_GPUS" == "$train_world_size" ]] || \
    die "TRAIN_GPUS=$TRAIN_GPUS must equal TP × DP × CP ($TRAIN_TP × $TRAIN_DP × $TRAIN_CP = $train_world_size)"
if (( TRAIN_CP > 1 )); then
    [[ "$TRAIN_TP" == 1 ]] || die "Qwen3.5 Ascend CP requires TRAIN_TP=1"
    TITAN_CONFIG=${TITAN_CONFIG:-vime_qwen35_4b_cp}
else
    TITAN_CONFIG=${TITAN_CONFIG:-vime_qwen35_4b}
fi
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
VLLM_RANKS_PER_ENGINE=${VLLM_RANKS_PER_ENGINE:-4}
[[ "$ROLLOUT_GPUS" =~ ^(0|[1-9][0-9]*)$ ]] || die "ROLLOUT_GPUS must be a nonnegative integer"
[[ "$VLLM_RANKS_PER_ENGINE" =~ ^[1-9][0-9]*$ ]] || die "VLLM_RANKS_PER_ENGINE must be a positive integer"

if [[ "$COLOCATE" == on ]]; then
    required_npus=$((TRAIN_GPUS > ROLLOUT_GPUS ? TRAIN_GPUS : ROLLOUT_GPUS))
else
    required_npus=$((TRAIN_GPUS + ROLLOUT_GPUS))
fi
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
    die "ASCEND_RT_VISIBLE_DEVICES must list comma-separated NPU IDs"
IFS=, read -r -a visible_npus <<< "$ASCEND_RT_VISIBLE_DEVICES"
seen_ids=,
for id in "${visible_npus[@]}"; do
    id=$((10#$id))
    [[ "$seen_ids" != *",$id,"* ]] || die "Duplicate NPU ID $id in ASCEND_RT_VISIBLE_DEVICES"
    seen_ids+="$id,"
done
(( ${#visible_npus[@]} >= required_npus )) || \
    die "COLOCATE=$COLOCATE (train=$TRAIN_GPUS, rollout=$ROLLOUT_GPUS) requires $required_npus NPUs;" \
        "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES exposes ${#visible_npus[@]}"
PROMPT_DATA=${PROMPT_DATA:-${VIME_DATA_ROOT}/dapo-math-17k.jsonl}

# Inherit the common device capability; keep the existing vLLM and machine overrides.
for index in "${!RT_ENV[@]}"; do
    key=${RT_ENV[index]%%=*}
    if [[ "$key" == VLLM_VERSION || "$key" == VLLM_ASCEND_NPUGRAPH_CAPTURE_LIMIT ]]; then
        unset 'RT_ENV[index]'
    elif [[ "$key" != TORCH_NPU_DEVICE_CAPABILITY && "${!key+x}" ]]; then
        RT_ENV[index]="${key}=${!key}"
    fi
done
unset index key

PLACEMENT_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node "$TRAIN_GPUS"
    --rollout-num-gpus "$ROLLOUT_GPUS"
    --num-gpus-per-node "$((TRAIN_GPUS + ROLLOUT_GPUS))"
)

CKPT_ARGS=(
    --hf-checkpoint "$MODEL_PATH"
    --checkpoint.folder "$CKPT_DIR"
)

# The parser derives global batch size from the final rollout and sample counts.
ROLLOUT_ARGS=(
    --prompt-data "$PROMPT_DATA"
    --input-key prompt
    --label-key label
    --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT:-2}"
    --rollout-batch-size "${ROLLOUT_BATCH:-16}"
    --update-weights-interval 1
    --n-samples-per-prompt "${N_SAMPLES:-8}"
    --num-steps-per-rollout 1
    --rollout-max-response-len "$MAX_RESPONSE_LEN"
    --rollout-max-context-len "$SEQ_LEN"
    --rollout-max-prompt-len "$MAX_PROMPT_LEN"
    --rollout-temperature 1.0
    --rollout-top-k -1
    --balance-data
    --rollout-function-path vime.rollout.vllm_rollout.generate_rollout
)
# AgenticRL entries configure their own prompting, rewards and rollout dumps.
if [[ "$VIME_RL_TASK" == singleturn ]]; then
    ROLLOUT_ARGS+=(
        --apply-chat-template
        --rm-type deepscaler
    )
    if [[ -n "$REPLAY_DIR" ]]; then
        ROLLOUT_ARGS+=(--load-debug-rollout-data "${REPLAY_DIR}/{rollout_id}.pt")
    else
        DUMP_DIR=${DUMP_DIR:-${OUTPUT_DIR:-${ARCHIVE_DIR}}/rollout}
        ROLLOUT_ARGS+=(--save-debug-rollout-data "${DUMP_DIR}/{rollout_id}.pt")
    fi
fi

VLLM_ARGS=(
    --rollout-num-gpus-per-engine "$VLLM_RANKS_PER_ENGINE"
    --vllm-server-concurrency "${SERVER_CONCURRENCY:-8}"
    --vllm-gpu-memory-utilization "${ROLLOUT_MEMORY_FRACTION:-0.85}"
    --vllm-max-model-len "$SEQ_LEN"
    --no-vllm-weight-sync-packed
    --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --no-vllm-async-scheduling
)

PERF_ARGS=(
    --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-$SEQ_LEN}"
    --log-probs-chunk-size 64
)
# Each contiguous CP shard must satisfy the Ascend Gated DeltaNet alignment.
if (( TRAIN_CP > 1 )); then
    PERF_ARGS+=(--packed-seq-alignment 64)
fi

TRAINING_ARGS=(
    --freeze-params-name-list vision_encoder
    --parallelism.data_parallel_replicate_degree 1
    --parallelism.data_parallel_shard_degree "$TRAIN_DP"
    --parallelism.tensor_parallel_degree "$TRAIN_TP"
    --parallelism.pipeline_parallel_degree 1
    --parallelism.context_parallel_degree "$TRAIN_CP"
    --training.seq_len "$SEQ_LEN"
)

RECIPE_ARGS=(
    --module vime.backends.torchtitan_utils.models.qwen3_5
    --config "$TITAN_CONFIG"
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-coef 0.0
    --kl-loss-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --calculate-per-token-loss
)
add_when TIS_ON 1 GRPO_ARGS -- --use-tis --tis-clip 2.0 --tis-clip-low 0.0

TRAIN_ARGS=(
    "${PLACEMENT_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${VLLM_ARGS[@]}"
    "${PERF_ARGS[@]}" "${TRAINING_ARGS[@]}" "${RECIPE_ARGS[@]}" "${GRPO_ARGS[@]}"
)
start_run
