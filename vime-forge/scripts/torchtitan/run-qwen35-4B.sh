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

# Text-only asynchronous GRPO parameters; runtime comes from the prepared wrapper.
set -euo pipefail
[[ "${VIME_QWEN35_WRAPPER:-0}" == 1 ]] || {
    echo 'Use scripts/run-qwen35-4B-torchtitan.sh [--dry-run]' >&2; exit 2;
}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXTRA_ARGS=()
if [[ "${1:-}" == -- ]]; then shift; EXTRA_ARGS=("$@");
elif (( $# )); then echo 'Expected -- TRAIN_ARGS...' >&2; exit 2; fi

SEQ_LEN=${SEQ_LEN:-10240}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
NUM_ROLLOUT=${NUM_ROLLOUT:-2}
ROLLOUT_BATCH=${ROLLOUT_BATCH:-16}
N_SAMPLES=${N_SAMPLES:-8}
TRAIN_GPUS=${TRAIN_GPUS:-4}
TRAIN_TP=${TRAIN_TP:-1}
TRAIN_DP=${TRAIN_DP:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
VLLM_RANKS_PER_ENGINE=${VLLM_RANKS_PER_ENGINE:-4}
for name in SEQ_LEN MAX_RESPONSE_LEN MAX_PROMPT_LEN NUM_ROLLOUT ROLLOUT_BATCH N_SAMPLES \
    TRAIN_GPUS TRAIN_TP TRAIN_DP ROLLOUT_GPUS VLLM_RANKS_PER_ENGINE; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done
(( TRAIN_TP * TRAIN_DP == TRAIN_GPUS )) || { echo 'TRAIN_TP * TRAIN_DP must equal TRAIN_GPUS' >&2; exit 2; }
(( ROLLOUT_GPUS % VLLM_RANKS_PER_ENGINE == 0 )) || { echo 'Rollout device count must be divisible by engine TP' >&2; exit 2; }
(( MAX_RESPONSE_LEN < SEQ_LEN && MAX_PROMPT_LEN + MAX_RESPONSE_LEN <= SEQ_LEN )) || {
    echo 'Prompt and response limits must fit inside SEQ_LEN' >&2; exit 2;
}
IFS=, read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
(( ${#devices[@]} >= TRAIN_GPUS + ROLLOUT_GPUS )) || { echo 'Not enough visible NPUs for disjoint training and rollout' >&2; exit 2; }
GLOBAL_BATCH=$((ROLLOUT_BATCH * N_SAMPLES))

if [[ "$VIME_QWEN35_DRY_RUN" != 1 ]]; then
    source "$SCRIPT_DIR/common/launcher.sh"
    configure_run -- ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi

# Replace common/default.sh's environment table; capability=8.0 belongs only to
# training workers, and VLLM_VERSION must come from the installed inference stack.
RT_ENV=()
for name in PATH LD_LIBRARY_PATH PYTHONPATH VIME_CANN_HOME ASCEND_HOME_PATH ASCEND_OPP_PATH \
    ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME VIME_TRAIN_PYTHON VIME_VLLM_EXECUTABLE VIME_VLLM_PYTHON \
    ROLLOUT_ENV CC TRITON_CACHE_DIR \
    VLLM_ASCEND_ENABLE_NZ VLLM_USE_AOT_COMPILE VLLM_ASCEND_NPUGRAPH_CAPTURE_LIMIT \
    PYTHONUNBUFFERED PYTHONDONTWRITEBYTECODE RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES \
    HCCL_WHITELIST_DISABLE HCCL_CONNECT_TIMEOUT HCCL_DETERMINISTIC HCCL_HOST_SOCKET_PORT_RANGE \
    HCCL_NPU_SOCKET_PORT_RANGE PYTORCH_NPU_ALLOC_CONF CUDA_DEVICE_MAX_CONNECTIONS \
    TASK_QUEUE_ENABLE STREAMS_PER_DEVICE MULTI_STREAM_MEMORY_RESERVE CPU_AFFINITY_CONF \
    OMP_NUM_THREADS TRANSFORMERS_VERBOSITY no_proxy NO_PROXY TMPDIR; do
    [[ "${!name+x}" ]] && RT_ENV+=("${name}=${!name}")
done

TRAIN_ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node "$TRAIN_GPUS"
    --rollout-num-gpus "$ROLLOUT_GPUS" --num-gpus-per-node "$((TRAIN_GPUS + ROLLOUT_GPUS))"
    --hf-checkpoint "$MODEL_PATH" --freeze-params-name-list vision_encoder
    --checkpoint.folder "${OUTPUT_DIR}/checkpoints"
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label
    --apply-chat-template --rollout-shuffle --rm-type deepscaler
    --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH"
    --update-weights-interval 1
    --n-samples-per-prompt "$N_SAMPLES" --global-batch-size "$GLOBAL_BATCH"
    --rollout-max-response-len "$MAX_RESPONSE_LEN" --rollout-max-context-len "$SEQ_LEN"
    --rollout-max-prompt-len "$MAX_PROMPT_LEN" --rollout-temperature 1.0 --rollout-top-k -1
    --disable-rewards-normalization --balance-data
    --rollout-function-path vime.rollout.vllm_rollout.generate_rollout
    --rollout-num-gpus-per-engine "$VLLM_RANKS_PER_ENGINE"
    --vllm-server-concurrency 8 --vllm-gpu-memory-utilization 0.85
    --vllm-max-model-len "$SEQ_LEN" --no-vllm-weight-sync-packed
    --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --no-vllm-async-scheduling
    --advantage-estimator grpo --kl-coef 0.0 --kl-loss-coef 0.0 --entropy-coef 0.0
    --eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss --log-probs-chunk-size 64
    --train-env-vars '{"TORCH_NPU_DEVICE_CAPABILITY":"8.0"}'
    --module vime.backends.torchtitan_utils.models.qwen3_5 --config vime_qwen35_4b
    --parallelism.data_parallel_replicate_degree 1 --parallelism.data_parallel_shard_degree "$TRAIN_DP"
    --parallelism.tensor_parallel_degree "$TRAIN_TP" --parallelism.pipeline_parallel_degree 1
    --parallelism.context_parallel_degree 1 --training.seq_len "$SEQ_LEN"
    --micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu "$SEQ_LEN"
    --save-debug-rollout-data "${OUTPUT_DIR}/rollout/{rollout_id}.pt"
)

if [[ "$VIME_QWEN35_DRY_RUN" == 1 ]]; then
    "${VIME_JSON_PYTHON:-python3}" - "$REPO_ROOT" "${TRAIN_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
checks = {
    "qwen35_recipe_present": (root / "vime/backends/torchtitan_utils/models/qwen3_5/config_registry.py").is_file(),
    "prepared_environment_hook_present": "VIME_PREPARED_ENV" in (root / "scripts/torchtitan/common/launcher.sh").read_text(),
}
print(json.dumps({
    "mode": "async", "backend_validation": "pending; presence checks are not runtime validation",
    "integration_checks": checks,
    "environment": {key: os.environ[key] for key in (
        "MODEL_PATH", "PROMPT_DATA", "TRAIN_ENV", "VIME_TRAIN_PYTHON", "ROLLOUT_ENV",
        "VIME_CANN_HOME", "VIME_VLLM_PYTHON", "ASCEND_RT_VISIBLE_DEVICES",
        "OUTPUT_DIR", "VIME_SLOT", "VIME_RAY_PORT", "VIME_RAY_DASHBOARD_PORT", "VIME_RAY_TEMP_DIR")},
    "command": [os.environ["VIME_TRAIN_PYTHON"], str(root / "train_async.py"), "--train-backend", "torchtitan", *sys.argv[2:]],
}, indent=2))
PY
    exit 0
fi
mkdir -p "${OUTPUT_DIR}/rollout"
start_run
