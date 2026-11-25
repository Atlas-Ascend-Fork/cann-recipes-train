#!/bin/bash
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

# Qwen3.5-4B multi-turn Geo3K GRPO with TorchTitan and vLLM on separate NPUs.
# Usage: bash examples/geo3k_vlm_multi_turn/run-qwen35-4B-geo3k-mt-torchtitan.sh \
#            [--local local.mine.sh] [-- <extra training args>]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

source "${REPO_ROOT}/scripts/torchtitan/common/launcher.sh"
configure_run "$@"

# Dataset: VeraIsHere/geo3k_imgurl_processed, including Vime's tool-call prompts.
export GEO3K_TRAIN_DATA="${GEO3K_TRAIN_DATA:-${VIME_DATA_ROOT}/geo3k_imgurl_processed/train.parquet}"
# Optional evaluation uses a training subset, not a held-out benchmark.
export GEO3K_EVAL_DATA="${GEO3K_EVAL_DATA:-${VIME_DATA_ROOT}/geo3k_imgurl_processed/train.parquet@[0:64]}"
export MODEL_PATH="${MODEL_PATH:-${VIME_MODEL_ROOT}/Qwen3.5-4B}"
export VIME_MODEL_ROOT VIME_DATA_ROOT

# Shared training and rollout context limit.
SEQ_LEN=10240
MAX_RESPONSE_LEN=8192
ROLLOUT_BATCH=8
N_SAMPLES=4
GLOBAL_BATCH=$(( ROLLOUT_BATCH * N_SAMPLES ))

TRAIN_GPUS="${TRAIN_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
TRAIN_TP=1
TRAIN_DP=$(( TRAIN_GPUS / TRAIN_TP ))
VLLM_RANKS_PER_ENGINE=1

PLACEMENT_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node ${TRAIN_GPUS}
   --rollout-num-gpus ${ROLLOUT_GPUS}
)

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
)
add_when SAVE_ON on CKPT_ARGS -- \
   --save "${CKPT_DIR}" \
   --save-interval 20 \
   --checkpoint.keep_latest_k 2
# Resume only when a training checkpoint exists.
if compgen -G "${CKPT_DIR}/step-*" > /dev/null; then
    CKPT_ARGS+=(--load "${CKPT_DIR}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${GEO3K_TRAIN_DATA}"
   --input-key problem
   --label-key answer
   --multimodal-keys '{"image":"images"}'
   # The custom rollout renders conversations; do not apply the template here.
   --rollout-shuffle
   --balance-data
   --rm-type math
   --num-rollout 3000
   --rollout-temperature 1
   --rollout-batch-size       ${ROLLOUT_BATCH}
   --n-samples-per-prompt     ${N_SAMPLES}
   --global-batch-size        ${GLOBAL_BATCH}
   --rollout-max-response-len ${MAX_RESPONSE_LEN}
   --rollout-max-context-len  ${SEQ_LEN}
   --custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate
   --custom-config-path "${REPO_ROOT}/examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_config.yaml"
)
add_when DET on ROLLOUT_ARGS -- --debug.seed 1234 --debug.deterministic
add_when DUMP_DIR nonempty ROLLOUT_ARGS -- --save-debug-rollout-data "${DUMP_DIR}/{rollout_id}.pt"
add_when REPLAY_DIR nonempty ROLLOUT_ARGS -- --load-debug-rollout-data "${REPLAY_DIR}/{rollout_id}.pt"

VLLM_ARGS=(
   --rollout-num-gpus-per-engine ${VLLM_RANKS_PER_ENGINE}
   --vllm-max-model-len          ${SEQ_LEN}
   --vllm-limit-mm-per-prompt    '{"image":1,"video":0}'
   --vllm-gpu-memory-utilization 0.7
   --no-vllm-weight-sync-packed
   --vllm-additional-config '{"rl_config":{"enabled":true}}'
   --vllm-enable-scale-out
   --router-policy consistent_hash
)
add_when DET on VLLM_ARGS -- --vllm-enable-deterministic-inference --rollout-seed 1234

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data geo3k "${GEO3K_EVAL_DATA}"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len ${MAX_RESPONSE_LEN}
)
gate_when EVAL_ON on EVAL_ARGS

PERF_ARGS=(
   --max-tokens-per-gpu ${SEQ_LEN}
   --use-dynamic-batch-size
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-coef      0.0
   --kl-loss-coef 0.0
   --entropy-coef 0.0
   --eps-clip      0.2
   --eps-clip-high 0.28
   --log-probs-chunk-size 64
   --get-mismatch-metrics
   --custom-tis-function-path vime.backends.torchtitan_utils.loss.vanilla_tis_function
)
add_when TIS_ON 1 GRPO_ARGS -- --use-tis --tis-clip 2.0 --tis-clip-low 0.0

RECIPE_ARGS=(
   --module vime.backends.torchtitan_utils.models.qwen3_5
   --config vime_qwen35_4b_vl
   --parallelism.data_parallel_replicate_degree 1
   --parallelism.data_parallel_shard_degree ${TRAIN_DP}
   --parallelism.tensor_parallel_degree ${TRAIN_TP}
   --parallelism.pipeline_parallel_degree 1
   --parallelism.context_parallel_degree 1
   --training.seq_len ${SEQ_LEN}
)

TB_ARGS=(
   --use-tensorboard
   --log-multi-turn
   --tb-project-name    "${RUN_NAME}-torchtitan"
   --tb-experiment-name "${RUN_NAME}-${RUN_STAMP}"
)

ASYNC_ARGS=(--update-weights-interval 1)
add_when ROLLOUT_FN nonempty ASYNC_ARGS -- --rollout-function-path "${ROLLOUT_FN}"
gate_when IS_ASYNC 1 ASYNC_ARGS

TRAIN_ARGS=(
   "${PLACEMENT_ARGS[@]}" "${ASYNC_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}"
   "${VLLM_ARGS[@]}"      "${EVAL_ARGS[@]}"  "${PERF_ARGS[@]}" "${GRPO_ARGS[@]}"
   "${RECIPE_ARGS[@]}"    "${TB_ARGS[@]}"
)

rt_env_add TORCHDYNAMO_DISABLE=1 PYTHONFAULTHANDLER=1
[[ -z "${VIME_VLLM_PYTHON:-}" ]] || rt_env_add "VIME_VLLM_PYTHON=${VIME_VLLM_PYTHON}"
[[ ! -v VIME_VLLM_LD_PRELOAD ]] || rt_env_add "VIME_VLLM_LD_PRELOAD=${VIME_VLLM_LD_PRELOAD}"
[[ -z "${CC:-}" ]] || rt_env_add "CC=${CC}"
start_run
