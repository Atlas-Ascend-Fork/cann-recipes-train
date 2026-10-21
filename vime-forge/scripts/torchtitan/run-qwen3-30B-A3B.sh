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

# Qwen3-30B-A3B (MoE) on torchtitan with a vLLM rollout (NPU). Training and rollout
# run on disjoint cards.
#
#   bash scripts/torchtitan/run-qwen3-30B-A3B.sh [--local <name>] [-- <extra training args>]
#   See --help for options and common/local.sh for machine config.
#
# Same layout as run-qwen3-4B.sh, so the two diff cleanly; only the values a MoE
# model needs are different, each marked "MoE:".
#
# ⚠ marks values that break things when changed (divergence from megatron
#   sampling, non-convergence, crashes).
#
# Config is split in two: what a run varies lives here, and what tyro cannot express (model
# flavor, optimizer hyper-parameters, precision, activation checkpointing) lives in the recipe
# named by RECIPE_ARGS below.
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/common/launcher.sh"
configure_run "$@"


# The rollout context bound. vLLM's max-model-len and torchtitan's
# training.seq_len (the RoPE cache size) are both taken from it.
SEQ_LEN=10240
MAX_RESPONSE_LEN=4096           # must stay below SEQ_LEN; the launcher checks

# GLOBAL_BATCH is the product of the two; do not write the third number by hand.
ROLLOUT_BATCH=32
N_SAMPLES=8
GLOBAL_BATCH=$(( ROLLOUT_BATCH * N_SAMPLES ))

# Rollout-side cards. VLLM_RANKS_PER_ENGINE must divide this, and the launcher reads it to check.
ROLLOUT_GPUS=8
# MoE: one TP=8 engine; the weights do not fit a single 64 GB card.
VLLM_RANKS_PER_ENGINE=8


# Training and rollout run on disjoint cards; the sum should not exceed the
# visible ones (ASCEND_RT_VISIBLE_DEVICES, set in common/local.sh).
# MoE: 8 + 8 needs a 16-card machine config.
PLACEMENT_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --rollout-num-gpus ${ROLLOUT_GPUS}
)

CKPT_ARGS=(
   --hf-checkpoint "${VIME_MODEL_ROOT}/Qwen3-30B-A3B"
   # No --load: an empty --save directory seeds from --hf-checkpoint, a non-empty one resumes from
   # its largest step. --load only names a DIFFERENT directory to read from, see LOAD_FROM below.
   # The KL loss below needs a ref model; vime checks the path whenever KL is on.
   --ref-load "${VIME_MODEL_ROOT}/Qwen3-30B-A3B"
)
add_when SAVE_ON on CKPT_ARGS -- \
   --save "${CKPT_DIR}" \
   --save-interval 20 \
   --checkpoint.keep_latest_k 2  # ⚠ 2 is torchtitan's floor: the newest copy may be
                                 #   mid-write, so an older one has to survive. 1 loses ckpts.

# Read a checkpoint other than the newest one in ${CKPT_DIR}, writing to ${CKPT_DIR} as usual:
#   LOAD_FROM=/path/to/other-ckpt LOAD_STEP=40 bash scripts/torchtitan/run-qwen3-30B-A3B.sh
#   LOAD_STEP=40                                 same directory, an earlier step
#   LOAD_MODE=weights                            weights only, optimizer and step start over
add_when LOAD_FROM nonempty CKPT_ARGS -- --load "${LOAD_FROM:-}"
add_when LOAD_STEP nonempty CKPT_ARGS -- --load-step "${LOAD_STEP:-}"
add_when LOAD_MODE nonempty CKPT_ARGS -- --load-mode "${LOAD_MODE:-}"

ROLLOUT_ARGS=(
   --prompt-data "${VIME_DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key  prompt
   --label-key  label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler       # ⚠ changing rm_type shifts the reward scale a lot;
                              #   check both sides match before comparing runs
   --num-rollout 200          # short by default for smoke runs; override after --
   --rollout-temperature 1
   --rollout-batch-size       ${ROLLOUT_BATCH}
   --n-samples-per-prompt     ${N_SAMPLES}
   --global-batch-size        ${GLOBAL_BATCH}
   --rollout-max-response-len ${MAX_RESPONSE_LEN}
   --rollout-max-context-len  ${SEQ_LEN}
)
add_when DET on ROLLOUT_ARGS -- --debug.seed 1234 --debug.deterministic
add_when DUMP_DIR nonempty ROLLOUT_ARGS -- --save-debug-rollout-data "${DUMP_DIR}/{rollout_id}.pt"
# --load-debug-rollout-data forces debug_train_only: no vLLM, no eval,
# update_weights becomes a no-op, and the placement group only gets actor cards.
add_when REPLAY_DIR nonempty ROLLOUT_ARGS -- --load-debug-rollout-data "${REPLAY_DIR}/{rollout_id}.pt"

VLLM_ARGS=(
   --rollout-num-gpus-per-engine ${VLLM_RANKS_PER_ENGINE}
   --vllm-max-model-len          ${SEQ_LEN}
   --vllm-gpu-memory-utilization 0.7
   # Decode-only graphs; prefill runs eager.
   --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
)
add_when DET on VLLM_ARGS -- --vllm-enable-deterministic-inference --rollout-seed 1234

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "${VIME_DATA_ROOT}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 4   # megatron uses 1; higher damps eval sampling noise
   --eval-max-response-len 16384
   --eval-top-p 1
)
# The fully-async rollout raises in eval mode, so MODE=fully-async drops the group.
gate_when EVAL_ON on EVAL_ARGS

PERF_ARGS=(
   --max-tokens-per-gpu 4096
   --use-dynamic-batch-size
   # MoE: experts are sharded 8 ways; 128 experts stay divisible. tp/pp/cp remain 1.
   # The AdamW moments go to host memory through the recipe's swap-optimizer override, which is
   # what makes 30.5B fit a 64 GB card.
   --parallelism.expert_parallel_degree 8
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip      0.2
   --eps-clip-high 0.28
   # KL at coefficient 0 contributes nothing to the gradient; the ref model is still loaded and
   # its log-probs computed every step.
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   # MoE: replay the rollout's expert choices during training, so the two policies route
   # identically and the ratio measures the policy gap rather than routing noise
   # (https://arxiv.org/abs/2510.11370). Needs a rollout that returns routed_experts.
   --use-rollout-routing-replay
)
add_when TIS_ON 1 GRPO_ARGS -- --use-tis --tis-clip 2.0 --tis-clip-low 0.0

# The torchtitan recipe: model flavor, optimizer, precision, activation checkpointing and the NPU
# kernel overrides all live in vime/backends/torchtitan_utils/models/qwen3/config_registry.py.
RECIPE_ARGS=(
   --module vime.backends.torchtitan_utils.models.qwen3
   --config vime_qwen3_30b_a3b
)

MISC_ARGS=(
   --attention-dropout 0.0    # megatron defaults to 0.1
   --hidden-dropout    0.0
)

TB_ARGS=(
   --use-tensorboard
   --tb-project-name    "${RUN_NAME}-torchtitan"
   --tb-experiment-name "${RUN_NAME}-${RUN_STAMP}"
)

# Appended only for async / fully-async; the whole group is dropped under sync.
ASYNC_ARGS=(--update-weights-interval 1)   # bounds weight staleness
add_when ROLLOUT_FN nonempty ASYNC_ARGS -- --rollout-function-path "${ROLLOUT_FN}"
gate_when IS_ASYNC 1 ASYNC_ARGS

# The training command line, in this order. Empty groups drop out on their own.
TRAIN_ARGS=(
   "${PLACEMENT_ARGS[@]}" "${ASYNC_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}"
   "${VLLM_ARGS[@]}"      "${EVAL_ARGS[@]}"  "${PERF_ARGS[@]}" "${GRPO_ARGS[@]}"
   "${RECIPE_ARGS[@]}"    "${TB_ARGS[@]}"    "${MISC_ARGS[@]}"
)


start_run
