#!/bin/bash
# Qwen3-4B on torchtitan with a vLLM rollout (NPU). Training and rollout run on
# disjoint cards.
#
#   bash scripts/torchtitan/run-qwen3-4B.sh [--local <name>] [-- <extra training args>]
#   See --help for options and common/local.sh for machine config.
#
# Between configure_run and start_run this file only declares parameters -- no
# control flow. A value is pulled out into a variable only when it is used more
# than once, or when the launcher has to read it to validate something;
# everything else is written where it is used.
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
MAX_RESPONSE_LEN=8192  # must stay below SEQ_LEN; the launcher checks. Small respones_len(<2048) makes sample zero reward

# GLOBAL_BATCH is the product of the two; do not write the third number by hand.
ROLLOUT_BATCH=32
N_SAMPLES=8
GLOBAL_BATCH=$(( ROLLOUT_BATCH * N_SAMPLES ))

# Rollout-side cards. VLLM_RANKS_PER_ENGINE must divide this, and the launcher reads it to check.
ROLLOUT_GPUS=4
# ⚠ megatron uses 2; 4 removes the TP-induced numerical divergence in sampling.
VLLM_RANKS_PER_ENGINE=4


# Training and rollout run on disjoint cards; the sum should not exceed the
# visible ones (ASCEND_RT_VISIBLE_DEVICES, set in common/local.sh).
PLACEMENT_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 4
   --rollout-num-gpus ${ROLLOUT_GPUS}
)

CKPT_ARGS=(
   --hf-checkpoint "${VIME_MODEL_ROOT}/Qwen3-4B"
   # No --load: an empty --save directory seeds from --hf-checkpoint, a non-empty one resumes from
   # its largest step. --load only names a DIFFERENT directory to read from, see LOAD_FROM below.
   # No --ref-load: matches megatron, which loads no ref model (see: no KL loss below)
)
add_when SAVE_ON on CKPT_ARGS -- \
   --save "${CKPT_DIR}" \
   --save-interval 20 \
   --checkpoint.keep_latest_k 2  # ⚠ 2 is torchtitan's floor: the newest copy may be
                                 #   mid-write, so an older one has to survive. 1 loses ckpts.

# Read a checkpoint other than the newest one in ${CKPT_DIR}, writing to ${CKPT_DIR} as usual:
#   LOAD_FROM=/path/to/other-ckpt LOAD_STEP=40 bash scripts/torchtitan/run-qwen3-4B.sh
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
   --vllm-gpu-memory-utilization 0.6   # smaller KV cache, leaving room for generation activations
   --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
   # --vllm-enable-sleep-mode     # ⚠ conflicts with PYTORCH_NPU_ALLOC_CONF=expandable_segments:True;
                                  #  vLLM asserts and exits. Memory only, no correctness impact.
   # --no-vllm-async-scheduling   megatron defaults async on, so this stays off. If rollout
   #                                misbehaves on NPU, this is the first line to restore.

   # --vllm-enable-expert-parallel
   # --vllm-data-parallel-size ${VLLM_RANKS_PER_ENGINE}
)
add_when DET on VLLM_ARGS -- --vllm-enable-deterministic-inference --rollout-seed 1234

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "${VIME_DATA_ROOT}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 4   # megatron uses 1; higher damps eval sampling noise
   --eval-max-response-len 16384   # megatron uses 16384
   --eval-top-p 1
)
# The fully-async rollout raises in eval mode, so MODE=fully-async drops the group.
gate_when EVAL_ON on EVAL_ARGS

PERF_ARGS=(
   --max-tokens-per-gpu 9216       # matches megatron; also torchtitan's per-micro-batch budget
   --use-dynamic-batch-size
   # --log-probs-chunk-size 8192

   # tp/pp/cp/ep all default to 1 in torchtitan, so they need no explicit flags.
   # compile stays off: NPU forbids torch.compile, and tyro's bool has no "False" form.
   # --parallelism.expert_parallel_degree 4   # tp/pp/cp=1 默认可省
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip      0.2
   --eps-clip-high 0.28
   # No KL loss, matching megatron. Leaving it out also stops the ref model
   # from being loaded and computed.
)
add_when TIS_ON 1 GRPO_ARGS -- --use-tis --tis-clip 2.0 --tis-clip-low 0.0

# The torchtitan recipe: model flavor, optimizer, precision, activation checkpointing and the NPU
# kernel overrides all live in vime/backends/torchtitan_utils/models/qwen3/config_registry.py.
RECIPE_ARGS=(
   --module vime.backends.torchtitan_utils.models.qwen3
   --config vime_qwen3_4b
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
