#!/bin/bash
# DeepSeek-V4 RL entry point with stable TorchTitan launch arguments.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common/launcher.sh"

# Runtime paths remain configurable because they vary between installations.
export VIME_VENV=${VIME_VENV:-/app/uv-train}
export VIME_ROLLOUT_VENV=${VIME_ROLLOUT_VENV:-/app/uv-vllm}
export TRAIN_PYTHON=${TRAIN_PYTHON:-${VIME_VENV}/bin/python}
export VIME_VLLM_EXECUTABLE=${VIME_VLLM_EXECUTABLE:-${VIME_ROLLOUT_VENV}/bin/vllm}

# Output roots; use shared storage for multi-node checkpoints.
export VIME_WORKSPACE=${VIME_WORKSPACE:-/tmp/vime}
export VIME_CKPT_ROOT=${VIME_CKPT_ROOT:-${VIME_WORKSPACE}/ckpt}
export VIME_TB_ROOT=${VIME_TB_ROOT:-${VIME_WORKSPACE}/tb}
# launcher derives CKPT_DIR=<ckpt root>/<run>-<mode> and
# TENSORBOARD_DIR=<tb root>/<run>/<timestamp>; configure the roots above.

# Preserve the original default checkpoint/archive naming after renaming this file.
export RUN_NAME=${RUN_NAME:-run-dsv4-torchtitan}
configure_run "$@"

# These defaults differ between direct and node-list launches.
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
SEQ_LEN=${SEQ_LEN:-4096}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-2048}
VLLM_RANKS_PER_ENGINE=${VLLM_RANKS_PER_ENGINE:-${ROLLOUT_GPUS}}
DSV4_RECIPE=${DSV4_RECIPE:-vime_dsv4_flash_4layers_256experts}
EP_DEGREE=${EP_DEGREE:-4}
ENABLE_MXFP4=${ENABLE_MXFP4:-0}
CP_DEGREE=${CP_DEGREE:-1}
# The scheduler multiplies this per-rank budget by CP_DEGREE when packing samples.
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-$(( (SEQ_LEN + CP_DEGREE - 1) / CP_DEGREE ))}
if (( CP_DEGREE > 1 )); then
  SPMD_BACKEND=${SPMD_BACKEND:-spmd_types}
else
  SPMD_BACKEND=${SPMD_BACKEND:-partial_dtensor}
fi

# Model, data, checkpoint, and configuration paths remain configurable.
DSV4_MODEL=${DSV4_MODEL:-${DSV4_MODEL_PATH:-${VIME_MODEL_ROOT}/DeepSeek-V4}}
DATASET=${DATASET:-${DATA_PATH:-${VIME_DATA_ROOT}/dapo-math-17k.jsonl}}
# Set VLLM_CONFIG explicitly to use a YAML deployment (including PD separation).
VLLM_CONFIG=${VLLM_CONFIG:-}

PLACEMENT_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${ACTOR_GPUS}"
  --rollout-num-gpus "${ROLLOUT_GPUS}"
)
MODEL_ARGS=(
  --hf-checkpoint "${DSV4_MODEL}"
  --ref-load "${DSV4_MODEL}"
)
CKPT_ARGS=(
  --save "${CKPT_DIR}"
  --save-interval 100
  --checkpoint.keep_latest_k 2
)
ROLLOUT_ARGS=(
  --prompt-data "${DATASET}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --balance-data
  --rm-type math
  --num-rollout 200
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --global-batch-size 256
  --rollout-temperature 1
  --rollout-max-response-len "${MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${SEQ_LEN}"
)
VLLM_ARGS=(
  --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --vllm-server-concurrency 64
  --vllm-max-model-len "${SEQ_LEN}"
  --vllm-gpu-memory-utilization 0.7
  --vllm-kv-cache-dtype bfloat16
  --vllm-enable-expert-parallel
  --no-vllm-async-scheduling
  --vllm-load-format dummy
)
if [[ -n "${VLLM_CONFIG}" ]]; then
  # Group-specific DP values are read and applied by Vime for each instance.
  VLLM_ARGS+=(--vllm-config "${VLLM_CONFIG}")
  VLLM_ARGS+=(--vllm-data-parallel-size "$((VLLM_RANKS_PER_ENGINE / 2))")
else
  VLLM_ARGS+=(--vllm-data-parallel-size "${VLLM_RANKS_PER_ENGINE}")
  VLLM_ARGS+=(--rollout-num-gpus-per-engine "${VLLM_RANKS_PER_ENGINE}")
fi
if [[ "${ENABLE_MXFP4}" == "1" ]]; then
  VLLM_ARGS+=(--vllm-enable-mxfp4)
  VLLM_ARGS+=(--vllm-additional-config '{"enable_mxfp4_li":true}')
fi
# Release cached allocations between training phases for DSV4 memory pressure.
# Pass -- --no-fine-grained-memory-cleanup to disable it.
PERF_ARGS=(
  --fine-grained-memory-cleanup
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --parallelism.expert_parallel_degree "${EP_DEGREE}"
  --parallelism.context_parallel_degree "${CP_DEGREE}"
  --parallelism.spmd_backend "${SPMD_BACKEND}"
  --use-rollout-routing-replay
)
if (( CP_DEGREE > 1 )); then
  PERF_ARGS+=(--packed-seq-alignment "${PACKED_SEQ_ALIGNMENT:-64}")
fi
TRAINING_ARGS=(
  --training.seq_len "${SEQ_LEN}"
  --lr_scheduler.decay_type linear
  --lr_scheduler.min_lr_factor 1.0
  --lr_scheduler.warmup_steps 0
  --use-dynamic-batch-size
  --checkpoint.initial_load_in_hf_quantized
)
RECIPE_ARGS=(
  --module vime.backends.torchtitan_utils.models.deepseek_v4
  --config "${DSV4_RECIPE}"
  --extension.quantization.recipe all_block_fp8
  --extension.quantization.enable-quantized-training
)
[[ "${ENABLE_MXFP4}" == "1" ]] && RECIPE_ARGS+=(
  --extension.quantization.enable-mxfp4-qat
  --extension.quantization.li-quantization mxfp4
)
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-type low_var_kl
  --kl-loss-coef 0.0
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
)
MISC_ARGS=(--use-tensorboard)
REPRODUCIBILITY_ARGS=(
  --debug.seed 1234
  --debug.deterministic
  --vllm-enable-deterministic-inference
  --rollout-seed 1234
  --rl-batch-invariant
)

TRAIN_ARGS=(
  "${PLACEMENT_ARGS[@]}" "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${VLLM_ARGS[@]}"
  "${PERF_ARGS[@]}" "${TRAINING_ARGS[@]}" "${RECIPE_ARGS[@]}" "${GRPO_ARGS[@]}"
  "${MISC_ARGS[@]}" "${REPRODUCIBILITY_ARGS[@]}"
)
start_run
