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

# DeepSeek-V4 RL entry point (TorchTitan, NPU). Single node by default; the same
# command also covers a multi-node run.
#
#   bash scripts/torchtitan/run-dsv4.sh [--local <name>] [-- <extra training args>]
#
# Single node: placement comes from ACTOR_GPUS / ROLLOUT_GPUS below (4 + 4 by
# default), cards picked by ASCEND_RT_VISIBLE_DEVICES in common/local.sh.
#
# Multi node: run this SAME command on every node with NODE_IP_LIST set, head
# first. launcher.sh works out which node it is; the head launches the job and
# the others join its ray cluster and stay alive as workers. Everything
# positional (MASTER_ADDR, the node counts, ACTOR_GPUS / ROLLOUT_GPUS, the
# no_proxy list) is then derived, and the recipe/EP defaults scale up:
#
#   NODE_IP_LIST="10.0.0.1 10.0.0.2" NPUS_PER_NODE=8 ROLLOUT_NODE_NUM=1 \
#     bash scripts/torchtitan/run-dsv4.sh
#
# Config is split in two: what a run varies lives here, and what tyro cannot
# express (optimizer hyper-parameters, precision, quantization) lives in the
# recipe named by RECIPE_ARGS.
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/common/launcher.sh"
configure_run "$@"


# Set VLLM_CONFIG explicitly to use a YAML deployment (including PD separation).
VLLM_CONFIG=${VLLM_CONFIG:-}

# The rollout context bound; torchtitan's training.seq_len and the vLLM context
# are both taken from it.
SEQ_LEN=${SEQ_LEN:-4096}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-3072}      # must stay below SEQ_LEN; the launcher checks
ROLLOUT_BATCH=128
N_SAMPLES=8
GLOBAL_BATCH=$(( ROLLOUT_BATCH * N_SAMPLES ))

# Training and rollout cards. launcher.sh derives these three for a multi-node
# run (actor nodes, actor cards per node, rollout cards); the values here are the
# single-node defaults. VLLM_RANKS_PER_ENGINE must divide ROLLOUT_GPUS; the
# launcher checks.
ACTOR_NODE_NUM=${ACTOR_NODE_NUM:-1}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
VLLM_RANKS_PER_ENGINE=${VLLM_RANKS_PER_ENGINE:-${ROLLOUT_GPUS}}

# ⚠ Scale-dependent: this is the 4-layer smoke recipe. A real (multi-node) run needs
# the full stack, set in common/local.sh:
#   DSV4_RECIPE=vime_dsv4_flash_43layers_256experts EP_DEGREE=32
EP_DEGREE=${EP_DEGREE:-4}
CP_DEGREE=${CP_DEGREE:-1}
DSV4_RECIPE=${DSV4_RECIPE:-vime_dsv4_flash_4layers_256experts_fa_mxfp8}

# The scheduler multiplies this per-rank budget by CP_DEGREE when packing samples.
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-$(( (SEQ_LEN + CP_DEGREE - 1) / CP_DEGREE ))}
if (( CP_DEGREE > 1 )); then
    SPMD_BACKEND=${SPMD_BACKEND:-spmd_types}
else
    SPMD_BACKEND=${SPMD_BACKEND:-partial_dtensor}
fi

ENABLE_MXFP4=${ENABLE_MXFP4:-0}

# Training and rollout run on disjoint cards; the sum should not exceed the
# visible ones (ASCEND_RT_VISIBLE_DEVICES, set in common/local.sh).
PLACEMENT_ARGS=(
   --actor-num-nodes "${ACTOR_NODE_NUM}"
   --actor-num-gpus-per-node "${ACTOR_GPUS}"
   --rollout-num-gpus "${ROLLOUT_GPUS}"
)

CKPT_ARGS=(
   # --ref-load is the frozen reference model. No --load: an empty --save directory
   # seeds from --hf-checkpoint, a non-empty one resumes from its largest step.
   --hf-checkpoint "${VIME_MODEL_ROOT}/DeepSeek-V4-Flash-MXFP8/"
   --ref-load      "${VIME_MODEL_ROOT}/DeepSeek-V4-Flash-MXFP8/"
)
add_when SAVE_ON on CKPT_ARGS -- \
   --save "${CKPT_DIR}" \
   --save-interval 20 \
   --checkpoint.keep_latest_k 2   # ⚠ 2 is torchtitan's floor: the newest copy may be
                                  #   mid-write, so an older one has to survive.

ROLLOUT_ARGS=(
   --prompt-data "${VIME_DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type math
   --num-rollout 200
   --rollout-batch-size ${ROLLOUT_BATCH}
   --n-samples-per-prompt ${N_SAMPLES}
   --global-batch-size ${GLOBAL_BATCH}
   --rollout-temperature 1
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"
   --rollout-max-context-len "${SEQ_LEN}"
)
add_when DUMP_DIR nonempty ROLLOUT_ARGS -- --save-debug-rollout-data "${DUMP_DIR}/{rollout_id}.pt"
add_when REPLAY_DIR nonempty ROLLOUT_ARGS -- --load-debug-rollout-data "${REPLAY_DIR}/{rollout_id}.pt"

VLLM_ARGS=(
   --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
   --vllm-server-concurrency 1024
   --vllm-max-model-len "${SEQ_LEN}"
   --vllm-gpu-memory-utilization 0.7
   --vllm-kv-cache-dtype fp8
   --vllm-attention-config '{"indexer_kv_dtype":"mxfp4"}'
   --vllm-enable-expert-parallel
   --no-vllm-async-scheduling
   --vllm-load-format dummy
)

if [[ -n "${VLLM_CONFIG}" ]]; then
   VLLM_ARGS+=(--vllm-config "${VLLM_CONFIG}")
else
   VLLM_ARGS+=(--vllm-data-parallel-size "${VLLM_RANKS_PER_ENGINE}")
   VLLM_ARGS+=(--rollout-num-gpus-per-engine "${VLLM_RANKS_PER_ENGINE}")
fi

add_when ENABLE_MXFP4 1 VLLM_ARGS -- --vllm-enable-mxfp4

EVAL_ARGS=(
   --eval-interval 20
   # Datasets and their per-dataset graders live in the config, so eval can use
   # `math` while training keeps its own --rm-type.
   --eval-config "${REPO_ROOT}/scripts/configs/eval.yaml"
   --eval-top-p 1
)
# The fully-async rollout raises in eval mode, so MODE=fully-async drops the group.
gate_when EVAL_ON on EVAL_ARGS

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
   --extension.quantization.li-quantization mxfp4
)
add_when ENABLE_MXFP4 1 RECIPE_ARGS -- --extension.quantization.enable-mxfp4-qat

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-type low_var_kl
   --kl-loss-coef 0.0
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

TB_ARGS=(
   --use-tensorboard
   --tb-project-name    "${RUN_NAME}-torchtitan"
   --tb-experiment-name "${RUN_NAME}-${RUN_STAMP}"
)

# The training command line, in this order. Empty groups drop out on their own.
TRAIN_ARGS=(
   "${PLACEMENT_ARGS[@]}" "${CKPT_ARGS[@]}"  "${ROLLOUT_ARGS[@]}"
   "${VLLM_ARGS[@]}"      "${EVAL_ARGS[@]}"  "${PERF_ARGS[@]}"     "${TRAINING_ARGS[@]}"
   "${RECIPE_ARGS[@]}"    "${GRPO_ARGS[@]}"  "${TB_ARGS[@]}"
)


start_run
