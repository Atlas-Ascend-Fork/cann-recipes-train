# coding=utf-8
# Adapted from cann-recipes-train/llm_rl/qwen2_5/verl_npu_demo/run_qwen2_5_1_5b.sh
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates. All rights reserved.
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

# Qwen2.5-1.5B MATH RL on A5, two cards. The actor trains in BF16 either way;
# only the vLLM rollout precision changes. Everything below is identical for the
# two runs except MODEL_PATH, the rollout quantization flag and the run names.
#
# Usage:
#   bash run_qwen2_5_1_5b_A5.sh bf16
#   bash run_qwen2_5_1_5b_A5.sh hif8

set -euo pipefail

PRECISION=${1:-}
if [ "${PRECISION}" != "bf16" ] && [ "${PRECISION}" != "hif8" ]; then
    echo "usage: bash $(basename "$0") {bf16|hif8}" >&2
    exit 2
fi

export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}
if [ ! -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    echo "missing ${ASCEND_HOME_PATH}/set_env.sh; point ASCEND_HOME_PATH at the CANN toolkit root" >&2
    exit 1
fi
source "${ASCEND_HOME_PATH}/set_env.sh"

export HYDRA_FULL_ERROR=1
export ASCEND_LAUNCH_BLOCKING=0
export RAY_DEDUP_LOGS=1

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
# The A2/A3 recipe pins HCCL_IF_BASE_PORT=64033, which collides on a shared host.
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export CUDA_DEVICE_MAX_CONNECTIONS=1

export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_V1=1
export TORCHDYNAMO_DISABLE=1

QUANT_ARG=""
if [ "${PRECISION}" = "hif8" ]; then
    # vLLM-Ascend reads the scheme from quant_model_description.json inside the model
    # directory; rollout.quantization_config_file cannot carry it. Build the directory
    # with make_quant_model_dir.py; see the sample README.md for the keys.
    MODEL_PATH=${MODEL_PATH:-./model/Qwen2_5_1_5B_Instruct_hif8}
    if [ ! -f "${MODEL_PATH}/quant_model_description.json" ]; then
        echo "missing ${MODEL_PATH}/quant_model_description.json" >&2
        echo "build the directory MODEL_PATH points at; see the sample README. For the default path," >&2
        echo "run this from the sample directory:" >&2
        echo "    python3 make_quant_model_dir.py --model \"$(pwd)/model/Qwen2_5_1_5B_Instruct\" --scheme hif8" >&2
        exit 2
    fi
    QUANT_ARG="actor_rollout_ref.rollout.quantization=ascend"
else
    MODEL_PATH=${MODEL_PATH:-./model/Qwen2_5_1_5B_Instruct}
fi
RUN_NAME="qwen2_5_1_5b_math_a5_${PRECISION}"

mkdir -p ./run_log

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=./data/math/data/train.parquet \
    data.val_files=./data/math/data/val_subset.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=1234 \
    reward.custom_reward_function.path=./a5_math_reward_threadsafe.py \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    ${QUANT_ARG} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='a5_rollout_precision_cmp' \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10 \
    trainer.device=npu 2>&1 | tee "./run_log/${RUN_NAME}.log"
