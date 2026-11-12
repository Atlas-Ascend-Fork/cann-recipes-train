#!/bin/bash
# Adapted from
# https://github.com/volcengine/verl/blob/v0.6.0/recipe/retool/run_qwen2_7b_sft_npu.sh
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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


export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export HYDRA_FULL_ERROR=1                   # display the accurate error stack
export ASCEND_LAUNCH_BLOCKING=0             # debug usage, which seriously affects performance after use, but the error stack is accurate
export RAY_DEDUP_LOGS=1                     # 0: disable ray's log folding 1: enable ray's log folding

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15    # specify certain devices in the environment for use
export ASCEND_GLOBAL_EVENT_ENABLE=0         # whether to display the Ascend EVENT log; 1 is for display
export ASCEND_SLOG_PRINT_TO_STDOUT=0        # does the Ascend log screen out? 0 does not screen. By default, it is at /root/ascend/log. 1 directly screens
export ASCEND_GLOBAL_LOG_LEVEL=3            # log level: 3: ERROR  2:EVENT  1:INFO   0:DEBUG

export HCCL_CONNECT_TIMEOUT=360
export HCCL_EXEC_TIMEOUT=360
export HCCL_IF_BASE_PORT=64033
export CUDA_DEVICE_MAX_CONNECTIONS=1

set -x

nnodes=1
nproc_per_node=16

project_name=retool_sft
experiment_name=multiturn-sft-qwen-3-4b-instruct

TRAIN_DATA=ReTool-SFT/data/train-00000-of-00001.parquet
EVAL_DATA=ReTool-SFT/data/train-00000-of-00001.parquet
MODEL_PATH=ReTool-SFT/model/Qwen3-4B-Instruct-2507
SAVE_PATH=checkpoint/$experiment_name

torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=64 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console", "tensorboard"]' \
    trainer.total_epochs=6 \
    trainer.save_freq=31 \
    trainer.device=npu \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true 2>&1 | tee log/sft_run_log/multiturn-sft-qwen-3-4b-instruct.log
