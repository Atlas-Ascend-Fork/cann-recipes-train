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

# 8-NPU node: 4 training NPUs (DP shard=1, CP=4) + 4 rollout NPUs (TP=4).
set -euo pipefail
export TITAN_CONFIG="${TITAN_CONFIG:-vime_qwen35_9b_cp}"
export TRAIN_NPUS="${TRAIN_NPUS:-4}"
export ROLLOUT_NPUS="${ROLLOUT_NPUS:-4}"
export TP_SIZE="${TP_SIZE:-1}" CP_SIZE="${CP_SIZE:-4}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_single_node_npu.sh" "$@"
