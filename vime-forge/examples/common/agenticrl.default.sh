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

# AgenticRL argument defaults: only what a multi-turn run adds to the chosen
# model script. Source after local configuration and task runtime resolution.
# Shared RL parameters (data path, placement, batch, sampling, context lengths)
# stay in the model script; configure_agentic_runtime sets the variables it reads.
AGENTIC_ARGS=(
    # Task generation: the Agent adapter owns prompting, tools and the reward.
    --metadata-key metadata
    --custom-generate-function-path "$GENERATOR_MODULE"
    --router-policy "$VIME_RL_ROUTER_POLICY" --vllm-max-num-seqs "$VIME_RL_VLLM_MAX_NUM_SEQS"
    --use-tensorboard --tb-project-name "$TB_PROJECT" --tb-experiment-name "$RUN_NAME"
)

# TIS
add_when TIS_ON 1 AGENTIC_ARGS -- --use-tis --tis-clip 2.0 --tis-clip-low 0.0

if [[ "$VIME_RL_ENABLE_PREFIX_CACHING" == 1 ]]; then AGENTIC_ARGS+=(--vllm-enable-prefix-caching); fi
if [[ "$VIME_RL_KL_COV" == on ]]; then
    AGENTIC_ARGS+=(--use-kl-cov --kl-cov-ratio "$VIME_RL_KL_COV_RATIO" --kl-cov-coef "$VIME_RL_KL_COV_COEF" --keep-old-actor)
fi
if [[ "$VIME_RL_SAVE_INTERVAL" != off ]]; then
    AGENTIC_ARGS+=(--save "$CKPT_DIR" --save-interval "$VIME_RL_SAVE_INTERVAL"
                  --checkpoint.interval 1 --checkpoint.keep_latest_k 2)
fi
# Trajectory dumps are large at AgenticRL context lengths; VIME_RL_DUMP_MODE=off
# leaves them out. configure_agentic_runtime resolved DUMP_DIR accordingly.
if [[ -n "$REPLAY_DIR" ]]; then
    AGENTIC_ARGS+=(--load-debug-rollout-data "$REPLAY_DIR/{rollout_id}.pt")
elif [[ -n "$DUMP_DIR" ]]; then
    AGENTIC_ARGS+=(--save-debug-rollout-data "$DUMP_DIR/{rollout_id}.pt")
fi
