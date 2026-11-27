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


# Qwen3.5-4B + terminalbench: configure the experiment, then reuse the model entry.
set -euo pipefail
AGENTIC_ENTRY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$AGENTIC_ENTRY_DIR/../common/agenticrl.runtime.sh"
source "$AGENTIC_REPO_ROOT/scripts/torchtitan/common/launcher.sh"
VIME_RL_TASK="${VIME_RL_TASK:-terminalbench}"
AGENTIC_RUN_PREFIX=qwen35-4B
configure_run "$@"

# These settings belong to this Qwen Agent experiment, not to the model interface.
export VIME_AGENT_CODEX_MODEL_PROFILE="${VIME_AGENT_CODEX_MODEL_PROFILE:-Qwen3.5-4B}"
configure_agentic_runtime
source "$AGENTIC_COMMON_DIR/agenticrl.default.sh"
AGENTIC_ARGS+=(--vllm-tool-call-parser qwen3_coder --vllm-reasoning-parser qwen3)
source "$AGENTIC_REPO_ROOT/scripts/torchtitan/run-qwen35-4B.sh" "$@"
