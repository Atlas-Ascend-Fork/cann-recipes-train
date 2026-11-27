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

# Task runtime helpers. Python/CANN/ATB activation belongs to the shared launcher.
fail() { echo "$*" >&2; exit 2; }

AGENTIC_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTIC_REPO_ROOT="$(cd "$AGENTIC_COMMON_DIR/../.." && pwd)"

load_agentic_config() {
    local task="$1" resolved
    shift
    export REPO_ROOT="${REPO_ROOT:-$AGENTIC_REPO_ROOT}"
    # Check the producer's status before evaluating its shell-quoted assignments.
    resolved=$("${VIME_JSON_PYTHON:-python3}" "$AGENTIC_REPO_ROOT/vime/agent/config.py" \
        shell --task "$task" --repo-root "$REPO_ROOT" "$@") || return $?
    eval "$resolved"
}

check_agentic_paths() {
    local path
    for path in "$MODEL_PATH/config.json" "$VIME_RL_PROMPT_DATA"; do
        [[ -r "$path" ]] || fail "Required file not readable: $path"
    done
    if [[ "$VIME_RL_TASK" == swegym ]]; then
        [[ -r "$VIME_SWEGYM_SOURCE_PATH/swegym/harness/grading.py" ]] || fail 'SWE-Gym evaluator path is not readable'
    fi
    if [[ "$VIME_AGENT_SANDBOX_BACKEND" == agentenv ]]; then
        [[ -r "$VIME_AGENTENV_API_KEY_FILE" ]] || fail 'AgentENV API key file is not readable'
    fi
}

initialize_agentic_run() {
    umask 077
    if [[ "${LOCAL_NODE_INDEX:-0}" == 0 ]]; then
        mkdir -p "$(dirname "$OUTPUT_DIR")"
        mkdir "$OUTPUT_DIR"  # Refuse to overwrite an earlier run.
    fi
    # The key is inherited by the Ray head, not serialized in runtime-env JSON.
    if [[ "${VIME_AGENT_SANDBOX_BACKEND:-agentenv}" == agentenv ]]; then
        [[ -r "$VIME_AGENTENV_API_KEY_FILE" ]] || fail "AgentENV API key file is not readable"
        export E2B_API_KEY="$(<"$VIME_AGENTENV_API_KEY_FILE")"
        [[ -n "$E2B_API_KEY" ]] || fail 'AgentENV API key is empty.'
    fi
}

configure_agentic_workers() {
    local resolved
    resolved=$("${VIME_JSON_PYTHON:-python3}" "$AGENTIC_REPO_ROOT/vime/agent/config.py" workers) || return $?
    # config.py emits only task settings; shared runtime defaults stay intact.
    eval "${resolved/RT_ENV=/AGENT_WORKER_ENV=}"
    rt_env_add "${AGENT_WORKER_ENV[@]}"
}

configure_task_options() {
    TASK_BUNDLE=''; TASK_AGENT=''; TASK_LAUNCH_ARGS=()
    while (( $# )); do
        case "$1" in
            --bundle|--agent)
                need_val "$@"
                case "$1" in
                    --bundle) TASK_BUNDLE="$2" ;;
                    --agent) TASK_AGENT="$2" ;;
                esac
                shift 2 ;;
            --local|--dump-dir|--replay-dir)
                need_val "$@"; TASK_LAUNCH_ARGS+=("$1" "$2"); shift 2 ;;
            -h|--help)
                cli_help
                printf '\nTask options (before --):\n  --bundle <path>    prepared AgenticRL bundle\n  --agent codex|claude_code\n\nThe task comes from the entry script; set VIME_RL_TASK to override it.\n'
                exit 0 ;;
            --) TASK_LAUNCH_ARGS+=("$@"); break ;;
            *) TASK_LAUNCH_ARGS+=("$1"); shift ;;
        esac
    done
    case "$VIME_RL_TASK" in
        swegym|terminalbench) ;;
        singleturn) fail 'This entry runs AgenticRL; single-turn runs use scripts/torchtitan/<model>.sh' ;;
        *) fail "Unknown task: $VIME_RL_TASK" ;;
    esac
    MODE=${MODE:-async}
    EVAL_ON=${EVAL_ON:-off}; SAVE_ON=${SAVE_ON:-off}
    RUN_NAME=${RUN_NAME:-${AGENTIC_RUN_PREFIX:-agenticrl}-${VIME_RL_TASK}-$(date +%Y%m%d-%H%M%S)}
}


# Resolves the task configuration and sets the run paths, context lengths and
# task data that the model script expands. Called before sourcing the model.
configure_agentic_runtime() {
    [[ "$MODE" == async ]] || fail 'AgenticRL requires MODE=async'
    [[ "$COLOCATE" == off ]] || fail 'AgenticRL requires COLOCATE=off'
    [[ -z "$TASK_BUNDLE" ]] || export VIME_RL_PREPARED_BUNDLE="$TASK_BUNDLE"
    [[ -z "$TASK_AGENT" ]] || export VIME_RL_AGENT="$TASK_AGENT"
    load_agentic_config "$VIME_RL_TASK" || return $?
    export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/$RUN_NAME}"
    CKPT_DIR="$OUTPUT_DIR/checkpoints"
    ARCHIVE_DIR="$OUTPUT_DIR/tensorboard/$RUN_NAME/$RUN_STAMP"
    # The Agent context budget bounds training and inference together. A whole
    # trajectory is the response, so the usual prompt/response split does not apply.
    SEQ_LEN="${SEQ_LEN:-$VIME_AGENT_MAX_CONTEXT_TOKENS}"
    [[ "$SEQ_LEN" == "$VIME_AGENT_MAX_CONTEXT_TOKENS" ]] || \
        fail 'Set VIME_AGENT_MAX_CONTEXT_TOKENS to change the AgenticRL context length'
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-$((SEQ_LEN - 1))}"
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
    # Task data is derived from the prepared bundle, not the model's own dataset.
    PROMPT_DATA="$VIME_RL_PROMPT_DATA"
    case "$VIME_RL_TASK" in
        swegym) GENERATOR_MODULE=examples.swe_agent_rl.generate.generate ;;
        terminalbench) GENERATOR_MODULE=examples.terminal_agent_rl.generate.generate ;;
    esac
    # Resolved before agenticrl.default.sh reads it; a model script supplies a
    # dump path for single-turn runs only. VIME_RL_DUMP_MODE=off keeps it unset.
    if [[ -n "$REPLAY_DIR" || "$VIME_RL_DUMP_MODE" == off ]]; then DUMP_DIR=''
    else DUMP_DIR="${DUMP_DIR:-$OUTPUT_DIR/rollout_dumps}"; fi
    # Derived from the resolved task so a task override cannot leave a stale project name.
    TB_PROJECT=${TB_PROJECT:-${AGENTIC_RUN_PREFIX:-agenticrl}-${VIME_RL_TASK}-async}
    export HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
    export HCCL_DETERMINISTIC=strict
    export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}" STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
    export MULTI_STREAM_MEMORY_RESERVE="${MULTI_STREAM_MEMORY_RESERVE:-0}"
    export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export PYTHONDONTWRITEBYTECODE=1
}

# Derives inference concurrency from the final device placement, then verifies
# paths and creates the run. Called by start_run, after the model script.
prepare_agentic_run() {
    [[ "${AGENTIC_RUNTIME_READY:-0}" != 1 ]] || return 0
    # Trajectory concurrency is a task limit spread over the rollout engines. An
    # explicit SERVER_CONCURRENCY keeps the model value; CLI arguments still win.
    if [[ -z "${SERVER_CONCURRENCY:-}" ]]; then
        local engines=$((ROLLOUT_GPUS / VLLM_RANKS_PER_ENGINE))
        (( engines > 0 )) || fail 'Rollout devices must cover at least one inference engine'
        SERVER_CONCURRENCY=$((VIME_RL_TRIAL_CONCURRENCY / engines))
        (( SERVER_CONCURRENCY > 0 )) || fail 'Trajectory concurrency must cover the rollout engine count'
        AGENTIC_ARGS+=(--vllm-server-concurrency "$SERVER_CONCURRENCY")
    fi
    check_agentic_paths
    initialize_agentic_run
    configure_agentic_workers
    AGENTIC_RUNTIME_READY=1
}
