#!/usr/bin/env bash
# Sourced by the SWE-Gym launcher. No experiment defaults or training command here.
# Keep training activation, bundle validation and Ray env transport out
# of the user-facing parameter list. Sourcing this file has no side effects.

validate_swegym_config() {
    if (( ! DRY_RUN )); then
        [[ "$ADAPTER_PUBLIC_HOST" != 192.0.2.1 ]] || fail 'Set ADAPTER_PUBLIC_HOST to the adapter/relay host reachable from your sandboxes; 192.0.2.1 is only a placeholder.'
    fi
    [[ "$VIME_SLOT" =~ ^(0|[1-9][0-9]*)$ ]] && (( VIME_SLOT <= 44 )) || fail "VIME_SLOT must be an integer from 0 to 44 (HCCL port range)."
    for name in SWE_TRAIN_TP SWE_TRAIN_DP SWE_ROLLOUT_GPUS SWE_ROLLOUT_TP SWE_NUM_EPOCHS \
        SWE_ROLLOUT_BATCH_SIZE SWE_N_SAMPLES_PER_PROMPT SWE_TRIAL_CONCURRENCY SWE_BOOT_CONCURRENCY \
        SWE_BOOT_RETRIES SWE_AGENT_TIME_BUDGET_SEC SWE_EVAL_TIMEOUT_SEC SWE_ROLLOUT_GUARD_SEC \
        SWE_VLLM_MAX_NUM_SEQS SWE_MAX_TOKENS_PER_GPU VIME_AGENT_MAX_TURNS_PER_SESSION \
        VIME_AGENT_SANDBOX_LIFETIME_SEC VIME_AGENT_SANDBOX_RPC_RETRIES; do
        [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer."
    done
    [[ "$SWE_SAVE_INTERVAL" == off || "$SWE_SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]] || fail 'SWE_SAVE_INTERVAL must be off or positive.'
    case "$SWE_DUMP_MODE" in off|rollout) ;; *) fail 'SWE_DUMP_MODE must be off or rollout (detailed tracing is disabled).' ;; esac
    [[ "$SWE_ENABLE_PREFIX_CACHING" == 0 || "$SWE_ENABLE_PREFIX_CACHING" == 1 ]] || fail 'SWE_ENABLE_PREFIX_CACHING must be 0 or 1.'
    (( SWE_ROLLOUT_GPUS % SWE_ROLLOUT_TP == 0 )) || fail 'Rollout TP must divide rollout devices.'
    (( SWE_TRIAL_CONCURRENCY % (SWE_ROLLOUT_GPUS / SWE_ROLLOUT_TP) == 0 )) || fail 'Trial concurrency must be divisible by rollout engine count.'
    (( SWE_MAX_TOKENS_PER_GPU >= SWE_ROLLOUT_SEQ_LEN )) || fail 'Packing budget must cover a complete 32768-token segment.'
    (( SWE_ROLLOUT_GUARD_SEC > SWE_AGENT_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC )) || fail 'Rollout guard must exceed agent plus verifier budgets.'
    [[ -n "$VIME_AGENT_TASK_CONDA_SH" && -n "$VIME_AGENT_TASK_CONDA_PREFIX" || -z "$VIME_AGENT_TASK_CONDA_SH" && -z "$VIME_AGENT_TASK_CONDA_PREFIX" ]] || fail 'Set both guest Conda variables, or leave both empty.'
    if [[ -n "$VIME_AGENT_TASK_CONDA_SH" ]]; then
        [[ "$VIME_AGENT_TASK_CONDA_SH" == /* && "$VIME_AGENT_TASK_CONDA_PREFIX" == /* ]] || fail 'Guest Conda paths must be absolute.'
    fi
    export SWE_TOTAL_NPUS=$((SWE_TRAIN_TP * SWE_TRAIN_DP + SWE_ROLLOUT_GPUS))
    # Explicit physical IDs are respected, including non-contiguous allocations.
    if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        ASCEND_RT_VISIBLE_DEVICES=0
        for ((device=1; device<SWE_TOTAL_NPUS; device++)); do ASCEND_RT_VISIBLE_DEVICES+=",${device}"; done
    fi
    export ASCEND_RT_VISIBLE_DEVICES
    "${VIME_JSON_PYTHON:-python3}" - <<'PY'
import math, os
e = os.environ
temperature, top_k = float(e['SWE_ROLLOUT_TEMPERATURE']), int(e['SWE_ROLLOUT_TOP_K'])
if not math.isfinite(temperature) or temperature < 0 or (top_k != -1 and top_k <= 0):
    raise SystemExit('Temperature must be finite/nonnegative; Top-K must be -1 or positive')
if e['SWE_LEARNING_RATE'] and (not math.isfinite(float(e['SWE_LEARNING_RATE'])) or float(e['SWE_LEARNING_RATE']) <= 0):
    raise SystemExit('Learning rate must be finite and positive')
ids = e['ASCEND_RT_VISIBLE_DEVICES'].split(',')
if any(not i.isdecimal() for i in ids) or len(set(map(int, ids))) != len(ids):
    raise SystemExit('Visible NPU IDs must be distinct nonnegative integers')
if len(ids) < int(e['SWE_TOTAL_NPUS']):
    raise SystemExit('Not enough visible NPUs for the selected training and rollout allocation')
for name in ('ADAPTER_PORT', 'ADAPTER_PUBLIC_PORT'):
    if not 1 <= int(e[name]) <= 65535:
        raise SystemExit(name + ' must be a valid TCP port')
PY
    # Count actual prompts; a dry-run can still describe the command when its
    # dataset only exists on the remote training host.
    SWE_PROMPT_COUNT=$("${VIME_JSON_PYTHON:-python3}" - <<'PY'
import json, os, sys
from pathlib import Path
dry_run = os.environ['VIME_QWEN35_DRY_RUN'] == '1'
def rows(key):
    return [json.loads(line) for line in Path(os.environ[key]).read_text().splitlines() if line.strip()]
try:
    prompts = rows('SWE_PROMPT_DATA')
except OSError:
    if dry_run:
        sys.exit(0)  # Empty count means unknown, never zero tasks.
    raise
ids = [row.get('label') or row.get('metadata', {}).get('instance_id') for row in prompts]
if not ids or not all(isinstance(i, str) and i for i in ids) or len(ids) != len(set(ids)):
    raise SystemExit('Prompt IDs must be nonempty and unique')
if not dry_run:
    for key in ('SWE_AGENTENV_TEMPLATE_MAP', 'SWE_VERIFIER_SPEC_MAP'):
        mapped = [row.get('instance_id') for row in rows(key)]
        if len(mapped) != len(ids) or set(mapped) != set(ids):
            raise SystemExit(key + ': task IDs differ from prompt data')
    json.loads(Path(os.environ['SWE_PREPARATION_MANIFEST']).read_text())
print(len(ids))
PY
    )
    export SWE_PROMPT_COUNT
    if [[ -n "$SWE_PROMPT_COUNT" ]]; then
        (( SWE_PROMPT_COUNT % SWE_ROLLOUT_BATCH_SIZE == 0 )) || fail 'Prompt batch size must divide the dataset.'
    fi
}

prepare_swegym_runtime() {
    for executable in "$VIME_TRAIN_PYTHON" "$TRAIN_ENV/bin/ray" "$ROLLOUT_ENV/bin/python"; do
        [[ -x "$executable" ]] || fail "Missing executable: $executable"
    done
    for path in "$TRAIN_ENV/bin/activate" "$VIME_CANN_HOME/set_env.sh" \
        "$MODEL_PATH/config.json" "$PROMPT_DATA" \
        "$VIME_AGENTENV_API_KEY_FILE" "$VIME_AGENT_NODE_TARBALL" "$VIME_AGENT_CODEX_TARBALL" \
        "$VIME_SWEGYM_SOURCE_PATH/swegym/harness/grading.py"; do
        [[ -r "$path" ]] || fail "Required file not readable: $path"
    done
    export REPO_ROOT

    NPU_SMI_DIR="$(dirname "$(command -v npu-smi)")"
    unset VIRTUAL_ENV PYTHONPATH PYTHONHOME CC CXX CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
    unset TORCH_NPU_DEVICE_CAPABILITY VLLM_VERSION
    unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME
    unset ASCEND_CUSTOM_OPP_PATH ASCEND_HOME NNAL_HOME ATB_HOME_PATH ATB_OPSRUNNER_PATH
    export LD_LIBRARY_PATH="/opt/buildtools/python-3.12.9/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
    export PATH="${NPU_SMI_DIR}:${TRAIN_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    set +u
    source "$TRAIN_ENV/bin/activate"
    source "$VIME_CANN_HOME/set_env.sh"
    set -u
    export PATH="$(dirname "$VIME_TRAIN_PYTHON"):$PATH"
    export VIME_JSON_PYTHON="$VIME_TRAIN_PYTHON"
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    export TRITON_CACHE_DIR="${OUTPUT_DIR}/triton-cache"
    export TMPDIR=/tmp PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    export VLLM_ASCEND_ENABLE_NZ=0 VLLM_USE_AOT_COMPILE=0 VLLM_ASCEND_NPUGRAPH_CAPTURE_LIMIT=512
    export HCCL_WHITELIST_DISABLE=1 HCCL_CONNECT_TIMEOUT=3600 HCCL_DETERMINISTIC=strict
    export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-$((60000 + VIME_SLOT * 100))-$((60050 + VIME_SLOT * 100))}"
    export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-$((61000 + VIME_SLOT * 100))-$((61050 + VIME_SLOT * 100))}"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_MAX_CONNECTIONS=1
    export TASK_QUEUE_ENABLE=1 STREAMS_PER_DEVICE=32 MULTI_STREAM_MEMORY_RESERVE=0
    export CPU_AFFINITY_CONF=1 OMP_NUM_THREADS=1 TRANSFORMERS_VERBOSITY=error
    export MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
    export no_proxy="127.0.0.1,localhost,::1,${MASTER_ADDR}" NO_PROXY="127.0.0.1,localhost,::1,${MASTER_ADDR}"
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
    ulimit -Sn "$VIME_NOFILE_LIMIT"
    umask 077
    # Use the actual training interpreter for SDK, grader and bundle preflight.
    # Imports do not create a sandbox or load model weights.
    "$VIME_TRAIN_PYTHON" - <<'PY_AGENT'
import json, os
from pathlib import Path
from vime.agent.sandbox import _append_optional_sdk_path
from vime.agent.harness.common import TaskCondaEnvironment
_append_optional_sdk_path('VIME_AGENT_E2B_SDK_PATH')
from e2b import AsyncSandbox
import examples.coding_agent_rl.generate as generation
TaskCondaEnvironment.from_env()
config = json.loads((Path(os.environ['MODEL_PATH']) / 'config.json').read_text())
text = config.get('text_config', {})
actual = (text.get('hidden_size'), text.get('intermediate_size'),
          text.get('tie_word_embeddings', config.get('tie_word_embeddings', False)))
if config.get('model_type') != 'qwen3_5' or actual != (2560, 9216, True):
    raise SystemExit(f'Expected a dense Qwen3.5-4B checkpoint, found {actual}')
if generation.swe._SWEGYM_IMPORT_ERROR is not None:
    raise RuntimeError('SWE-Gym evaluator is unavailable') from generation.swe._SWEGYM_IMPORT_ERROR
if generation.TEMPLATE_MAPPING is None or generation.VERIFIER_SPECS is None:
    raise SystemExit('Prepared template and verifier mappings are required')
count = 0
for line in Path(os.environ['SWE_PROMPT_DATA']).read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    sample = generation.Sample(prompt=row['prompt'], label=row.get('label'), metadata=row.get('metadata', {}))
    md = generation.swe.get_metadata(sample, generation.swe.PROTOCOL_SWEBENCH)
    md = generation.apply_template_mapping(md, generation.TEMPLATE_MAPPING)
    md = generation.apply_verifier_spec(md, generation.VERIFIER_SPECS)
    if md['grading']['verifier_spec'].evaluator != 'swegym':
        raise SystemExit(f"Expected SWE-Gym verifier for {md['instance_id']}")
    reason = generation.swe.evaluability_check(md)
    if reason:
        raise SystemExit(f"Invalid prepared task {md['instance_id']}: {reason}")
    count += 1
if count != int(os.environ['SWE_PROMPT_COUNT']):
    raise SystemExit('Prepared task count changed during preflight')
print(f'SWE-Gym preflight passed for {count} tasks', flush=True)
PY_AGENT
    mkdir -p "$(dirname "$OUTPUT_DIR")"
    mkdir "$OUTPUT_DIR"  # Refuse to overwrite an earlier run.
    # The key is inherited by the Ray head, not serialized in runtime-env JSON.
    export E2B_API_KEY="$(<"$VIME_AGENTENV_API_KEY_FILE")"
    [[ -n "$E2B_API_KEY" ]] || fail 'AgentENV API key is empty.'
}

configure_swegym_workers() {
    RT_ENV=()
    for name in PATH LD_LIBRARY_PATH PYTHONPATH VIME_CANN_HOME ASCEND_HOME_PATH ASCEND_OPP_PATH \
        ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME VIME_TRAIN_PYTHON VIME_VLLM_EXECUTABLE VIME_VLLM_PYTHON \
        ROLLOUT_ENV CC TRITON_CACHE_DIR \
        VLLM_ASCEND_ENABLE_NZ VLLM_USE_AOT_COMPILE VLLM_ASCEND_NPUGRAPH_CAPTURE_LIMIT \
        PYTHONUNBUFFERED PYTHONDONTWRITEBYTECODE RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES \
        HCCL_WHITELIST_DISABLE HCCL_CONNECT_TIMEOUT HCCL_DETERMINISTIC HCCL_HOST_SOCKET_PORT_RANGE \
        HCCL_NPU_SOCKET_PORT_RANGE PYTORCH_NPU_ALLOC_CONF CUDA_DEVICE_MAX_CONNECTIONS \
        TASK_QUEUE_ENABLE STREAMS_PER_DEVICE MULTI_STREAM_MEMORY_RESERVE CPU_AFFINITY_CONF \
        OMP_NUM_THREADS TRANSFORMERS_VERBOSITY no_proxy NO_PROXY TMPDIR; do
        [[ "${!name+x}" ]] && RT_ENV+=("${name}=${!name}")
    done

    # Explicit task configuration, including guest Conda values, reaches Ray workers.
    # E2B_API_KEY is inherited from the fresh Ray head, not serialized here.
    for name in SWE_QWEN35_SIZE SWE_TRAIN_MODE SWE_TITAN_CONFIG SWE_TRAIN_TP SWE_TRAIN_DP \
        SWE_ROLLOUT_TP SWE_ROLLOUT_GPUS SWE_TOTAL_NPUS SWE_NUM_EPOCHS SWE_ROLLOUTS_PER_EPOCH \
        SWE_NUM_ROLLOUTS SWE_MAX_TOKENS_PER_GPU SWE_DUMP_MODE SWE_SAVE_INTERVAL RUN_NAME \
        SWE_AGENT SWE_TRAIN_PROTOCOL SWE_PROMPT_DATA SWE_AGENTENV_TEMPLATE_MAP SWE_VERIFIER_SPEC_MAP \
        SWE_PREPARATION_MANIFEST VIME_SWEGYM_PREPARED_BUNDLE SWE_PROMPT_COUNT SWE_ROLLOUT_BATCH_SIZE \
        SWE_N_SAMPLES_PER_PROMPT SWE_TRIAL_CONCURRENCY SWE_BOOT_CONCURRENCY SWE_BOOT_RETRIES \
        SWE_AGENT_TIME_BUDGET_SEC SWE_EVAL_TIMEOUT_SEC SWE_ROLLOUT_GUARD_SEC SWE_VLLM_MAX_NUM_SEQS \
        SWE_ROLLOUT_TEMPERATURE SWE_ROLLOUT_TOP_K SWE_LEARNING_RATE SWE_ENABLE_PREFIX_CACHING \
        SWE_ROLLOUT_SEQ_LEN VIME_AGENT_MAX_CONTEXT_TOKENS \
        VIME_AGENT_MAX_TURNS_PER_SESSION VIME_AGENT_SANDBOX_LIFETIME_SEC VIME_AGENT_SANDBOX_RPC_RETRIES \
        VIME_AGENT_DEBUG_TRACE_DIR VIME_AGENT_TRACE_WRITER VIME_SWEGYM_SOURCE_PATH \
        E2B_API_URL E2B_SANDBOX_URL VIME_AGENT_E2B_SDK_PATH VIME_AGENT_E2B_SELECTOR_MODE \
        VIME_AGENT_TASK_CONDA_SH VIME_AGENT_TASK_CONDA_PREFIX VIME_AGENT_NODE_TARBALL \
        VIME_AGENT_CODEX_TARBALL VIME_AGENT_CODEX_MODEL_PROFILE ADAPTER_BIND_HOST ADAPTER_PORT \
        ADAPTER_PUBLIC_HOST ADAPTER_PUBLIC_PORT; do
        RT_ENV+=("${name}=${!name}")
    done

    rt_env_add "${RT_ENV[@]}"
    rt_env_add "TENSORBOARD_DIR=${ARCHIVE_DIR}" \
        "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}" \
        "VIME_VLLM_EXECUTABLE=${VIME_VLLM_EXECUTABLE}" \
        "VIME_VLLM_PYTHON=${VIME_VLLM_PYTHON}" \
        "VIME_LAUNCHER_RUN_ID=${VIME_LAUNCHER_RUN_ID}" "TMPDIR=${TMPDIR}"
    [[ -z "${LD_PRELOAD:-}" ]] || rt_env_add "LD_PRELOAD=${LD_PRELOAD}"
    rt_env_finalize
    export MASTER_ADDR ASCEND_RT_VISIBLE_DEVICES HCCL_HOST_SOCKET_PORT_RANGE HCCL_NPU_SOCKET_PORT_RANGE
    export TMPDIR TRITON_CACHE_DIR HF_HOME ASCEND_CACHE_PATH
}

print_swegym_plan() {
    "${VIME_JSON_PYTHON:-python3}" - "${TRAIN_CMD[@]}" <<'PY_PLAN'
import json, os, sys
# Deliberate allowlist: never dump credentials or the complete host environment.
keys = (
    'MODEL_PATH', 'TRAIN_ENV', 'VIME_TRAIN_PYTHON', 'ROLLOUT_ENV',
    'VIME_CANN_HOME', 'VIME_VLLM_PYTHON', 'VIME_VLLM_EXECUTABLE',
    'ASCEND_RT_VISIBLE_DEVICES', 'OUTPUT_DIR', 'VIME_SLOT', 'VIME_RAY_PORT',
    'VIME_RAY_DASHBOARD_PORT', 'VIME_RAY_TEMP_DIR', 'VIME_SWEGYM_PREPARED_BUNDLE',
    'SWE_PROMPT_DATA', 'SWE_AGENTENV_TEMPLATE_MAP', 'SWE_VERIFIER_SPEC_MAP',
    'SWE_PREPARATION_MANIFEST', 'SWE_TRAIN_TP', 'SWE_TRAIN_DP', 'SWE_ROLLOUT_GPUS',
    'SWE_ROLLOUT_TP', 'SWE_TRIAL_CONCURRENCY', 'SWE_BOOT_CONCURRENCY',
    'SWE_BOOT_RETRIES', 'SWE_AGENT_TIME_BUDGET_SEC', 'SWE_EVAL_TIMEOUT_SEC',
    'SWE_ROLLOUT_GUARD_SEC', 'VIME_AGENT_SANDBOX_LIFETIME_SEC',
    'VIME_AGENT_SANDBOX_RPC_RETRIES', 'VIME_AGENT_MAX_TURNS_PER_SESSION',
    'VIME_SWEGYM_SOURCE_PATH', 'VIME_AGENT_E2B_SDK_PATH',
    'E2B_API_URL', 'E2B_SANDBOX_URL', 'VIME_AGENT_TASK_CONDA_SH',
    'VIME_AGENT_TASK_CONDA_PREFIX', 'VIME_AGENT_NODE_TARBALL', 'VIME_AGENT_CODEX_TARBALL',
    'VIME_AGENT_CODEX_MODEL_PROFILE', 'ADAPTER_BIND_HOST', 'ADAPTER_PORT',
    'ADAPTER_PUBLIC_HOST', 'ADAPTER_PUBLIC_PORT', 'SWE_DUMP_MODE', 'SWE_SAVE_INTERVAL',
)
print(json.dumps({
    'mode': 'async', 'task': 'swegym',
    'validation': 'Plan only: readable prompt data is checked; template/verifier mappings, runtime dependencies, connectivity and devices are not checked. Null counts mean the prompt data is unavailable locally.',
    'prompt_count': int(os.environ['SWE_PROMPT_COUNT']) if os.environ['SWE_PROMPT_COUNT'] else None,
    'planned_updates': int(os.environ['SWE_NUM_ROLLOUTS']) if os.environ['SWE_NUM_ROLLOUTS'] else None,
    'detailed_tracing': False, 'rewards_normalization': False,
    'environment': {key: os.environ[key] for key in keys},
    'command': sys.argv[1:],
}, indent=2))
PY_PLAN
}
