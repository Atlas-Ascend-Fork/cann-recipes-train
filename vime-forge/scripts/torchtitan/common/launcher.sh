#!/bin/bash
# torchtitan launcher. Sourced by a model script, which declares its parameters
# between configure_run and start_run. Machine config lives in local.sh, ray worker
# environment is assembled here by rt_env_add. Only mechanism here, no parameters.

LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${LAUNCHER_DIR}/../../.." &>/dev/null && pwd)"

source "${LAUNCHER_DIR}/reclaim.sh"

die() { echo "✗ $*" >&2; exit 2; }

# ── Defaults ──────────────────────────────────────────────────────────

cli_defaults() {
    DUMP_DIR=''
    REPLAY_DIR=''
    TRAIN_ENTRY=''; TIS_ON=0
    ROLLOUT_FN=''
    IS_ASYNC=0

    EXTRA_ARGS=()
}

# ── Command line ──────────────────────────────────────────────────────

need_val() { [[ $# -ge 2 ]] || die "$1 needs a value"; }
onoff() { [[ "$2" == on || "$2" == off ]] || die "$1 takes on|off, got: $2"; }

cli_parse() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --local)        need_val "$@"; shift 2 ;;   # already read by cli_peek_local
            --dump-dir)     need_val "$@"; DUMP_DIR="$2"; shift 2 ;;
            --replay-dir)   need_val "$@"; REPLAY_DIR="$2"; shift 2 ;;
            --)             shift; EXTRA_ARGS=("$@"); break ;;
            -h|--help)      cli_help; exit 0 ;;
            *)              die "unknown option: $1 (see --help)" ;;
        esac
    done
}

# --local names the machine config, which has to be read before local.sh is
# sourced -- earlier than the main parse. So it is scanned out separately.
cli_peek_local() {
    LOCAL_NAME=''
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --)      return ;;
            --local) need_val "$@"; LOCAL_NAME="$2"; return ;;
            *)       shift ;;
        esac
    done
}

cli_derive_mode() {
    local eval_default
    case "$MODE" in
        sync)        TRAIN_ENTRY=train.py; IS_ASYNC=0
                     eval_default=on ;;
        async)       TRAIN_ENTRY=train_async.py; IS_ASYNC=1
                     TIS_ON=1
                     eval_default=on ;;
        fully-async) TRAIN_ENTRY=train_async.py; IS_ASYNC=1
                     TIS_ON=1
                     ROLLOUT_FN=vime.rollout.fully_async_rollout.generate_rollout_fully_async
                     eval_default=off ;;   # that rollout raises in eval mode
        *)           die "unknown --mode: ${MODE} (sync | async | fully-async)" ;;
    esac
    # Follows MODE unless something already set it (local.sh, env).
    EVAL_ON="${EVAL_ON:-$eval_default}"
}

cli_help() {
    cat <<'EOF'
usage: bash scripts/torchtitan/<model>.sh [options] [-- <extra training args>]

  --local <name>       machine config under common/ [local.sh]
  --dump-dir <path>    record rollout data
  --replay-dir <path>  replay recorded data (mutually exclusive with --dump-dir)
  -h, --help

Everything after -- is appended verbatim to the training command line:
  bash scripts/torchtitan/run-qwen3-4B.sh -- --num-rollout 5 --training.max_norm 0.5

Finer adjustments go straight into the files, each a single-point edit:
  run name / mode / determinism / params  <model>.sh (copy it for a new experiment)
  ray worker environment                  common/default.sh
  paths / visible cards / ports / venv    common/local.sh (template alongside it)

What local.sh can set: every line in common/default.sh is ${VAR:-default}.
EOF
}

# ── Validation ────────────────────────────────────────────────────────

preflight_cli() {
    [[ -z "$DUMP_DIR" || -z "$REPLAY_DIR" ]] || \
        die "--dump-dir and --replay-dir are mutually exclusive (one writes, one reads)."

    [[ "$MODE" != fully-async || "$EVAL_ON" != on ]] || \
        die "MODE=fully-async cannot run eval: generate_rollout_fully_async raises in eval mode. Drop the EVAL_ON=on override."

    [[ "$VIME_NOFILE_LIMIT" =~ ^[0-9]+$ ]] || die "VIME_NOFILE_LIMIT must be an integer"

    # ray's AF_UNIX socket paths cap at 107 bytes; over that the dashboard agent
    # dies with an error unrelated to the real cause.
    (( ${#VIME_RAY_TEMP_DIR} <= 90 )) || \
        die "VIME_RAY_TEMP_DIR is too long (${#VIME_RAY_TEMP_DIR} chars): ray's AF_UNIX socket path caps at 107 bytes. Shorten it, or use a shallower VIME_WORKSPACE."

    [[ -z "$REPLAY_DIR" || -d "$REPLAY_DIR" ]] || die "--replay-dir does not exist: ${REPLAY_DIR}"
}

# Relations that can only be checked once the model script has run.
preflight_params() {
    [[ "${DET:-}" == on || "${DET:-}" == off ]] || die "DET must be on or off, got: ${DET:-<unset>}"
    (( MAX_RESPONSE_LEN < SEQ_LEN )) || \
        die "MAX_RESPONSE_LEN(${MAX_RESPONSE_LEN}) must be below SEQ_LEN(${SEQ_LEN}), which bounds prompt+response together."
    (( ROLLOUT_GPUS % VLLM_RANKS_PER_ENGINE == 0 )) || \
        die "VLLM_RANKS_PER_ENGINE(${VLLM_RANKS_PER_ENGINE}) must divide ROLLOUT_GPUS(${ROLLOUT_GPUS}): vime floor-divides to get the engine count, so the remainder sits idle."
}

preflight_env() {
    [[ -f "${VIME_CANN_HOME}/set_env.sh" ]] || \
        die "no ${VIME_CANN_HOME}/set_env.sh. Point VIME_CANN_HOME at the CANN install."
    [[ -f "${VIME_NNAL_HOME}/set_env.sh" ]] || \
        die "no ${VIME_NNAL_HOME}/set_env.sh. Point VIME_NNAL_HOME at the nnal/atb directory."
    [[ -z "$VIME_TRAIN_ENV" || -f "${VIME_TRAIN_ENV}/bin/activate" ]] || \
        die "VIME_TRAIN_ENV=${VIME_TRAIN_ENV} has no bin/activate, so it is not a venv."
    [[ -f "${RAY_CLI}" ]] || \
        die "RAY_CLI=${RAY_CLI:-<empty>} is not a Ray CLI script."
    [[ -z "$VIME_VLLM_ENV" || -x "${VIME_VLLM_ENV}/bin/vllm" ]] || \
        die "VIME_VLLM_ENV=${VIME_VLLM_ENV} has no executable bin/vllm."
}

# ── Declarative helpers for the model script ──────────────────────────

# add_when <var> <value|nonempty> <array name> -- <args...>
add_when() {
    local var="$1" want="$2" arr="$3"; shift 4   # the 4th is the literal --
    local cur="${!var:-}"
    if [[ "$want" == nonempty ]]; then [[ -n "$cur" ]] || return 0
    else [[ "$cur" == "$want" ]] || return 0; fi
    eval "${arr}+=(\"\$@\")"
}

# gate_when <var> <value> <array name>: clear the whole array unless it matches
gate_when() {
    local var="$1" want="$2" arr="$3"
    [[ "${!var:-}" == "$want" ]] || eval "${arr}=()"
}

# ── Environment: declared once, both exported and sent to ray workers ──

RT_ENV_KEYS=(); RT_ENV_VALS=()

rt_env_add() {
    local kv
    for kv in "$@"; do
        [[ "$kv" == *=* ]] || die "rt_env_add takes K=V, got: $kv"
        RT_ENV_KEYS+=("${kv%%=*}"); RT_ENV_VALS+=("${kv#*=}")
    done
}

rt_env_init() { RT_ENV_KEYS=(); RT_ENV_VALS=(); }

configure_colocation() {
    COLOCATE_ARGS=()
    if [[ "$COLOCATE" == on ]]; then
        COLOCATE_ARGS=(--colocate --vllm-enable-sleep-mode)
        # vLLM sleep mode uses its memory pool and rejects expandable segments.
        # Added after RT_ENV so this value overrides the default for both the
        # launcher process and Ray workers. max_split_size_mb keeps large
        # blocks whole so the backward transient stays reusable for big
        # requests; garbage_collection_threshold proactively returns cached
        # blocks once the pool passes 80% of device capacity. Without these,
        # expandable_segments:False fragments the pool (29G reserved-but-
        # unallocated, 594M requests failing).
        rt_env_add PYTORCH_NPU_ALLOC_CONF=expandable_segments:False,max_split_size_mb:256,garbage_collection_threshold:0.8
    fi
}

rt_env_finalize() {
    local i
    for i in "${!RT_ENV_KEYS[@]}"; do export "${RT_ENV_KEYS[i]}=${RT_ENV_VALS[i]}"; done
    RUNTIME_ENV_JSON=$(rt_env_json)
}

# Built by python, never string-concatenated: LD_PRELOAD / PYTHONPATH can carry
# characters that need escaping.
rt_env_json() {
    local i
    {
        for i in "${!RT_ENV_KEYS[@]}"; do
            printf '%s\0%s\0' "${RT_ENV_KEYS[i]}" "${RT_ENV_VALS[i]}"
        done
    } | "${VIME_JSON_PYTHON:-python3}" -c '
import json, sys
parts = sys.stdin.buffer.read().split(b"\0")[:-1]
env = {parts[i].decode(): parts[i + 1].decode() for i in range(0, len(parts), 2)}
sys.stdout.write(json.dumps({"env_vars": env}))
'
}

# The three sourced files are vendor scripts that routinely reference undefined
# variables; under set -u that aborts with an "unbound variable" message
# unrelated to the real cause, so relax it here.
platform_activate() {
    set +u
    [[ -z "${VIME_TRAIN_ENV:-}" ]] || source "${VIME_TRAIN_ENV}/bin/activate"
    source "${VIME_CANN_HOME}/set_env.sh"
    source "${VIME_NNAL_HOME}/set_env.sh"
    set -u
    ulimit -n "${VIME_NOFILE_LIMIT}" || true
}

# TRAIN_PYTHON is only spliced into the command line as a string; it is never
# executed by this process.
# Either binary can also be named outright in local.sh, bypassing the venv
# derivation; that is how the dual-venv setups pin a specific vllm.
platform_resolve_interpreters() {
    # VIME_TRAIN_ENV is the trainer/vime python env (activated before ray
    # start so raylet workers run under it); VIME_VLLM_ENV is the vllm python
    # env, defaulting to the train env. Both are set in local.sh.
    local train_env="${VIME_TRAIN_ENV:-}" rollout_env="${VIME_VLLM_ENV:-${VIME_TRAIN_ENV:-}}"
    if [[ -z "${TRAIN_PYTHON:-}" ]]; then
        if [[ -n "$train_env" ]]; then TRAIN_PYTHON="${train_env}/bin/python"
        else TRAIN_PYTHON="$(command -v python3 || echo python3)"; fi
    fi
    if [[ -z "${RAY_CLI:-}" || ! -f "${RAY_CLI}" ]]; then
        if [[ -n "$train_env" && -f "${train_env}/bin/ray" ]]; then RAY_CLI="${train_env}/bin/ray"
        else RAY_CLI="$(command -v ray || true)"; fi
    fi
    if [[ -z "${VIME_VLLM_EXECUTABLE:-}" ]]; then
        if [[ -n "$rollout_env" ]]; then VIME_VLLM_EXECUTABLE="${rollout_env}/bin/vllm"
        else VIME_VLLM_EXECUTABLE="$(command -v vllm || echo vllm)"; fi
    fi
    if [[ -z "${VIME_VLLM_PYTHON:-}" && -n "$rollout_env" ]]; then
        VIME_VLLM_PYTHON="${rollout_env}/bin/python"
    fi
}

# ── Launch and archive ────────────────────────────────────────────────

# The words of TRAIN_CMD, escaped for the shell ray runs the entrypoint through. ray joins the
# words with subprocess.list2cmdline, which escapes the double quotes itself, so everything else
# is escaped here and the quotes are left alone:
#   {"a":1,"b":2}  ->  \{"a":1\,"b":2\}  ->  ray  ->  \{\"a\":1\,\"b\":2\}  ->  shell  ->  {"a":1,"b":2}
quoted_train_cmd() {
    local w q
    QUOTED_TRAIN_CMD=()
    for w in "${TRAIN_CMD[@]}"; do
        q=$(printf '%q' "$w")
        QUOTED_TRAIN_CMD+=("${q//\\\"/\"}")
    done
}

# The archive stores something you can re-run, not just something you can read.
launch_archive() {
    local dir="$1"
    mkdir -p "$dir"
    {
        echo '#!/bin/bash'
        echo "# generated $(date '+%F %T'); run it to reproduce this launch."
        echo '# It replays exactly what ran, so later edits to the model script'
        echo '# or local.sh do not change it.'
        echo 'set -euo pipefail'
        echo
        echo '# Drop the ray start line if a cluster is already up on these ports.'
        printf '%q %q start' "${TRAIN_PYTHON}" "${RAY_CLI}"
        printf ' %q' "${RAY_HEAD_ARGS[@]}"
        printf '

'
        printf '%q job submit --address=%q --runtime-env-json=%q --' \
            "${RAY_CLI}" "http://127.0.0.1:${VIME_RAY_DASHBOARD_PORT}" "$RUNTIME_ENV_JSON"
        # One more %q layer: the shell running this line consumes it and hands ray the same words.
        printf ' %q' "${QUOTED_TRAIN_CMD[@]}"
        printf '
'
    } > "${dir}/rerun.sh"
    chmod +x "${dir}/rerun.sh"
}

ray_cli() { "${TRAIN_PYTHON}" "${RAY_CLI}" "$@"; }

launch_ray_head() { ray_cli start "${RAY_HEAD_ARGS[@]}"; }

# ── The two halves the model script sits between ──────────────────────

# Before the parameters: read local.sh, parse the command line, and compute the
# CKPT_DIR / ARCHIVE_DIR that the parameters refer to.
configure_run() {
    OVERRIDE_ARGS=()   # local.sh may append to this; declared before it is sourced
    # BASH_SOURCE[1] is the model script that called this shared launcher. It is
    # combined with the slot-specific Ray temp dir as a cleanup ownership marker.
    LAUNCHER_SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")" &>/dev/null && pwd)/$(basename -- "${BASH_SOURCE[1]}")"

    cli_peek_local "$@"
    if [[ "${VIME_PREPARED_ENV:-0}" == 1 ]]; then
        [[ -z "$LOCAL_NAME" ]] || die "--local cannot override a prepared runtime"
        LOCAL_SH=''
    elif [[ -n "${LOCAL_NAME:-}" ]]; then
        [[ "$LOCAL_NAME" == /* ]] && LOCAL_SH="$LOCAL_NAME" \
                                  || LOCAL_SH="${LAUNCHER_DIR}/${LOCAL_NAME}"
        [[ -f "$LOCAL_SH" ]] || die "--local names a file that does not exist: ${LOCAL_SH}"
    else
        LOCAL_SH="${LAUNCHER_DIR}/local.sh"
    fi

    if [[ "${VIME_PREPARED_ENV:-0}" == 1 ]]; then
        LOCAL_NOTE="prepared environment from outer wrapper"
    elif [[ -f "$LOCAL_SH" ]]; then
        source "$LOCAL_SH"
        LOCAL_NOTE="${LOCAL_SH##*/} ($(date -r "$LOCAL_SH" '+%F %T'))"
    else
        LOCAL_NOTE="no local.sh; built-in defaults throughout"
    fi

    cli_defaults
    source "${LAUNCHER_DIR}/default.sh"
    VIME_LAUNCHER_RUN_ID="${LAUNCHER_SCRIPT}:${VIME_RAY_TEMP_DIR}"
    RUN_NAME="${RUN_NAME:-$(basename "$0" .sh)}"
    [[ "$VIME_SLOT" =~ ^[0-9]+$ ]] || die "VIME_SLOT must be a non-negative integer, got: ${VIME_SLOT}"
    cli_parse "$@"
    cli_derive_mode
    preflight_cli

    platform_resolve_interpreters
    # The outer wrapper owns validation and activation in prepared mode. Re-sourcing
    # generic CANN/ATB or a venv here can replace its selected runtime libraries.
    if [[ "${VIME_PREPARED_ENV:-0}" != 1 ]]; then
        preflight_env
        platform_activate
    fi

    CKPT_DIR="${VIME_CKPT_ROOT}/${RUN_NAME}-${MODE}"
    RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
    ARCHIVE_DIR="${VIME_TB_ROOT}/${RUN_NAME}/${RUN_STAMP}"
    # Declared once; launch_ray_head runs it and launch_archive replays it.
    # No --num-gpus: passing it suppresses Ascend NPU auto-detection.
    RAY_HEAD_ARGS=(--head --node-ip-address "${MASTER_ADDR}"
                   --port="${VIME_RAY_PORT}"
                   --temp-dir="${VIME_RAY_TEMP_DIR}"
                   --dashboard-host=0.0.0.0
                   --dashboard-port="${VIME_RAY_DASHBOARD_PORT}"
                   --dashboard-agent-listen-port="${VIME_RAY_DASHBOARD_AGENT_PORT}"
                   --min-worker-port="${RAY_MIN_WORKER_PORT}"
                   --max-worker-port="${RAY_MAX_WORKER_PORT}"
                   --disable-usage-stats)

    rt_env_init
    # Armed as soon as the marker exists, so a failure anywhere below still clears the slot.
    trap reclaim_slot EXIT
    RUN_CONFIGURED=1
}

# After the parameters: check the relations between them, assemble, submit.
start_run() {
    [[ "${RUN_CONFIGURED:-0}" == 1 ]] || die "call configure_run before start_run"
    preflight_params
    [[ -z "$DUMP_DIR" ]] || mkdir -p "$DUMP_DIR"

    rt_env_add "${RT_ENV[@]}"
    configure_colocation
    rt_env_add "TENSORBOARD_DIR=${ARCHIVE_DIR}" \
               "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}" \
               "VIME_VLLM_EXECUTABLE=${VIME_VLLM_EXECUTABLE}" \
               "VIME_VLLM_PYTHON=${VIME_VLLM_PYTHON}" \
               "VIME_LAUNCHER_RUN_ID=${VIME_LAUNCHER_RUN_ID}" \
    # ⚠ Must match the rollout side: vllm_ascend's batch-invariant path forces the
    # rollout worker to strict, and a training side left at "true" deadlocks the
    # HCCL collective used for cross-process weight transfer.
    [[ "$DET" != on ]] || rt_env_add HCCL_DETERMINISTIC=strict LCCL_DETERMINISTIC=1
    [[ -z "${LD_PRELOAD:-}" ]] || rt_env_add "LD_PRELOAD=${LD_PRELOAD}"
    rt_env_finalize

    export MASTER_ADDR ASCEND_RT_VISIBLE_DEVICES
    export HCCL_HOST_SOCKET_PORT_RANGE HCCL_NPU_SOCKET_PORT_RANGE
    export TRITON_CACHE_DIR HF_HOME ASCEND_CACHE_PATH

    # TRAIN_ARGS comes from the model script. The two overrides land after it, so
    # they win: argparse keeps the last value.
    TRAIN_CMD=("${TRAIN_PYTHON}" "${REPO_ROOT}/${TRAIN_ENTRY}" --train-backend torchtitan "${COLOCATE_ARGS[@]}"
               "${TRAIN_ARGS[@]}" "${OVERRIDE_ARGS[@]}" "${EXTRA_ARGS[@]}")

    # An earlier run of this slot may have left ray or vLLM behind; a busy ray port stays fatal.
    reclaim_slot || die "could not reclaim slot ${VIME_SLOT}"

    echo "config: ${LOCAL_NOTE}"
    echo "mode: ${MODE} (${TRAIN_ENTRY})  ckpt: ${CKPT_DIR}  archive: ${ARCHIVE_DIR}"
    quoted_train_cmd
    launch_archive "${ARCHIVE_DIR}"
    launch_ray_head
    ray_cli job submit \
        --address="http://127.0.0.1:${VIME_RAY_DASHBOARD_PORT}" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- "${QUOTED_TRAIN_CMD[@]}" 2>&1 | tee "${ARCHIVE_DIR}/train.log"
}
