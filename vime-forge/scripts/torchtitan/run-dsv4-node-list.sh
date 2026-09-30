#!/bin/bash
# Start the DSV4 job from a shared node list.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
die() { echo "ERROR: $*" >&2; exit 2; }

read -r -a NODE_IP_LIST <<< "${NODE_IP_LIST:-}"
(( ${#NODE_IP_LIST[@]} >= 2 )) || die "NODE_IP_LIST must contain at least two IPs"
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
[[ "${NPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "NPUS_PER_NODE must be a positive integer"
NODE_COUNT=${#NODE_IP_LIST[@]}
HEAD_NODE_IP=${NODE_IP_LIST[0]}
ROLLOUT_NODE_NUM=${ROLLOUT_NODE_NUM:-1}
[[ "${ROLLOUT_NODE_NUM}" =~ ^[0-9]+$ ]] || die "ROLLOUT_NODE_NUM must be an integer"
(( ROLLOUT_NODE_NUM >= 1 && ROLLOUT_NODE_NUM < NODE_COUNT )) || \
  die "ROLLOUT_NODE_NUM must be between 1 and node_count-1"

validate_node_list() {
  local node
  declare -A seen=()
  for node in "${NODE_IP_LIST[@]}"; do
    [[ -n "${node}" ]] || die "NODE_IP_LIST contains an empty address"
    [[ -z "${seen[${node}]:-}" ]] || die "NODE_IP_LIST contains duplicate address: ${node}"
    seen["${node}"]=1
  done
}

# Runtime and data paths remain configurable because they vary by deployment.
export DSV4_MODEL=${DSV4_MODEL:-/home/models/DeepSeek-V4-Flash-MXFP8/}
export DATASET=${DATASET:-/tmp/dataset/dapo-math-17k.jsonl}
export VLLM_CONFIG=${VLLM_CONFIG:-}
export VIME_VENV=${VIME_VENV:-/app/uv-train}
export VIME_ROLLOUT_VENV=${VIME_ROLLOUT_VENV:-/app/uv-vllm}
export TRAIN_PYTHON=${TRAIN_PYTHON:-${VIME_VENV}/bin/python}
export RAY_CLI=${RAY_CLI:-${VIME_VENV}/bin/ray}
export VIME_VLLM_EXECUTABLE=${VIME_VLLM_EXECUTABLE:-${VIME_ROLLOUT_VENV}/bin/vllm}
export VIME_CANN_HOME=${VIME_CANN_HOME:-/usr/local/Ascend/cann}
export VIME_NNAL_HOME=${VIME_NNAL_HOME:-/usr/local/Ascend/nnal/atb}

# Output roots; use shared storage for multi-node checkpoints.
export VIME_WORKSPACE=${VIME_WORKSPACE:-/tmp/vime}
export VIME_CKPT_ROOT=${VIME_CKPT_ROOT:-${VIME_WORKSPACE}/ckpt}
export VIME_TB_ROOT=${VIME_TB_ROOT:-${VIME_WORKSPACE}/tb}
export RUN_NAME=${RUN_NAME:-run-dsv4-torchtitan}
# The head derives checkpoint/log directories from these roots and RUN_NAME.

# These defaults differ from the direct launch script.
export ACTOR_GPUS=$(((NODE_COUNT - ROLLOUT_NODE_NUM) * NPUS_PER_NODE))
export ROLLOUT_GPUS=$((ROLLOUT_NODE_NUM * NPUS_PER_NODE))
export DSV4_RECIPE=${DSV4_RECIPE:-vime_dsv4_flash_43layers_256experts}
export EP_DEGREE=${EP_DEGREE:-32}
export ENABLE_MXFP4=${ENABLE_MXFP4:-0}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp34s0f1}
export VIME_TORCHTITAN_WEIGHT_SYNC_LEADER_RELAY=${VIME_TORCHTITAN_WEIGHT_SYNC_LEADER_RELAY:-0}

source "${SCRIPT_DIR}/common/default.sh"
export MASTER_ADDR=${HEAD_NODE_IP}
export VIME_RAY_PORT VIME_RAY_TEMP_DIR RAY_MIN_WORKER_PORT RAY_MAX_WORKER_PORT
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
NODE_IP_CSV=$(IFS=,; printf '%s' "${NODE_IP_LIST[*]}")
export no_proxy="${no_proxy:-},localhost,127.0.0.1,${NODE_IP_CSV}"
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,${NODE_IP_CSV}"

activate_runtime() {
  set +u
  [[ -z "${VIME_VENV}" ]] || source "${VIME_VENV}/bin/activate"
  source "${VIME_CANN_HOME}/set_env.sh"
  source "${VIME_NNAL_HOME}/set_env.sh"
  set -u
  if [[ ! -f "${RAY_CLI}" ]]; then RAY_CLI=$(command -v ray || true); fi
  [[ -x "${TRAIN_PYTHON}" ]] || die "Training Python not found: ${TRAIN_PYTHON}"
  [[ -f "${RAY_CLI}" ]] || die "Ray CLI script not found: ${RAY_CLI:-<empty>}"
  export RAY_CLI
}

ray_cli() { "${TRAIN_PYTHON}" "${RAY_CLI}" "$@"; }

detect_local_node() {
  local addresses=" ${LOCAL_NODE_IP:-} $(hostname -I 2>/dev/null || true) "
  local index candidate
  for index in "${!NODE_IP_LIST[@]}"; do
    candidate=${NODE_IP_LIST[${index}]}
    if [[ "${addresses}" == *" ${candidate} "* ]]; then
      LOCAL_NODE_IP=${candidate}; LOCAL_NODE_INDEX=${index}; return
    fi
  done
  die "Local addresses do not match NODE_IP_LIST: ${addresses}"
}

start_worker() {
  local worker_temp_dir="${VIME_RAY_TEMP_DIR}/worker-${LOCAL_NODE_INDEX}"
  mkdir -p "${worker_temp_dir}"
  local deadline=$((SECONDS + ${WAIT_TIMEOUT:-180}))
  until ray_cli start \
      --address="${HEAD_NODE_IP}:${VIME_RAY_PORT}" \
      --node-ip-address="${LOCAL_NODE_IP}" \
      --temp-dir="${worker_temp_dir}" \
      --min-worker-port="${RAY_MIN_WORKER_PORT}" \
      --max-worker-port="${RAY_MAX_WORKER_PORT}" \
      --disable-usage-stats; do
    (( SECONDS < deadline )) || die "Unable to join Ray head ${HEAD_NODE_IP}:${VIME_RAY_PORT}"
    sleep 2
  done
  echo "Worker ${LOCAL_NODE_IP} joined Ray; keeping the container process alive."
  while ray_cli status --address="${HEAD_NODE_IP}:${VIME_RAY_PORT}" >/dev/null 2>&1; do sleep 10; done
  die "Lost connection to Ray head"
}

activate_runtime
validate_node_list
detect_local_node
echo "local_ip=${LOCAL_NODE_IP}, index=${LOCAL_NODE_INDEX}, head=${HEAD_NODE_IP}"
if [[ "${LOCAL_NODE_INDEX}" == 0 ]]; then
  exec bash "${SCRIPT_DIR}/run-dsv4.sh" "$@"
fi
start_worker
