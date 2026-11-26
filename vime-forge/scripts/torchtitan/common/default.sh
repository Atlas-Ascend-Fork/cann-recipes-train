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

# Defaults for everything that is not a training parameter:
#   - switches that control which training parameters get added
#   - environment: paths, cards, ports
#
# Sourced after local.sh, so anything local.sh already set is kept; this only
# fills the gaps and derives the rest. Override any of it in local.sh.

# ── Switches ──────────────────────────────────────────────────────────

COLOCATE="${COLOCATE:-off}"

# sync | async | fully-async. async switches the entry point to train_async.py
# and turns TIS on; fully-async additionally routes rollout through the
# in-flight pool and drops eval (that rollout raises in eval mode).
MODE="${MODE:-sync}"

# Checkpoint saving: off by default for local/smoke runs. Set SAVE_ON=on in
# common/local.sh to enable periodic checkpoint saves.
SAVE_ON="${SAVE_ON:-off}"

# EVAL_ON is not set here: it follows MODE (on for sync/async, off for
# fully-async, whose rollout raises in eval mode). Set EVAL_ON=off in local.sh
# to skip eval on a throughput or smoke run.

# ── Environment: roots ────────────────────────────────────────────────

VIME_WORKSPACE="${VIME_WORKSPACE:-/tmp/vime}"        # shared root for data/ckpt/tb/tmp/caches
VIME_MODEL_ROOT="${VIME_MODEL_ROOT:-/tmp/models}"    # model store, usually a different disk
VIME_SLOT="${VIME_SLOT:-0}"                          # which concurrent instance on this host

VIME_CANN_HOME="${VIME_CANN_HOME:-/usr/local/Ascend/ascend-toolkit}"
VIME_NNAL_HOME="${VIME_NNAL_HOME:-/usr/local/Ascend/nnal/atb}"
VIME_TRAIN_ENV="${VIME_TRAIN_ENV:-}"                 # trainer + vime python env
VIME_VLLM_ENV="${VIME_VLLM_ENV:-}"                   # vllm python env; empty => same as train
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
VIME_NOFILE_LIMIT="${VIME_NOFILE_LIMIT:-65535}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# ── Environment: derived from the roots ───────────────────────────────

VIME_DATA_ROOT="${VIME_DATA_ROOT:-${VIME_WORKSPACE}/data}"
VIME_CKPT_ROOT="${VIME_CKPT_ROOT:-${VIME_WORKSPACE}/ckpt}"
VIME_TB_ROOT="${VIME_TB_ROOT:-${VIME_WORKSPACE}/tb}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${VIME_WORKSPACE}/.cache/triton}"
VLLM_TORCH_EXTENSIONS_DIR="${VLLM_TORCH_EXTENSIONS_DIR:-${VIME_WORKSPACE}/.cache/torch_extensions/vllm}"
TORCHTITAN_TORCH_EXTENSIONS_DIR="${VIME_WORKSPACE}/.cache/torch_extensions/torchtitan"
HF_HOME="${HF_HOME:-${VIME_WORKSPACE}/.cache/huggingface}"
ASCEND_CACHE_PATH="${ASCEND_CACHE_PATH:-${VIME_WORKSPACE}/.ascend_cache}"

# One non-colliding port set per slot. Ray's default dashboard-agent port 52365
# collides whenever two clusters share a host; the symptom is RuntimeEnvAgent
# failing to start and job submission failing, neither pointing at the cause.
VIME_RAY_PORT="${VIME_RAY_PORT:-$(( 6379 + VIME_SLOT ))}"
VIME_RAY_DASHBOARD_PORT="${VIME_RAY_DASHBOARD_PORT:-$(( 8265 + VIME_SLOT ))}"
VIME_RAY_DASHBOARD_AGENT_PORT="${VIME_RAY_DASHBOARD_AGENT_PORT:-$(( 52365 + VIME_SLOT ))}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-30000}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-30999}"
HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-$(( 60000 + VIME_SLOT * 100 ))-$(( 60050 + VIME_SLOT * 100 ))}"
HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-$(( 61000 + VIME_SLOT * 100 ))-$(( 61050 + VIME_SLOT * 100 ))}"
# Kept short on purpose: ray's AF_UNIX socket paths cap at 107 bytes.
VIME_RAY_TEMP_DIR="${VIME_RAY_TEMP_DIR:-${VIME_WORKSPACE}/ray-s${VIME_SLOT}}"

# set CANN env
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
# Disable the HCCL watchdog for long checkpoint loading on the A5 setup.
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"

# ── Environment handed to ray workers ─────────────────────────────────
# The launcher appends the computed ones (TENSORBOARD_DIR, PYTHONPATH,
RT_ENV=(
   RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1  # ray assigns NPUs per worker itself
   RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.95}"
   PYTORCH_NPU_ALLOC_CONF=expandable_segments:True     # ⚠ torchtitan needs it; incompatible
                                                       #   with vLLM sleep-mode, so that stays off
   TORCH_NPU_DEVICE_CAPABILITY=0.0                     # torch_npu returns None; <9 avoids CUDA-only paths
   VLLM_VERSION="${VLLM_VERSION:-0.30.0}"              # vllm-ascend branches on vllm_version_is()
   VLLM_ASCEND_ENABLE_NZ=0                             # ⚠ RL weight hot-update needs FRACTAL_NZ off
   VLLM_USE_AOT_COMPILE=0                              # ⚠ torch2.12 <-> torch_npu aot path crashes
   VLLM_ASCEND_NPUGRAPH_CAPTURE_LIMIT=512              # keeps rollout off the eager fallback
   ATB_MATMUL_SHUFFLE_K_ENABLE=0                       # vllm-ascend's official determinism recipe;
   ATB_LLM_LCOC_ENABLE=0                               # always on
   VIME_DATA_ROOT="${VIME_DATA_ROOT}"                  # --eval-config resolves it via ${oc.env:...}
   VIME_TORCHTITAN_WEIGHT_SYNC_LEADER_RELAY="${VIME_TORCHTITAN_WEIGHT_SYNC_LEADER_RELAY:-0}"  # 0 = no relay
   PYTHONUNBUFFERED=1
   HYDRA_FULL_ERROR=1
   DISABLE_L2_CACHE=1
   CUDA_DEVICE_MAX_CONNECTIONS=1                       # carried over from the original script;
                                                       #   no evidence torchtitan depends on it
   FLA_NPU_DISABLE_PTH="${FLA_NPU_DISABLE_PTH:-1}"     # disable fla_npu pth
   HCCL_ASYNC_ERROR_HANDLING=0                         # disable HCCL watchdog 
   HCCL_DETERMINISTIC="strict"
   LCCL_DETERMINISTIC=1
)

# ── Launcher/process environment ─────────────────────────────────────
# RLIMIT_MEMLOCK: the image default (~94 GiB) is below what torchtitan's
# pinned-host novaswap pool needs, and the failure surfaces as
# "rtsMallocHost ... driver error:out of memory" / "Pinned pool failed to
# allocate slab". Raise it here — this file is sourced by the launcher before
# it starts Ray, so the raylet and every Ray worker inherit the limit.
ulimit -l unlimited 2>/dev/null || true

# TMPDIR is cleared on reboot (and the Ascend bishengir/hivmc compiler plus
# triton need it to exist), so recreate it every launch.
[[ -z "${TMPDIR:-}" ]] || mkdir -p "${TMPDIR}"

# ── Multi-node ────────────────────────────────────────────────────────
# NODE_IP_LIST: the nodes of the run, head first, space-separated. Empty keeps
# the single-node path. When set, launcher.sh validates the list, works out
# which node this process runs on, derives MASTER_ADDR / the placement counts /
# the no_proxy exclusions, and -- on a non-head node -- joins this node to the
# head's ray cluster instead of launching the job. Run the SAME command on every
# node; the placement numbers below are then inputs, not derived.
NODE_IP_LIST="${NODE_IP_LIST:-}"                  # e.g. "10.0.0.1 10.0.0.2 10.0.0.3"
NPUS_PER_NODE="${NPUS_PER_NODE:-8}"               # training cards per node
ROLLOUT_NODE_NUM="${ROLLOUT_NODE_NUM:-1}"         # how many nodes host the rollout
# GLOO_SOCKET_IFNAME pins the host NIC the CPU collectives use. Only defaulted
# for a multi-node run: naming an interface that does not exist makes gloo fail
# to initialise even on a single node, where the loopback is fine. Override it in
# local.sh when the cluster NIC is named differently.
if [[ -n "${NODE_IP_LIST:-}" ]]; then
    GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp34s0f1}"
else
    GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"
fi

# TorchTitan's cross-process weight-sync relay. 0 is the multi-node-safe setting
# the old node-list launcher forced; set 1 in local.sh to route the sync through
# a leader relay.
VIME_TORCHTITAN_WEIGHT_SYNC_LEADER_RELAY="${VIME_TORCHTITAN_WEIGHT_SYNC_LEADER_RELAY:-0}"
# The head waits this long for every node in NODE_IP_LIST to join ray before
# submitting, then submits with a warning instead of hanging silently.
VIME_MULTINODE_WAIT_TIMEOUT="${VIME_MULTINODE_WAIT_TIMEOUT:-600}"
# How long a worker node keeps retrying to join the head's ray cluster.
VIME_MULTINODE_JOIN_TIMEOUT="${VIME_MULTINODE_JOIN_TIMEOUT:-180}"

# Set by launcher.sh for a multi-node run: VIME_MULTINODE=1, NODE_COUNT,
# HEAD_NODE_IP, LOCAL_NODE_IP, LOCAL_NODE_INDEX, ACTOR_NODE_NUM, ACTOR_GPUS,
# ROLLOUT_GPUS. Single-node leaves VIME_MULTINODE=0 and the model script's own
# ACTOR_GPUS / ROLLOUT_GPUS / ACTOR_NODE_NUM defaults in place.
