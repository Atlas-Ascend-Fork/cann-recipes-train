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

# Determinism at all three sites at once: the training seed, vLLM sampling and
# HCCL. Training and rollout must agree on HCCL or weight transfer deadlocks.
DET="${DET:-on}"

# Checkpoint saving: off by default for local/smoke runs. Set SAVE_ON=on in
# common/local.sh to enable periodic checkpoint saves.
SAVE_ON="${SAVE_ON:-on}"

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
VIME_VLLM_ENV="${VIME_VLLM_ENV:-}"                    # vllm python env; empty => same as train
VIME_VLLM_PYTHON="${VIME_VLLM_PYTHON:-}"             # rollout interpreter; derived by launcher
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
VIME_NOFILE_LIMIT="${VIME_NOFILE_LIMIT:-65535}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# ── Environment: derived from the roots ───────────────────────────────

VIME_DATA_ROOT="${VIME_DATA_ROOT:-${VIME_WORKSPACE}/data}"
VIME_CKPT_ROOT="${VIME_CKPT_ROOT:-${VIME_WORKSPACE}/ckpt}"
VIME_TB_ROOT="${VIME_TB_ROOT:-${VIME_WORKSPACE}/tb}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${VIME_WORKSPACE}/.triton}"
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

# ── Environment handed to ray workers ─────────────────────────────────
# The launcher appends the computed ones (TENSORBOARD_DIR, PYTHONPATH,
# VIME_VLLM_EXECUTABLE, VIME_VLLM_PYTHON) plus the DET-dependent ones.

RT_ENV=(
   RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1  # ray assigns NPUs per worker itself
   PYTORCH_NPU_ALLOC_CONF=expandable_segments:True     # ⚠ torchtitan needs it; incompatible
                                                       #   with vLLM sleep-mode, so that stays off
   TORCH_NPU_DEVICE_CAPABILITY=0.0                     # torch_npu returns None; <9 avoids
                                                       #   CUDA-only fast paths
   VLLM_VERSION="${VLLM_VERSION:-0.23.0}"              # vllm-ascend branches on vllm_version_is()
   VLLM_ASCEND_ENABLE_NZ=0                             # ⚠ RL weight hot-update needs FRACTAL_NZ off
   VLLM_USE_AOT_COMPILE=0                              # ⚠ torch2.12 <-> torch_npu aot path crashes
   VLLM_ASCEND_NPUGRAPH_CAPTURE_LIMIT=512              # keeps rollout off the eager fallback
   ATB_MATMUL_SHUFFLE_K_ENABLE=0                       # vllm-ascend's official determinism recipe;
   ATB_LLM_LCOC_ENABLE=0                               #   always on, not gated by DET
   PYTHONUNBUFFERED=1
   HYDRA_FULL_ERROR=1
   DISABLE_L2_CACHE=1
   CUDA_DEVICE_MAX_CONNECTIONS=1                       # carried over from the original script;
                                                       #   no evidence torchtitan depends on it
   RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.95}"
)
