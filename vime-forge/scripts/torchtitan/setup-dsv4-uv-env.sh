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

# Build two isolated uv environments for dsv4 RL.  CANN and torch-npu wheels
# are intentionally left to the caller because they are host/image-specific.

source /usr/local/Ascend/cann/set_env.sh

# SOURCE_ROOT is the original source directory (usually a host path mapped
# into the container). CONTAINER_ROOT is the only installation target.
SOURCE_ROOT=${SOURCE_ROOT:?"Set SOURCE_ROOT to the original source directory"}
VIME_PATH=${VIME_PATH:-"$SOURCE_ROOT/vime-forge"}
CONTAINER_ROOT="/app"
WORKSPACE=${WORKSPACE:-${CONTAINER_ROOT}/third_party}
VIME_VENV=${VIME_VENV:-/app/uv-train}
VIME_ROLLOUT_VENV=${VIME_ROLLOUT_VENV:-/app/uv-vllm}
PYTHON=${PYTHON:-python3}
UV=${UV:-uv}
VLLM_TORCH_WHEEL=${VLLM_TORCH_WHEEL:-${TORCH_WHEEL:-}}
VLLM_TORCH_NPU_WHEEL=${VLLM_TORCH_NPU_WHEEL:-${TORCH_NPU_WHEEL:-}}
TRAIN_TORCH_WHEEL=${TRAIN_TORCH_WHEEL:-}
TRAIN_TORCH_NPU_WHEEL=${TRAIN_TORCH_NPU_WHEEL:-}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.27.0+cpu}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.11.0+cpu}

die() { echo "ERROR: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null || die "command not found: $1"; }
need_dir() { [[ -d "$1" ]] || die "directory not found: $1"; }

need_cmd "$UV"
need_cmd "$PYTHON"
need_dir "$VIME_PATH"
need_dir "$SOURCE_ROOT/vime-3rdparty/torchtitan"
need_dir "$SOURCE_ROOT/vime-3rdparty/torchtitan-npu"
need_dir "$SOURCE_ROOT/vime-3rdparty/vllm"
need_dir "$SOURCE_ROOT/vime-3rdparty/vllm-ascend"

mkdir -p "$WORKSPACE"
for repo in torchtitan torchtitan-npu vllm vllm-ascend; do
  rm -rf "$WORKSPACE/$repo"
  cp -a "$SOURCE_ROOT/vime-3rdparty/$repo" "$WORKSPACE/$repo"
done

if [[ -z "$VLLM_TORCH_WHEEL" || -z "$VLLM_TORCH_NPU_WHEEL" || -z "$TRAIN_TORCH_WHEEL" || -z "$TRAIN_TORCH_NPU_WHEEL" ]]; then
  echo "Set VLLM_TORCH_WHEEL, VLLM_TORCH_NPU_WHEEL, TRAIN_TORCH_WHEEL and TRAIN_TORCH_NPU_WHEEL." >&2
  exit 2
fi
for wheel in "$VLLM_TORCH_WHEEL" "$VLLM_TORCH_NPU_WHEEL" "$TRAIN_TORCH_WHEEL" "$TRAIN_TORCH_NPU_WHEEL"; do
  [[ -f "$wheel" ]] || die "wheel not found: $wheel"
done

# Install only the genuinely shared stack once into the container's base
# Python. The UV environments inherit it through --system-site-packages.
"$PYTHON" -m pip install setuptools-rust
"$PYTHON" -m pip install --no-deps "$VLLM_TORCH_WHEEL" "$VLLM_TORCH_NPU_WHEEL"
[[ -f "$VIME_PATH/requirements.txt" ]] || die "missing $VIME_PATH/requirements.txt"
"$PYTHON" -m pip install -r "$VIME_PATH/requirements.txt"

install_vllm_into_python() {
  local python_bin=$1
  "$python_bin" -m pip install \
    --trusted-host download-r2.pytorch.org \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"
  "$python_bin" -m pip install \
    --index-url https://pypi.org/simple \
    --find-links https://triton-ascend.osinfra.cn/pypi/simple \
    --trusted-host triton-ascend.osinfra.cn \
    triton-ascend==3.2.1
  VLLM_TARGET_DEVICE=empty \
  VLLM_VERSION_OVERRIDE=0.23.0 \
    "$python_bin" -m pip install \
      --no-build-isolation \
      --no-deps \
      -e "$WORKSPACE/vllm"
  COMPILE_CUSTOM_KERNELS=1 \
  CMAKE_BUILD_TYPE=Release \
  MAX_JOBS=${MAX_JOBS:-16} \
    "$python_bin" -m pip install \
      --no-build-isolation \
      --no-deps \
      -v \
      -e "$WORKSPACE/vllm-ascend"
}

 # vLLM is shared by rollout and inference, so install it once in the base
 # interpreter using the inference Torch version.
install_vllm_into_python "$PYTHON"

# Create UV environments only after the complete public Python environment is
# ready, so both environments inherit the same vLLM stack.
[[ -x "$VIME_VENV/bin/python" ]] || "$UV" venv \
  "$VIME_VENV" \
  --python "$PYTHON" \
  --system-site-packages
[[ -x "$VIME_ROLLOUT_VENV/bin/python" ]] || "$UV" venv \
  "$VIME_ROLLOUT_VENV" \
  --python "$PYTHON" \
  --system-site-packages
SYSTEM_VLLM=$(which vllm)
ln -sf "$SYSTEM_VLLM" "$VIME_ROLLOUT_VENV/bin/vllm"

# Training overrides the inherited inference Torch stack, then adds Titan.
"$UV" pip install \
  --python "$VIME_VENV/bin/python" \
  --no-deps \
  "$TRAIN_TORCH_WHEEL" \
  "$TRAIN_TORCH_NPU_WHEEL"
"$UV" pip install \
  --python "$VIME_VENV/bin/python" \
  --no-build-isolation \
  --no-deps \
  -e "$WORKSPACE/torchtitan"
"$UV" pip install \
  --python "$VIME_VENV/bin/python" \
  --no-build-isolation \
  --no-deps \
  -e "$WORKSPACE/torchtitan-npu"
"$UV" pip install \
  --python "$VIME_VENV/bin/python" \
  --no-build-isolation \
  --no-deps \
  -e "$WORKSPACE/torchtitan-npu/experiments/torchao-npu"

for env in "$VIME_VENV" "$VIME_ROLLOUT_VENV"; do
  "$env/bin/python" - <<'PY'
import torch, torch_npu
print(torch.__version__, torch_npu.__version__)
PY
done
echo "Ready: VIME_VENV=$VIME_VENV VIME_ROLLOUT_VENV=$VIME_ROLLOUT_VENV"
