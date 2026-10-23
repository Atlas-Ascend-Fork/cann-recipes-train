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

# Fetch the three framework trees at the commits the sample README.md reports;
# the patches under patches/ apply to exactly these trees.
# WORKSPACE_DIR may hold your own mounts, so this script never deletes anything.
# A tree already sitting on the pinned commit is left alone, so re-running is a no-op.

set -ex

HOME_DIR=$(pwd)

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}

mkdir -p "${WORKSPACE_DIR}"
cd "${WORKSPACE_DIR}"

for spec in \
    "https://gitcode.com/GitHub_Trending/ve/verl.git verl 483b8a009ba3a97563edee3a19887e4862b8094a" \
    "https://gitcode.com/gh_mirrors/vl/vllm.git vllm 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665" \
    "https://gitcode.com/gh_mirrors/vl/vllm-ascend.git vllm-ascend f042ad88882e22a43af323b0df5691467bad8553"
do
    set -- ${spec}
    if [ -e "$2" ]; then
        # Keep an existing tree only when it is already on the pinned commit; anything
        # else is the user's own and must be moved aside by hand, never by this script.
        have=$(git -C "$2" rev-parse HEAD 2>/dev/null || echo none)
        [ "${have}" = "$3" ] \
            || { echo "[FAIL]: ${WORKSPACE_DIR}/$2 is at ${have}, not the pinned $3; check it out at $3, move it aside, or set WORKSPACE_DIR elsewhere" >&2; exit 1; }
        echo "[SKIP]: $2 already at the pinned commit"
        continue
    fi
    git clone "$1"
    ( cd "$2" && git checkout "$3" )
done

# Catlass arrives as a submodule and is needed to build the custom kernels.
( cd vllm-ascend && git submodule update --init --recursive )

cd "${HOME_DIR}"
