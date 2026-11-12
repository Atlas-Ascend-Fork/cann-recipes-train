#!/bin/bash
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

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
PATCH_ROOT="llm_rl/qwen3/verl-torchtitan/patches"

cd "${REPO_DIR}"

echo "Applying patches in numerical order..."

find "./${PATCH_ROOT}" -type f -name "*.patch" | \
sort -V | \
while IFS= read -r PATCH_FILE; do
    [[ -z "$PATCH_FILE" ]] && continue
    PATCH_REL_PATH=$(realpath --relative-to=. "$PATCH_FILE")

    echo -n "Applying $PATCH_REL_PATH ... "

    if [[ "$PATCH_REL_PATH" == llm_rl/qwen3/verl-torchtitan/patches/torchair/* ]]; then
        if ! command -v patch >/dev/null 2>&1; then
            echo "[FAIL]: patch command not found" >&2
            exit 1
        fi

        PATCH_ABS_PATH="${REPO_DIR}/${PATCH_REL_PATH}"
        TORCHAIR_DIR=$(python3 -c 'import pathlib, site, sys, sysconfig; paths = [sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib"), *site.getsitepackages()]; paths = [p for p in paths if p]; matches = [pathlib.Path(p) / "torch_npu" / "dynamo" / "torchair" for p in paths]; matches = [p.resolve() for p in matches if p.is_dir()]; sys.exit(1) if not matches else print(matches[0])') || {
            echo "[FAIL]: torch_npu torchair dir not found" >&2
            exit 1
        }
        if ! patch --dry-run -p1 -d "${TORCHAIR_DIR}" < "${PATCH_ABS_PATH}" >/dev/null; then
            echo "[FAIL]: $PATCH_REL_PATH dry run failed" >&2
            exit 1
        fi

        if ! patch -p1 -d "${TORCHAIR_DIR}" --backup --version-control=numbered < "${PATCH_ABS_PATH}" >/dev/null; then
            echo "[FAIL]: $PATCH_REL_PATH" >&2
            exit 1
        fi
    else
        if ! git apply -v --ignore-whitespace "$PATCH_REL_PATH"; then
            echo "[FAIL]: $PATCH_REL_PATH" >&2
            exit 1
        fi
    fi
    echo "[SUCCESS]: $PATCH_REL_PATH"
done
