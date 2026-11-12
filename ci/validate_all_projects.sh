#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

set -e
set -o pipefail

# --- Colors ---
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RESET="\033[0m"

# Resolve root directory
CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CI_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

SCAN_LIST=(
    "llm_rl/qwen3"
    # Other paths that needed check...
)

REQUIRED_FILES=(
    "download_frameworks_source_code.sh"
    "build_project.sh"
    "apply_all_patches.sh"
)

echo -e "${CYAN}=== CI Starts ===${RESET}"

set +e
echo -e "Checking git environment..."
git rev-parse --is-inside-work-tree 2>&1
GIT_STATUS=$?

if [ ! -d "${ROOT_DIR}/.git" ] || [ $GIT_STATUS -ne 0 ]; then
    git config --global --add safe.directory "$ROOT_DIR"
    git init "${ROOT_DIR}"
    echo -e "${YELLOW}[Warning] Git environment unavailable. Creating a shadow environment at the project root.${RESET}"
else
    echo -e "${GREEN}Git environment available.${RESET}"
fi
set -e

validate_project() {

    # --- Step 1: Check patch naming ---
    echo -e "${CYAN}=== Step 1: Checking patch naming ===${RESET}"
    if ! bash "${CI_DIR}/check_patch_names.sh"; then
        echo -e "${RED}[ERROR] Patch naming validation failed.${RESET}" >&2
        return 1
    fi
    echo -e "${GREEN}[OK] Patch names are valid.${RESET}"

    # --- Step 2: Download framework source code ---
    echo -e "${CYAN}=== Step 2: Download framework source code ===${RESET}"
    if ! bash download_frameworks_source_code.sh; then
        echo -e "${RED}[ERROR] Failed to download framework source code.${RESET}" >&2
        return 1
    fi
    echo -e "${GREEN}[OK] Framework source code downloaded.${RESET}"

    # --- Step 3: Build project ---
    echo -e "${CYAN}=== Step 3: Build project ===${RESET}"
    if ! bash build_project.sh; then
        echo -e "${RED}[ERROR] Project build failed.${RESET}" >&2
        return 1
    fi
    ls -l
    echo -e "${GREEN}[OK] Project built.${RESET}"

    echo -e "${CYAN}=== Step 4: Apply patches ===${RESET}"

    if grep -q "apply_all_patches.sh" build_project.sh; then
        echo -e "${YELLOW}[SKIP] build_project.sh already applies patches.${RESET}"
        echo -e "${GREEN}=== Project CI completed successfully ===${RESET}"
        return 0
    fi
    
    set +e

    PATCH_LOG=$(bash apply_all_patches.sh 2>&1)
    PATCH_STATUS=$?
    set -e

    echo "$PATCH_LOG"

    # Patch application failed. Some patch failed during application.
    FAILED_PATCHES=$(echo "$PATCH_LOG" | grep -i "\[FAIL\]" || true)
    SKIPPED_PATCHES=$(echo "$PATCH_LOG" | grep -E 'Applying .* \.\.\. Skipped patch' || true)

    if [ $PATCH_STATUS -ne 0 ] || [ -n "$FAILED_PATCHES" ] || [ -n "$SKIPPED_PATCHES" ]; then

        echo -e "${RED}[ERROR] Patch application failed.${RESET}"
        if [ -n "$FAILED_PATCHES" ]; then
            echo -e "${YELLOW}The following patches failed during application:${RESET}"

            echo "$FAILED_PATCHES" | while IFS= read -r line; do
                PATCH_NAME=$(echo "$line" | sed -E 's/.*(patches\/[^ ]+\.patch).*/\1/')
                echo "  $PATCH_NAME"
            done
            echo ""
        fi

        if [ -n "$SKIPPED_PATCHES" ]; then
            echo -e "${YELLOW}The following patches were skipped silently, breaking patch application:${RESET}"

            echo "$SKIPPED_PATCHES" | while IFS= read -r line; do
                PATCH_NAME=$(echo "$line" | sed -E 's/.*Applying (patches\/[^ ]+\.patch).*/\1/')
                echo "  $PATCH_NAME"
            done
            echo ""
        fi

        return 1
    fi

    echo -e "${GREEN}[OK] All patches applied successfully.${RESET}"
    echo -e "${GREEN}=== Project CI completed successfully ===${RESET}"
}

has_required_files() {
    local project_dir="$1"

    for f in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "${project_dir}/${f}" ]; then
            return 1
        fi
    done

    return 0
}

for SCAN_PATH in "${SCAN_LIST[@]}"; do
    FULL_PATH="${ROOT_DIR}/${SCAN_PATH}"

    if [ ! -d "$FULL_PATH" ]; then
        echo -e "${RED}[ERROR] Project directory not found: ${FULL_PATH}${RESET}"
        exit 1
    fi

    PROJECT_DIRS=()
    if has_required_files "$FULL_PATH"; then
        PROJECT_DIRS+=("$FULL_PATH")
    else
        while IFS= read -r SUBDIR; do
            if has_required_files "$SUBDIR"; then
                PROJECT_DIRS+=("$SUBDIR")
            fi
        done < <(find "$FULL_PATH" -mindepth 1 -maxdepth 1 -type d | sort)
    fi

    if [ ${#PROJECT_DIRS[@]} -eq 0 ]; then
        SCAN_BASENAME=$(basename "$FULL_PATH")
        echo -e "${RED}[ERROR] Missing required project scripts under ${SCAN_BASENAME}${RESET}"
        for f in "${REQUIRED_FILES[@]}"; do
            echo -e "${RED}[ERROR] Missing ${f} in project ${SCAN_BASENAME}${RESET}"
        done
        exit 1
    fi

    for PROJECT_DIR in "${PROJECT_DIRS[@]}"; do
        PROJECT="${PROJECT_DIR#${ROOT_DIR}/}"
        PROJECT_BASENAME=$(basename "$PROJECT_DIR")

        echo -e "${CYAN}--- Running CI for project: ${PROJECT} ---${RESET}"

        echo -e "${CYAN}Validating project ${PROJECT}${RESET}"
        pushd "${PROJECT_DIR}" >/dev/null

        if ! validate_project; then
            echo -e "${RED}[ERROR] CI pipeline failed for ${PROJECT_BASENAME}${RESET}"
            popd >/dev/null
            exit 1
        fi

        echo -e "${GREEN}[OK] Project ${PROJECT_BASENAME} passed CI.${RESET}"
        popd >/dev/null
    done
done

echo -e "${GREEN}=== All projects passed CI ===${RESET}"
exit 0
