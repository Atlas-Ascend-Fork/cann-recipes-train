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

ROOT="patches"
ERRORS=()

# -----------------------------------------
# Validate root-level patches
#   NNNN-(bugfix|feature)-description.patch
#   description must use underscores, not dashes
# -----------------------------------------
check_root_patches() {
    local files=("$@")
    local nums=()

    for f in "${files[@]}"; do
        if [[ ! "$f" =~ ^([0-9]{4})\-(bugfix|feature)\-[A-Za-z0-9_]+\.patch$ ]]; then
            ERRORS+=("[Naming error] $ROOT/$f must follow: NNNN-(bugfix|feature)-description.patch (description must use underscores)")
            continue
        fi

        nums+=("${BASH_REMATCH[1]}:$ROOT/$f")
    done

    # Sequence check: patch numbers must strictly increase by 1 (0001, 0002, 0003, ...)
    if [[ ${#nums[@]} -gt 0 ]]; then
        IFS=$'\n'
        sorted=($(sort <<< "${nums[*]}"))
        unset IFS

        # Ensure first patch starts at 0001
        first_seq="${sorted[0]%%:*}"
        if (( 10#$first_seq != 1 )); then
            ERRORS+=("[Number error] First patch must be 0001, but got $first_seq in ${sorted[0]#*:}")
        fi

        # Check sequence number continuity.
        for ((i=1; i < ${#sorted[@]}; i++)); do
            prev_seq="${sorted[$((i-1))]%%:*}"
            curr_seq="${sorted[$i]%%:*}"
            expected=$(( 10#$prev_seq + 1 ))
            actual=10#$curr_seq

            if (( actual != expected )); then
                ERRORS+=("[Number error] Expected sequence $expected, but got $curr_seq in ${sorted[$i]#*:}")
            fi
        done
    fi
}

# -----------------------------------------
# Validate subdirectory patches
#   NNNN-module-(bugfix|feature)-description.patch
#   description must use underscores, not dashes
# -----------------------------------------
check_subdir_patches() {
    local subdir="$1"
    shift
    local files=("$@")

    local nums=()

    for f in "${files[@]}"; do
        local full="$subdir/$f"

        if [[ ! "$f" =~ ^([0-9]{4})\-([A-Za-z0-9_]+)\-(bugfix|feature)\-([A-Za-z0-9_]+)\.patch$ ]]; then
            ERRORS+=("[Naming error] $full must follow: NNNN-module-(bugfix|feature)-description.patch (description must use underscores)")
            continue
        fi

        nums+=("${BASH_REMATCH[1]}:$full")
        seq="${BASH_REMATCH[1]}"
        module="${BASH_REMATCH[2]}"
        expected_module="$(basename "$subdir")"
    
        if [[ "$module" != "$expected_module" ]]; then
            ERRORS+=("[Module mismatch] $full: module '$module' does not match directory '$expected_module'")
            continue
        fi

    done

    # Sequence check: patch numbers must strictly increase by 1 (0001, 0002, 0003, ...)
    if [[ ${#nums[@]} -gt 0 ]]; then
        IFS=$'\n'
        sorted=($(sort <<< "${nums[*]}"))
        unset IFS

        # Ensure first patch starts at 0001
        first_seq="${sorted[0]%%:*}"
        if (( 10#$first_seq != 1 )); then
            ERRORS+=("[Number error] First patch must be 0001, but got $first_seq in ${sorted[0]#*:}")
        fi

        # Check continuity: each should be previous + 1
        for ((i=1; i < ${#sorted[@]}; i++)); do
            prev_seq="${sorted[$((i-1))]%%:*}"
            curr_seq="${sorted[$i]%%:*}"
            expected=$(( 10#$prev_seq + 1 ))
            actual=10#$curr_seq

            if (( actual != expected )); then
                ERRORS+=("[Number error] Expected sequence $expected, but got $curr_seq in ${sorted[$i]#*:}")
            fi
        done
    fi
}

# -----------------------------------------
# Check if patch names appear in README.md.
# -----------------------------------------
check_patch_references_in_readme() {
    local readme_path="README.md"
    local -a patches=("$@")

    if [[ ! -f "$readme_path" ]]; then
        ERRORS+=("[Missing README.md] README.md is not found")
        return
    fi
    
    # Read the entire README file into a variable
    readme_content=$(cat "$readme_path")
    
    # Check each patch file name appears in the README
    for patch in "${patches[@]}"; do
        if [[ "$readme_content" != *"$patch"* ]]; then
            ERRORS+=("[Missing reference] Patch '$patch' is not mentioned in README.md.")
        fi
    done
}

# -----------------------------------------
# Main logic
# -----------------------------------------
if [ ! -d "$ROOT" ]; then
    echo "[ERROR] Directory '$ROOT' does not exist"
    exit 1
fi

# Root-level patches
root_files=()
while IFS= read -r f; do
    root_files+=("$f")
done < <(find "$ROOT" -maxdepth 1 -type f -name "*.patch" -printf "%f\n")

check_root_patches "${root_files[@]}"

# Check root-level patches against root README.md (if it exists)
check_patch_references_in_readme "${root_files[@]}"

# Subdirectories (only level-1)
while IFS= read -r sub; do
    subname=$(basename "$sub")

    mod_files=()
    while IFS= read -r f; do
        mod_files+=("$f")
    done < <(find "$sub" -maxdepth 1 -type f -name "*.patch" -printf "%f\n")

    check_subdir_patches "$sub" "${mod_files[@]}"
    
    # Check subdirectory patches against their respective README.md (if it exists)
    check_patch_references_in_readme "${mod_files[@]}"

done < <(find "$ROOT" -maxdepth 1 -mindepth 1 -type d)

# -----------------------------------------
# Final report
# -----------------------------------------
if (( ${#ERRORS[@]} > 0 )); then
    echo "Validation issues found:"
    for e in "${ERRORS[@]}"; do
        echo " - $e"
    done
    exit 1
fi

echo "All patches follow the naming rules: "
echo " - NNNN-(bugfix|feature)-description.patch/NNNN-module-(bugfix|feature)-description.patch"
echo " - NNNN (Sequence number) are consecutive integers starting from 1."
echo " - Description should use only underscores(_)."
echo " - Appeared at least once in README.md."
exit 0
