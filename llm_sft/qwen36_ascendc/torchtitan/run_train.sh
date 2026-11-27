#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

EXAMPLE_ROOT=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

die() {
  printf '%s\n' "$1" >&2
  exit 2
}

require_positive_integer() {
  local name=$1 value=$2
  [[ $value =~ ^[0-9]+$ && $value =~ [1-9] ]] || die "$name must be a positive integer"
}

if [[ ${1:-} == export-hf ]]; then
  shift
  (( $# == 3 )) || die "usage: bash run_train.sh export-hf DCP SOURCE_HF OUTPUT"
  exec python3 "$EXAMPLE_ROOT/cannbot_recipe.py" export-hf -- "$@"
fi

(( $# == 0 )) || die "This sample accepts only the export-hf subcommand"
: "${HF_ASSETS_PATH:?Set the pinned Qwen3.6-27B HF directory}"
: "${DUMP_FOLDER:?Set a new output directory}"
: "${DATA_FILES:?Set comma-separated JSONL paths}"
[[ -d "$HF_ASSETS_PATH" ]] || die "HF_ASSETS_PATH must point to a model directory: $HF_ASSETS_PATH"
[[ "$DUMP_FOLDER" != *,* ]] || die "DUMP_FOLDER must not contain commas"
[[ "$HF_ASSETS_PATH" != -* ]] || HF_ASSETS_PATH="./$HF_ASSETS_PATH"
[[ "$DUMP_FOLDER" != -* ]] || DUMP_FOLDER="./$DUMP_FOLDER"

case "$DATA_FILES" in
  ,*|*,|*,,*|*$'\n'*|*$'\r'*)
    die "DATA_FILES must be a single line of comma-separated paths with no empty entries"
    ;;
esac
IFS=',' read -r -a data_files <<< "$DATA_FILES"
for index in "${!data_files[@]}"; do
  data_file=${data_files[index]}
  [[ -f "$data_file" && -r "$data_file" ]] ||
    die "DATA_FILES entry must be a readable file: $data_file"
  if [[ "$data_file" == -* ]]; then
    data_files[index]="./$data_file"
  fi
done

if [[ ${CANNBOT_SMOKE:-0} == 1 ]]; then
  CANNBOT_TRAIN_STEPS=3
  CANNBOT_CHECKPOINT_INTERVAL=3
else
  require_positive_integer CANNBOT_TRAIN_STEPS "${CANNBOT_TRAIN_STEPS:-}"
  require_positive_integer CANNBOT_CHECKPOINT_INTERVAL "${CANNBOT_CHECKPOINT_INTERVAL:-}"
fi
export CANNBOT_TRAIN_STEPS CANNBOT_CHECKPOINT_INTERVAL

mkdir -p -- "$DUMP_FOLDER"
export DATA_MANIFEST="$DUMP_FOLDER/data_manifest.json"
export DATA_FILES="$DUMP_FOLDER/train.jsonl"
python3 "$EXAMPLE_ROOT/data_process.py" \
  --data "${data_files[@]}" --output "$DATA_MANIFEST" \
  --merge-jsonl "$DATA_FILES"

export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_VISIBLE_DEVICES}"
export PYTHONPATH="$EXAMPLE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$DUMP_FOLDER/pycache}"
mkdir -p -- "$PYTHONPYCACHEPREFIX"

python3 -m py_compile "$EXAMPLE_ROOT"/{cannbot_recipe,data_process}.py

exec torchrun --nnodes 1 --node-rank 0 --nproc-per-node 16 \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" \
  --master-port "${MASTER_PORT:-29851}" \
  --local-ranks-filter "${LOG_RANK:-0}" --role cannbot_qwen36_sft --tee 3 \
  -m torchtitan.train --module cannbot_recipe \
  --config qwen3_6_27b_sft_cp8_64k \
  --hf-assets-path "$HF_ASSETS_PATH" \
  --checkpoint.initial-load-path "$HF_ASSETS_PATH" \
  --checkpoint.initial-load-in-hf --checkpoint.initial-load-model-only \
  --checkpoint.folder checkpoint \
  --dump-folder "$DUMP_FOLDER" \
  --debug.print-config --debug.save-config-file resolved_config.json
