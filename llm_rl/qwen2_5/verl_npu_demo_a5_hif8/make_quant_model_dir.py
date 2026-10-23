# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Build the model directory a low-precision rollout run needs.

vLLM-Ascend reads quant_model_description.json from inside the model directory;
verl's rollout.quantization_config_file cannot carry it. This symlinks the BF16
weights and adds only the description, so both runs start from the same weights.

Usage:
    python3 make_quant_model_dir.py --model "${WORKSPACE_DIR}/verl/model/Qwen2_5_1_5B_Instruct" --scheme hif8
    -> ${WORKSPACE_DIR}/verl/model/Qwen2_5_1_5B_Instruct_hif8
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from safetensors import safe_open

# Quantize the transformer linears; embeddings, norms and the output head stay BF16.
LINEAR_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

logger = logging.getLogger(__name__)

SCHEME = "W8A8_HIF8_PTS"
# The code default for hif8_act_mode is "pts"; cast must be set explicitly.
EXTRA = {"hif8_act_mode": "cast", "hif8_dst_type_max": 16.0}


def weight_names(model_dir: Path) -> list[str]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return sorted(json.loads(index.read_text())["weight_map"])
    with safe_open(model_dir / "model.safetensors", "pt") as handle:
        return sorted(handle.keys())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="BF16 model directory")
    parser.add_argument("--scheme", required=True, choices=["hif8"])
    args = parser.parse_args()

    base = Path(args.model).resolve()
    if not (base / "config.json").is_file():
        raise SystemExit(f"not a model directory: {base}")
    target = base.parent / f"{base.name}_{args.scheme}"
    target.mkdir(parents=True, exist_ok=True)

    for item in sorted(base.iterdir()):
        if item.name == "quant_model_description.json":
            continue
        link = target / item.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(item.resolve())

    description = {"quant_method": "ascend", "model_quant_type": SCHEME}
    description.update(EXTRA)

    quantized = 0
    for name in weight_names(base):
        if not name.endswith(".weight"):
            continue
        if any(f".{suffix}." in name for suffix in LINEAR_SUFFIXES):
            description[name] = SCHEME
            quantized += 1
        else:
            description[name] = "FLOAT"

    (target / "quant_model_description.json").write_text(json.dumps(description, indent=2) + "\n")

    logger.info("model dir : %s", target)
    logger.info("scheme    : %s on %d linear weights", SCHEME, quantized)
    logger.info("extra     : %s", json.dumps(EXTRA))
    logger.info("Point the launch script at this directory with MODEL_PATH.")


if __name__ == "__main__":
    main()
