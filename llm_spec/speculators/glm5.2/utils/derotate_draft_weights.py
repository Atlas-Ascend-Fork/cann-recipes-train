#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""发布前对 GLM-5.2 DSpark 草稿的 fc 反向旋转（de-QuaRot，见 glm5.2/README.md §9）。

用法：
  python3 derotate_draft_weights.py \\
      --draft  ./output/dspark_glm52_800k/checkpoints/epoch4_end \\
      --target /data/GLM-5.2-w4a8c8 \\
      --output /path/to/publish/GLM-5.2-DSpark-epoch4_end_derotated
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LOG = logging.getLogger("derotate_draft_weights")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)
LOG.propagate = False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", required=True, help="训练产出的存档目录（含 model.safetensors）")
    ap.add_argument("--target", default="/data/GLM-5.2-w4a8c8", help="QuaRot 量化 verifier 目录")
    ap.add_argument("--output", required=True, help="发布目录（新建）")
    a = ap.parse_args()

    desc = json.loads((Path(a.target) / "quant_model_description.json").read_text())
    rel = desc.get("optional", {}).get("quarot", {}).get("rotation_map", {}).get("global_rotation")
    if not rel:
        raise SystemExit(f"{a.target} 清单里没有 quarot.rotation_map.global_rotation，"
                         "该 verifier 未旋转，不需要反向旋转")
    rot = load_file(str(Path(a.target) / rel))["global_rotation"].to(torch.float32)

    draft, out = Path(a.draft), Path(a.output)
    sd = load_file(str(draft / "model.safetensors"))
    keys = [k for k in sd if k.endswith("fc.weight")]
    if not keys:
        raise SystemExit(f"{draft} 里没有以 fc.weight 结尾的权重，检查 --draft")
    blocks = []
    worst_rt = 0.0      # 反旋转后再右乘 rot 与原始权重的最大绝对误差（fp32，验证旋转方向）
    worst_cast = 0.0    # 舍入回存储 dtype 后同一指标的最大绝对误差
    for k in keys:
        w = sd[k].to(torch.float32)
        if w.ndim != 2:
            raise SystemExit(f"{k} 不是二维权重（ndim={w.ndim}），检查 --draft")
        if w.shape[1] % rot.shape[0]:
            raise SystemExit(f"{k} 输入维 {w.shape[1]} 不是 {rot.shape[0]} 的整数倍")
        n = w.shape[1] // rot.shape[0]
        # 引擎侧（README §9 引用的 vllm_ascend/models/qwen3_dspark.py 的 process_weight()）
        # 假定草稿未旋转，会把 fc.weight 按 hidden_size 分块右乘 rot；发布前先把每块右乘
        # rot.T（rot 正交，rot.T = rot⁻¹），引擎再右乘 rot 即还原训练权重。旋转方向不能
        # 靠行范数验证（对任意正交变换都不变），直接做还原比对：r 分块右乘 rot 应还原为 w。
        r = torch.matmul(w.reshape(w.shape[0], n, rot.shape[0]), rot.T).reshape(w.shape)
        rt = (torch.matmul(r.reshape(w.shape[0], n, rot.shape[0]), rot).reshape(w.shape) - w).abs().max()
        worst_rt = max(worst_rt, float(rt))
        r_cast = r.to(sd[k].dtype)
        cast_rt = (
            torch.matmul(r_cast.to(torch.float32).reshape(w.shape[0], n, rot.shape[0]), rot)
            .reshape(w.shape)
            - w
        ).abs().max()
        worst_cast = max(worst_cast, float(cast_rt))
        blocks.append(n)
        sd[k] = r_cast

    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "model.safetensors"))
    # checkpoint 由 save_pretrained 写出：config.json / config.py（自定义代码）/
    # generation_config.json（若有）；tokenizer 不在其中（草稿共享 verifier 词表）
    for f in ("config.json", "config.py", "generation_config.json"):
        src = draft / f
        if src.exists():
            shutil.copy2(src, out / f)
    LOG.info(f"[derotate] {len(keys)} 个 fc 权重，每权重块数 {sorted(set(blocks))}；"
             f"反旋转还原最大绝对误差 fp32 {worst_rt:.2e} / 舍入回存储 dtype 后 {worst_cast:.2e} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
