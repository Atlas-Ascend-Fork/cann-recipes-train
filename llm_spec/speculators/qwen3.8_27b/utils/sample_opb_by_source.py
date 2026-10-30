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

"""按 source 分层抽样英文源池（Qwen3.8-27B 样例 §6.2.2 用：先抽 65w，再重生成）。

用法：
python3 sample_opb_by_source.py \
    --src-dir ./data/opb_by_source \
    --out ./data/opb_650k.jsonl \
    --num-samples 650000 \
    --seed 42

为什么按 source 分层、为什么在重生成之前抽：
- open-perfectblend 是多子集合并的数据集（ultrachat / math / code / ...），
  各子集的原始比例是有意义的，不能只按总量随便截一段；
- 重生成是整条管线里最贵的一步，先抽 65w 再重生成，比全量 1.42M 重生成
  之后再截省一半以上算力；
- 重生成产物的行里没有 source 字段（只有 id / primary_id / input_ids /
  loss_mask / text / metadata），所以分层抽样必须在源池上做——源池按
  "一个 source 一个（或多个分片）文件"布局，source 靠文件名归组保留，
  抽完合并成单文件即可整体重生成。

做法：
1. 统计每个 source 的总条数（分片行数相加）；
2. 按各 source 的条数比例把目标条数拆成整数（largest-remainder 分配，
   保证各 source 的占比与源池一致，且整数之和恰为目标条数）；
3. 每个 source 内部按 --seed 随机抽对应条数。

源池总条数不足目标条数时全取，并打印提示。
"""
import argparse
import glob
import logging
import os
import random
import re
import sys
from collections import defaultdict

LOG = logging.getLogger("sample_opb_by_source")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)
LOG.propagate = False

# 分片文件名后缀：..._s0of10 / ...s0of10 / ..._shard3 / ..._part2
SHARD_SUFFIX = re.compile(
    r"(?:[_.-]?(?:s|shard|part)\d+of\d+|[_.-](?:shard|part)\d+)$", re.IGNORECASE
)


def parse_args():
    p = argparse.ArgumentParser(description="按 source 分层抽样 regen 产物")
    p.add_argument(
        "--src-dir",
        action="append",
        required=True,
        help="regen 产物目录，可重复传；目录下的 *.jsonl 按文件名归到各 source",
    )
    p.add_argument("--out", required=True, help="输出 jsonl（抽中的行）")
    p.add_argument(
        "--num-samples", type=int, required=True, help="目标条数（超过池子则全取）"
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    return p.parse_args()


def source_of(path):
    """文件名 -> source 名：去掉 .jsonl 与分片后缀（s0of10 之类）。"""
    stem = os.path.basename(path)
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    return SHARD_SUFFIX.sub("", stem)


def largest_remainder(shares, total):
    """按 shares 的比例把 total 拆成整数：各份之和恰为 total，占比尽量贴近份额。

    先按比例取整，再把余数按小数部分从大到小逐个补齐；平局按 key 排序，
    保证同一输入永远得到同一分配。
    """
    pool = sum(shares.values())
    if pool <= 0 or total <= 0:
        return {k: 0 for k in shares}
    raw = {k: v / pool * total for k, v in shares.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    rest = total - sum(alloc.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    return alloc


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def collect_files(src_dirs):
    """收集各 source 的分片文件：{source: [path, ...]}。"""
    files = defaultdict(list)
    for d in src_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            files[source_of(path)].append(path)
    return dict(files)


def take_from_file(path, n, take, rng):
    """从单个分片文件里等概率抽 take 行（0 < take <= n）。"""
    want = set(rng.sample(range(n), take))
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in want:
                out.append(line.rstrip("\n"))
    return out


def sample_source(paths, counts, k, rng):
    """在若干分片里随机抽 k 行（分片之间也按行数比例分配）。

    counts 与 paths 一一对应，是各分片的行数。
    """
    total = sum(counts)
    if k >= total:
        per_file = dict(zip(paths, counts))
    else:
        per_file = largest_remainder(dict(zip(paths, counts)), k)
    out = []
    for p, n in zip(paths, counts):
        take = min(per_file[p], n)
        if take > 0:
            out += take_from_file(p, n, take, rng)
    return out


def main():
    args = parse_args()
    if args.num_samples <= 0:
        sys.exit(f"--num-samples 必须为正，当前 {args.num_samples}")

    files = collect_files(args.src_dir)
    if not files:
        sys.exit(f"在 {args.src_dir} 下没找到 *.jsonl")

    file_counts = {p: count_lines(p) for paths in files.values() for p in paths}
    counts = {
        src: sum(file_counts[p] for p in paths) for src, paths in files.items()
    }
    pool = sum(counts.values())
    target = min(args.num_samples, pool)
    alloc = largest_remainder(counts, target)

    LOG.info(f"source 数: {len(counts)}    池子总条数: {pool:,}    目标: {target:,}")
    for src in sorted(alloc, key=lambda s: -alloc[s]):
        share = counts[src] / pool if pool else 0.0
        got = alloc[src] / target if target else 0.0
        LOG.info(
            f"  {src:40s} 池 {counts[src]:>9,} ({share:6.2%})"
            f" -> 抽 {alloc[src]:>9,} ({got:6.2%})"
        )
    if target < args.num_samples:
        LOG.info(f"池子不足目标，全取（{target:,} < {args.num_samples:,}）")

    rng = random.Random(args.seed)
    n_written = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for src in sorted(files):
            for line in sample_source(
                files[src], [file_counts[p] for p in files[src]], alloc[src], rng
            ):
                fout.write(line + "\n")
                n_written += 1
    LOG.info(f"写出 {n_written:,} 条 -> {args.out}")


if __name__ == "__main__":
    main()
