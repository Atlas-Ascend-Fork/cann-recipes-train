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

"""CMB-Eagle-Mix-330k -> 中文 prompt 集合（speculators/utils/ 共用脚本，glm5.2 与 qwen3.8_27b 两个样例共用）。

用法（占比 >= 0.24、分母=总字符数，降序取前 10w）：
python3 select_cmb_chinese_prompts.py \
    --src ./data/cmb/cmb_eagle_mix_330k.jsonl \
    --out ./data/cmb_zh_prompt_100k.jsonl \
    --min-zh-ratio 0.24 \
    --num-prompts 100000

选取规则：
1. 每条只取第一个 human 轮的文本作 prompt（assistant 轮丢弃，由 target 重新生成，on-policy）；
2. 按 prompt 的中文占比过滤（占比 = CJK 字符数 / 总字符数），占比低于 --min-zh-ratio 的整条丢弃；
3. 剩下的按中文占比降序取前 --num-prompts 条（不传则全部保留；占比相同的保持原文件顺序）。

输出 jsonl，每行一个 {"prompt": "<中文 prompt>"}，供 regen 的 --dataset 直接读取
（regen 侧按 prompt_field 包装成单轮 user 消息）。
"""
import argparse
import json
import logging
import statistics
import sys

HUMAN_ROLES = ("human", "user")

LOG = logging.getLogger("select_cmb_chinese_prompts")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)
LOG.propagate = False


def parse_args():
    p = argparse.ArgumentParser(description="从 CMB-Eagle-Mix-330k 挑中文 prompt")
    p.add_argument("--src", required=True, help="CMB jsonl（ShareGPT 风格，conversations 或 messages 字段）")
    p.add_argument("--out", required=True, help="输出 jsonl，每行 {\"prompt\": ...}")
    p.add_argument("--min-zh-ratio", type=float, default=0.24, help="中文占比下限（默认 0.24）")
    p.add_argument("--num-prompts", type=int, default=None, help="按中文占比降序取前 N 条；不传则保留全部过阈值的行")
    return p.parse_args()


def conversation_list(record):
    """取对话列表：CMB 原始文件用 conversations，清洗后的用 messages。"""
    convs = record.get("conversations")
    if not convs:
        convs = record.get("messages")
    return convs or []


def first_human_text(record):
    """返回第一个 human 轮的文本；没有则返回 None。"""
    for msg in conversation_list(record):
        if not isinstance(msg, dict):
            continue
        role = msg.get("from") or msg.get("role")
        if role not in HUMAN_ROLES:
            continue
        text = msg.get("value")
        if text is None:
            text = msg.get("content")
        if not text:
            continue
        return text
    return None


def zh_ratio(text):
    """中文占比：CJK 统一表意文字区间的字符数 / 总字符数。"""
    if not text:
        return 0.0
    zh = sum(1 for c in text if "一" <= c <= "鿿")
    return zh / len(text)


def main():
    args = parse_args()
    if args.min_zh_ratio < 0 or args.min_zh_ratio > 1:
        sys.exit(f"--min-zh-ratio 必须在 [0, 1]，当前 {args.min_zh_ratio}")

    total = 0
    with_human = 0
    kept = []  # (zh_ratio, 原始行号, prompt)
    with open(args.src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = first_human_text(record)
            if text is None:
                continue
            with_human += 1
            r = zh_ratio(text)
            if r >= args.min_zh_ratio:
                kept.append((r, i, text))

    # 占比降序；占比相同的按原始行号稳定排序
    kept.sort(key=lambda x: (-x[0], x[1]))
    selected = kept if args.num_prompts is None else kept[: args.num_prompts]

    with open(args.out, "w", encoding="utf-8") as fout:
        for _, _, text in selected:
            fout.write(json.dumps({"prompt": text}, ensure_ascii=False) + "\n")

    ratios = [r for r, _, _ in selected]
    lens = sorted(len(t) for _, _, t in selected)
    LOG.info(f"源文件记录数        : {total:,}")
    LOG.info(f"有 human 首轮        : {with_human:,}")
    LOG.info(f"中文占比 >= {args.min_zh_ratio:<5}: {len(kept):,}")
    LOG.info(f"实际写出            : {len(selected):,} -> {args.out}")
    if ratios:
        LOG.info(f"中文占比            : min={min(ratios):.4f} p50={statistics.median(ratios):.4f} max={max(ratios):.4f}")
        LOG.info(f"prompt 字符长度      : p50={lens[len(lens) // 2]} p90={lens[len(lens) * 9 // 10]} max={lens[-1]}")


if __name__ == "__main__":
    main()
