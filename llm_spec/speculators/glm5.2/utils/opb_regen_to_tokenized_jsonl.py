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

"""open-perfectblend-glm5.2-regen 的 parquet -> 对话 jsonl + 分词后的 jsonl。

用法：
python3 opb_regen_to_tokenized_jsonl.py \
    --src ./open-perfectblend-glm5.2-regen \
    --tokenizer /path/to/GLM-5.2-W4A8C8 \
    --out-conversations ./data/openperfectblend_regen.jsonl \
    --out-tokenized ./data/openperfectblend_regen_tokenized.jsonl \
    --max-samples 600000 --seed 42 --seq-length 8192
"""
import argparse
import bisect
import glob
import json
import logging
import re
import sys

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

LOG = logging.getLogger("opb_regen_to_tokenized_jsonl")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)
LOG.propagate = False


def detect_assistant_pattern(tok):
    """从 chat template 里推出「assistant 轮内容」的正则。

    用 4 轮探针会话渲染一次，从第二轮 assistant 的 role 标记、以及两轮
    assistant 尾部的公共前缀，拼出「role 标记 + 到下一轮标记之前」的匹配式。
    """
    probe = [
        {"role": "user", "content": "USER_MSG_1"},
        {"role": "assistant", "content": "ASSISTANT_MSG_1"},
        {"role": "user", "content": "USER_MSG_2"},
        {"role": "assistant", "content": "ASSISTANT_MSG_2"},
    ]
    text = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)

    first_start = text.find("ASSISTANT_MSG_1")
    second_start = text.find("ASSISTANT_MSG_2")
    if first_start == -1 or second_start == -1:
        raise ValueError("chat template 里找不到 assistant 消息，无法推出 assistant 边界")
    first_end = first_start + len("ASSISTANT_MSG_1")
    second_end = second_start + len("ASSISTANT_MSG_2")

    # 第二轮 assistant 之前的那一段，取最后一个 '<'/'[' 起头的 role 标记
    user2_end = text.find("USER_MSG_2") + len("USER_MSG_2")
    prefix = text[user2_end:second_start]
    pos = prefix.rfind("assistant")
    if pos != -1:
        role_start = -1
        for ch in ("<", "["):
            role_start = max(role_start, prefix.rfind(ch, 0, pos))
        role_marker = prefix[role_start:] if role_start != -1 else prefix[pos:]
    else:
        role_marker = prefix

    # 思考型模板会在 assistant 内容外包 <think>...</think>，探针下它是空的
    role_marker = re.sub(r"<think>.*?</think>\s*", "", role_marker, flags=re.DOTALL)

    # 两轮 assistant 尾部的公共前缀 = 到下一轮标记之前的那段
    suffix1 = text[first_end:text.find("USER_MSG_2")]
    suffix2 = text[second_end:]
    common = 0
    for c1, c2 in zip(suffix1, suffix2):
        if c1 != c2:
            break
        common += 1
    suffix = suffix1[:common] or suffix1 or "\n"

    m = re.search(r"((<\|?[a-zA-Z0-9_]+[\|>]?)|(\[[a-zA-Z0-9_]+\]))", role_marker)
    lookahead = f"(?!{re.escape(m.group(1))})" if m else r"(?!<\|start\|)"
    # 末轮 assistant 后面没有下一轮标记时，允许直接到字符串末尾
    terminator = (
        r"(?:" + re.escape(suffix) + r"|$)" if not suffix2.strip() else re.escape(suffix)
    )

    return re.escape(role_marker) + r"((?:" + lookahead + r".)*?)" + terminator


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


def normalize_conversation(conv):
    """把 ShareGPT 的 from/value 换成 role/content。

    只认白名单里的角色；未识别的角色（如 function_call / observation）返回
    None，由调用方丢弃该样本——不能一概当成 assistant，否则工具调用与工具
    返回会被写进监督区间。
    """
    out = []
    for turn in conv:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value") or turn.get("content") or ""
        if role not in ROLE_MAP:
            return None
        out.append({"role": ROLE_MAP[role], "content": content})
    return out


def loss_mask_from_offsets(text, offsets, pattern):
    """按字符偏移把落在 assistant 内容区间内的 token 标 1。"""
    mask = [0] * len(offsets)
    starts = [o[0] for o in offsets]
    for m in re.finditer(pattern, text, re.DOTALL):
        span_start, span_end = m.start(1), m.end(1)
        idx = bisect.bisect_left(starts, span_start)
        for i in range(max(0, idx - 1), len(offsets)):
            tok_start, tok_end = offsets[i]
            if tok_start >= span_end:
                break
            if tok_end > span_start and tok_start < span_end:
                mask[i] = 1
    return mask


def tokenize_row(conv, tok, seq_length, pattern):
    """渲染成文本再分词，返回 (input_ids, loss_mask, 是否被截断)，按 seq_length 截断。"""
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    # 不交给 tokenizer 截断：先拿完整长度记截断标记（恰好等于上限不算截断），再手动截断
    enc = tok(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    ids_full = enc["input_ids"]
    ids = ids_full[:seq_length]
    mask = loss_mask_from_offsets(text, enc["offset_mapping"], pattern)[:seq_length]
    if len(ids) != len(mask):
        raise ValueError(f"input_ids/loss_mask 长度不一致：{len(ids)} vs {len(mask)}")
    return ids, mask, len(ids_full) > seq_length


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="regen 数据集的 parquet 目录")
    ap.add_argument("--tokenizer", required=True, help="目标模型目录（提供 chat template 与分词器）")
    ap.add_argument("--out-conversations", required=True, help="中间产物：对话 jsonl")
    ap.add_argument("--out-tokenized", required=True, help="产物：带 input_ids/loss_mask 的 jsonl")
    ap.add_argument("--max-samples", type=int, default=None, help="抽样条数，不给则全量")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    ap.add_argument("--seq-length", type=int, default=8192, help="序列长度上限")
    return ap.parse_args()


def pick_rows(files, max_samples, seed):
    """等概率抽样：等价于先 shuffle(seed) 再取前 n_take 条。返回 (总行数, 抽样条数, 行号集合)。"""
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    n_take = total if max_samples is None else min(max_samples, total)
    picked = set(
        np.random.default_rng(seed).permutation(total)[:n_take].tolist()
    )
    return total, n_take, picked


def iter_rows(files):
    """按文件顺序产出 (全局行号, 原始行)；行号对被丢弃的样本同样推进。"""
    g = 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=2000):
            for r in b.to_pylist():
                yield g, r
                g += 1


def convert_row(r, tok, seq_length, pattern):
    """单条 -> (对话行, 分词行, 是否截断, 监督 token 数)；角色未识别返回 None。"""
    conv = normalize_conversation(r["conversations"])
    if conv is None:
        return None
    ids, mask, was_truncated = tokenize_row(conv, tok, seq_length, pattern)
    conv_line = json.dumps({"conversations": r["conversations"]}, ensure_ascii=False)
    tok_line = json.dumps({"input_ids": ids, "loss_mask": mask})
    return conv_line, tok_line, was_truncated, sum(mask)


def main():
    args = parse_args()

    files = sorted(glob.glob(args.src + "/**/*.parquet", recursive=True))
    if not files:
        raise ValueError(f"{args.src} 下没有 parquet 文件")
    total, n_take, picked = pick_rows(files, args.max_samples, args.seed)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    pattern = detect_assistant_pattern(tok)
    LOG.info(f"[opb] parquet {len(files)} 个文件共 {total} 行，抽 {n_take} 条")
    LOG.info(f"[opb] assistant 边界正则：{pattern}")

    n_conv = n_tok = n_trunc = n_empty = n_skip = 0
    sup_tokens = 0
    with open(args.out_conversations, "w", encoding="utf-8") as fo, open(
        args.out_tokenized, "w", encoding="utf-8"
    ) as ft:
        for g, r in iter_rows(files):
            if g not in picked:
                continue
            row = convert_row(r, tok, args.seq_length, pattern)
            if row is None:
                # 角色不认识：丢弃该样本（不写中间产物，也不进训练集）
                n_skip += 1
                continue
            conv_line, tok_line, was_truncated, sup = row
            # --out-conversations 只写抽中的样本（中间产物，便于核对）
            fo.write(conv_line + "\n")
            ft.write(tok_line + "\n")
            n_conv += 1
            n_tok += 1
            sup_tokens += sup
            n_trunc += was_truncated
            n_empty += sup == 0

    LOG.info(f"[opb] 对话 {n_conv} 行 -> {args.out_conversations}")
    if n_skip:
        LOG.info(f"[opb] 角色未识别而丢弃的样本 {n_skip} 条")
    LOG.info(
        f"[opb] 分词 {n_tok} 行 -> {args.out_tokenized}"
        f"（截断 {n_trunc}，零监督 {n_empty}，平均监督 {sup_tokens / max(n_tok, 1):.1f} token）"
    )


if __name__ == "__main__":
    main()
