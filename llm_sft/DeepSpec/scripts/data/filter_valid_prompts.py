#!/usr/bin/env python3
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

"""Filter prompt JSONL using the producer's token constraints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running this file directly from the repository checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepspec.data.parser import set_tokenizer_chat_template


def _record_label(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get("sample_id", record.get("id"))
    return None if value is None else str(value)


def validate_record(record, tokenizer, *, max_length: int, min_loss_tokens: int):
    """Return ``(True, None)`` or ``(False, reason)``.

    This intentionally mirrors ``data_utils.tokenize_conversation`` in the
    producer.  DeepSeek-V4 tokenizer directories may omit a HF chat template;
    ``main`` installs the same local DeepSpec fallback before validation.
    """
    if not isinstance(record, dict):
        return False, "record is not a JSON object"
    conversation = record.get("conversations")
    if not conversation:
        return False, "missing or empty conversations"

    try:
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        encoding = tokenizer(
            formatted_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        input_ids = encoding["input_ids"]
        seq_len = int(input_ids.shape[-1])
    except Exception as exc:  # producer treats template/tokenizer failures as invalid
        return False, f"tokenization failed: {exc}"

    if seq_len <= 0:
        return False, "empty token sequence"

    # Producer marks input_ids[len(input_ids)//2:] as loss tokens.
    num_loss_tokens = seq_len - seq_len // 2
    if num_loss_tokens < min_loss_tokens:
        return False, (
            f"loss tokens {num_loss_tokens} < required {min_loss_tokens} "
            f"(token length after truncation: {seq_len})"
        )
    return True, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-filter prompt JSONL with producer token constraints."
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Clean JSONL path")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Same tokenizer directory used by the producer",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N non-empty records (default: 1000)",
    )
    parser.add_argument(
        "--chat-template",
        default="deepseek",
        help="Template fallback when tokenizer metadata has none (default: deepseek)",
    )
    parser.add_argument(
        "--rejected-output",
        help="Rejected-record report (default: <output>.rejected.jsonl)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rejected_path = Path(args.rejected_output or f"{output_path}.rejected.jsonl")

    if args.max_length <= 0 or args.min_loss_tokens < 0 or args.progress_every <= 0:
        raise SystemExit("--max-length must be positive and --min-loss-tokens non-negative")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("--output must be different from --input")
    if input_path.resolve() == rejected_path.resolve():
        raise SystemExit("--rejected-output must be different from --input")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("filtering requires the installed transformers package") from exc

    print(f"loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    set_tokenizer_chat_template(tokenizer, args.chat_template)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    total = valid = rejected = 0
    started = time.monotonic()
    with (
        input_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8", newline="") as clean,
        rejected_path.open("w", encoding="utf-8") as report,
    ):
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            total += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                record = None
                ok = False
                reason = f"invalid JSON: {exc.msg} (column {exc.colno})"
            else:
                ok, reason = validate_record(
                    record,
                    tokenizer,
                    max_length=args.max_length,
                    min_loss_tokens=args.min_loss_tokens,
                )
            if ok:
                clean.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                valid += 1
            else:
                rejected += 1
                report.write(
                    json.dumps(
                        {
                            "line": line_number,
                            "sample_id": _record_label(record),
                            "reason": reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if total % args.progress_every == 0:
                elapsed = time.monotonic() - started
                rate = total / elapsed if elapsed else 0.0
                print(
                    f"progress={total} valid={valid} rejected={rejected} "
                    f"rate={rate:.1f}/s elapsed={elapsed:.1f}s",
                    flush=True,
                )

    print(f"input:    {input_path}")
    print(f"output:   {output_path}")
    print(f"rejected: {rejected_path}")
    print(f"total={total} valid={valid} rejected={rejected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
