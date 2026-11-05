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

"""Convert HuggingFaceH4/MATH-500 into the vime evaluation JSONL schema.

Vime reads eval prompts from local `.jsonl`/`.parquet` files (see
`vime/utils/data.py`), not from a Hub dataset id, and its generic eval path
expects a prompt field plus a label field. MATH-500 already ships as JSONL, so
this only rewrites the record schema:

    {"problem": ..., "answer": ..., "subject": ..., "level": ..., "unique_id": ...}
->  {"prompt": [{"role": "user", "content": <problem>}], "label": <answer>,
     "metadata": {"subject": ..., "level": ..., "unique_id": ...}}

`prompt` is a chat message list to match the existing eval set
(zhuzilin/aime-2024), and `label` carries the ground-truth answer used by the
`math` reward grader. The original subject/level/unique_id are kept under
`metadata` for per-topic breakdowns.

Usage:
    python tools/build_math500_eval_data.py                 # -> $VIME_DATA_ROOT/math-500/math-500.jsonl
    python tools/build_math500_eval_data.py --output /data/math-500.jsonl
    python tools/build_math500_eval_data.py --input raw.jsonl  # skip the download
"""

from __future__ import annotations

import logging
import argparse
import json
import os
from urllib.request import urlopen

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "HuggingFaceH4/MATH-500"
DEFAULT_FILENAME = "test.jsonl"
DEFAULT_DATA_ROOT = "/tmp/vime/data"  # matches VIME_DATA_ROOT's default in scripts/torchtitan/common/default.sh


def _download(repo_id: str, filename: str) -> str:
    # hf_hub_url honors HF_ENDPOINT, so a mirror works without extra flags. The
    # direct resolve URL is used instead of hf_hub_download because the latter's
    # HEAD pre-check fails behind some mirrors.
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    with urlopen(url, timeout=120) as resp:
        return resp.read().decode("utf-8")


def _parse(text: str):
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _convert(records, output: str) -> int:
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for record in records:
            row = {
                "prompt": [{"role": "user", "content": record["problem"]}],
                "label": record["answer"],
                "metadata": {key: record[key] for key in ("subject", "level", "unique_id") if key in record},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="Local raw MATH-500 jsonl; download from the Hub when omitted.")
    parser.add_argument(
        "--output",
        default=os.path.join(os.environ.get("VIME_DATA_ROOT") or DEFAULT_DATA_ROOT, "math-500", "math-500.jsonl"),
        help="Output jsonl path (default: $VIME_DATA_ROOT/math-500/math-500.jsonl).",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hub dataset id (default: {DEFAULT_REPO_ID}).")
    parser.add_argument(
        "--filename", default=DEFAULT_FILENAME, help=f"File inside the repo (default: {DEFAULT_FILENAME})."
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            text = f.read()
    else:
        text = _download(args.repo_id, args.filename)
    count = _convert(_parse(text), args.output)
    logger.info(f"wrote {count} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
