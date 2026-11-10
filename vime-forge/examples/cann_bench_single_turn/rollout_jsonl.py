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

"""Per-sample snapshots: generated immediately, atomically finalized after reward."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def _record(sample, snapshot, reward):
    return {
        "schema": "vime_cann_single_turn_rollout_sample_v1",
        "snapshot": snapshot,
        "rollout_step": sample.metadata["single_turn_rollout_step"],
        "evaluation": sample.metadata["single_turn_evaluation"],
        "group_index": sample.group_index,
        "sample_index": sample.index,
        "sample_rollout_id": sample.rollout_id,
        "session_id": sample.session_id,
        "status": sample.status.value,
        "remove_sample": sample.remove_sample,
        "prompt": sample.prompt,
        "response": sample.response,
        "reward": reward,
        # Do not copy a reused reference dataset's answer into online archives.
        "label": {"op_name": sample.metadata["op_name"], "format": "4string"},
        "response_token_count": sample.response_length,
        "total_token_count": len(sample.tokens),
        "prompt_token_count": len(sample.tokens) - sample.response_length,
        "trainable_token_count": sample.effective_response_length,
        "rollout_logprob_count": len(sample.rollout_log_probs or []),
        "metadata": sample.metadata,
    }


def _atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def archive_generated(sample, directory):
    if not directory:
        return
    metadata = sample.metadata
    op = re.sub(r"[^A-Za-z0-9_.-]", "_", metadata["op_name"])
    identity = metadata.setdefault("single_turn_record_id", uuid.uuid4().hex)
    phase = "eval" if metadata["single_turn_evaluation"] else "train"
    path = Path(directory).expanduser() / phase / f"rollout_{metadata['single_turn_rollout_step'] + 1:06d}"
    path /= f"{op}_s{sample.index}_{identity}.jsonl"
    metadata["single_turn_rollout_jsonl_path"] = str(path.resolve())
    try:
        _atomic_json(path, _record(sample, "sampled", None))
        path.with_suffix(".txt").write_text(
            f"===== PROMPT =====\n{sample.prompt}\n\n===== RESPONSE =====\n{sample.response}\n", encoding="utf-8"
        )
        submission = metadata.get("single_turn_submission")
        if submission:
            submission_path = path.with_name(path.stem + "_submission.txt")
            submission_path.write_text(submission + "\n", encoding="utf-8")
            metadata["single_turn_submission_archive_path"] = str(submission_path.resolve())
    except (OSError, TypeError, ValueError):
        logger.exception("Failed to archive generated CANN sample %s", identity)


def archive_final(sample, reward):
    path = sample.metadata.get("single_turn_rollout_jsonl_path")
    if path:
        try:
            _atomic_json(Path(path), _record(sample, "final", reward))
        except (OSError, TypeError, ValueError):
            logger.exception("Failed to finalize CANN sample archive %s", path)
