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

"""Opaque Responses reasoning round-trip helpers.

The Responses API calls the field ``encrypted_content``. Vime does not need
cryptographic secrecy here: it is both the producer and consumer, and Codex
only has to replay the value opaquely. The names below deliberately say
``pack``/``unpack`` to avoid implying encryption at rest or in transit.
"""

from __future__ import annotations

import base64
from typing import Any


_PREFIX = "vime:"


def pack_reasoning(reasoning_text: str) -> str:
    if not reasoning_text:
        return ""
    return _PREFIX + base64.urlsafe_b64encode(reasoning_text.encode("utf-8")).decode("ascii")


def unpack_reasoning(packed: Any) -> str:
    if not isinstance(packed, str) or not packed.startswith(_PREFIX):
        return ""
    try:
        return base64.urlsafe_b64decode(packed[len(_PREFIX) :].encode("ascii")).decode("utf-8")
    except Exception:
        return ""


def extract_responses_reasoning(item: dict[str, Any]) -> str:
    """Read full content, summary, or Vime's opaque packed fallback."""
    for field in ("content", "summary"):
        blocks = item.get(field)
        if not isinstance(blocks, list):
            continue
        text = "\n".join(
            block["text"]
            for block in blocks
            if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"]
        )
        if text:
            return text
    return unpack_reasoning(item.get("encrypted_content"))


__all__ = ["extract_responses_reasoning", "pack_reasoning", "unpack_reasoning"]
