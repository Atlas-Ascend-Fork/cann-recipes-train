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
