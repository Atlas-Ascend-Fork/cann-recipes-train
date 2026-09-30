"""Model-side parsing and chat-template policy for agent rollouts."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Literal


ReasoningMode = Literal["visible_verbatim", "separate"]
logger = logging.getLogger(__name__)


def load_checkpoint_model_type(checkpoint: str) -> str | None:
    """Read model identity from local metadata without loading code or downloading.

    Unavailable metadata leaves model-specific policies disabled. Hub identifiers
    are supported only when they also resolve to a local checkpoint directory.
    """
    config_path = Path(checkpoint) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        logger.warning("Cannot read checkpoint model_type from %s: %s", config_path, exc)
        return None
    model_type = config.get("model_type") if isinstance(config, dict) else None
    if not isinstance(model_type, str) or not model_type.strip():
        logger.warning("Checkpoint config %s has no nonempty string model_type", config_path)
        return None
    return model_type


def uses_qwen3_apply_patch_policy(model_type: str | None) -> bool:
    """Select the shared JSON patch contract for dense and MoE Qwen3 checkpoints."""
    return model_type in {"qwen3", "qwen3_moe"}


def uses_single_system_message_policy(model_type: str | None) -> bool:
    """Qwen3.5's validated dense checkpoint accepts only an initial system message."""
    return model_type == "qwen3_5"


@dataclasses.dataclass(frozen=True)
class ModelProfile:
    """Configuration owned by the served model, not by the Agent harness.

    ``visible_verbatim`` keeps reasoning in normal assistant text and therefore
    requires no reasoning parser. ``separate`` parses reasoning into the
    canonical ``reasoning_content`` field so an API adapter can round-trip it
    without exposing it as user-visible text.
    """

    name: str
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    reasoning_mode: ReasoningMode = "visible_verbatim"
    chat_template_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    normalize_apply_patch_description: bool = False
    coalesce_system_messages: bool = False

    def __post_init__(self) -> None:
        if self.reasoning_mode == "separate" and not self.reasoning_parser:
            raise ValueError("reasoning_mode='separate' requires a reasoning_parser")
        if self.reasoning_mode == "visible_verbatim" and self.reasoning_parser:
            raise ValueError("visible_verbatim cannot silently discard output through a reasoning_parser")

    @classmethod
    def from_runtime(
        cls,
        *,
        name: str = "runtime",
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        normalize_apply_patch_description: bool = False,
        coalesce_system_messages: bool = False,
    ) -> ModelProfile:
        return cls(
            name=name,
            tool_parser=tool_parser or None,
            reasoning_parser=reasoning_parser or None,
            reasoning_mode="separate" if reasoning_parser else "visible_verbatim",
            chat_template_kwargs=dict(chat_template_kwargs or {}),
            normalize_apply_patch_description=normalize_apply_patch_description,
            coalesce_system_messages=coalesce_system_messages,
        )


__all__ = [
    "ModelProfile",
    "ReasoningMode",
    "load_checkpoint_model_type",
    "uses_qwen3_apply_patch_policy",
    "uses_single_system_message_policy",
]
