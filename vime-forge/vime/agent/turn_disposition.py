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

"""Classify Codex Responses turns and track whether completions were adopted.

Codex performs context-checkpoint requests through the same Responses endpoint
as ordinary agent turns. Their sampled summaries follow the same adoption and
training rules as other model outputs; re-injected summary context is masked by
the trajectory builder. Superseded retries, length truncations, and ambiguous
concurrent branches remain non-trainable. Protocol state stays separate from
the trajectory tree so adapters can mask outputs without losing diagnostics.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any


COMPACTION_REQUEST_MARKER = "CONTEXT CHECKPOINT COMPACTION"
COMPACTION_SUMMARY_MARKER = "Another language model started to solve this problem"


class TurnKind(str, enum.Enum):
    AGENT = "agent_turn"
    COMPACTION = "compaction"


class TurnDisposition(str, enum.Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    AUXILIARY = "auxiliary"  # Legacy diagnostic value; compaction now uses accepted/terminal.
    DISCARDED = "discarded"
    TERMINAL = "terminal"


@dataclasses.dataclass(frozen=True)
class RequestObservation:
    kind: TurnKind
    fingerprint: str
    contains_compaction_summary: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "turn_kind": self.kind.value,
            "turn_disposition": TurnDisposition.PENDING.value,
            "request_fingerprint": self.fingerprint,
            # Fail closed until the following request (or session finalization)
            # proves how the completion was used.
            "trainable": False,
        }


@dataclasses.dataclass
class TurnDispositionState:
    session_id: str
    turn_index: int
    kind: TurnKind
    request_fingerprint: str
    finish_reason: str
    disposition: TurnDisposition = TurnDisposition.PENDING
    reason: str = "awaiting_next_request"

    @property
    def trainable(self) -> bool:
        # Sampled compaction summaries are policy outputs too. Their prompt
        # copies are masked separately by TrajectoryManager, not by turn kind.
        # Keep the common truncation/retry/ambiguity guards for both kinds.
        return (
            self.finish_reason != "length"
            and self.disposition in {TurnDisposition.ACCEPTED, TurnDisposition.TERMINAL}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "turn_kind": self.kind.value,
            "request_fingerprint": self.request_fingerprint,
            "finish_reason": self.finish_reason,
            "turn_disposition": self.disposition.value,
            "disposition_reason": self.reason,
            "trainable": self.trainable,
        }


@dataclasses.dataclass(frozen=True)
class DispositionUpdate:
    session_id: str
    turn_index: int
    turn_kind: str
    disposition: str
    reason: str
    trainable: bool
    finish_reason: str
    request_fingerprint: str

    @classmethod
    def from_state(cls, state: TurnDispositionState) -> DispositionUpdate:
        return cls(
            session_id=state.session_id,
            turn_index=state.turn_index,
            turn_kind=state.kind.value,
            disposition=state.disposition.value,
            reason=state.reason,
            trainable=state.trainable,
            finish_reason=state.finish_reason,
            request_fingerprint=state.request_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class TurnDispositionTracker:
    """Per-session state machine for completion adoption.

    A completion becomes final only when the next request reveals whether Codex
    advanced, retried the same request, or the session ended. Pending and
    ambiguous states are non-trainable by default.
    """

    def __init__(self) -> None:
        self._states: dict[str, dict[int, TurnDispositionState]] = {}
        self._pending: dict[str, list[int]] = {}

    def observe_request(
        self, session_id: str, body: dict[str, Any]
    ) -> tuple[RequestObservation, list[DispositionUpdate]]:
        observation = observe_codex_request(body)
        updates: list[DispositionUpdate] = []
        pending_indices = self._pending.pop(session_id, [])
        if len(pending_indices) > 1:
            # Concurrent requests form branches that this protocol-only state
            # machine cannot causally associate with one completion. Keep all
            # raw turns for diagnostics, but do not train ambiguous outputs.
            for pending_index in pending_indices:
                state = self._states[session_id][pending_index]
                updates.append(
                    self._resolve(
                        state,
                        TurnDisposition.DISCARDED,
                        "ambiguous_concurrent_branch",
                    )
                )
        elif pending_indices:
            state = self._states[session_id][pending_indices[0]]
            if state.kind is TurnKind.COMPACTION:
                if observation.kind is TurnKind.COMPACTION and observation.fingerprint == state.request_fingerprint:
                    disposition = TurnDisposition.DISCARDED
                    reason = "same_compaction_request_retried"
                elif state.finish_reason == "length":
                    disposition = TurnDisposition.DISCARDED
                    reason = "downstream_after_truncated_completion"
                elif observation.contains_compaction_summary:
                    disposition = TurnDisposition.ACCEPTED
                    reason = "compaction_summary_replayed"
                else:
                    disposition = TurnDisposition.ACCEPTED
                    reason = "downstream_request_without_retry"
            elif observation.fingerprint == state.request_fingerprint:
                disposition = TurnDisposition.DISCARDED
                reason = "same_agent_request_retried"
            elif state.finish_reason == "length":
                disposition = TurnDisposition.DISCARDED
                reason = "downstream_after_truncated_completion"
            else:
                disposition = TurnDisposition.ACCEPTED
                reason = "downstream_request_observed"
            updates.append(self._resolve(state, disposition, reason))
        return observation, updates

    def record_completion(
        self,
        session_id: str,
        turn_index: int,
        observation: RequestObservation,
        finish_reason: str,
    ) -> DispositionUpdate:
        state = TurnDispositionState(
            session_id=session_id,
            turn_index=int(turn_index),
            kind=observation.kind,
            request_fingerprint=observation.fingerprint,
            finish_reason=finish_reason,
        )
        self._states.setdefault(session_id, {})[state.turn_index] = state
        self._pending.setdefault(session_id, []).append(state.turn_index)
        return DispositionUpdate.from_state(state)

    def finalize_session(self, session_id: str) -> list[DispositionUpdate]:
        pending_indices = self._pending.pop(session_id, [])
        if not pending_indices:
            return []
        updates: list[DispositionUpdate] = []
        ambiguous = len(pending_indices) > 1
        for pending_index in pending_indices:
            state = self._states[session_id][pending_index]
            if ambiguous:
                disposition = TurnDisposition.DISCARDED
                reason = "ambiguous_concurrent_branch"
            elif state.finish_reason == "length":
                disposition = TurnDisposition.DISCARDED
                reason = "terminal_truncated_completion"
            else:
                disposition = TurnDisposition.TERMINAL
                reason = (
                    "terminal_compaction_completion"
                    if state.kind is TurnKind.COMPACTION
                    else "terminal_agent_completion"
                )
            updates.append(self._resolve(state, disposition, reason))
        return updates

    def states(self, session_id: str | None = None) -> list[TurnDispositionState]:
        if session_id is not None:
            return list(self._states.get(session_id, {}).values())
        return [state for states in self._states.values() for state in states.values()]

    def clear_session(self, session_id: str) -> None:
        self._pending.pop(session_id, None)
        self._states.pop(session_id, None)

    @staticmethod
    def _resolve(
        state: TurnDispositionState,
        disposition: TurnDisposition,
        reason: str,
    ) -> DispositionUpdate:
        state.disposition = disposition
        state.reason = reason
        return DispositionUpdate.from_state(state)


def observe_codex_request(body: dict[str, Any]) -> RequestObservation:
    text = "\n".join(_iter_text(body.get("input")))
    kind = TurnKind.COMPACTION if COMPACTION_REQUEST_MARKER in text else TurnKind.AGENT
    return RequestObservation(
        kind=kind,
        fingerprint=semantic_request_fingerprint(body),
        contains_compaction_summary=COMPACTION_SUMMARY_MARKER in text,
    )


def semantic_request_fingerprint(body: dict[str, Any]) -> str:
    """Hash request semantics while excluding volatile Codex correlation ids."""
    semantic_keys = (
        "model",
        "instructions",
        "input",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "include",
    )
    semantic = {key: body.get(key) for key in semantic_keys if key in body}
    payload = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iter_text(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_text(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_text(child)


__all__ = [
    "COMPACTION_REQUEST_MARKER",
    "COMPACTION_SUMMARY_MARKER",
    "DispositionUpdate",
    "RequestObservation",
    "TurnDisposition",
    "TurnDispositionState",
    "TurnDispositionTracker",
    "TurnKind",
    "observe_codex_request",
    "semantic_request_fingerprint",
]
