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

"""OpenAI Responses API adapter for Codex agent rollouts.

This is the Responses counterpart of :mod:`vime.agent.adapters.openai`.  It
keeps the PR's ``BaseAdapter`` lifecycle (session routing, vLLM generation,
trajectory capture and cleanup) and only implements Responses wire translation
and response/SSE framing.

The Codex CLI currently sends a mixture of ordinary ``function`` tools,
freeform ``custom`` tools and provider-native tools (for example ``namespace``
and ``web_search``).  ``apply_patch`` is translated between Codex's freeform
custom-tool protocol and the served model's JSON function-tool protocol.  Other
provider-native tools are deliberately omitted rather than exposed as tools the
model cannot actually invoke through this adapter.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from typing import Any

from aiohttp import web

from vime.agent.adapters.common import (
    BaseAdapter,
    Reply,
    flatten_content,
    manager_finish_reason,
    sid_from_bearer,
    sid_from_body,
    tool_call_dict,
)
from vime.agent.adapters.openai import _arguments_as_dict, _tools_to_chat_tools
from vime.agent.parsing import ParsedModelOutput
from vime.agent.reasoning import extract_responses_reasoning, pack_reasoning
from vime.agent.turn_disposition import (
    DispositionUpdate,
    TurnDispositionTracker,
    observe_codex_request,
)

logger = logging.getLogger(__name__)

_APPLY_PATCH_TOOL_NAME = "apply_patch"
_JSON_APPLY_PATCH_DESCRIPTION = (
    "Edit files using a complete patch. Supply the tool arguments as a JSON object "
    "with a required string field `patch`. The string must contain the complete "
    "patch text, including `*** Begin Patch` and `*** End Patch`. Do not include "
    "Markdown fences or an additional JSON wrapper inside the patch string. "
    "The adapter forwards the string as the input to Codex's freeform apply_patch tool."
)
_APPLY_PATCH_INSTRUCTION_PREFIX = (
    "Use the `apply_patch` tool to edit files "
    "(NEVER try `applypatch` or `apply-patch`, only `apply_patch`). "
)
_FREEFORM_APPLY_PATCH_INSTRUCTION = (
    _APPLY_PATCH_INSTRUCTION_PREFIX + "This is a FREEFORM tool, so do not wrap the patch in JSON."
)
_JSON_APPLY_PATCH_INSTRUCTION = (
    _APPLY_PATCH_INSTRUCTION_PREFIX
    + "For `apply_patch`, supply the tool arguments as a JSON object with a required "
    "string field `patch`. The string must contain the complete patch text, "
    "including `*** Begin Patch` and `*** End Patch`, without Markdown fences "
    "or an additional JSON wrapper inside the string. The adapter forwards the "
    "string as Codex's freeform tool input."
)
_APPLY_PATCH_GUIDE_ANCHOR = (
    "## apply_patch\n\n"
    "Use the `apply_patch` tool to edit files. Your patch language is a "
    "stripped‑down, file‑oriented diff format designed to be easy to parse and "
    "safe to apply. You can think of it as a high‑level envelope:\n\n"
)
_APPLY_PATCH_HUNK_CLARIFICATION = (
    "Inside the JSON `patch` string, use Codex's apply_patch format. For an update "
    "hunk, use a bare `@@` or `@@ ` followed by an exact source-code context line. "
    "Do not use unified-diff line-number headers such as `@@ -15,7 +15,7 @@`. "
    "Text after `@@ ` is interpreted as source context to find, not line numbers. "
    "Copy context from the actual file; do not invent it. Prefix unchanged lines "
    "with a space, removed lines with `-`, and added lines with `+`.\n\n"
    "Incorrect: `@@ -15,7 +15,7 @@ def add_parser(subparsers):`\n\n"
    "Correct: `@@ def add_parser(subparsers):`"
)
_CLARIFIED_APPLY_PATCH_GUIDE_ANCHOR = _APPLY_PATCH_GUIDE_ANCHOR.replace(
    "## apply_patch\n\n", "## apply_patch\n\n" + _APPLY_PATCH_HUNK_CLARIFICATION + "\n\n", 1
)


class ResponsesAdapter(BaseAdapter):
    """OpenAI ``/v1/responses`` adapter used by current Codex releases."""

    logger = logger
    log_prefix = "responses_adapter"
    max_token_keys = ("max_output_tokens",)
    stop_keys: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs) -> None:
        self.turn_dispositions = TurnDispositionTracker()
        super().__init__(*args, **kwargs)

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/responses", self._run_turn)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return sid_from_bearer(request) or sid_from_body(body) or "default"

    def _observe_request(self, sid: str, body: dict) -> dict[str, Any]:
        observation, updates = self.turn_dispositions.observe_request(sid, body)
        self._apply_disposition_updates(updates)
        return observation.metadata()

    def _on_turn_recorded(self, sid, turn_index, body, turn, request_metadata) -> None:
        observation = observe_codex_request(body)
        update = self.turn_dispositions.record_completion(
            sid,
            turn_index,
            observation,
            turn.finish_reason,
        )
        self._apply_disposition_updates([update])

    def _finalize_turn_dispositions(self, sid: str) -> None:
        self._apply_disposition_updates(self.turn_dispositions.finalize_session(sid))
        self.turn_dispositions.clear_session(sid)

    def _clear_turn_dispositions(self, sid: str) -> None:
        self.turn_dispositions.clear_session(sid)

    def _apply_disposition_updates(self, updates: list[DispositionUpdate]) -> None:
        for update in updates:
            found = self.manager.set_turn_disposition(
                update.session_id,
                update.turn_index,
                disposition=update.disposition,
                reason=update.reason,
                trainable=update.trainable,
            )
            if not found:
                logger.warning(
                    "[%s] sid=%s turn=%s disposition target not found",
                    self.log_prefix,
                    update.session_id,
                    update.turn_index,
                )

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        instructions = body.get("instructions")
        if self.model_profile.normalize_apply_patch_description:
            instructions = _normalize_apply_patch_instructions(instructions)
        translated = _translate_input(body.get("input"), instructions)
        if self.model_profile.coalesce_system_messages:
            translated = _coalesce_system_messages(translated)
        tools_schema = _responses_tools_to_chat_tools(
            body.get("tools"),
            normalize_apply_patch_description=self.model_profile.normalize_apply_patch_description,
        )
        return translated, tools_schema

    def _build_reply(self, parsed, raw_finish, translated, tools_schema, body) -> Reply:
        visible_text = _trim_terminal_special_token(parsed.text, self.tokenizer)
        output, manager_message = _build_reply_parts(
            parsed,
            visible_text=visible_text,
            custom_tool_names=_responses_custom_tool_names(body.get("tools")),
        )
        return Reply(
            manager_message=manager_message,
            finish_reason=manager_finish_reason(parsed.tool_uses, raw_finish),
            wire={"output": output, "raw_finish": raw_finish},
        )

    async def _prepare_stream_response(self, request, body, in_tok) -> web.StreamResponse:
        return await _prepare_stream(request, body)

    async def _write_stream_heartbeat(self, response: web.StreamResponse) -> None:
        # SSE comments are protocol-valid keepalives and are ignored by clients.
        await response.write(b": ping\n\n")

    async def _fail_stream_response(self, response: web.StreamResponse) -> None:
        # Headers are already sent; report a protocol error without exposing
        # upstream exception details or recording a successful completion.
        sequence = int(getattr(response, "_vime_sequence_number"))
        await _emit(
            response,
            "error",
            {
                "code": "server_error",
                "message": "Model turn failed; see adapter logs.",
                "param": None,
                "sequence_number": sequence,
            },
        )
        setattr(response, "_vime_sequence_number", sequence + 1)
        await response.write_eof()

    async def _finish_stream_response(self, response, body, reply, in_tok, out_tok) -> None:
        await _finish_stream(response, body, reply.wire, in_tok, out_tok)

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        if stream:
            out = await _prepare_stream(request, body)
            await _finish_stream(out, body, reply.wire, in_tok, out_tok)
            return out
        data = _render_response(body, reply.wire, in_tok, out_tok)
        out = web.json_response(data)
        setattr(out, "_vime_debug_response", data)
        return out


# --- Translation (Responses input -> chat-template messages) ----------------


def _normalize_apply_patch_instructions(instructions: str | None) -> str | None:
    """Normalize known apply-patch instructions in a model-facing copy.

    The name warning anchors the replacement to apply_patch, leaving other
    freeform tools untouched. Clarify only the known patch guide, preserving
    its original text and examples. This
    also applies to repeated instructions on requests without tools (compaction).
    """
    if not isinstance(instructions, str) or not instructions:
        return instructions
    if _FREEFORM_APPLY_PATCH_INSTRUCTION in instructions:
        instructions = instructions.replace(_FREEFORM_APPLY_PATCH_INSTRUCTION, _JSON_APPLY_PATCH_INSTRUCTION)
    elif _JSON_APPLY_PATCH_INSTRUCTION not in instructions:
        logger.warning(
            "[responses_adapter] expected apply_patch instruction paragraph not found; "
            "that paragraph left unchanged"
        )
    instructions, guide_count = re.subn(
        r"(?m)^" + re.escape(_APPLY_PATCH_GUIDE_ANCHOR),
        lambda _: _CLARIFIED_APPLY_PATCH_GUIDE_ANCHOR,
        instructions,
    )
    if not guide_count and not re.search(r"(?m)^" + re.escape(_CLARIFIED_APPLY_PATCH_GUIDE_ANCHOR), instructions):
        logger.warning(
            "[responses_adapter] expected apply_patch guide not found; hunk clarification not inserted"
        )
    return instructions


def _responses_tools_to_chat_tools(
    tools: list[dict] | None,
    *,
    normalize_apply_patch_description: bool = False,
) -> list[dict] | None:
    """Translate Responses tools to the served model's function-tool schema.

    Codex 0.153 exposes ``apply_patch`` as a grammar-constrained freeform
    custom tool. Qwen's chat template and vLLM parser consume JSON function
    tools, so the adapter presents an internal ``patch`` string parameter and
    converts it back to freeform text on the Responses wire.
    """
    if not tools:
        return None
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "custom" and tool.get("name") == _APPLY_PATCH_TOOL_NAME:
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": _APPLY_PATCH_TOOL_NAME,
                        "description": (
                            _JSON_APPLY_PATCH_DESCRIPTION
                            if normalize_apply_patch_description
                            else (
                                f"{tool.get('description', '')}\n"
                                "Return the complete patch, including *** Begin Patch and "
                                "*** End Patch markers, in the patch field."
                            ).strip()
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "patch": {
                                    "type": "string",
                                    "description": "Complete apply_patch freeform payload.",
                                }
                            },
                            "required": ["patch"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
            continue
        ordinary = _tools_to_chat_tools([tool])
        if ordinary:
            normalized.extend(ordinary)
    return normalized or None


def _responses_custom_tool_names(tools: list[dict] | None) -> set[str]:
    if not tools:
        return set()
    return {
        str(tool["name"])
        for tool in tools
        if isinstance(tool, dict)
        and tool.get("type") == "custom"
        and tool.get("name") == _APPLY_PATCH_TOOL_NAME
    }


def _response_content_text(content: Any) -> str:
    """Flatten Responses message/function output content into plain text."""
    if not isinstance(content, list):
        return flatten_content(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "refusal":
            parts.append(str(block.get("refusal") or ""))
        else:
            text = flatten_content([block])
            if text:
                parts.append(text)
    return "\n".join(part for part in parts if part)


def _coalesce_system_messages(messages: list[dict]) -> list[dict]:
    """Fit system/developer instructions into a template with one system slot.

    _translate_input has already mapped developer to system. Preserve their
    text and relative order, including instructions introduced mid-history.
    Moving those later instructions to the front changes the prompt prefix;
    the trajectory manager must see this translated form as well as the
    tokenizer so it can track the resulting branch correctly.
    """
    system_messages = [message for message in messages if message["role"] == "system"]
    if not system_messages:
        return messages
    return [
        {"role": "system", "content": "\n\n".join(message["content"] for message in system_messages)},
        *(message for message in messages if message["role"] != "system"),
    ]


def _translate_input(input_value: Any, instructions: Any = None) -> list[dict]:
    """Translate a Responses ``input`` value into canonical chat messages.

    Consecutive assistant ``message`` and tool-call output items are folded into
    one canonical assistant message.  This is important for the trajectory
    manager: the representation replayed by Codex on the following turn must
    equal the assistant leaf recorded for the generated turn.
    """
    translated: list[dict[str, Any]] = []
    if instructions:
        translated.append({"role": "system", "content": flatten_content(instructions)})

    if isinstance(input_value, str):
        return [*translated, {"role": "user", "content": input_value}]
    if not isinstance(input_value, list):
        raise web.HTTPBadRequest(text="input must be a string or a list")

    pending_assistant: dict[str, Any] | None = None
    pending_reasoning = ""

    def flush_assistant() -> None:
        nonlocal pending_assistant, pending_reasoning
        if pending_assistant is None and pending_reasoning:
            pending_assistant = {
                "role": "assistant",
                "content": "",
                "reasoning_content": pending_reasoning,
            }
            pending_reasoning = ""
        if pending_assistant is not None:
            translated.append(pending_assistant)
            pending_assistant = None

    def assistant_message() -> dict[str, Any]:
        nonlocal pending_assistant, pending_reasoning
        if pending_assistant is None:
            pending_assistant = {"role": "assistant", "content": ""}
        if pending_reasoning:
            prior = pending_assistant.get("reasoning_content") or ""
            pending_assistant["reasoning_content"] = (
                f"{prior}\n{pending_reasoning}" if prior else pending_reasoning
            )
            pending_reasoning = ""
        return pending_assistant

    for item in input_value:
        if isinstance(item, str):
            flush_assistant()
            translated.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "message" or (item_type is None and item.get("role")):
            role = item.get("role")
            text = _response_content_text(item.get("content"))
            if role == "assistant":
                assistant = assistant_message()
                if text:
                    prior = assistant.get("content") or ""
                    assistant["content"] = f"{prior}\n{text}" if prior else text
            elif role in {"system", "developer", "user", "tool"}:
                flush_assistant()
                translated.append(
                    {
                        "role": "system" if role == "developer" else role,
                        "content": text,
                    }
                )
            continue

        if item_type in {"function_call", "custom_tool_call"}:
            assistant = assistant_message()
            name = item.get("name") or "tool"
            arguments = item.get("arguments")
            if item_type == "custom_tool_call":
                argument_name = "patch" if name == _APPLY_PATCH_TOOL_NAME else "_raw_arguments"
                arguments = {argument_name: item.get("input") or ""}
            assistant.setdefault("tool_calls", []).append(
                tool_call_dict(
                    name,
                    _arguments_as_dict(arguments),
                )
            )
            continue

        if item_type == "reasoning":
            # Reasoning is hidden from visible content but must remain attached
            # to the same canonical assistant/tool-call turn. Otherwise the
            # next chat-template render silently changes the model history.
            if pending_assistant is not None and (
                pending_assistant.get("content") or pending_assistant.get("tool_calls")
            ):
                flush_assistant()
            reasoning = extract_responses_reasoning(item)
            if reasoning:
                pending_reasoning = f"{pending_reasoning}\n{reasoning}" if pending_reasoning else reasoning
            continue

        flush_assistant()
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            translated.append(
                {
                    "role": "tool",
                    "content": _response_content_text(item.get("output")),
                }
            )

    flush_assistant()
    return translated


# --- Reply building (parsed model output -> Responses output items) ---------


def _trim_terminal_special_token(text: str, tokenizer: Any) -> str:
    """Remove a decoded terminal EOS marker from wire-visible text.

    ``TurnRecord.output_ids`` is untouched, so the stop token and its logprob
    remain trainable.  Only the text rendered back to Codex is normalized.
    """
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos_token, str) or not eos_token:
        return text
    cleaned = text.rstrip()
    while cleaned.endswith(eos_token):
        cleaned = cleaned[: -len(eos_token)].rstrip()
    return cleaned


def _build_reply_parts(
    parsed: ParsedModelOutput,
    *,
    visible_text: str | None = None,
    custom_tool_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = parsed.text if visible_text is None else visible_text
    output: list[dict[str, Any]] = []

    if parsed.reasoning:
        output.append(
            {
                "id": f"rs_{secrets.token_hex(12)}",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": parsed.reasoning}],
                "content": [{"type": "reasoning_text", "text": parsed.reasoning}],
                "encrypted_content": pack_reasoning(parsed.reasoning),
            }
        )

    # Responses represents visible assistant text and each function call as
    # separate output items.  Empty text is emitted only when there is no call.
    if text or not parsed.tool_uses:
        output.append(
            {
                "id": f"msg_{secrets.token_hex(12)}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text or "",
                        "annotations": [],
                    }
                ],
            }
        )

    manager_message: dict[str, Any] = {"role": "assistant", "content": text or ""}
    if parsed.reasoning:
        manager_message["reasoning_content"] = parsed.reasoning
    manager_calls: list[dict[str, Any]] = []
    custom_tool_names = custom_tool_names or set()
    for tool_use in parsed.tool_uses:
        name = tool_use.get("name") or "tool"
        arguments = tool_use.get("input") or {}
        if not isinstance(arguments, dict):
            arguments = {"_raw_arguments": str(arguments)}
        call_id = f"call_{secrets.token_hex(12)}"
        if name in custom_tool_names:
            patch = arguments.get("patch")
            if not isinstance(patch, str):
                patch = arguments.get("_raw_arguments")
            if not isinstance(patch, str):
                patch = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
            output.append(
                {
                    "id": f"ct_{secrets.token_hex(12)}",
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "input": patch,
                }
            )
        else:
            output.append(
                {
                    "id": f"fc_{secrets.token_hex(12)}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                }
            )
        manager_calls.append(tool_call_dict(name, arguments))
    if manager_calls:
        manager_message["tool_calls"] = manager_calls

    return output, manager_message


# --- Response object and SSE framing ---------------------------------------


def _usage(in_tok: int, out_tok: int) -> dict[str, Any]:
    return {
        "input_tokens": in_tok,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": out_tok,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": in_tok + out_tok,
    }


def _response_object(
    body: dict,
    *,
    response_id: str,
    created_at: int,
    status: str,
    output: list[dict[str, Any]],
    in_tok: int | None,
    out_tok: int | None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    usage = None if in_tok is None or out_tok is None else _usage(in_tok, out_tok)
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "completed_at": int(time.time()) if status in {"completed", "incomplete"} else None,
        "error": None,
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "model": body.get("model", "vime-actor"),
        "output": output,
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "previous_response_id": body.get("previous_response_id"),
        "reasoning": body.get("reasoning") or {"effort": None, "summary": None},
        "store": body.get("store", False),
        "temperature": body.get("temperature", 1),
        "text": body.get("text") or {"format": {"type": "text"}},
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools") or [],
        "top_p": body.get("top_p", 1),
        "truncation": body.get("truncation", "disabled"),
        "usage": usage,
        "metadata": body.get("metadata") or {},
    }


def _completion_state(wire: dict[str, Any]) -> tuple[str, str | None]:
    if wire.get("raw_finish") == "length":
        return "incomplete", "max_output_tokens"
    return "completed", None


def _render_response(body: dict, wire: dict[str, Any], in_tok: int, out_tok: int) -> dict[str, Any]:
    status, reason = _completion_state(wire)
    return _response_object(
        body,
        response_id=f"resp_{secrets.token_hex(16)}",
        created_at=int(time.time()),
        status=status,
        output=wire["output"],
        in_tok=in_tok,
        out_tok=out_tok,
        incomplete_reason=reason,
    )


async def _emit(out: web.StreamResponse, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = {"type": event_type, **payload}
    await out.write(
        f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()
    )
    return data


async def _prepare_stream(request: web.Request, body: dict) -> web.StreamResponse:
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)
    response_id = f"resp_{secrets.token_hex(16)}"
    created_at = int(time.time())
    setattr(out, "_vime_response_id", response_id)
    setattr(out, "_vime_created_at", created_at)
    setattr(out, "_vime_sequence_number", 0)
    response = _response_object(
        body,
        response_id=response_id,
        created_at=created_at,
        status="in_progress",
        output=[],
        in_tok=None,
        out_tok=None,
    )
    debug_events = [
        await _emit(out, "response.created", {"response": response, "sequence_number": 0}),
        await _emit(out, "response.in_progress", {"response": response, "sequence_number": 1}),
    ]
    setattr(out, "_vime_debug_events", debug_events)
    setattr(out, "_vime_sequence_number", 2)
    return out


async def _finish_stream(
    out: web.StreamResponse,
    body: dict,
    wire: dict[str, Any],
    in_tok: int,
    out_tok: int,
) -> None:
    sequence = int(getattr(out, "_vime_sequence_number"))

    async def emit(event_type: str, payload: dict[str, Any]) -> None:
        nonlocal sequence
        event = await _emit(out, event_type, {**payload, "sequence_number": sequence})
        getattr(out, "_vime_debug_events", []).append(event)
        sequence += 1
        setattr(out, "_vime_sequence_number", sequence)

    for output_index, complete_item in enumerate(wire["output"]):
        item_type = complete_item["type"]
        if item_type == "message":
            added_item = {**complete_item, "status": "in_progress", "content": []}
            await emit("response.output_item.added", {"output_index": output_index, "item": added_item})
            part = complete_item["content"][0]
            empty_part = {**part, "text": ""}
            await emit(
                "response.content_part.added",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "part": empty_part,
                },
            )
            await emit(
                "response.output_text.delta",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": part["text"],
                },
            )
            await emit(
                "response.output_text.done",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "text": part["text"],
                },
            )
            await emit(
                "response.content_part.done",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "part": part,
                },
            )
        elif item_type == "reasoning":
            added_item = {**complete_item, "status": "in_progress", "summary": [], "content": []}
            await emit("response.output_item.added", {"output_index": output_index, "item": added_item})
            part = complete_item["summary"][0]
            await emit(
                "response.reasoning_summary_part.added",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {**part, "text": ""},
                },
            )
            await emit(
                "response.reasoning_summary_text.delta",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "summary_index": 0,
                    "delta": part["text"],
                },
            )
            await emit(
                "response.reasoning_summary_text.done",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "summary_index": 0,
                    "text": part["text"],
                },
            )
            await emit(
                "response.reasoning_summary_part.done",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": part,
                },
            )
        elif item_type == "function_call":
            added_item = {**complete_item, "status": "in_progress", "arguments": ""}
            await emit("response.output_item.added", {"output_index": output_index, "item": added_item})
            await emit(
                "response.function_call_arguments.delta",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "delta": complete_item["arguments"],
                },
            )
            await emit(
                "response.function_call_arguments.done",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "name": complete_item["name"],
                    "arguments": complete_item["arguments"],
                },
            )
        elif item_type == "custom_tool_call":
            added_item = {**complete_item, "status": "in_progress", "input": ""}
            await emit("response.output_item.added", {"output_index": output_index, "item": added_item})
            await emit(
                "response.custom_tool_call_input.delta",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "delta": complete_item["input"],
                },
            )
            await emit(
                "response.custom_tool_call_input.done",
                {
                    "item_id": complete_item["id"],
                    "output_index": output_index,
                    "input": complete_item["input"],
                },
            )
        await emit(
            "response.output_item.done",
            {"output_index": output_index, "item": complete_item},
        )

    status, reason = _completion_state(wire)
    completed = _response_object(
        body,
        response_id=str(getattr(out, "_vime_response_id")),
        created_at=int(getattr(out, "_vime_created_at")),
        status=status,
        output=wire["output"],
        in_tok=in_tok,
        out_tok=out_tok,
        incomplete_reason=reason,
    )
    await emit(f"response.{status}", {"response": completed})
    setattr(out, "_vime_sequence_number", sequence)
    setattr(out, "_vime_debug_response", completed)
