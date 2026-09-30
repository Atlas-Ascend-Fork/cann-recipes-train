"""Anthropic-compatible upstream provider for local CodeAgent smoke tests.

This path deliberately reconstructs tokens and fills logprobs with zeros. It
validates rollout control flow only and must never feed the training entrypoint.
"""

from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import aiohttp

from vime.agent.adapters.common import ProvidedTurn, Reply, _render_token_ids, tool_call_dict
from vime.agent.trajectory import TurnRecord


class AnthropicProxyTurnProvider:
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout_sec: int = 900) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self._capture_lock = asyncio.Lock()
        self._request_index: dict[str, int] = {}

    @classmethod
    def from_env(cls) -> AnthropicProxyTurnProvider:
        required = {
            "VIME_AGENT_UPSTREAM_ANTHROPIC_BASE_URL": os.environ.get("VIME_AGENT_UPSTREAM_ANTHROPIC_BASE_URL"),
            "VIME_AGENT_UPSTREAM_ANTHROPIC_API_KEY": os.environ.get("VIME_AGENT_UPSTREAM_ANTHROPIC_API_KEY"),
            "VIME_AGENT_UPSTREAM_ANTHROPIC_MODEL": os.environ.get("VIME_AGENT_UPSTREAM_ANTHROPIC_MODEL"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError("Missing Anthropic proxy settings: " + ", ".join(missing))
        return cls(
            base_url=str(required["VIME_AGENT_UPSTREAM_ANTHROPIC_BASE_URL"]),
            api_key=str(required["VIME_AGENT_UPSTREAM_ANTHROPIC_API_KEY"]),
            model=str(required["VIME_AGENT_UPSTREAM_ANTHROPIC_MODEL"]),
            timeout_sec=int(os.environ.get("VIME_AGENT_UPSTREAM_ANTHROPIC_TIMEOUT_SEC", "900")),
        )

    @property
    def messages_url(self) -> str:
        if self.base_url.endswith("/v1/messages"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    async def __call__(
        self,
        *,
        prompt_ids: list[int],
        body: dict,
        adapter,
        translated: list[dict],
        tools_schema: list[dict] | None,
        **_kwargs,
    ) -> ProvidedTurn:
        session_id = str(_kwargs.get("session_id") or "unknown")
        await self._capture_prompt(session_id, prompt_ids, body, translated, tools_schema)
        payload = deepcopy(body)
        payload["model"] = self.model
        payload["stream"] = False
        headers = {
            "x-api-key": self.api_key,
            "authorization": f"Bearer {self.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        try:
            async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session, session.post(
                self.messages_url,
                headers=headers,
                json=payload,
            ) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"anthropic proxy upstream {response.status}: {text[:400]}")
                data = json.loads(text)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise RuntimeError(f"anthropic proxy request failed: {type(exc).__name__}: {exc}") from exc

        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise RuntimeError("anthropic proxy response is missing a content block list")
        stop_reason = str(data.get("stop_reason") or "end_turn")
        manager_message, manager_finish = _manager_message(blocks, stop_reason)
        output_ids = _retokenize_output(
            adapter.tokenizer,
            prompt_ids=prompt_ids,
            translated=translated,
            tools_schema=tools_schema,
            manager_message=manager_message,
            blocks=blocks,
        )
        turn = TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=output_ids,
            output_log_probs=[0.0] * len(output_ids),
            finish_reason=manager_finish,
        )
        reply = Reply(
            manager_message=manager_message,
            finish_reason=manager_finish,
            wire=(blocks, stop_reason),
        )
        return ProvidedTurn(
            turn=turn,
            reply=reply,
            metadata={"trajectory_source": "retokenized_proxy"},
        )

    async def _capture_prompt(
        self,
        session_id: str,
        prompt_ids: list[int],
        body: dict,
        translated: list[dict],
        tools_schema: list[dict] | None,
    ) -> None:
        """Optionally persist request diagnostics for local smoke analysis.

        The capture contains model-visible request content but never transport
        headers or API credentials. It is disabled unless an explicit path is
        provided and must not be enabled for production training data.
        """
        capture_path = os.environ.get("VIME_AGENT_PROMPT_CAPTURE_PATH", "").strip()
        if not capture_path:
            return
        async with self._capture_lock:
            request_index = self._request_index.get(session_id, 0) + 1
            self._request_index[session_id] = request_index
            record = {
                "session_id": session_id,
                "request_index": request_index,
                "prompt_tokens": len(prompt_ids),
                "model": body.get("model"),
                "system": body.get("system"),
                "messages": body.get("messages") or [],
                "tools": body.get("tools") or [],
                "translated": translated,
                "translated_tools": tools_schema or [],
            }
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            await asyncio.to_thread(_append_capture_line, Path(capture_path), line)


def _append_capture_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(line)


def _manager_message(blocks: list[dict], stop_reason: str) -> tuple[dict[str, Any], str]:
    texts: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(str(block.get("text") or ""))
        elif block.get("type") == "thinking":
            reasoning.append(str(block.get("thinking") or ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(tool_call_dict(str(block.get("name") or "tool"), block.get("input")))
    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = "tool_calls" if tool_calls else ("length" if stop_reason == "max_tokens" else "stop")
    return message, finish


def _retokenize_output(
    tokenizer,
    *,
    prompt_ids: list[int],
    translated: list[dict],
    tools_schema: list[dict] | None,
    manager_message: dict,
    blocks: list[dict],
) -> list[int]:
    full_ids = _render_token_ids(
        [*translated, manager_message],
        tokenizer,
        tools=tools_schema,
        add_generation_prompt=False,
    )
    common = 0
    for left, right in zip(prompt_ids, full_ids, strict=False):
        if left != right:
            break
        common += 1
    output_ids = list(full_ids[common:])
    if output_ids:
        return output_ids
    fallback = json.dumps(blocks, ensure_ascii=False, separators=(",", ":"))
    try:
        encoded = tokenizer.encode(fallback, add_special_tokens=False)
    except TypeError:  # lightweight test tokenizers need not expose HF kwargs
        encoded = tokenizer.encode(fallback)
    return list(encoded)


def get_turn_provider_from_env(agent_name: str):
    backend = os.environ.get("VIME_AGENT_GENERATION_BACKEND", "vllm").strip().lower()
    if backend == "vllm":
        return None
    if backend != "anthropic_proxy":
        raise ValueError(f"Unsupported VIME_AGENT_GENERATION_BACKEND={backend!r}")
    if agent_name != "claude_code":
        raise RuntimeError("anthropic_proxy supports only the Claude Code harness")
    if os.environ.get("VIME_LOCAL_ROLLOUT_SMOKE") != "1":
        raise RuntimeError("anthropic_proxy is smoke-only; set VIME_LOCAL_ROLLOUT_SMOKE=1")
    return AnthropicProxyTurnProvider.from_env()


def reject_synthetic_training_backend() -> None:
    if os.environ.get("VIME_AGENT_GENERATION_BACKEND", "vllm").strip().lower() == "anthropic_proxy":
        raise RuntimeError(
            "VIME_AGENT_GENERATION_BACKEND=anthropic_proxy produces retokenized zero-logprob trajectories "
            "and is not allowed in train.py"
        )
