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

"""Terminal-Agent RL: per-sample generate() function for vime.

    --custom-generate-function-path examples.terminal_agent_rl.generate.generate

generate() is a three-stage orchestrator: terminalbench.prepare_workspace +
harness.run -> terminalbench.run_evaluation (same sandbox) ->
adapter.finish_session. The (harness, adapter) pair is chosen by VIME_RL_AGENT
(legacy: TB_AGENT; claude_code | codex); see _AGENTS below.
Sandbox-side work is split across three layers: the provider-agnostic sandbox
contract (vime.agent.sandbox), the swappable harness lifecycle
(vime.agent.harness), and the Terminal-Bench task layer
(examples.terminal_agent_rl.terminalbench -- dataset parsing, workspace prep,
oracle grading). LLM plumbing (Anthropic / Responses <-> vLLM
``/inference/v1/generate``, token capture, segment split) is the matching
vime.agent.adapters adapter. terminalbench.get_metadata documents the dataset
row schema and produces the md dict consumed below.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from vime.agent.adapters import AnthropicAdapter, ResponsesAdapter
from vime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from vime.agent.config import bundle_path, effective_sandbox_env, rl_value, sandbox_backend, validate_sandbox_config
from vime.agent.harness import ClaudeCodeHarness, CodexHarness
from vime.agent.model_profile import (
    ModelProfile,
    load_checkpoint_model_type,
    uses_qwen3_apply_patch_policy,
    uses_single_system_message_policy,
)
from vime.agent.sandbox import E2BSandbox, Sandbox, SandboxAdmissionTimeout, sandbox_preparation
from vime.utils.misc import SingletonMeta
from vime.utils.processing_utils import load_tokenizer
from vime.utils.types import Sample


from . import terminalbench
from .sandbox import create_sandbox
from .template_mapping import TemplateMappingError, apply_template_mapping, load_template_mapping
from .verifier_spec import VerifierSpecError, apply_verifier_spec, load_verifier_specs

logger = logging.getLogger(__name__)
logging.getLogger("e2b").setLevel(logging.WARNING)

_AGENTS = {
    "claude_code": (ClaudeCodeHarness, AnthropicAdapter),
    "codex": (CodexHarness, ResponsesAdapter),
}
AGENT_NAME = rl_value("terminalbench", "AGENT", "claude_code")
if AGENT_NAME not in _AGENTS:
    raise ValueError(f"VIME_RL_AGENT={AGENT_NAME!r} not in {sorted(_AGENTS)}")
HARNESS_CLS, ADAPTER_CLS = _AGENTS[AGENT_NAME]


@dataclass(frozen=True)
class TbConfig:
    adapter_public_host: str | None
    adapter_public_port: int | None
    adapter_bind_host: str
    adapter_port: int
    max_turns_per_session: int | None
    fork_merge_threshold: int | None
    agent_time_budget_sec: int
    eval_timeout_sec: int
    rollout_guard_sec: int
    boot_concurrency: int
    boot_retries: int
    agentenv_template_map: str | None
    verifier_spec_map: str | None

    @classmethod
    def from_env(cls) -> TbConfig:
        agent_time_budget = int(rl_value("terminalbench", "AGENT_TIME_BUDGET_SEC", "1800"))
        eval_timeout = int(rl_value("terminalbench", "EVAL_TIMEOUT_SEC", "900"))
        guard = int(rl_value("terminalbench", "ROLLOUT_GUARD_SEC", "0") or 0) or (
            agent_time_budget + eval_timeout + 180
        )
        fork = int(v) if (v := os.environ.get("VIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        selector_mode = effective_sandbox_env().get(
            "VIME_AGENT_E2B_SELECTOR_MODE", E2BSandbox.default_selector_mode
        ).strip()
        template_map = (bundle_path("terminalbench", "environment") or "").strip() or None
        verifier_map = (bundle_path("terminalbench", "verifier") or "").strip() or None
        public_port = os.environ.get("ADAPTER_PUBLIC_PORT", "").strip()
        max_turns = os.environ.get("VIME_AGENT_MAX_TURNS_PER_SESSION", "").strip()
        if sandbox_backend() == "agentenv" and template_map and selector_mode != "template":
            raise ValueError(
                "A prepared environment mapping requires AgentENV selector_mode=template; "
                f"got {selector_mode!r}"
            )
        return cls(
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_public_port=int(public_port) if public_port else None,
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            max_turns_per_session=int(max_turns) if max_turns else None,
            fork_merge_threshold=fork,
            agent_time_budget_sec=agent_time_budget,
            eval_timeout_sec=eval_timeout,
            rollout_guard_sec=guard,
            boot_concurrency=int(rl_value("terminalbench", "BOOT_CONCURRENCY", "16")),
            boot_retries=int(rl_value("terminalbench", "BOOT_RETRIES", "2")),
            agentenv_template_map=template_map,
            verifier_spec_map=verifier_map,
        )


validate_sandbox_config()
CONFIG = TbConfig.from_env()
TEMPLATE_MAPPING = load_template_mapping(CONFIG.agentenv_template_map) if CONFIG.agentenv_template_map else None
VERIFIER_SPECS = load_verifier_specs(CONFIG.verifier_spec_map) if CONFIG.verifier_spec_map else None

_BOOT_SEM = asyncio.Semaphore(CONFIG.boot_concurrency)


@asynccontextmanager
async def boot_agent_sandbox(image: str, instance_id: str, md: dict) -> AsyncIterator[Sandbox]:
    """Boot a fresh sandbox and prepare its toolchain and task workspace.

    Create the sandbox from the dataset image, install Node 22 + the harness CLI
    from host tarballs and prepare ownership under the preparation limit. Retry
    failures in a fresh sandbox, never by replaying a possibly executed command.
    """
    sb = None
    last_err: Exception | None = None
    for attempt in range(CONFIG.boot_retries):
        cand = create_sandbox(image)
        try:
            async with _BOOT_SEM:
                try:
                    async with sandbox_preparation(cand):
                        await cand.__aenter__()
                        await HARNESS_CLS().install_cli(cand)
                        await terminalbench.prepare_workspace(cand, md["workdir"], md)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except SandboxAdmissionTimeout:
            # Capacity waiting retains its FIFO ticket and original deadline.
            # Expiry is explicit; recreating would reset its place and budget.
            raise
        except Exception as e:
            last_err = e
            logger.warning(
                "[terminal_agent_rl] %s: provision attempt %d/%d failed: %s: %s",
                instance_id,
                attempt + 1,
                CONFIG.boot_retries,
                type(e).__name__,
                str(e)[:200],
            )
            await asyncio.sleep(1 + attempt + random.random())
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "vllm_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "vllm_reasoning_parser", None) or None
        checkpoint_model_type = load_checkpoint_model_type(str(args.hf_checkpoint))
        self.model_profile = ModelProfile.from_runtime(
            name=str(args.hf_checkpoint),
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            normalize_apply_patch_description=uses_qwen3_apply_patch_policy(checkpoint_model_type),
            coalesce_system_messages=uses_single_system_message_policy(checkpoint_model_type),
        )
        logger.info(
            "[terminal_agent_rl] checkpoint_model_type=%s normalize_apply_patch_description=%s "
            "coalesce_system_messages=%s",
            checkpoint_model_type,
            self.model_profile.normalize_apply_patch_description,
            self.model_profile.coalesce_system_messages,
        )
        vllm_url = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"
        if not CONFIG.adapter_public_host:
            raise RuntimeError(
                "ADAPTER_PUBLIC_HOST is not set. Export it to the host IP that "
                "sandboxes can reach for reverse-connection to the adapter; "
                "without it the sandbox cannot dial back and the rollout aborts."
            )
        # model_profile carries tool/reasoning parser selection (built via
        # ModelProfile.from_runtime); passing the parsers again is rejected by
        # the adapter base class. Same construction as swe_agent_rl.
        self.adapter = ADAPTER_CLS(
            tokenizer=self.tokenizer,
            vllm_url=vllm_url,
            model_profile=self.model_profile,
            fork_threshold_tokens=CONFIG.fork_merge_threshold,
            max_turns_per_sid=CONFIG.max_turns_per_session,
        )
        # handler_cancellation=True so a client disconnect cancels the handler
        # coroutine, tearing down the in-flight engine ``/inference/v1/generate``
        # request. Without it a cancelled client leaves an inflight generate that
        # races with the next release_memory_occupation.
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name="anthropic-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        public_port = CONFIG.adapter_public_port or self.app_handle.port
        self.adapter_url = f"http://{CONFIG.adapter_public_host}:{public_port}"
        logger.info(
            "[terminal_agent_rl] tokenizer=%s adapter=%s max_context_len=%s tool_parser=%s reasoning_parser=%s",
            args.hf_checkpoint,
            self.adapter_url,
            self.max_context_len,
            self.tool_parser,
            self.reasoning_parser,
        )


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Run a terminal task, grade its final environment, and finalize segments."""
    md = terminalbench.get_metadata(base_sample)
    task_id = md["task_id"]
    try:
        if TEMPLATE_MAPPING is None or VERIFIER_SPECS is None:
            return _abort_result(base_sample, "missing_prepared_bundle_maps", task_id)
        md = apply_template_mapping(md, TEMPLATE_MAPPING)
        md = apply_verifier_spec(md, VERIFIER_SPECS)
    except (TemplateMappingError, VerifierSpecError) as error:
        logger.warning("[terminal_agent_rl] %s: %s", task_id, error)
        return _abort_result(base_sample, "invalid_prepared_task", task_id)
    reason = terminalbench.evaluability_check(md)
    if reason:
        return _abort_result(base_sample, reason, task_id)

    state = _AdapterService(args)
    session_id = base_sample.session_id = _session_id(base_sample, task_id)
    state.adapter.open_session(session_id, sampling_defaults=sampling_params,
                               max_context_tokens=state.max_context_len)
    try:
        async with asyncio.timeout(CONFIG.rollout_guard_sec):
            async with boot_agent_sandbox(md["image"], task_id, md) as sb:
                agent_exit_code = await HARNESS_CLS().run(
                    sb, workdir=md["workdir"], session_id=session_id,
                    adapter_url=state.adapter_url, time_budget_sec=CONFIG.agent_time_budget_sec,
                    prompt=terminalbench.TB_PROMPT,
                )
                result = await terminalbench.run_evaluation(sb, md, timeout_sec=CONFIG.eval_timeout_sec)
            metadata = {
                "task_id": task_id, "verifier_spec_id": md.get("verifier_spec_id"),
                "verifier_status": result.status, "verifier_infra_failure": result.infra_failure,
                "verifier_reason": result.reason, "grading_solved": result.reward == 1.0,
                "agent_exit_code": agent_exit_code,
            }
            if evaluation:
                if result.infra_failure:
                    return _abort_result(base_sample, f"verifier_infra_error:{result.reason}", task_id)
                return _eval_result(base_sample, result.reward, metadata)
            samples = await state.adapter.finish_session(
                session_id, base_sample=base_sample, reward=result.reward, extra_metadata=metadata,
            )
            if not samples:
                return _abort_result(base_sample, "empty_session", task_id)
            if result.infra_failure:
                for sample in samples:
                    sample.reward = 0.0
                    sample.remove_sample = True
                    sample.status = Sample.Status.ABORTED
            return samples
    except TimeoutError:
        return _abort_result(base_sample, "wall_clock_timeout", task_id)
    except Exception as error:
        logger.warning("[terminal_agent_rl] %s: %s", task_id, error, exc_info=True)
        return _abort_result(base_sample, f"exception:{type(error).__name__}", task_id)
    finally:
        await state.adapter.drop_session(session_id, wait_timeout=30)


def _session_id(sample: Sample, task_id: str) -> str:
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"tagent-{task_id}-{sample.index}-{sample.group_index}"
    return f"tagent-{task_id}-{secrets.token_hex(8)}"


def _placeholder(sample: Sample) -> None:
    if sample.rollout_id is None:
        sample.rollout_id = sample.index
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.remove_sample = True


def _abort_result(sample: Sample, reason: str, task_id: str) -> list[Sample]:
    _placeholder(sample)
    sample.reward = 0.0
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason, "task_id": task_id}
    logger.warning("[terminal_agent_rl] %s aborted: %s", task_id, reason)
    return [sample]


def _eval_result(sample: Sample, reward: float, metadata: dict) -> list[Sample]:
    _placeholder(sample)
    sample.reward = float(reward)
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {**(sample.metadata or {}), **metadata}
    return [sample]
