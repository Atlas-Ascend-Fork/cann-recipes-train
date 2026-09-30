"""Codex harness.

Two non-obvious bits: the provider base_url must be inline in the TOML (Codex
only honours env vars for the default OpenAI provider), and the config is written
via a base64 round-trip to dodge shell-quoting traps.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
from pathlib import Path

from vime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, install_npm_cli, run_agent


class CodexHarness(BaseHarness):
    name = "codex"

    # host paths + CLI knobs, all under the agent-layer VIME_AGENT_* prefix
    node_tarball_env = "VIME_AGENT_NODE_TARBALL"
    cli_tarball_env = "VIME_AGENT_CODEX_TARBALL"
    extra_args_env = "VIME_AGENT_CODEX_EXTRA_ARGS"
    extra_envs_env = "VIME_AGENT_CODEX_EXTRA_ENVS"
    context_window_env = "VIME_AGENT_MAX_CONTEXT_TOKENS"
    context_reserve_tokens_env = "VIME_AGENT_CODEX_CONTEXT_RESERVE_TOKENS"
    auto_compact_tokens_env = "VIME_AGENT_CODEX_AUTO_COMPACT_TOKENS"
    model_profile_env = "VIME_AGENT_CODEX_MODEL_PROFILE"

    # Codex 0.153.0 decides which built-in tools to expose from its model
    # metadata.  The generic ``vime-actor`` fallback has no apply_patch tool,
    # while this bundled profile advertises the native freeform tool.  This is
    # only a CLI capability profile: the adapter still serves the model loaded
    # by vLLM and intentionally ignores the request's model field.
    default_model_profile = "gpt-5.2-codex"

    # SWE rollouts must be able to edit the checked-out workspace. Codex
    # otherwise defaults to read-only in non-interactive execution.
    # --skip-git-repo-check also lets it run in brittle/shallow workdirs.
    exec_flags = "--sandbox workspace-write --skip-git-repo-check"

    # config.toml written into the sandbox. base_url MUST be inline here (Codex
    # only honours env vars for the default OpenAI provider). {model} / {base_url}
    # are filled per run in write_config; the rest is fixed wiring.
    config_toml = (
        'model = "{model}"\n'
        'model_provider = "vime"\n'
        "\n"
        "[model_providers.vime]\n"
        'name = "vime"\n'
        'base_url = "{base_url}"\n'
        'env_key = "OPENAI_API_KEY"\n'
        'wire_api = "responses"\n'
        "\n"
        "[features]\n"
        "apply_patch_freeform = true\n"
    )

    async def install_cli(self, sb: Sandbox) -> None:
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ[self.node_tarball_env]),
            npm_package=Path(os.environ[self.cli_tarball_env]),
            check_cmd="codex --version",
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        context_window = int(ctx.max_context_tokens or os.environ.get(self.context_window_env, "0"))
        context_config = ""
        if context_window > 0:
            configured_reserve = os.environ.get(self.context_reserve_tokens_env, "").strip()
            context_reserve = (
                int(configured_reserve)
                if configured_reserve
                else min(1024, max(1, context_window // 10))
            )
            if not 0 <= context_reserve < context_window:
                raise ValueError(
                    f"{self.context_reserve_tokens_env} must be between 0 and "
                    f"model context - 1 ({context_window - 1}), got {context_reserve}"
                )
            advertised_context = context_window - context_reserve
            configured_compact = os.environ.get(self.auto_compact_tokens_env, "").strip()
            auto_compact = (
                int(configured_compact)
                if configured_compact
                else max(1, advertised_context * 80 // 100)
            )
            if not 0 < auto_compact < advertised_context:
                raise ValueError(
                    f"{self.auto_compact_tokens_env} must be between 1 and "
                    f"advertised model context - 1 ({advertised_context - 1}), got {auto_compact}"
                )
            context_config = (
                f"model_context_window = {advertised_context}\n"
                f"model_auto_compact_token_limit = {auto_compact}\n"
            )
        model_profile = (
            os.environ.get(self.model_profile_env, "").strip()
            or self.default_model_profile
        )
        toml = context_config + self.config_toml.format(
            model=model_profile,
            base_url=f"{ctx.adapter_url}/v1",
        )
        toml_b64 = base64.b64encode(toml.encode("utf-8")).decode("ascii")
        await sb.exec(
            "mkdir -p /home/agent/.codex && "
            # base64 round-trip avoids any single-quote / heredoc shell-quoting trap
            f"echo {shlex.quote(toml_b64)} | base64 -d > /home/agent/.codex/config.toml && "
            "chown -R agent:agent /home/agent/.codex",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        # ``codex exec`` is the non-interactive entrypoint
        cmd = f"codex exec {self.exec_flags} {shlex.quote(prompt)}"
        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"
        env = {
            # Codex propagates OPENAI_API_KEY into Authorization: Bearer; the
            # vime adapter resolves the sid from that header.
            "OPENAI_API_KEY": ctx.session_id,
            "OPENAI_BASE_URL": f"{ctx.adapter_url}/v1",
        }
        # extra env vars as a JSON object, merged last so callers can override
        # the defaults above
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))
        return await run_agent(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)
