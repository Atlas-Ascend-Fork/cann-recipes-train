"""Harness-agnostic coding-agent lifecycle in a sandbox.

A harness is a swappable coding agent (Claude Code, Codex, ...). Each one
installs a CLI, writes its own config, and runs the agent against a prompt. The
shared parts (create the agent user, the run skeleton, the launch-detached-and-
poll transport) live here; adding a CLI-style harness means subclassing
BaseHarness and implementing install_cli, write_config and launch_and_wait.
Two module-level helpers cover the common cases: install_npm_cli for
npm-packaged CLIs, and run_agent for the launch-the-agent-to-completion case.

The base knows nothing about the task: run() takes only generic fields
(workdir / session_id / adapter_url / prompt). Task-specific workspace prep and
scoring live in the example layer.
"""

from __future__ import annotations

import asyncio
import lzma
import os
import secrets
import shlex
import shutil
import tempfile
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from vime.agent import sandbox as _sandbox
from vime.agent.sandbox import Sandbox, exec_and_wait
from vime.utils.misc import SingletonMeta


class SingletonABCMeta(ABCMeta, SingletonMeta):
    pass


# In-sandbox retry budget for the npm global install (transient flakes like
# exit 217). Cheaper than a full sandbox recreate by the caller.
NPM_INSTALL_RETRIES = 3
NPM_INSTALL_BACKOFF_SEC = 2.0
NODE_INSTALL_RETRIES = 3


class TaskEnvironmentError(RuntimeError):
    """The configured guest task environment could not be selected."""


@dataclass(frozen=True)
class TaskCondaEnvironment:
    """Optional guest paths; never activate or inspect Conda on the host."""

    init_script: str
    prefix: str

    @classmethod
    def from_env(cls) -> TaskCondaEnvironment | None:
        init_script = os.environ.get("VIME_AGENT_TASK_CONDA_SH", "").strip()
        prefix = os.environ.get("VIME_AGENT_TASK_CONDA_PREFIX", "").strip()
        if not init_script and not prefix:
            return None
        if not init_script or not prefix:
            raise TaskEnvironmentError("Set both VIME_AGENT_TASK_CONDA_SH and VIME_AGENT_TASK_CONDA_PREFIX")
        if any(not value.startswith("/") or "\x00" in value or "\n" in value for value in (init_script, prefix)):
            raise TaskEnvironmentError("Task Conda settings must be absolute, single-line guest paths")
        return cls(init_script, prefix)

    def wrap(self, command: str, failure_marker: str) -> str:
        # Keep setup and CLI in one subshell: activation reaches the CLI, while
        # an exit here cannot bypass exec_and_wait's outer completion marker.
        verify = (
            "import os,sys; from pathlib import Path; "
            "p=Path(sys.argv[1]).resolve(); "
            "sys.exit(0 if Path(sys.prefix).resolve()==p and "
            "Path(sys.executable).resolve()==(p/'bin/python').resolve() and "
            "Path(os.environ.get('CONDA_PREFIX','')).resolve()==p else 1)"
        )
        return (
            "(\n"
            f"if test -r {shlex.quote(self.init_script)} && "
            f"test -x {shlex.quote(self.prefix + '/bin/python')} &&\n"
            f"   source {shlex.quote(self.init_script)} &&\n"
            f"   conda activate {shlex.quote(self.prefix)} &&\n"
            f"   python -B -c {shlex.quote(verify)} {shlex.quote(self.prefix)}; then\n"
            f"{command}\n"
            "else\n"
            f"    printf '\\n%s\\n' {shlex.quote(failure_marker)} >&2\n"
            "    exit 86\n"
            "fi\n"
            ")"
        )


@dataclass(frozen=True)
class HarnessContext:
    """Generic run context, free of any task fields.

    model_label is the model name the harness advertises to its CLI. The vime
    adapter ignores it and serves whatever upstream vllm has loaded, so it is
    not a run() parameter. max_context_tokens is the served model's actual
    context budget; harnesses may use it to compact before reaching that limit.
    """

    workdir: str
    session_id: str
    adapter_url: str
    model_label: str = "vime-actor"
    max_context_tokens: int = 0


class BaseHarness(ABC, metaclass=SingletonABCMeta):
    """Base lifecycle for a sandbox-resident coding agent."""

    # short identifier set by each subclass (claude_code / codex)
    name: str = ""

    @abstractmethod
    async def install_cli(self, sb: Sandbox) -> None:
        """Install the harness CLI into the sandbox.
        npm-packaged harnesses delegate to install_npm_cli."""

    @abstractmethod
    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Write any CLI config files into the sandbox."""

    @abstractmethod
    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        """Run the agent to completion and return its exit code.

        A non-interactive CLI builds one shell command and hands it to
        run_agent. An interactive or long-running harness drives its own loop
        here instead.
        """

    async def run(
        self,
        sb: Sandbox,
        *,
        workdir: str,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
        prompt: str,
        max_context_tokens: int = 0,
    ) -> int:
        """Run the harness in the sandbox and return its exit code.

        Steps: ensure the agent user -> write config -> launch and wait.
        Workspace prep (writing the problem statement etc.) is the caller's job
        and must run before this.
        """
        await _sandbox.ensure_agent_user(sb, workdir)
        ctx = HarnessContext(
            workdir=workdir,
            session_id=session_id,
            adapter_url=adapter_url,
            max_context_tokens=int(max_context_tokens or 0),
        )
        await self.write_config(sb, ctx)
        return await self.launch_and_wait(sb, ctx, prompt, time_budget_sec)


async def run_agent(
    sb: Sandbox,
    *,
    workdir: str,
    start_cmd: str,
    env: dict[str, str],
    time_budget_sec: int,
) -> int:
    """Launch the agent (start_cmd) and run it to completion, returning its exit code."""
    task_environment = TaskCondaEnvironment.from_env()
    failure_marker = "VIME_TASK_ENV_SETUP_FAILED_" + secrets.token_hex(16)
    if task_environment is not None:
        # This preflight uses the same account, HOME and supplied environment as
        # the final launcher. Activation is repeated inside that launcher because
        # changes in this short-lived shell cannot propagate to a later RPC.
        probe = (
            f"cd {shlex.quote(workdir)} && export HOME=/home/agent &&\n"
            + task_environment.wrap(":", failure_marker)
        )
        try:
            code, _, _ = await sb.exec(probe, user="agent", env=env, timeout=30, check=False)
        except Exception as exc:
            raise TaskEnvironmentError("Task environment preflight could not complete") from exc
        if code != 0:
            raise TaskEnvironmentError(f"Task environment preflight failed for {task_environment.prefix} (exit={code})")
        start_cmd = task_environment.wrap(start_cmd, failure_marker)
    meta_dir = f"{workdir}/.harness"
    await sb.exec(f"mkdir -p {meta_dir} && chown agent:agent {meta_dir}", user="root", check=True, timeout=30)
    exit_code, output = await exec_and_wait(
        sb,
        cmd=start_cmd,
        user="agent",
        env=env,
        workdir=workdir,
        out_file=f"{meta_dir}/trajectory.jsonl",
        time_budget_sec=time_budget_sec,
        tag="run",
        want_output=False,
    )
    if task_environment is not None and exit_code == 86 and failure_marker in output.splitlines():
        raise TaskEnvironmentError(f"Task environment activation failed at launch for {task_environment.prefix}")
    return exit_code


async def install_npm_cli(
    sb: Sandbox,
    *,
    node_runtime: Path,
    npm_package: Path,
    check_cmd: str,
) -> None:
    """Install an npm-packaged CLI into the sandbox: the Node 22 runtime first,
    then the CLI's npm package (global install, then self-check via check_cmd).
    Non-npm harnesses write their own install_cli."""
    await install_node22(sb, node_runtime)

    await sb.write_file("/tmp/harness-cli.tgz", npm_package)
    install_cmd = "npm install -g --prefix=/usr/local --no-audit --no-fund /tmp/harness-cli.tgz && " + check_cmd
    # Detached install with a few in-place retries for transient disk flakes.
    last_log = ""
    for attempt in range(NPM_INSTALL_RETRIES):
        exit_code, last_log = await exec_and_wait(
            sb, cmd=install_cmd, user="root", time_budget_sec=300, tag="harness-npm-install"
        )
        if exit_code == 0:
            return
        if attempt + 1 < NPM_INSTALL_RETRIES:
            await asyncio.sleep(NPM_INSTALL_BACKOFF_SEC * (attempt + 1))
    raise RuntimeError(
        f"npm install failed after {NPM_INSTALL_RETRIES} attempts (exit={exit_code}):\n{last_log[-1000:]}"
    )


async def install_node22(sb: Sandbox, host_tarball: Path) -> None:
    """Install Node 22 over the base image (some base images ship a version too
    old for the CLI). A .xz tarball is decompressed on the host (cached) so
    sandboxes without xz-utils can still run a plain `tar xf`."""
    host_tarball = Path(host_tarball)
    if host_tarball.suffix == ".xz":
        plain = Path(tempfile.gettempdir()) / f"coding_agent_rl.{host_tarball.stem}.tar"
        if not plain.exists():
            tmp = plain.with_suffix(".tar.partial")
            with lzma.open(host_tarball, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp, plain)
        host_tarball = plain
    for attempt in range(NODE_INSTALL_RETRIES):
        try:
            # AgentENV can acknowledge an upload just before the file becomes
            # visible to envd under load. Re-upload the complete artifact when
            # the immediately following existence/extract check fails.
            await sb.write_file("/tmp/node22.tar", host_tarball)
            await sb.exec(
                "set -e && test -s /tmp/node22.tar && mkdir -p /opt/node22 && "
                "tar xf /tmp/node22.tar -C /opt/node22 --strip-components=1 && "
                "ln -sf /opt/node22/bin/node /usr/local/bin/node && "
                "ln -sf /opt/node22/bin/npm  /usr/local/bin/npm && "
                "ln -sf /opt/node22/bin/npx  /usr/local/bin/npx && "
                "node --version && npm --version",
                user="root",
                timeout=180,
                check=True,
            )
            return
        except Exception:
            if attempt + 1 >= NODE_INSTALL_RETRIES:
                raise
            await asyncio.sleep(NPM_INSTALL_BACKOFF_SEC * (attempt + 1))
