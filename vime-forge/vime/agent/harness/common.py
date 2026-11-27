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
import logging
import os
import secrets
import shlex
import shutil
import tempfile
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from vime.agent import sandbox as _sandbox
from vime.agent.sandbox import Sandbox, exec_and_wait, EXIT_TIME_BUDGET_EXCEEDED
from vime.utils.misc import SingletonMeta


logger = logging.getLogger(__name__)
ARTIFACT_BASE_URL_ENV = "VIME_AGENT_ARTIFACT_BASE_URL"
ARTIFACT_FETCH_TIMEOUT_SEC = 600


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
    quoted_meta_dir = shlex.quote(meta_dir)
    await sb.exec(
        f"mkdir -p {quoted_meta_dir} && chown agent:agent {quoted_meta_dir}", user="root", check=True, timeout=30
    )
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
    """Install an npm-packaged CLI into the sandbox: the configured Node runtime first,
    then the CLI's npm package (global install, then self-check via check_cmd).
    Non-npm harnesses write their own install_cli."""
    await install_node_runtime(sb, node_runtime)

    base = _artifact_base_url()
    if not (base and await _fetch_cli_tarball_via_http(sb, base, npm_package)):
        await sb.write_file("/tmp/harness-cli.tgz", npm_package)
    npm_flags = "--no-audit --no-fund"
    if os.environ.get("VIME_AGENT_NPM_ONLINE", "").strip() != "1":
        npm_flags += " --offline --fetch-timeout=5000 --fetch-retries=1"
    install_cmd = f"npm install -g --prefix=/usr/local {npm_flags} /tmp/harness-cli.tgz && {check_cmd}"
    budget = int(os.environ.get("VIME_AGENT_NPM_INSTALL_TIMEOUT_SEC", "300"))
    if budget <= 0:
        raise ValueError("VIME_AGENT_NPM_INSTALL_TIMEOUT_SEC must be positive")
    # Detached install with a few in-place retries for transient disk flakes.
    last_log = ""
    for attempt in range(NPM_INSTALL_RETRIES):
        exit_code, last_log = await exec_and_wait(
            sb, cmd=install_cmd, user="root", time_budget_sec=budget, tag="harness-npm-install"
        )
        if exit_code == 0:
            return
        if exit_code == EXIT_TIME_BUDGET_EXCEEDED:
            # The detached command may still be running. The caller destroys
            # this sandbox; never start a second npm against uncertain state.
            raise RuntimeError("npm installation exceeded its budget; discard this sandbox")
        if attempt + 1 < NPM_INSTALL_RETRIES:
            await asyncio.sleep(NPM_INSTALL_BACKOFF_SEC * (attempt + 1))
    raise RuntimeError(
        f"npm install failed after {NPM_INSTALL_RETRIES} attempts (exit={exit_code}):\n{last_log[-1000:]}"
    )


def _artifact_base_url() -> str:
    return os.environ.get(ARTIFACT_BASE_URL_ENV, "").strip().rstrip("/")


def _artifact_fetch_sh(url: str, out: str, expected_bytes: int | None) -> str:
    """POSIX-sh snippet: download `url` to `out` with the first available of
    curl/wget/python3. When `expected_bytes` is known (a readable local
    reference file), the byte count is verified to catch a stale/mismatched
    artifact server; without a local reference the server's content is the
    authority and integrity rides on the fetch tool's error codes plus the
    archive's own checksums (xz/gzip CRC, tar). Exit codes: 99 = no fetch
    tool in the image, 98 = size mismatch."""
    script = (
        f'url={shlex.quote(url)} out={shlex.quote(out)}\n'
        'if command -v curl >/dev/null 2>&1; then '
        'curl -fsSL --retry 2 --connect-timeout 10 -o "$out" "$url"\n'
        'elif command -v wget >/dev/null 2>&1; then '
        'wget -q -T 30 -t 2 -O "$out" "$url"\n'
        'elif command -v python3 >/dev/null 2>&1; then '
        "python3 -c 'import sys,urllib.request,shutil; "
        "u=urllib.request.urlopen(sys.argv[1],timeout=60); "
        "f=open(sys.argv[2],\"wb\"); shutil.copyfileobj(u,f); f.close()' "
        '"$url" "$out"\n'
        'else echo "no fetch tool (curl/wget/python3) in sandbox" >&2; exit 99; fi\n'
    )
    if expected_bytes is not None:
        script += (
            f'want={int(expected_bytes)}\n'
            'size=$(wc -c < "$out")\n'
            '[ "$size" = "$want" ] || { echo "fetched $size bytes, expected $want" >&2; exit 98; }\n'
        )
    return script


async def _run_artifact_script(sb: Sandbox, script: str, what: str) -> bool:
    """Run an in-sandbox fetch/extract script; any failure logs a warning and
    returns False so the caller falls back to the gateway upload path."""
    # Only a confirmed nonzero exit permits fallback. Unknown execution state
    # (transport error/cancellation/command timeout) must discard the sandbox.
    code, _, _ = await sb.exec(script, user="root", timeout=ARTIFACT_FETCH_TIMEOUT_SEC, check=False, idempotent=False)
    if code:
        logger.warning("Artifact download/extraction failed for %s (exit=%s); using upload fallback", what, code)
    return code == 0


async def _install_node_runtime_via_http(sb: Sandbox, base_url: str, host_tarball: Path) -> bool:
    """Fetch and extract the node runtime entirely inside the sandbox: the
    data path becomes sandbox <-> artifact server (loopback when both live on
    the AgentENV host) instead of training host <-> gateway write_file. The
    .xz is transferred as-is (a third of the decompressed .tar); sandboxes
    without xz-utils decompress it with python3's lzma."""
    expected: int | None = None
    try:
        expected = host_tarball.stat().st_size
    except OSError:
        pass
    url = f"{base_url}/{host_tarball.name}"
    if host_tarball.suffix == ".xz":
        extract = (
            'if command -v xz >/dev/null 2>&1; then '
            'tar -xJf /tmp/node.dl -C /opt/node --strip-components=1\n'
            'elif command -v python3 >/dev/null 2>&1; then '
            "python3 -c 'import lzma,sys,shutil; "
            'shutil.copyfileobj(lzma.open(sys.argv[1],"rb"), open(sys.argv[2],"wb"))\' '
            '/tmp/node.dl /tmp/node.tar && '
            'tar -xf /tmp/node.tar -C /opt/node --strip-components=1 && rm -f /tmp/node.tar\n'
            'else echo "no xz or python3 to extract .xz" >&2; exit 97; fi\n'
        )
    else:
        extract = "tar -xf /tmp/node.dl -C /opt/node --strip-components=1\n"
    script = (
        "set -e\n"
        + _artifact_fetch_sh(url, "/tmp/node.dl", expected)
        + "mkdir -p /opt/node\n"
        + extract
        + "ln -sf /opt/node/bin/node /usr/local/bin/node\n"
        + "ln -sf /opt/node/bin/npm  /usr/local/bin/npm\n"
        + "ln -sf /opt/node/bin/npx  /usr/local/bin/npx\n"
        + "node --version && npm --version\n"
        + "rm -f /tmp/node.dl\n"
    )
    return await _run_artifact_script(sb, script, f"Node runtime tarball {url}")


async def _fetch_cli_tarball_via_http(sb: Sandbox, base_url: str, npm_package: Path) -> bool:
    expected: int | None = None
    try:
        expected = npm_package.stat().st_size
    except OSError:
        pass
    url = f"{base_url}/{npm_package.name}"
    script = "set -e\n" + _artifact_fetch_sh(url, "/tmp/harness-cli.tgz", expected)
    return await _run_artifact_script(sb, script, f"cli tarball {url}")


async def install_node_runtime(sb: Sandbox, host_tarball: Path) -> None:
    """Install Node over HTTP or upload a tarball, decompressing .xz on the host for upload."""
    host_tarball = Path(host_tarball)
    base = _artifact_base_url()
    if base and await _install_node_runtime_via_http(sb, base, host_tarball):
        return
    if host_tarball.suffix == ".xz":
        plain = Path(tempfile.gettempdir()) / f"swe_agent_rl.{host_tarball.stem}.tar"
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
            await sb.write_file("/tmp/node.tar", host_tarball)
            await sb.exec(
                "set -e && test -s /tmp/node.tar && mkdir -p /opt/node && "
                "tar xf /tmp/node.tar -C /opt/node --strip-components=1 && "
                "ln -sf /opt/node/bin/node /usr/local/bin/node && "
                "ln -sf /opt/node/bin/npm  /usr/local/bin/npm && "
                "ln -sf /opt/node/bin/npx  /usr/local/bin/npx && "
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
