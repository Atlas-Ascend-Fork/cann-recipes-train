"""Sandbox backends for agent rollouts.

The public sandbox contract is intentionally small: async context management,
command execution, and file read/write. Agent examples can build task-specific
setup, runner, and evaluator logic on top of this without depending directly on
one sandbox provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
import io
import logging
import os
import random
import shlex
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .config import apply_agentenv_config, effective_sandbox_env

if TYPE_CHECKING:
    from vime.observability.trace_utils import TraceHandle

logger = logging.getLogger(__name__)


ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.

    ``idempotent`` is a hint for the backend's transport-retry policy: callers
    mark whether re-sending the command after a severed response is safe to
    replay (see ``E2BSandbox._rpc_retry``). Backends without retries may
    ignore it.
    """

    sandbox_id: str

    async def __aenter__(self) -> Sandbox: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


EXIT_TIME_BUDGET_EXCEEDED = -1
# Task-local scope, inherited by wait_for's task, never by unrelated sandbox RPCs.
_MARKER_RPC: ContextVar[str | None] = ContextVar("marker_rpc", default=None)
_SUPPRESS_COMMAND_SPAN: ContextVar[bool] = ContextVar("suppress_command_span", default=False)


def _command_prefix(cmd: str) -> str:
    """Return a compact, whitespace-normalized command preview."""
    return " ".join(cmd.split())[:50]


@contextmanager
def _suppress_command_span():
    token = _SUPPRESS_COMMAND_SPAN.set(True)
    try:
        yield
    finally:
        _SUPPRESS_COMMAND_SPAN.reset(token)


@contextmanager
def _sandbox_command_span(sandbox: Sandbox, cmd: str):
    trace = getattr(sandbox, "trace", None)
    if trace is None or _MARKER_RPC.get() is not None or _SUPPRESS_COMMAND_SPAN.get():
        yield None
        return
    from vime.observability.trace_utils import trace_span

    attrs = {
        "sandbox_id": sandbox.sandbox_id,
        "sandbox_role": getattr(sandbox, "observation_role", "agent"),
        "command_prefix": _command_prefix(cmd),
    }
    with trace_span(trace, "sandbox_command", attrs=attrs, record_error=False) as span:
        yield span


def _set_command_exit_code(span: Any, exit_code: int) -> None:
    if span is not None and hasattr(span, "set"):
        span.set("exit_code", int(exit_code))


class SandboxAdmissionTimeout(TimeoutError):
    """Capacity admission expired before setup; do not recreate/requeue blindly."""


class SandboxCommandTimeout(RuntimeError):
    """The worker returned a complete command-timeout reply; execution is unknown.

    Only read-only completion checks may retry this error. It is deliberately
    distinct from a transport timeout, which can leave an unread RPC response.
    """


def _positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@asynccontextmanager
async def sandbox_preparation(sb: Sandbox):
    """Optional provider admission control, held through all agent preparation."""
    prepare = getattr(sb, "preparation", None)
    if prepare is None:
        yield
    else:
        async with prepare():
            yield


def _exception_chain(error: BaseException):
    seen = set()
    for _ in range(12):
        if error is None or id(error) in seen:
            break
        seen.add(id(error))
        yield error
        # SDK exception conversion uses implicit context; connectrpc uses cause.
        error = error.__cause__ or error.__context__


def _terminal_marker_error(error: BaseException) -> bool:
    for item in _exception_chain(error):
        if isinstance(item, (asyncio.CancelledError, TimeoutError)):
            return True
        if getattr(getattr(item, "code", None), "name", "") in {"DEADLINE_EXCEEDED", "CANCELED"}:
            return True
        if type(item).__name__ in {"SandboxException", "TimeoutException", "NotFoundException"}:
            if any(text in str(item).lower() for text in (
                "not proxyable", "does not exist", "stopped state", "sandbox was killed", "end of life",
            )):
                return True
    return False


async def _await_done_marker(sb: Sandbox, done_file: str, *, user: str, time_budget_sec: int) -> int:
    """Poll a detached command's exit-code marker until it appears, returning the
    exit code (or ``EXIT_TIME_BUDGET_EXCEEDED`` if the budget runs out first).

    Polling does not renew AgentENV's absolute sandbox lifetime. The caller's
    budget includes polling, transport retries and backoff, using a monotonic
    clock. Only this read-only operation opts into wrapped-transport recovery.
    """
    deadline = time.monotonic() + time_budget_sec
    poll_timeout = _positive_env("VIME_AGENT_MARKER_TIMEOUT_SEC", 15)
    retry_limit = _positive_env("VIME_AGENT_MARKER_TIMEOUT_RETRIES", 3)
    timeout_count = 0
    cmd = f"test -f {done_file} && cat {done_file}"
    while time.monotonic() < deadline:
        await asyncio.sleep(min(5, max(0, deadline - time.monotonic())))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        token = _MARKER_RPC.set(f"exec({cmd[:60]!r})")
        try:
            ec, out, _ = await asyncio.wait_for(
                sb.exec(cmd, user=user, timeout=min(poll_timeout, remaining), check=False),
                timeout=remaining,
            )
        except SandboxCommandTimeout:
            timeout_count += 1
            if time.monotonic() >= deadline:
                return EXIT_TIME_BUDGET_EXCEEDED
            if timeout_count > retry_limit:
                raise
            delay = min(2 ** timeout_count + random.random(), 15, deadline - time.monotonic())
            logger.warning("sandbox=%s marker_timeout retry=%d/%d backoff_sec=%.2f",
                           sb.sandbox_id, timeout_count, retry_limit, max(0, delay))
            await asyncio.sleep(max(0, delay))
            continue
        except asyncio.TimeoutError:
            if time.monotonic() < deadline:
                raise  # A genuine request timeout, not exhaustion of our budget.
            return EXIT_TIME_BUDGET_EXCEEDED
        finally:
            _MARKER_RPC.reset(token)
        timeout_count = 0
        if ec == 0 and (out or "").strip():
            return int(out.strip())
    return EXIT_TIME_BUDGET_EXCEEDED


async def exec_and_wait(
    sb: Sandbox,
    *,
    cmd: str,
    time_budget_sec: int,
    tag: str,
    user: str = "root",
    env: dict[str, str] | None = None,
    workdir: str | None = None,
    out_file: str | None = None,
    want_output: bool = False,
) -> tuple[int, str]:
    """Run ``cmd`` to completion detached, returning ``(exit_code, output)``.

    A plain ``sb.exec`` keeps an HTTP/2 stream open for the command's whole
    runtime, so a long-running command (build, test suite) outlives what the
    E2B gateway will hold a single response stream open for: the stream gets
    severed mid-run and we lose the exit code with no safe way to retry a
    non-idempotent command. Instead we ``setsid`` the command fully detached,
    redirect its output to a file, and have it drop its exit code into a marker
    file. The caller side then becomes a sequence of short, idempotent RPCs --
    write the launcher, fire-and-forget the spawn, then poll for the marker (see
    ``_await_done_marker``) -- none of which depend on a stream staying alive,
    but polling does not extend the sandbox lifetime.
    """
    out_file = out_file or f"/tmp/.{tag}.out"
    quoted_out_file = shlex.quote(out_file)
    done_file = f"/tmp/.{tag}.done"
    launcher = f"/tmp/.{tag}.sh"
    lock_dir = f"/tmp/.{tag}.spawned"
    prefix = f"cd {shlex.quote(workdir)}\nexport HOME=/home/{user}\n" if workdir else ""
    launcher_body = f"#!/bin/bash\n{prefix}{cmd}\necho $? > {done_file}\n"
    await sb.write_file(launcher, launcher_body, user=user)

    # Clear the previous invocation's state in its own idempotent RPC, *before*
    # the guarded spawn. The mkdir guard below exists only to dedupe transport
    # retries of this one spawn (a severed response replayed by _rpc_retry); it
    # must not survive into the next logical invocation of the same tag (e.g.
    # install_npm_cli's retry loop), which would skip the spawn entirely and
    # read the previous run's stale exit-code marker. Callers must not overlap
    # two exec_and_wait calls with the same tag.
    with _sandbox_command_span(sb, cmd) as span:
        with _suppress_command_span():
            await sb.exec(
                f"rm -rf {lock_dir}; rm -f {quoted_out_file} {done_file}",
                user=user,
                timeout=30,
                check=True,
                idempotent=True,
            )
            await sb.exec(
                f"chmod +x {launcher}; "
                f"mkdir {lock_dir} 2>/dev/null || exit 0; "
                f"setsid bash {launcher} < /dev/null > {quoted_out_file} 2>&1 &",
                user=user,
                env=env,
                timeout=30,
                check=True,
                idempotent=True,
            )
        exit_code = await _await_done_marker(sb, done_file, user=user, time_budget_sec=time_budget_sec)
        if exit_code != EXIT_TIME_BUDGET_EXCEEDED:
            _set_command_exit_code(span, exit_code)
    if exit_code == 0 and not want_output:
        return exit_code, ""
    if want_output:
        return exit_code, await sb.read_file(out_file, user=user)
    with _suppress_command_span():
        _, tail, _ = await sb.exec(
            f"tail -c 512 {quoted_out_file} 2>/dev/null", user=user, timeout=15, check=False
        )
    return exit_code, tail or ""


def _getenv(*names: str, default: str = "", env: Mapping[str, str] | None = None) -> str:
    """First non-empty environment value among ``names`` (else ``default``).

    Lets a setting carry a primary name plus legacy aliases: list the canonical
    ``VIME_AGENT_*`` name first, older names after."""
    source = os.environ if env is None else env
    for name in names:
        value = source.get(name)
        if value is not None and value.strip():
            return value
    return default


def _append_optional_sdk_path(env_name: str) -> None:
    """Append an existing SDK site-packages directory without overriding the
    active Python environment.

    This is intended for deployments that keep sandbox-provider clients in a
    dedicated runtime environment.  Appending (rather than prepending) keeps
    the rollout environment authoritative for packages it already provides.
    """
    raw_path = os.environ.get(env_name, "").strip()
    if not raw_path:
        return
    sdk_path = Path(raw_path).expanduser()
    if not sdk_path.is_dir():
        raise RuntimeError(f"{env_name} is not a directory: {sdk_path}")
    resolved = str(sdk_path.resolve())
    if resolved not in sys.path:
        sys.path.append(resolved)


class E2BSandbox:
    """Async context manager around e2b.AsyncSandbox."""

    image_metadata_key_env = ("VIME_AGENT_SANDBOX_IMAGE_METADATA_KEY", "SWE_SANDBOX_IMAGE_METADATA_KEY")
    selector_mode_env = ("VIME_AGENT_E2B_SELECTOR_MODE",)
    lifetime_sec_env = ("VIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("VIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")
    size_env = ("VIME_AGENT_E2B_SANDBOX_SIZE", "SWE_E2B_SANDBOX_SIZE")

    default_lifetime_sec = 3600
    default_rpc_retries = 6
    default_size = "md"
    default_selector_mode = "metadata"
    rpc_backoff_base_sec = 1.0
    rpc_backoff_cap_sec = 32.0

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        image_metadata_key: str | None = None,
        selector_mode: str | None = None,
        rpc_retries: int | None = None,
        size: str | None = None,
        observation_role: str = "agent",
        trace: TraceHandle | None = None,
    ) -> None:
        env = effective_sandbox_env()
        self.image = image
        self.observation_role = observation_role
        self.trace = trace
        self.timeout = timeout if timeout is not None else self._lifetime_sec_from_env(env)
        self.image_metadata_key = image_metadata_key or self._image_metadata_key_from_env(env)
        self.selector_mode = selector_mode or _getenv(
            *self.selector_mode_env, default=self.default_selector_mode, env=env
        )
        if self.selector_mode not in {"metadata", "template"}:
            raise ValueError(
                "VIME_AGENT_E2B_SELECTOR_MODE must be 'metadata' or 'template', "
                f"got {self.selector_mode!r}"
            )
        self.rpc_retries = rpc_retries if rpc_retries is not None else self._rpc_retries_from_env(env)
        self.size = size if size is not None else self._size_from_env(env)
        self._sb = None
        self.sandbox_id = ""

    @classmethod
    def _image_metadata_key_from_env(cls, env: Mapping[str, str] | None = None) -> str | None:
        return _getenv(*cls.image_metadata_key_env, env=env) or None

    @classmethod
    def _lifetime_sec_from_env(cls, env: Mapping[str, str] | None = None) -> int:
        return int(_getenv(*cls.lifetime_sec_env, default=str(cls.default_lifetime_sec), env=env))

    @classmethod
    def _rpc_retries_from_env(cls, env: Mapping[str, str] | None = None) -> int:
        return int(_getenv(*cls.rpc_retries_env, default=str(cls.default_rpc_retries), env=env))

    @classmethod
    def _size_from_env(cls, env: Mapping[str, str] | None = None) -> str:
        return _getenv(*cls.size_env, default=cls.default_size, env=env)

    # Transient client-side failures safe to retry.
    _TRANSIENT_RPC_ERRORS = frozenset(
        {
            "ProtocolError",
            "LocalProtocolError",
            "WriteError",
            "ReadError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "SSLError",
        }
    )

    @classmethod
    def _is_transient_rpc_error(cls, e: BaseException) -> bool:
        """True if e is a transient E2B client-side failure safe to retry."""
        name = type(e).__name__
        if name in cls._TRANSIENT_RPC_ERRORS:
            return True
        msg = str(e)
        if name == "SandboxException":
            if "does not exist" in msg or "STOPPED state" in msg:
                return False
            return True
        return False

    @staticmethod
    def _marker_transport_cause(op_name: str, error: BaseException) -> BaseException | None:
        """Recover typed connection failures only inside completion-marker polls.

        A generic TimeoutException/UNAVAILABLE response is insufficient. Do not
        infer transport failures from the SDK's generic timeout advice.
        """
        if _MARKER_RPC.get() != op_name or _terminal_marker_error(error):
            return None
        for item in _exception_chain(error):
            if type(item).__name__ in {
                "RemoteProtocolError", "ReadError", "WriteError", "StreamError",
                "ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError",
            }:
                return item
        return None

    async def _rpc_retry(self, op_name: str, coro_factory, *, idempotent: bool = True):
        started = time.monotonic()
        try:
            return await self._rpc_retry_impl(op_name, coro_factory, idempotent=idempotent)
        except Exception as e:
            if type(e).__name__ == "CommandExitException":
                raise
            if self.trace is not None:
                from vime.observability.trace_utils import trace_event

                trace_event(
                    self.trace,
                    "agentic.sandbox_rpc_failed",
                    attrs={
                        "operation": op_name.split("(", 1)[0],
                        "sandbox_id": self.sandbox_id,
                        "sandbox_role": self.observation_role,
                        "idempotent": idempotent,
                        "error_type": type(e).__name__,
                        "duration_ms": (time.monotonic() - started) * 1000,
                    },
                )
            raise

    async def _reset_conn_pool(self) -> None:
        """Tear down the sandbox's httpcore pool so the next RPC reconnects."""
        try:
            pool = self._sb._transport.pool  # httpcore.AsyncConnectionPool
            await pool.aclose()
        except Exception as e:
            logger.debug("[agent.sandbox] conn-pool reset skipped: %s", e)

    async def __aenter__(self) -> E2BSandbox:
        if self.selector_mode == "metadata" and self.image_metadata_key is None:
            raise RuntimeError(
                "VIME_AGENT_SANDBOX_IMAGE_METADATA_KEY is not set. Export it "
                "to the metadata key your E2B gateway uses for image routing. "
                "The legacy SWE_SANDBOX_IMAGE_METADATA_KEY name is also "
                "accepted for coding-agent examples."
            )
        apply_agentenv_config()
        _append_optional_sdk_path("VIME_AGENT_E2B_SDK_PATH")
        from e2b import AsyncSandbox  # type: ignore

        if self.selector_mode == "template":
            # AgentENV's E2B-compatible API selects a prebuilt sandbox template
            # through the standard ``template`` argument instead of provider-
            # specific image-routing metadata.
            create = lambda: AsyncSandbox.create(template=self.image, timeout=self.timeout)
        else:
            assert self.image_metadata_key is not None
            md = {self.image_metadata_key: self.image}
            if self.size:
                prefix = self.image_metadata_key.rsplit("/", 1)[0] if "/" in self.image_metadata_key else ""
                size_key = f"{prefix}/size" if prefix else "size"
                md[size_key] = self.size
            create = lambda: AsyncSandbox.create(timeout=self.timeout, metadata=md)

        self._sb = await self._rpc_retry("create", create)
        self.sandbox_id = self._sb.sandbox_id
        if self.trace is not None:
            from vime.observability.trace_utils import trace_event

            trace_event(
                self.trace,
                "agentic.sandbox_ready",
                attrs={
                    "sandbox_id": self.sandbox_id,
                    "sandbox_role": self.observation_role,
                    "selector_mode": self.selector_mode,
                },
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._sb.kill()
        except Exception as e:
            if self.trace is not None:
                from vime.observability.trace_utils import trace_event

                trace_event(
                    self.trace,
                    "agentic.sandbox_destroy_failed",
                    attrs={
                        "sandbox_id": self.sandbox_id,
                        "sandbox_role": self.observation_role,
                        "error_type": type(e).__name__,
                    },
                )
            logger.warning("[agent.sandbox] kill %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        with _sandbox_command_span(self, cmd) as span:
            try:
                res = await self._rpc_retry(
                    f"exec({cmd[:60]!r})",
                    lambda: self._sb.commands.run(
                        cmd,
                        user=user,
                        envs=env,
                        timeout=timeout,
                        on_stdout=lambda s: None,
                        on_stderr=lambda s: None,
                    ),
                    idempotent=idempotent,
                )
                _set_command_exit_code(span, res.exit_code)
                return res.exit_code, res.stdout or "", res.stderr or ""
            except CommandExitException as e:
                _set_command_exit_code(span, e.exit_code)
                if check:
                    raise RuntimeError(
                        f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                    ) from None
                return e.exit_code, e.stdout or "", e.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            host_path = content

            async def _do_path():
                with open(host_path, "rb") as fp:
                    await self._sb.files.write(
                        sandbox_path,
                        fp,
                        user=user,
                        gzip=False,
                        use_octet_stream=True,
                        request_timeout=600,
                    )

            await self._rpc_retry(f"write_file({sandbox_path} <- {host_path.name})", _do_path)
            return

        if isinstance(content, bytes):

            async def _do_bytes():
                await self._sb.files.write(
                    sandbox_path,
                    io.BytesIO(content),
                    user=user,
                    gzip=False,
                    use_octet_stream=True,
                    request_timeout=600,
                )

            await self._rpc_retry(f"write_file({sandbox_path}, bytes={len(content)})", _do_bytes)
            return

        await self._rpc_retry(
            f"write_file({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            return ""

    async def _rpc_retry_impl(self, op_name: str, coro_factory, *, idempotent: bool = True):
        """Run coro_factory() with retries for transient E2B RPC failures.

        :param idempotent: When False, a transient failure is re-raised instead
            of retried: re-running a non-idempotent op (e.g. a process-spawning
            exec) after a severed response could double-execute it. Idempotent
            ops (the default: create / read_file / write_file / short read-only
            execs) retry as before.
        """
        last_err = None
        for attempt in range(self.rpc_retries):
            try:
                result = await coro_factory()
                if attempt and _MARKER_RPC.get() == op_name:
                    logger.info("[agent.sandbox] marker RPC recovered sandbox=%s attempts=%d",
                                self.sandbox_id, attempt + 1)
                return result
            except Exception as e:
                marker_cause = self._marker_transport_cause(op_name, e) if idempotent else None
                if _MARKER_RPC.get() == op_name and _terminal_marker_error(e):
                    raise
                if marker_cause is None and not self._is_transient_rpc_error(e):
                    raise
                if not idempotent:
                    raise
                last_err = e
                if attempt + 1 < self.rpc_retries:
                    await self._wait_rpc_retry(op_name, attempt, e, marker_cause)
        assert last_err is not None
        raise last_err

    async def _wait_rpc_retry(self, op_name: str, attempt: int, error: Exception, marker_cause) -> None:
        """Reset transport state, record the retry, and wait before retrying."""
        # The installed pyqwest SDK shares process-wide pools; its connect()
        # can resume expired sandboxes. Do not flush shared pools or renew the
        # sandbox lifetime for marker transport retries.
        if marker_cause is None:
            await self._reset_conn_pool()
        else:
            logger.warning(
                "[agent.sandbox] marker transport retry sandbox=%s cause=%s attempt=%d/%d",
                self.sandbox_id, type(marker_cause).__name__, attempt + 1, self.rpc_retries,
            )
        ceiling = min(self.rpc_backoff_cap_sec, self.rpc_backoff_base_sec * (2**attempt))
        backoff = random.uniform(0.0, ceiling)
        self._record_rpc_retry(op_name, attempt, error, backoff)
        logger.debug(
            "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
            op_name,
            type(marker_cause).__name__ if marker_cause is not None else type(error).__name__,
            attempt + 1,
            self.rpc_retries,
            backoff,
            "connection-level failure" if marker_cause is not None else str(error)[:120],
        )
        await asyncio.sleep(backoff)

    def _record_rpc_retry(self, op_name: str, attempt: int, error: Exception, backoff: float) -> None:
        if self.trace is not None:
            from vime.observability.trace_utils import trace_event

            trace_event(
                self.trace,
                "agentic.sandbox_rpc_retry",
                attrs={
                    "operation": op_name.split("(", 1)[0],
                    "sandbox_id": self.sandbox_id,
                    "sandbox_role": self.observation_role,
                    "attempt": attempt + 1,
                    "error_type": type(error).__name__,
                    "backoff_sec": backoff,
                },
            )


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Prepare ownership once per sandbox identity/workdir, after success only.

    Subsequent workspace writes must use the agent user. No mutation is replayed
    on timeout; the orchestrator discards the failed sandbox before retrying.
    """
    if not hasattr(sb, "_agent_ownership_lock"):
        sb._agent_ownership_lock = asyncio.Lock()
        sb._agent_owned_workspaces = set()
    async with sb._agent_ownership_lock:
        key = (sb.sandbox_id, workdir)
        if key in sb._agent_owned_workspaces:
            return
        timeout = _positive_env("VIME_AGENT_WORKSPACE_PREP_TIMEOUT_SEC", 60)
        started = time.monotonic()
        try:
            await sb.exec(
                "id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
                f"chown -R agent:agent /home/agent {shlex.quote(workdir)} && "
                "(if command -v git >/dev/null 2>&1; "
                "then git config --system --add safe.directory '*'; fi) && id agent",
                user="root", check=True, timeout=timeout, idempotent=False,
            )
        except BaseException as error:
            logger.warning("sandbox=%s stage=ownership status=%s elapsed_sec=%.3f timeout_sec=%s",
                           sb.sandbox_id, type(error).__name__, time.monotonic() - started, timeout)
            raise
        sb._agent_owned_workspaces.add(key)
        logger.info("sandbox=%s stage=ownership status=ok elapsed_sec=%.3f",
                    sb.sandbox_id, time.monotonic() - started)
