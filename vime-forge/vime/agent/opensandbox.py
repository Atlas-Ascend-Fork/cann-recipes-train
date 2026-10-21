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

"""OpenSandbox implementation of :mod:`vime.agent.sandbox`.

The dependency is intentionally isolated in this module. OpenSandbox owns the
remote Docker lifecycle; Vime only maps its small sandbox contract onto the
SDK. SDK 0.1.14 requires explicit ``kill()`` followed by ``close()`` for a
create-use-discard sandbox.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import shlex
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from vime.agent.sandbox import ExecResult, FileContent, _getenv

logger = logging.getLogger(__name__)

USER_IDS = {"root": (0, 0), "agent": (1000, 1000)}


def _metadata_label(value: object) -> str:
    """Normalize a value to the OpenSandbox metadata-label contract."""
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("._-")
    label = label[:63].rstrip("._-")
    return label or "unknown"


class OpenSandboxSandbox:
    default_cpu = "8"
    default_memory = "32Gi"
    default_lifetime_sec = 3600
    default_ready_timeout_sec = 120
    default_rpc_retries = 6
    rpc_backoff_base_sec = 1.0
    rpc_backoff_cap_sec = 32.0

    def __init__(
        self,
        image: str,
        *,
        purpose: str = "work",
        timeout: int | None = None,
        cpu: str | None = None,
        memory: str | None = None,
        arch: str | None = None,
        metadata: dict[str, str] | None = None,
        rpc_retries: int | None = None,
        _create: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.image = image
        self.rpc_retries = max(
            1,
            rpc_retries if rpc_retries is not None else int(
                _getenv("VIME_AGENT_SANDBOX_RPC_RETRIES", default=str(self.default_rpc_retries))
            ),
        )
        self.purpose = purpose
        self.timeout = timeout or int(
            _getenv("VIME_AGENT_SANDBOX_LIFETIME_SEC", default=str(self.default_lifetime_sec))
        )
        self.cpu = cpu or _getenv("VIME_AGENT_SANDBOX_CPU", default=self.default_cpu)
        self.memory = memory or _getenv("VIME_AGENT_SANDBOX_MEMORY", default=self.default_memory)
        self.arch = arch or _getenv("VIME_AGENT_SANDBOX_ARCH", default="") or None
        raw_metadata = {"purpose": purpose, **(metadata or {})}
        self.metadata = {_metadata_label(key): _metadata_label(value) for key, value in raw_metadata.items()}
        self._create = _create
        self._sb: Any = None
        self.sandbox_id = ""

    @staticmethod
    def _connection_config():
        from opensandbox.config import ConnectionConfig

        domain = _getenv("VIME_OPEN_SANDBOX_DOMAIN")
        if not domain:
            raise RuntimeError("VIME_OPEN_SANDBOX_DOMAIN is required for the opensandbox backend")
        api_key = os.environ.get("OPEN_SANDBOX_API_KEY")
        if not api_key:
            raise RuntimeError("OPEN_SANDBOX_API_KEY is required for the opensandbox backend")
        return ConnectionConfig(
            domain=domain,
            protocol=_getenv("VIME_OPEN_SANDBOX_PROTOCOL", default="http"),
            api_key=api_key,
            request_timeout=timedelta(seconds=int(_getenv("VIME_OPEN_SANDBOX_REQUEST_TIMEOUT_SEC", default="60"))),
            use_server_proxy=True,
        )

    async def __aenter__(self) -> OpenSandboxSandbox:
        from opensandbox import Sandbox
        from opensandbox.models import PlatformSpec

        create = self._create or Sandbox.create
        arch = self.arch
        if arch is not None and arch not in ("amd64", "arm64"):
            raise ValueError(f"Unsupported sandbox architecture: {arch!r}; expected amd64 or arm64")
        platform = PlatformSpec(os="linux", arch=arch) if arch is not None else None
        create_kwargs = {
            "timeout": timedelta(seconds=self.timeout),
            "ready_timeout": timedelta(
                seconds=int(
                    _getenv(
                        "VIME_OPEN_SANDBOX_READY_TIMEOUT_SEC",
                        default=str(self.default_ready_timeout_sec),
                    )
                )
            ),
            "resource": {"cpu": self.cpu, "memory": self.memory},
            "metadata": self.metadata,
            "platform": platform,
            "connection_config": self._connection_config() if self._create is None else None,
        }
        attempts = max(1, int(_getenv("VIME_OPEN_SANDBOX_CREATE_RETRIES", default=str(self.rpc_retries))))
        for attempt in range(attempts):
            try:
                self._sb = await create(self.image, **create_kwargs)
                break
            except Exception as error:
                transient = self._is_infrastructure_error(error)
                if not transient or attempt + 1 >= attempts:
                    raise
                logger.warning(
                    "[agent.opensandbox] transient create failure (%s/%s): %s",
                    attempt + 1,
                    attempts,
                    str(error)[:300],
                )
                ceiling = min(self.rpc_backoff_cap_sec, self.rpc_backoff_base_sec * (2**attempt))
                await asyncio.sleep(random.uniform(0.0, ceiling))
        self.sandbox_id = str(self._sb.id)
        return self

    @staticmethod
    def _is_infrastructure_error(error: BaseException) -> bool:
        """Classify transport/service failures without hiding invalid requests."""
        if isinstance(error, (asyncio.TimeoutError, ConnectionError)):
            return True

        try:
            from opensandbox.exceptions import InvalidArgumentException, SandboxApiException, SandboxException

            if isinstance(error, InvalidArgumentException):
                return False
            if isinstance(error, SandboxApiException):
                status_code = getattr(error, "status_code", None)
                return status_code is None or status_code >= 500 or status_code in {408, 409, 425, 429}
            if isinstance(error, SandboxException):
                return True
        except ImportError:  # pragma: no cover - dependency is present for the backend
            pass

        message = str(error)
        return "DOCKER::SANDBOX_START_FAILED" in message or "ports are not available" in message

    @staticmethod
    def _is_missing_file_error(error: BaseException) -> bool:
        try:
            from opensandbox.exceptions import SandboxApiException

            return isinstance(error, SandboxApiException) and getattr(error, "status_code", None) == 404
        except ImportError:  # pragma: no cover - dependency is present for the backend
            return False

    async def _rpc_retry(self, operation: str, call, *, idempotent: bool = True):
        attempts = self.rpc_retries
        if not idempotent:
            attempts = 1
        for attempt in range(attempts):
            try:
                return await call()
            except Exception as error:
                if not self._is_infrastructure_error(error):
                    raise
                if attempt + 1 >= attempts:
                    raise
                ceiling = min(self.rpc_backoff_cap_sec, self.rpc_backoff_base_sec * (2**attempt))
                delay = random.uniform(0.0, ceiling)
                logger.warning(
                    "[agent.opensandbox] transient %s failure (%s/%s), retrying in %ss: %s",
                    operation,
                    attempt + 1,
                    attempts,
                    delay,
                    str(error)[:300],
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._sb is None:
            return
        try:
            await self._sb.kill()
        except Exception as error:
            logger.warning("[agent.opensandbox] kill %s failed: %s", self.sandbox_id[:8], error)
        finally:
            try:
                await self._sb.close()
            except Exception as error:
                logger.warning("[agent.opensandbox] close %s failed: %s", self.sandbox_id[:8], error)

    @staticmethod
    def _user_ids(user: str) -> tuple[int, int]:
        try:
            return USER_IDS[user]
        except KeyError as exc:
            raise ValueError(f"Unsupported sandbox user: {user!r}; expected root or agent") from exc

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
        from opensandbox.models.execd import RunCommandOpts

        self._user_ids(user)
        command = cmd if user == "root" else f"runuser -u agent -- /bin/bash -lc {shlex.quote(cmd)}"
        execution = await self._rpc_retry(
            "exec",
            lambda: self._sb.commands.run(
                command,
                opts=RunCommandOpts(
                    envs=env,
                    timeout=timedelta(seconds=timeout),
                ),
            ),
            idempotent=idempotent,
        )
        if execution.exit_code is None:
            raise RuntimeError(f"opensandbox exec returned no exit code: {cmd[:120]}")
        exit_code = int(execution.exit_code)
        stdout = "".join(line.text for line in execution.logs.stdout)
        stderr = "".join(line.text for line in execution.logs.stderr)
        if check and exit_code != 0:
            raise RuntimeError(f"opensandbox exec failed (exit={exit_code}): {cmd[:120]}\n{stderr[:400]}")
        return exit_code, stdout, stderr

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        from opensandbox.models import WriteEntry

        self._user_ids(user)
        data: str | bytes
        if isinstance(content, Path):
            data = content.read_bytes()
        else:
            data = content
        # execd runs without CAP_CHOWN in the hardened Docker profile. Asking
        # its file API to assign root ownership therefore fails even though the
        # upload itself is valid. Callers that need a specific owner already
        # perform their workspace-level chown through exec().
        await self._rpc_retry(
            "write_file",
            lambda: self._sb.files.write_files([WriteEntry(path=sandbox_path, data=data, mode=644)]),
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        self._user_ids(user)
        try:
            return await self._rpc_retry("read_file", lambda: self._sb.files.read_file(sandbox_path))
        except Exception as error:
            if self._is_missing_file_error(error):
                return ""
            raise
