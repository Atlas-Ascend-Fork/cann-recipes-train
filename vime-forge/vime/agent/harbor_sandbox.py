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

"""Harbor environments over SSH, without Harbor dependencies in the trainer.

Each sandbox owns one SSH worker and one Harbor Docker environment. Configuration
is external; host-side flock leases enforce capacity across rollout processes.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import base64
import json
import io
import logging
import math
import os
import random
import re
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import effective_sandbox_env
from .sandbox import _MARKER_RPC, SandboxAdmissionTimeout, SandboxCommandTimeout

logger = logging.getLogger(__name__)

# Allow 120s for worker cleanup and another 30s for SSH to exit.
_SSH_EXIT_TIMEOUT_SEC = 150


@dataclass(frozen=True)
class HarborConfig:
    hosts: tuple[dict, ...]
    lifetime: int
    startup_timeout: int
    acquire_timeout: int
    cpus: int
    memory_mb: int
    registry: str
    marker_cancel_grace: int
    prepare_concurrency: int = 0
    prepare_acquire_timeout: int = 600
    prepare_queue_limit: int = 64

    @classmethod
    def from_env(cls):
        # Direct HarborSandbox users historically did not need a factory selector.
        source = dict(os.environ)
        source.setdefault("VIME_AGENT_SANDBOX_BACKEND", "harbor")
        env = effective_sandbox_env(source)
        if env.get("VIME_HARBOR_ENVIRONMENT", "remote_docker") != "remote_docker":
            raise ValueError("This adapter supports VIME_HARBOR_ENVIRONMENT=remote_docker")
        raw = env.get("VIME_HARBOR_HOSTS", "")
        hosts = json.loads(raw) if raw else []
        if not isinstance(hosts, list) or not hosts:
            raise ValueError("VIME_HARBOR_HOSTS must be a nonempty JSON list of host settings")
        for host in hosts:
            if (
                not isinstance(host, dict)
                or not isinstance(host.get("ssh"), str)
                or not re.fullmatch(r"[\w.@:-]+", host["ssh"])
                or host["ssh"].startswith("-")
            ):
                raise ValueError("Each Harbor host needs a valid SSH destination")
            for key in ("python", "worker", "root"):
                if not isinstance(host.get(key), str) or not host[key].startswith("/"):
                    raise ValueError(f"Each Harbor host needs an absolute {key} path")
            if type(host.get("slots")) is not int or host["slots"] < 1:
                raise ValueError("Each Harbor host needs a positive slots count")
            if (
                not isinstance(host.get("ssh_args", []), list)
                or not all(isinstance(a, str) for a in host.get("ssh_args", []))
            ):
                raise ValueError("ssh_args must be a list of strings")
            overrides = host.get("docker_compose", [])
            if not isinstance(overrides, list) or not all(
                isinstance(path, str) and path.startswith("/") for path in overrides
            ):
                raise ValueError("docker_compose must be a list of absolute CPU-host paths")

        def positive(name, default):
            value = int(env.get(name, default))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            return value
        registry = env.get("VIME_HARBOR_IMAGE_REGISTRY", "").strip().rstrip("/")
        prepare_concurrency = int(env.get("VIME_HARBOR_PREPARE_CONCURRENCY", "0"))
        if prepare_concurrency < 0:
            raise ValueError("VIME_HARBOR_PREPARE_CONCURRENCY must be nonnegative")
        if registry and ("://" in registry or "/" in registry or not re.fullmatch(r"[\w.:-]+", registry)):
            raise ValueError("VIME_HARBOR_IMAGE_REGISTRY must be a registry host[:port]")
        return cls(tuple(hosts), positive("VIME_AGENT_SANDBOX_LIFETIME_SEC", 4500),
                   positive("VIME_HARBOR_STARTUP_TIMEOUT", 600),
                   positive("VIME_HARBOR_ACQUIRE_TIMEOUT", 600),
                   positive("VIME_HARBOR_CPUS", 16), positive("VIME_HARBOR_MEMORY_MB", 65536), registry,
                   positive("VIME_HARBOR_MARKER_CANCEL_GRACE_SEC", 30), prepare_concurrency,
                   positive("VIME_HARBOR_PREPARE_ACQUIRE_TIMEOUT_SEC", 600),
                   positive("VIME_HARBOR_PREPARE_QUEUE_LIMIT", 64))

    def image_reference(self, image):
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
            raise ValueError("Harbor requires the bundle's immutable source_image@sha256 digest")
        if not self.registry:
            return image
        # Rewrite only the registry location; the immutable digest is retained.
        parts = image.split("/", 1)
        if len(parts) == 2 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
            image = parts[1]
        return f"{self.registry}/{image}"


class HarborSandbox:
    def __init__(self, image: str, *, timeout: int | None = None):
        self.config = HarborConfig.from_env()
        self.image = self.config.image_reference(image)
        self.timeout = timeout or self.config.lifetime
        self.sandbox_id = ""
        self._process = None
        self._lock = asyncio.Lock()
        self._usable = True
        self._admitted = False
        self._host = None

    async def _rpc(self, request, timeout=120, *, cancel_grace=0):
        async with self._lock:
            if self._process is None or not self._usable:
                raise RuntimeError("Harbor transport is closed or interrupted; create a new sandbox")

            async def exchange():
                async with asyncio.timeout(timeout):
                    self._process.stdin.write(json.dumps(request).encode() + b"\n")
                    await self._process.stdin.drain()
                    line = await self._process.stdout.readline()
                    if not line:
                        raise RuntimeError("Harbor SSH worker disconnected; operation is not replayed")
                    return json.loads(line)

            # Only completion-marker reads opt in. Hold the lock until their
            # cancelled response is consumed, so the next RPC cannot read it.
            response = asyncio.create_task(exchange()) if cancel_grace else None
            try:
                result = await asyncio.shield(response) if response is not None else await exchange()
            except asyncio.CancelledError:
                if response is None:
                    self._usable = False
                else:
                    try:
                        await asyncio.wait_for(asyncio.shield(response), cancel_grace)
                    except BaseException:
                        self._usable = False
                        response.cancel()
                        await asyncio.gather(response, return_exceptions=True)
                # Preserve the caller's deadline/cancellation even after a
                # successful drain; never replay a command or report success.
                raise
            except BaseException:
                # An interrupted response must never be mistaken for the next
                # operation's response. Never retry a possibly executed command.
                self._usable = False
                raise
            if "error" in result:
                if result.get("error_kind") == "command_timeout" and request.get("op") == "exec":
                    raise SandboxCommandTimeout(f"Harbor: {result['error']}")
                raise RuntimeError(f"Harbor: {result['error']}")
            return result

    async def _connect(self, host):
        command = [host["python"], "-u", host["worker"], "--root", host["root"],
                   "--slots", str(host["slots"]), "--lifetime", str(self.timeout),
                   "--cpus", str(self.config.cpus), "--memory-mb", str(self.config.memory_mb)]
        for path in host.get("docker_compose", []):
            command.extend(["--docker-compose", path])
        self._process = await asyncio.create_subprocess_exec(
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            *host.get("ssh_args", []), host["ssh"], shlex.join(command),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            limit=64 * 1024 * 1024,
        )
        self._host, self._usable = host, True

    async def _queue_hosts(self):
        # Probe without reserving anything. Queue registration rechecks capacity
        # atomically; snapshots guide placement but cannot bypass FIFO order.
        hosts = list(self.config.hosts)
        random.shuffle(hosts)
        available, errors = [], []
        for host in hosts:
            try:
                await self._connect(host)
                status = await self._rpc(dict(op="admission_snapshot"), 30)
                score = ((status["waiting"] + status["preparing"]) / self.config.prepare_concurrency,
                         status["active"] / host["slots"])
                available.append((score, host))
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"{host['ssh']}: {exc}")
            finally:
                await self._disconnect()
        if not available:
            raise RuntimeError("No Harbor admission host available: " + "; ".join(errors))
        return [host for _, host in sorted(available, key=lambda item: item[0])]

    async def _admit(self):
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self.config.prepare_acquire_timeout
        request_id = uuid.uuid4().hex
        status = {}
        phase = "host_selection"
        budget = asyncio.timeout_at(deadline)
        try:
            async with budget:
                joined = False
                while not joined:
                    for host in await self._queue_hosts():
                        await self._connect(host)
                        phase = "queue_registration"
                        status = await self._rpc(dict(
                            op="admission_join", request_id=request_id,
                            limit=self.config.prepare_concurrency, queue_limit=self.config.prepare_queue_limit,
                            timeout=max(1, math.ceil(deadline - loop.time()))), 30)
                        if not status.get("full"):
                            joined = True
                            break
                        await self._disconnect()
                    if not joined:
                        await asyncio.sleep(1 + random.random())
                logger.info("host=%s request=%s stage=prepare_queue position=%s wait_sec=%.3f",
                            self._host["ssh"], request_id, status.get("position", 0), loop.time() - started)
                last_log = loop.time()
                phase = "queue_wait"
                while not status.get("admitted"):
                    if status.get("expired"):
                        raise SandboxAdmissionTimeout(f"Harbor admission expired: host={self._host['ssh']} "
                                                      f"request={request_id} wait_sec={loop.time() - started:.3f}")
                    await asyncio.sleep(1 + random.random())
                    status = await self._rpc(dict(op="admission_poll", request_id=request_id), 30)
                    if loop.time() - last_log >= 60:
                        logger.info("host=%s request=%s stage=prepare_queue position=%s wait_sec=%.3f",
                                    self._host["ssh"], request_id, status.get("position", 0), loop.time() - started)
                        last_log = loop.time()
                self._admitted = True
                logger.info("host=%s request=%s stage=prepare_admission wait_sec=%.3f limit=%d",
                            self._host["ssh"], request_id, loop.time() - started, self.config.prepare_concurrency)
        except TimeoutError as exc:
            if budget.expired():
                raise SandboxAdmissionTimeout(
                    f"Harbor admission deadline: phase={phase} host={self._host and self._host['ssh']} "
                    f"request={request_id} position={status.get('position')} "
                    f"wait_sec={loop.time() - started:.3f} budget_sec={self.config.prepare_acquire_timeout}"
                ) from exc
            raise  # A transport timeout is not normal capacity waiting.

    @asynccontextmanager
    async def preparation(self):
        """Enter before Docker creation; keep admission through workspace setup."""
        if not self.config.prepare_concurrency:
            yield
            return
        if self._process is not None or self.sandbox_id:
            raise RuntimeError("Harbor preparation admission must precede sandbox creation")
        try:
            await self._admit()
            yield
        except BaseException:
            # Closing stdin reconciles even a lost grant response: the worker
            # removes its ticket and releases both leases. Never replay a grant.
            await self._disconnect()
            raise
        else:
            if self.sandbox_id and self._usable:
                try:
                    await self._rpc(dict(op="prepare_end"), 30)
                except BaseException:
                    await self._disconnect()
                    raise
            else:
                await self._disconnect()
        finally:
            self._admitted = False

    async def __aenter__(self):
        if self._admitted:
            try:
                response = await self._rpc({"op": "start", "image": self.image}, self.config.startup_timeout)
                self.sandbox_id = f"{self._host['ssh']}:{response['sandbox_id']}"
                return self
            except BaseException:
                await self._disconnect()
                raise
        hosts = list(self.config.hosts)
        random.shuffle(hosts)
        deadline = asyncio.get_running_loop().time() + self.config.acquire_timeout
        try:
            while asyncio.get_running_loop().time() < deadline:
                errors = []
                busy = False
                for host in hosts:
                    await self._connect(host)
                    try:
                        response = await self._rpc({"op": "start", "image": self.image}, self.config.startup_timeout)
                        if not response.get("busy"):
                            self.sandbox_id = f"{host['ssh']}:{response['sandbox_id']}"
                            return self
                        busy = True
                    except (OSError, RuntimeError, ValueError) as exc:
                        errors.append(f"{host['ssh']}: {exc}")
                    await self._disconnect()
                if not busy:
                    raise RuntimeError("No Harbor host could start a sandbox: " + "; ".join(errors))
                await asyncio.sleep(1)
            raise TimeoutError("Timed out waiting for a Harbor host slot")
        except BaseException:
            await self._disconnect()
            raise

    async def _disconnect(self):
        if self._process is None:
            return
        process, self._process = self._process, None
        if process.stdin:
            process.stdin.close()  # EOF triggers host-side finally/cleanup.
        try:
            await asyncio.wait_for(process.wait(), _SSH_EXIT_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def __aexit__(self, exc_type, exc, tb):
        # Closing stdin also works after a cancelled/timed-out RPC: do not read
        # a stale response as if it acknowledged a new close request.
        await self._disconnect()

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False, idempotent=True):
        marker = idempotent and _MARKER_RPC.get() == f"exec({cmd[:60]!r})"
        rpc_timeout = None if timeout is None else timeout + 30
        result = await self._rpc(
            dict(op="exec", command=cmd, user=user, env=env, timeout=timeout), rpc_timeout,
            cancel_grace=self.config.marker_cancel_grace if marker else 0,
        )
        if check and result["code"]:
            raise RuntimeError(f"Harbor command exited {result['code']}: {result['stderr'][:400]}")
        return result["code"], result["stdout"], result["stderr"]

    async def write_file(self, sandbox_path, content, *, user="root"):
        # Upload IDs isolate interleaved writes; chunks bound memory for large artifacts.
        stream = (
            content.open("rb") if isinstance(content, Path)
            else io.BytesIO(content.encode() if isinstance(content, str) else content)
        )
        with stream:
            upload = (await self._rpc({"op": "upload_begin"}))["upload"]
            while chunk := stream.read(1024 * 1024):
                await self._rpc(dict(op="upload_chunk", upload=upload, data=base64.b64encode(chunk).decode()), 600)
        await self._rpc(dict(op="upload_end", upload=upload, path=sandbox_path, user=user), 600)

    async def read_file(self, sandbox_path, *, user="root"):
        result = await self._rpc(dict(op="read", path=sandbox_path, user=user))
        return result["text"]
