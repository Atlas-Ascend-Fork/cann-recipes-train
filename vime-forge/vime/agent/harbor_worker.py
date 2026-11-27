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

"""Host-side Harbor 0.24.0 worker, invoked over SSH (never a public daemon).

Deploy this file and requirements-harbor.txt to a CPU host. stdin/stdout carry
JSON RPC; stderr carries diagnostics. EOF, lifetime expiry and signals clean up
the environment. No model, training code, bundle rewrite or agent loop lives here.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import fcntl
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import shlex
import signal
import sys
import tempfile
import time
import uuid


def acquire_slot(root: Path, slots: int):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(slots):
        handle = (root / f"slot-{index}.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            handle.close()
    return None


class PreparationQueue:
    """FIFO admission shared by SSH workers on one host/local filesystem.

    A locked ticket is the worker's liveness token. All queue changes use one
    short mutex; no Docker or network operation holds it. Kernel locks release
    on worker death, so a subsequent operation can reap abandoned tickets.
    """
    def __init__(self, root: Path, slots: int):
        self.root, self.slots = root, slots
        self.directory = root / "preparation-queue"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ticket = self.active = self.permit = None
        self.request_id = None
        self.deadline = None
        self.limit = None
        self.granted = False

    @contextlib.contextmanager
    def locked(self):
        with (self.directory / "queue.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def pending(self):
        entries = []
        for path in sorted(self.directory.glob("*.ticket")):
            with path.open() as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    data = json.load(handle)
                    if data["deadline"] > time.monotonic():
                        entries.append((path, data))
                        continue
                path.unlink(missing_ok=True)
        return entries

    @staticmethod
    def occupied(root):
        count = 0
        for path in root.glob("slot-*.lock"):
            with path.open("a+") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    count += 1
        return count

    def snapshot(self):
        with self.locked():
            return dict(waiting=len(self.pending()),
                        preparing=self.occupied(self.root / "preparation-leases"),
                        active=self.occupied(self.root / "leases"), slots=self.slots)

    def join(self, request_id, limit, queue_limit, timeout):
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise ValueError("Invalid admission request ID")
        if any(type(value) is not int or value < 1 for value in (limit, queue_limit, timeout)):
            raise ValueError("Admission limits and timeout must be positive integers")
        if self.request_id is not None:
            if request_id != self.request_id:
                raise ValueError("Worker already owns a different admission request")
            return self.poll(request_id)
        with self.locked():
            pending = self.pending()
            if any(data["limit"] != limit for _, data in pending):
                raise ValueError("Workers sharing a queue must use the same preparation limit")
            if len(pending) >= queue_limit:
                return dict(full=True, waiting=len(pending))
            now = time.monotonic()
            path = self.directory / f"{time.monotonic_ns():020d}-{request_id}.ticket"
            self.ticket = path.open("x+")
            fcntl.flock(self.ticket, fcntl.LOCK_EX)
            self.request_id, self.limit, self.deadline = request_id, limit, now + timeout
            json.dump(dict(request_id=request_id, limit=limit, created=now, deadline=self.deadline), self.ticket)
            self.ticket.flush()
        return self.poll(request_id)

    def poll(self, request_id):
        if request_id != self.request_id or self.request_id is None:
            raise ValueError("Unknown admission request ID")
        with self.locked():
            if self.granted:
                return dict(admitted=True, request_id=request_id)
            entries = self.pending()
            if time.monotonic() >= self.deadline:
                self.remove_ticket()
                return dict(expired=True, request_id=request_id)
            position = next(i + 1 for i, (p, _) in enumerate(entries) if str(p) == self.ticket.name)
            if position == 1:
                active = acquire_slot(self.root / "leases", self.slots)
                if active is not None:
                    permit = acquire_slot(self.root / "preparation-leases", self.limit)
                    if permit is not None:
                        self.active, self.permit, self.granted = active, permit, True
                        self.remove_ticket()
                        return dict(admitted=True, request_id=request_id)
                    active.close()  # Never hold one capacity while waiting for the other.
            return dict(admitted=False, request_id=request_id, position=position, waiting=len(entries))

    def remove_ticket(self):
        if self.ticket is not None:
            Path(self.ticket.name).unlink(missing_ok=True)
            self.ticket.close()
            self.ticket = None

    def release_preparation(self):
        if self.permit is not None:
            self.permit.close()
            self.permit = None

    def close(self):
        with self.locked():
            self.remove_ticket()
            self.release_preparation()
            if self.active is not None:
                self.active.close()
                self.active = None


def command_error(exc, request, elapsed):
    """Classify the pinned Harbor Docker timeout, never a socket/SSH timeout."""
    kind = "command_error"
    if (request.get("op") == "exec" and isinstance(exc, RuntimeError)
            and re.fullmatch(r"Command timed out after [0-9.]+ seconds", str(exc))):
        kind = "command_timeout"
    return dict(error=f"{type(exc).__name__}: {exc}", error_kind=kind,
                operation=request.get("op"), elapsed_sec=round(elapsed, 3))


async def serve(args, reader, output):
    def respond(value):
        output.write(json.dumps(value) + "\n")
        output.flush()

    if importlib.metadata.version("harbor") != "0.24.0":
        raise RuntimeError("Install the pinned requirements-harbor.txt (Harbor 0.24.0)")
    root = Path(args.root)
    lease = None
    admission = PreparationQueue(root, args.slots)
    environment = None
    uploads = {}
    session = "vime-" + uuid.uuid4().hex
    work = root / "sessions" / session
    try:
        while line := await reader.readline():
            request = {}
            started = time.monotonic()
            try:
                request = json.loads(line)
                op = request["op"]
                if op == "admission_snapshot":
                    respond(admission.snapshot())
                elif op == "admission_join":
                    if environment is not None:
                        raise ValueError("Admission must precede Docker creation")
                    respond(admission.join(request["request_id"], request["limit"],
                                           request["queue_limit"], request["timeout"]))
                elif op == "admission_poll":
                    respond(admission.poll(request["request_id"]))
                elif op == "admission_cancel":
                    if environment is not None:
                        raise ValueError("Close the worker to cancel a started environment")
                    admission.close()
                    respond({"ok": True})
                    return
                elif op == "start":
                    if environment is not None:
                        raise ValueError("Environment already started")
                    if admission.request_id is not None:
                        if not admission.granted:
                            raise ValueError("FIFO admission has not been granted")
                    else:
                        lease = acquire_slot(root / "leases", args.slots)
                        if lease is None:
                            respond({"busy": True})
                            return
                    from harbor.environments.docker.docker import DockerEnvironment
                    from harbor.models.task.config import EnvironmentConfig
                    from harbor.models.trial.paths import TrialPaths
                    image = request["image"]
                    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
                        raise ValueError("Expected a digest-pinned image")
                    work.mkdir(parents=True)
                    definition = work / "environment"
                    definition.mkdir()
                    environment = DockerEnvironment(
                        environment_dir=definition, environment_name=session,
                        session_id=session, trial_paths=TrialPaths(trial_dir=work),
                        task_env_config=EnvironmentConfig(docker_image=image, cpus=args.cpus,
                                                          memory_mb=args.memory_mb, gpus=0),
                        mounts=[], enable_environment_dir_upload=False,
                        extra_docker_compose=args.docker_compose,
                    )
                    await environment.start(force_build=False)
                    respond({"sandbox_id": session})
                elif environment is None:
                    raise ValueError("start must be the first operation")
                elif op == "prepare_end":
                    admission.release_preparation()
                    respond({"ok": True})
                elif op == "exec":
                    result = await environment.exec(command=request["command"], user=request["user"],
                                                    env=request.get("env"), timeout_sec=request["timeout"])
                    respond(dict(code=result.return_code, stdout=result.stdout or "", stderr=result.stderr or ""))
                elif op == "read":
                    result = await environment.exec(command="cat -- " + shlex.quote(request["path"]),
                                                    user=request["user"], timeout_sec=120)
                    # Existing E2BSandbox reads are best-effort: a missing or
                    # unreadable grading/log file returns an empty string.
                    respond({"text": (result.stdout or "") if result.return_code == 0 else ""})
                elif op == "upload_begin":
                    upload = uuid.uuid4().hex
                    uploads[upload] = tempfile.NamedTemporaryFile(dir=work, prefix="upload-", delete=False)
                    respond({"upload": upload})
                elif op == "upload_chunk":
                    uploads[request["upload"]].write(base64.b64decode(request["data"], validate=True))
                    respond({"ok": True})
                elif op == "upload_end":
                    handle = uploads.pop(request["upload"])
                    handle.close()
                    staged = "/tmp/vime-upload-" + request["upload"]
                    target = request["path"]
                    try:
                        await environment.upload_file(Path(handle.name), staged)
                        result = await environment.exec(
                            command=f"chmod 644 {shlex.quote(staged)}", user="root", timeout_sec=120
                        )
                        if result.return_code:
                            raise RuntimeError(result.stderr)
                        result = await environment.exec(
                            command=(
                                f"mkdir -p -- {shlex.quote(str(Path(target).parent))} && "
                                f"cat -- {shlex.quote(staged)} > {shlex.quote(target)}"
                            ),
                            user=request["user"], timeout_sec=600)
                        if result.return_code:
                            raise RuntimeError(result.stderr)
                    finally:
                        Path(handle.name).unlink(missing_ok=True)
                        await environment.exec(command="rm -f -- " + shlex.quote(staged), user="root", timeout_sec=30)
                    respond({"ok": True})
                else:
                    raise ValueError(f"Unknown operation {op!r}")
            except Exception as exc:
                error = command_error(exc, request, time.monotonic() - started)
                logging.warning("sandbox=%s %s", session, json.dumps(error))
                respond(error)
                if request.get("op") == "start":
                    return
            finally:
                elapsed = time.monotonic() - started
                if elapsed >= 5:
                    logging.info("sandbox=%s op=%s elapsed_sec=%.3f", session, request.get("op"), elapsed)
    finally:
        try:
            if environment is not None:
                await asyncio.wait_for(environment.stop(delete=True), 120)
        finally:
            admission.close()
            for handle in uploads.values():
                handle.close()
                Path(handle.name).unlink(missing_ok=True)
            if lease is not None:
                lease.close()


async def main(args, output):
    reader = asyncio.StreamReader(limit=64 * 1024 * 1024)
    loop = asyncio.get_running_loop()
    transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    task = asyncio.create_task(serve(args, reader, output))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        await asyncio.wait_for(task, args.lifetime)
    finally:
        transport.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--lifetime", type=int, default=4500)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--memory-mb", type=int, default=65536)
    parser.add_argument("--docker-compose", type=Path, action="append", default=[],
                        help="Trusted, external CPU-host Compose override (repeatable)")
    args = parser.parse_args()
    if any(not path.is_absolute() or not path.is_file() for path in args.docker_compose):
        parser.error("Docker Compose overrides must be existing absolute CPU-host files")
    if min(args.slots, args.lifetime, args.cpus, args.memory_mb) <= 0:
        parser.error("Resource limits and timeouts must be positive")
    output = sys.stdout
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    with contextlib.redirect_stdout(sys.stderr):
        asyncio.run(main(args, output))
