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

import asyncio
import json
import os
import shlex
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from vime.agent.harbor_sandbox import HarborConfig, HarborSandbox
from vime.agent.harbor_worker import acquire_slot
from vime.agent.sandbox import (
    E2BSandbox,
    EXIT_TIME_BUDGET_EXCEEDED,
    SandboxAdmissionTimeout,
    SandboxCommandTimeout,
    _MARKER_RPC,
    exec_and_wait,
)
from vime.agent.config import sandbox_backend, validate_sandbox_config
from examples.swe_agent_rl.sandbox import create_sandbox as swe_create_sandbox
from examples.terminal_agent_rl.sandbox import create_sandbox as tb_create_sandbox
from examples.swe_agent_rl.template_mapping import TemplateMapping, TemplateMappingEntry, apply_template_mapping


IMAGE = "org/task@sha256:" + "a" * 64
# Each task owns its provider selection; both are covered so the two copies
# cannot drift apart unnoticed.
TASK_SANDBOXES = (("swegym", swe_create_sandbox), ("terminalbench", tb_create_sandbox))
HOST = dict(ssh="root@cpu", python="/opt/venv/bin/python", worker="/opt/worker.py", root="/opt/state", slots=1)
ENV = dict(VIME_AGENT_SANDBOX_BACKEND="harbor", VIME_HARBOR_HOSTS=json.dumps([HOST]))


class BackendTests(unittest.TestCase):
    def test_default_remains_agentenv(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(sandbox_backend(), "agentenv")
            for task, create_sandbox in TASK_SANDBOXES:
                with self.subTest(task=task):
                    self.assertIsInstance(create_sandbox("task"), E2BSandbox)

    def test_invalid_backend_never_falls_back(self):
        with patch.dict(os.environ, {"VIME_AGENT_SANDBOX_BACKEND": "typo"}, clear=True):
            for task, create_sandbox in TASK_SANDBOXES:
                with self.subTest(task=task), self.assertRaises(ValueError):
                    create_sandbox(IMAGE)

    def test_missing_host_configuration_fails(self):
        with patch.dict(os.environ, {"VIME_AGENT_SANDBOX_BACKEND": "harbor"}, clear=True):
            for task, create_sandbox in TASK_SANDBOXES:
                with self.subTest(task=task), self.assertRaises(ValueError):
                    create_sandbox(IMAGE)
            with self.assertRaises(ValueError):
                validate_sandbox_config()

    def test_harbor_selection_is_shared_by_both_tasks(self):
        with patch.dict(os.environ, ENV, clear=True):
            self.assertEqual(sandbox_backend(), "harbor")
            validate_sandbox_config()  # a complete configuration passes
            for task, create_sandbox in TASK_SANDBOXES:
                with self.subTest(task=task):
                    self.assertIsInstance(create_sandbox(IMAGE), HarborSandbox)

    def test_compose_overrides_require_absolute_host_paths(self):
        for value in ("/opt/runtime.yaml", ["relative.yaml"], [None]):
            with self.subTest(value=value), patch.dict(
                os.environ, {**ENV, "VIME_HARBOR_HOSTS": json.dumps([{**HOST, "docker_compose": value}])}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "docker_compose"):
                    HarborConfig.from_env()

    def test_mirror_preserves_digest_and_rejects_template(self):
        with patch.dict(os.environ, {**ENV, "VIME_HARBOR_IMAGE_REGISTRY": "mirror.example:443"}, clear=True):
            config = HarborConfig.from_env()
            self.assertEqual(config.image_reference("docker.io/" + IMAGE), "mirror.example:443/" + IMAGE)
            with self.assertRaises(ValueError):
                config.image_reference("01a0cf3a-b9fd-7563-bd27-04fdf557ec58")

    def test_mapping_keeps_verifier_identity_for_both_backends(self):
        record = dict(schema_version=1, instance_id="task", source_image=IMAGE,
                      source_image_tag="org/task:latest", image_digest="sha256:" + "a" * 64,
                      template_id="01a0cf3a-b9fd-7563-bd27-04fdf557ec58", repo="org/repo",
                      workdir="/testbed", base_commit="b" * 40)
        mapping = TemplateMapping([TemplateMappingEntry.from_record(record, location="test")])
        md = dict(instance_id="task", image="org/task:latest", workdir="/testbed", base_commit="b" * 40)
        for backend, image in [("agentenv", record["template_id"]), ("harbor", IMAGE)]:
            with patch.dict(os.environ, {"VIME_AGENT_SANDBOX_BACKEND": backend}, clear=True):
                resolved = apply_template_mapping(md, mapping)
            self.assertEqual(resolved["image"], image)
            self.assertEqual(resolved["template_id"], record["template_id"])
            self.assertEqual(resolved["image_digest"], record["image_digest"])
        self.assertEqual(md["image"], "org/task:latest")
        with patch.dict(os.environ, {"VIME_AGENT_SANDBOX_BACKEND": "harbor"}, clear=True):
            with self.assertRaises(ValueError):
                apply_template_mapping({**md, "base_commit": "c" * 40}, mapping)

    def test_host_leases_enforce_capacity_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = acquire_slot(Path(tmp), 1)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_slot(Path(tmp), 1))
            first.close()
            second = acquire_slot(Path(tmp), 1)
            self.assertIsNotNone(second)
            second.close()


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_command_timeout_preserves_transport_without_replay(self):
        with patch.dict(os.environ, ENV, clear=True):
            sandbox = HarborSandbox(IMAGE)
        replies = [dict(error="RuntimeError: Command timed out after 15 seconds", error_kind="command_timeout"),
                   dict(code=0, stdout="next", stderr="")]
        stdin = SimpleNamespace(write=Mock(), drain=AsyncMock())
        sandbox._process = SimpleNamespace(stdin=stdin, stdout=SimpleNamespace(
            readline=AsyncMock(side_effect=[json.dumps(r).encode() + b"\n" for r in replies])))
        with self.assertRaises(SandboxCommandTimeout):
            await sandbox.exec("possibly-mutating-command", idempotent=False)
        self.assertTrue(sandbox._usable)
        self.assertEqual(stdin.write.call_count, 1)
        self.assertEqual(await sandbox.exec("next command"), (0, "next", ""))

    async def test_unclassified_worker_error_stays_fatal(self):
        with patch.dict(os.environ, ENV, clear=True):
            sandbox = HarborSandbox(IMAGE)
        sandbox._process = SimpleNamespace(
            stdin=SimpleNamespace(write=Mock(), drain=AsyncMock()),
            stdout=SimpleNamespace(
                readline=AsyncMock(return_value=b'{"error":"Command timed out after 15 seconds"}\n')
            ),
        )
        with self.assertRaises(RuntimeError) as caught:
            await sandbox.exec("test")
        self.assertNotIsInstance(caught.exception, SandboxCommandTimeout)

    async def test_compose_override_survives_ssh_quoting(self):
        paths = ["/opt/runtime policy.yaml", "/opt/second.yaml"]
        host = {**HOST, "docker_compose": paths}
        with patch.dict(os.environ, {**ENV, "VIME_HARBOR_HOSTS": json.dumps([host])}, clear=True):
            sandbox = HarborSandbox(IMAGE)
        process = SimpleNamespace(
            stdin=SimpleNamespace(write=Mock(), drain=AsyncMock(), close=Mock()),
            stdout=SimpleNamespace(readline=AsyncMock(return_value=b'{"sandbox_id":"test"}\n')),
            wait=AsyncMock(return_value=0),
        )
        with patch(
            "vime.agent.harbor_sandbox.asyncio.create_subprocess_exec", AsyncMock(return_value=process)
        ) as spawn:
            async with sandbox:
                command = shlex.split(spawn.call_args.args[-1])
                self.assertEqual(command[-4:], ["--docker-compose", paths[0], "--docker-compose", paths[1]])

    async def test_busy_host_uses_other_host_and_closes_both_transports(self):
        def process(response):
            return SimpleNamespace(
                stdin=SimpleNamespace(write=Mock(), drain=AsyncMock(), close=Mock()),
                stdout=SimpleNamespace(readline=AsyncMock(return_value=json.dumps(response).encode() + b"\n")),
                wait=AsyncMock(return_value=0),
            )
        busy, ready = process({"busy": True}), process({"sandbox_id": "test"})
        hosts = [HOST, {**HOST, "ssh": "root@other"}]
        with patch.dict(os.environ, {**ENV, "VIME_HARBOR_HOSTS": json.dumps(hosts)}, clear=True):
            sandbox = HarborSandbox(IMAGE)
        with patch("vime.agent.harbor_sandbox.random.shuffle"), patch(
            "vime.agent.harbor_sandbox.asyncio.create_subprocess_exec", AsyncMock(side_effect=[busy, ready])
        ):
            async with sandbox:
                self.assertEqual(sandbox.sandbox_id, "root@other:test")
                busy.stdin.close.assert_called_once()
            ready.stdin.close.assert_called_once()

    async def test_interrupted_rpc_cannot_consume_a_stale_reply(self):
        with patch.dict(os.environ, ENV, clear=True):
            sandbox = HarborSandbox(IMAGE)

        async def delayed_reply():
            await asyncio.sleep(5)
            return b'{"text":"stale"}\n'
        stdin = SimpleNamespace(write=Mock(), drain=AsyncMock())
        sandbox._process = SimpleNamespace(stdin=stdin, stdout=SimpleNamespace(readline=delayed_reply))
        with self.assertRaises(TimeoutError):
            await sandbox._rpc({"op": "read"}, timeout=0.01)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            await sandbox._rpc({"op": "exec"})
        self.assertEqual(stdin.write.call_count, 1)

    async def test_command_result_and_check_semantics(self):
        with patch.dict(os.environ, ENV, clear=True):
            sandbox = HarborSandbox(IMAGE)
        sandbox._rpc = AsyncMock(return_value=dict(code=7, stdout="out", stderr="failure"))
        self.assertEqual(await sandbox.exec("exit 7", user="agent", env={"A": "B"}), (7, "out", "failure"))
        request = sandbox._rpc.call_args.args[0]
        self.assertEqual(request["user"], "agent")
        self.assertEqual(request["env"], {"A": "B"})
        with self.assertRaises(RuntimeError):
            await sandbox.exec("exit 7", check=True)

    async def test_upload_chunks_large_local_artifact(self):
        with patch.dict(os.environ, ENV, clear=True):
            sandbox = HarborSandbox(IMAGE)
        sandbox._rpc = AsyncMock(
            side_effect=lambda request, *a: {"upload": "id"} if request["op"] == "upload_begin" else {"ok": True}
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "runtime.tar"
            artifact.write_bytes(b"a" * (2 * 1024 * 1024 + 1))
            await sandbox.write_file("/tmp/runtime.tar", artifact, user="agent")
        requests = [call.args[0] for call in sandbox._rpc.call_args_list]
        self.assertEqual(
            [r["op"] for r in requests],
            ["upload_begin", "upload_chunk", "upload_chunk", "upload_chunk", "upload_end"],
        )
        self.assertEqual(requests[-1]["user"], "agent")


class AdmissionProtocolTests(unittest.IsolatedAsyncioTestCase):
    def sandbox(self):
        with patch.dict(os.environ, {**ENV, "VIME_HARBOR_PREPARE_CONCURRENCY": "1"}, clear=True):
            sb = HarborSandbox(IMAGE)
        sb._queue_hosts = AsyncMock(return_value=[HOST])

        async def connect(host):
            sb._host = host
        sb._connect = AsyncMock(side_effect=connect)
        sb._disconnect = AsyncMock()
        return sb

    async def test_fifo_poll_keeps_request_identity_then_starts_once(self):
        sb = self.sandbox()
        sb._rpc = AsyncMock(side_effect=[dict(admitted=False, position=1), dict(admitted=True),
                                        dict(sandbox_id="created"), dict(ok=True)])
        with patch("vime.agent.harbor_sandbox.asyncio.sleep", AsyncMock()):
            async with sb.preparation():
                await sb.__aenter__()
        requests = [c.args[0] for c in sb._rpc.call_args_list]
        self.assertEqual([r["op"] for r in requests],
                         ["admission_join", "admission_poll", "start", "prepare_end"])
        self.assertEqual(requests[0]["request_id"], requests[1]["request_id"])
        sb._connect.assert_awaited_once()

    async def test_deadline_during_grant_reply_closes_worker_and_classifies(self):
        sb = self.sandbox()
        sb.config = replace(sb.config, prepare_acquire_timeout=0.02)

        async def lost_reply(*args, **kwargs):
            await asyncio.sleep(1)
        sb._rpc = AsyncMock(side_effect=lost_reply)
        with self.assertRaisesRegex(SandboxAdmissionTimeout, "phase=queue_registration.*request=.*budget_sec"):
            async with sb.preparation():
                self.fail("must not start Docker")
        sb._disconnect.assert_awaited_once()
        sb._rpc.assert_awaited_once()  # Never replay a possibly granted request.

    async def test_transport_timeout_is_not_capacity_timeout(self):
        sb = self.sandbox()
        sb._rpc = AsyncMock(side_effect=TimeoutError("transport timeout"))
        with self.assertRaises(TimeoutError) as caught:
            async with sb.preparation():
                pass
        self.assertNotIsInstance(caught.exception, SandboxAdmissionTimeout)
        sb._disconnect.assert_awaited_once()

    async def test_full_queue_tries_other_host_before_registration(self):
        sb = self.sandbox()
        other = {**HOST, "ssh": "root@other"}
        sb._queue_hosts.return_value = [HOST, other]
        sb._rpc = AsyncMock(side_effect=[dict(full=True), dict(admitted=True)])
        await sb._admit()
        self.assertEqual(sb._host, other)
        self.assertEqual(sb._connect.await_count, 2)
        sb._disconnect.assert_awaited_once()

    async def test_probe_prefers_shorter_host_queue(self):
        sb = self.sandbox()
        other = {**HOST, "ssh": "root@other"}
        sb.config = replace(sb.config, hosts=(HOST, other))
        sb._rpc = AsyncMock(side_effect=[dict(waiting=8, preparing=1, active=1),
                                        dict(waiting=2, preparing=1, active=1)])
        with patch("vime.agent.harbor_sandbox.random.shuffle"):
            hosts = await HarborSandbox._queue_hosts(sb)
        self.assertEqual(hosts, [other, HOST])
        self.assertEqual(sb._disconnect.await_count, 2)

    async def test_admission_rejects_late_preparation_hook(self):
        sb = self.sandbox()
        sb.sandbox_id = "already-created"
        with self.assertRaisesRegex(RuntimeError, "precede"):
            async with sb.preparation():
                pass
        sb._queue_hosts.assert_not_awaited()


class MarkerCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with patch.dict(os.environ, ENV, clear=True):
            self.sandbox = HarborSandbox(IMAGE)
        self.sandbox.config = replace(self.sandbox.config, marker_cancel_grace=0.2)
        self.reader = asyncio.StreamReader()
        self.written = asyncio.Event()
        self.requests = []

        def write(data):
            self.requests.append(json.loads(data))
            self.written.set()

        self.sandbox._process = SimpleNamespace(
            stdin=SimpleNamespace(write=Mock(side_effect=write), drain=AsyncMock()),
            stdout=self.reader,
        )

    def reply(self, code=0, stdout=""):
        self.reader.feed_data(json.dumps(dict(code=code, stdout=stdout, stderr="")).encode() + b"\n")

    def marker_task(self, *, idempotent=True):
        cmd = "test -f /tmp/.run.done && cat /tmp/.run.done"
        token = _MARKER_RPC.set(f"exec({cmd[:60]!r})")
        try:
            return asyncio.create_task(self.sandbox.exec(cmd, idempotent=idempotent))
        finally:
            _MARKER_RPC.reset(token)

    async def test_agent_deadline_still_allows_log_and_patch_collection(self):
        self.sandbox.write_file = AsyncMock()

        def write(data):
            request = json.loads(data)
            self.requests.append(request)
            command = request["command"]
            if command.startswith("test -f"):
                asyncio.get_running_loop().call_later(0.08, self.reply, 1, "")
            elif command.startswith("tail"):
                self.reply(stdout="agent log tail")
            elif command == "git diff":
                self.reply(stdout="collected patch")
            else:
                self.reply()

        self.sandbox._process.stdin.write.side_effect = write
        # Skip only the five-second polling interval; real deadline and stream
        # waits reproduce cancellation in the production exec_and_wait path.
        with patch("vime.agent.sandbox.asyncio.sleep", AsyncMock()):
            code, tail = await exec_and_wait(
                self.sandbox, cmd="agent", tag="run", time_budget_sec=0.04,
            )
        self.assertEqual(code, EXIT_TIME_BUDGET_EXCEEDED)
        self.assertEqual(tail, "agent log tail")
        self.assertEqual(await self.sandbox.exec("git diff"), (0, "collected patch", ""))
        self.assertTrue(self.sandbox._usable)
        self.assertEqual(sum(r["command"].startswith("test -f") for r in self.requests), 1)

    async def test_cancelled_marker_drains_before_next_rpc_without_replay(self):
        marker = self.marker_task()
        await self.written.wait()
        marker.cancel()
        following = asyncio.create_task(self.sandbox.exec("git diff"))
        await asyncio.sleep(0)
        self.assertEqual(len(self.requests), 1)
        self.reply(1, "old marker reply")
        with self.assertRaises(asyncio.CancelledError):
            await marker
        self.written.clear()
        if len(self.requests) < 2:
            await self.written.wait()
        self.reply(stdout="new patch reply")
        self.assertEqual(await following, (0, "new patch reply", ""))
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(self.sandbox._usable)

    async def test_drain_grace_expiry_invalidates_stream(self):
        self.sandbox.config = replace(self.sandbox.config, marker_cancel_grace=0.01)
        marker = self.marker_task()
        await self.written.wait()
        marker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(marker, 1)
        self.reply(stdout="too late")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            await self.sandbox.exec("git diff")
        self.assertEqual(len(self.requests), 1)

    async def test_worker_eof_during_drain_invalidates_stream(self):
        marker = self.marker_task()
        await self.written.wait()
        marker.cancel()
        self.reader.feed_eof()
        with self.assertRaises(asyncio.CancelledError):
            await marker
        self.assertFalse(self.sandbox._usable)

    async def test_non_idempotent_command_does_not_opt_into_drain(self):
        marker = self.marker_task(idempotent=False)
        await self.written.wait()
        marker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await marker
        self.assertFalse(self.sandbox._usable)
        self.assertEqual(len(self.requests), 1)

    async def test_second_cancellation_stops_drain_and_invalidates_stream(self):
        marker = self.marker_task()
        await self.written.wait()
        marker.cancel()
        await asyncio.sleep(0)
        marker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await marker
        self.assertFalse(self.sandbox._usable)
        self.assertEqual(len(self.requests), 1)


if __name__ == "__main__":
    unittest.main()
