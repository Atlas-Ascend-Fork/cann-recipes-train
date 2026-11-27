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

"""CPU-only compatibility check using a real router and four loopback mock workers.

Run with the training interpreter from the repository root:
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. python tools/check_swegym_router.py --output /external/check-results

No Ray cluster, model, sandbox, rollout, or training job is started. The router
is launched through Vime's production wrapper and always stopped on exit.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import aiohttp
from aiohttp import web
from packaging.version import Version
from vllm_router.router_args import RouterArgs

from vime.agent.adapters.common import Session, call_vllm_generate


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def eventually(check, timeout=15):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        result = await check()
        if result:
            return result
        await asyncio.sleep(0.05)
    raise AssertionError(f"Condition did not become true within {timeout}s")


async def check_router(output: Path, report: dict, policy: str):
    output.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("vllm-router")
    assert Version(version) >= Version("0.1.15"), version
    report.update(version=version, python=sys.executable, policy=policy, checks=[])
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    args = parser.parse_args(["--router-policy", policy])
    config = RouterArgs.from_cli_args(args, use_router_prefix=True)
    assert config.policy == policy
    config.host = config.prometheus_host = "127.0.0.1"
    config.port, config.prometheus_port = free_port(), free_port()
    while config.prometheus_port == config.port:
        config.prometheus_port = free_port()
    config.log_level = "warning"
    config.request_timeout_secs = 5
    config.disable_circuit_breaker = True
    # Match Vime's add_vllm_arguments defaults.
    config.balance_abs_threshold = 10
    config.balance_rel_threshold = 1.2
    config.disable_retries = True
    config.health_check_interval_secs = 1
    config.health_success_threshold = 1
    report["checks"].append("prefixed RouterArgs API")
    router_url = f"http://127.0.0.1:{config.port}"
    report["router_url"] = router_url
    records, runners, workers, pending = [], [], [], []
    active = collections.Counter()
    gate = asyncio.Event()
    router = None

    async def health(request):
        return web.json_response({"status": "ok"})

    def generate_handler(index):
        async def generate(request):
            body = await request.json()
            records.append({"worker": index, "session": request.headers.get("x-session-id"), "body": body})
            active[index] += 1
            try:
                if body.get("compat_hold"):
                    await gate.wait()
                if body.get("compat_error"):
                    return web.json_response({"error": "deliberate compatibility-test error"}, status=400)
                return web.json_response({
                    "choices": [{"token_ids": [index, 42], "finish_reason": "length",
                                 "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]}}],
                    "compat_worker": index,
                })
            finally:
                active[index] -= 1
        return generate

    async def worker_list(client):
        async with client.get(router_url + "/workers") as response:
            response.raise_for_status()
            return (await response.json())["workers"]

    async def idle(client):
        return all(worker["load"] == 0 for worker in await worker_list(client))

    async def post(client, session_id, payload):
        async with client.post(router_url + "/inference/v1/generate", json=payload,
                               headers={"x-session-id": session_id}) as response:
            body = await response.text()
            try:
                body = json.loads(body)
            except ValueError:
                pass
            return response.status, body

    try:
        for index in range(4):
            app = web.Application()
            app.router.add_get("/health", health)
            app.router.add_post("/inference/v1/generate", generate_handler(index))
            runner = web.AppRunner(app)
            runners.append(runner)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            workers.append(f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}")

        config_path = output / "router-args.json"
        config_path.write_text(json.dumps(vars(config), indent=2))
        child_code = (
            "import json,sys; from vllm_router.router_args import RouterArgs; "
            "from vime.utils.http_utils import run_router; "
            "sys.exit(run_router(RouterArgs(**json.load(open(sys.argv[1])))))"
        )
        with (output / "router.log").open("w") as log:
            router = await asyncio.create_subprocess_exec(
                sys.executable, "-c", child_code, str(config_path), stdout=log, stderr=log,
                env={**os.environ, "TOKIO_WORKER_THREADS": "2"},
            )
        report["router_pid"] = router.pid
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), trust_env=False) as client:
            async def ready():
                if router.returncode is not None:
                    raise AssertionError(f"Router exited {router.returncode}; see router.log")
                try:
                    async with client.get(router_url + "/health") as response:
                        return response.status == 200
                except aiohttp.ClientError:
                    return False
            await eventually(ready)
            for url in workers:
                async with client.post(
                    router_url + "/workers", json={"url": url, "worker_type": "regular"}
                ) as response:
                    assert response.status in (200, 201, 202), await response.text()

            async def all_healthy():
                registered = await worker_list(client)
                return len(registered) == 4 and all(w["is_healthy"] for w in registered)
            await eventually(all_healthy)
            report["checks"].append("four dynamic worker registrations and health checks")

            payload = {"token_ids": [151644, 8948, 198], "sampling_params": {
                "max_tokens": 17, "temperature": 0.7, "top_p": 0.9, "top_k": -1,
                "seed": 123, "logprobs": 1, "stop_token_ids": [151645], "skip_special_tokens": False,
            }}
            assignments = {f"accounting-{i}": None for i in range(16)}
            if policy == "consistent_hash":
                # Realistic IDs: 32 prompts x 8 attempts. All first-turn prompts are
                # identical, then grow, proving that the session header drives affinity.
                assignments = {}
                for turn in range(3):
                    for prompt in range(32):
                        for sample in range(8):
                            sid = f"cagent-compat-task-{prompt}-{prompt * 8 + sample}-{prompt}"
                            body = {**payload, "token_ids": payload["token_ids"] + [turn] * turn}
                            status, result = await post(client, sid, body)
                            assert status == 200, result
                            worker = result["compat_worker"]
                            assert assignments.setdefault(sid, worker) == worker, (sid, turn)
                            record = records[-1]
                            assert record["session"] == sid
                            assert record["body"]["token_ids"] == body["token_ids"]
                            assert record["body"]["sampling_params"] == body["sampling_params"]
                            choice = result["choices"][0]
                            assert choice["token_ids"] == [worker, 42]
                            assert choice["logprobs"]["content"] == [{"logprob": -0.1}, {"logprob": -0.2}]
                distribution = collections.Counter(assignments.values())
                assert set(distribution) == set(range(4)), distribution
                report["sessions_per_worker"] = dict(distribution)
                report["synthetic_sessions"] = len(assignments)
                report["synthetic_turns"] = 3 * len(assignments)
                report["checks"] += ["session affinity across changing prompts", "all four workers used",
                                     "token IDs, sampling parameters, logprobs and session headers preserved"]

            adapter = SimpleNamespace(vllm_url=router_url, logger=logging.getLogger(__name__),
                                      log_prefix="router-compat", max_token_keys=("max_tokens",), stop_keys=("stop",))
            turn = await call_vllm_generate(
                [10, 11, 12], Session(sampling_defaults={"max_new_tokens": 7}, max_context_tokens=5),
                {"temperature": 0.5}, adapter=adapter, session_id="adapter-compat-session",
            )
            assert turn.prompt_ids == [10, 11, 12]
            assert turn.output_ids == [records[-1]["worker"], 42]
            assert turn.output_log_probs == [-0.1, -0.2] and turn.finish_reason == "length"
            assert records[-1]["body"]["sampling_params"]["max_tokens"] == 2
            report["checks"].append("production agent adapter and context-budget clamp")
            await eventually(lambda: idle(client))

            # Hold requests at the mock backends so load accounting is observable.
            pending = [asyncio.create_task(post(client, sid, {**payload, "compat_hold": True}))
                       for sid in list(assignments)[:16]]

            async def all_inflight():
                loads = sum(w["load"] for w in await worker_list(client))
                return sum(active.values()) == 16 and (policy != "cache_aware" or loads == 16)
            await eventually(all_inflight)
            report["inflight_loads"] = [w["load"] for w in await worker_list(client)]
            report["inflight_requests_per_worker"] = dict(active)
            gate.set()
            assert all(status == 200 for status, _ in await asyncio.gather(*pending))
            await eventually(lambda: idle(client))
            report["checks"].append("16 concurrent requests complete successfully")
            if policy == "cache_aware":
                report["checks"].append("in-flight load increments and completion releases")

            status, _ = await post(client, "error-session", {**payload, "compat_error": True})
            assert status == 400
            await eventually(lambda: idle(client))
            report["checks"].append("backend error forwarding and load cleanup")

            gate.clear()
            cancelled = asyncio.create_task(post(client, "cancel-session", {**payload, "compat_hold": True}))
            pending.append(cancelled)

            async def one_inflight():
                loads = sum(w["load"] for w in await worker_list(client))
                return sum(active.values()) == 1 and (policy != "cache_aware" or loads == 1)
            await eventually(one_inflight)
            cancelled.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancelled
            # Non-streaming inference may continue upstream after a client leaves.
            # Require bounded cleanup after the backend finishes, not instant abort.
            await asyncio.sleep(0.2)
            report["load_after_client_disconnect"] = sum(w["load"] for w in await worker_list(client))
            gate.set()
            await eventually(lambda: idle(client))

            async def backend_idle():
                return sum(active.values()) == 0
            await eventually(backend_idle)
            report["checks"].append("no leaked load after cancelled request finishes upstream")
            report["final_worker_loads"] = [w["load"] for w in await worker_list(client)]
    finally:
        gate.set()
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if router is not None and router.returncode is None:
            router.terminate()
            try:
                await asyncio.wait_for(router.wait(), timeout=10)
            except TimeoutError:
                router.kill()
                await router.wait()
        for runner in runners:
            await runner.cleanup()
        report["test_services_stopped"] = router is None or router.returncode is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "scope": "loopback router and mock workers only"}
    try:
        report["policies"] = {}
        for policy in ("consistent_hash", "cache_aware"):
            result = report["policies"][policy] = {}
            asyncio.run(check_router(args.output / policy, result, policy))
        report["status"] = "passed"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        # stdout exposes the JSON result for redirection and machine parsing.
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
