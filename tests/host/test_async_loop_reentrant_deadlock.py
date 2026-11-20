# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import importlib.util
import asyncio
import inspect
import subprocess
import sys
import textwrap
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "vime-forge/vime/utils"


def module(name):
    """Import a complete standalone utility module, without replacing imports."""
    path = UTILS / name
    identifier = "host_" + path.stem
    spec = importlib.util.spec_from_file_location(identifier, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[identifier] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_reentrant_run_fails_instead_of_deadlocking():
    program = textwrap.dedent("""
        import importlib.util, inspect, sys
        spec = importlib.util.spec_from_file_location('async_utils', sys.argv[1])
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        async def inner():
            return 1
        async def outer():
            pending = inner()
            try:
                loaded.run(pending)
            except RuntimeError as error:
                assert 'background event loop' in str(error)
                assert inspect.getcoroutinestate(pending) == inspect.CORO_CREATED
                pending.close()
                return
            raise AssertionError('reentrant synchronous run must report its deadlock')
        loaded.run(outer())
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", program, str(UTILS / "async_utils.py")],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("run() deadlocked inside its own background loop")
    assert result.returncode == 0, result.stderr


def test_run_from_other_thread_returns_result():
    loaded = module("async_utils.py")

    async def value():
        return 7

    try:
        assert loaded.run(value()) == 7
    finally:
        loaded.get_async_loop().close()


@pytest.mark.parametrize("_attempt", range(3))
def test_concurrent_initialization_owns_one_background_loop(monkeypatch, _attempt):
    loaded = module("async_utils.py")
    original = loaded.AsyncLoopThread

    def delayed_real_construction():
        actual = original()
        # Keep the real constructor and event loop; expose a scheduling window at publication.
        time.sleep(0.02)
        return actual

    monkeypatch.setattr(loaded, "AsyncLoopThread", delayed_real_construction)
    start = threading.Barrier(8)

    def get_loop(_):
        start.wait(timeout=2)
        return loaded.get_async_loop()

    loops = []
    try:
        with ThreadPoolExecutor(max_workers=8) as workers:
            loops = list(workers.map(get_loop, range(8)))
        assert len({id(loop) for loop in loops}) == 1
    finally:
        for loop in {id(value): value for value in loops}.values():
            loop.close()
            assert loop.is_closed()


def test_running_coroutine_sees_its_real_background_loop():
    loaded = module("async_utils.py")

    async def inspect_loop():
        return asyncio.get_running_loop(), asyncio.get_event_loop()

    runner = loaded.get_async_loop()
    try:
        first, second = loaded.run(inspect_loop())
        assert first is second
        assert loaded.run(inspect_loop()) == (first, first)
    finally:
        runner.close()


def test_closed_loop_is_recreated_for_next_run():
    loaded = module("async_utils.py")

    async def value():
        return 9

    original = loaded.get_async_loop()
    assert loaded.run(value()) == 9
    original.close()
    original.close()
    assert original.is_closed()
    replacement = loaded.get_async_loop()
    try:
        assert replacement is not original
        assert loaded.run(value()) == 9
    finally:
        replacement.close()


def test_close_from_running_loop_reports_error_without_stopping_it():
    loaded = module("async_utils.py")
    runner = loaded.get_async_loop()

    async def close_self():
        with pytest.raises(RuntimeError, match="own thread"):
            runner.close()
        return 11

    try:
        assert runner.run(close_self()) == 11
        assert runner.run(asyncio.sleep(0, result=12)) == 12
    finally:
        runner.close()


def test_closed_runner_rejects_without_consuming_the_coroutine():
    loaded = module("async_utils.py")
    runner = loaded.get_async_loop()
    runner.close()
    pending = asyncio.sleep(0, result=3)
    try:
        with pytest.raises(RuntimeError, match="closed background event loop"):
            runner.run(pending, timeout=0.1)
        assert inspect.getcoroutinestate(pending) == inspect.CORO_CREATED
    finally:
        pending.close()


def test_explicit_timeout_cancels_the_submitted_coroutine():
    loaded = module("async_utils.py")
    cancelled = threading.Event()
    started = threading.Event()

    async def wait_forever():
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.set()

    runner = loaded.get_async_loop()
    try:
        with pytest.raises(FutureTimeoutError):
            loaded.run(wait_forever(), timeout=0.1)
        assert started.is_set()
        assert cancelled.wait(2), "timed-out work was not cancelled on the real loop"
        assert loaded.run(asyncio.sleep(0, result=13), timeout=1) == 13
    finally:
        runner.close()


def test_coroutine_failure_propagates_and_runner_remains_usable():
    loaded = module("async_utils.py")

    async def failure():
        raise ValueError("original coroutine failure")

    runner = loaded.get_async_loop()
    try:
        with pytest.raises(ValueError, match="original coroutine failure"):
            runner.run(failure(), timeout=1)
        assert runner.run(asyncio.sleep(0, result=14), timeout=1) == 14
    finally:
        runner.close()
