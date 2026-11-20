# Copyright 2025 Zhipu AI
# Derived from THUDM/slime (Apache-2.0). Original notices: vime-forge/LICENSE.
# Source: https://github.com/THUDM/slime/blob/main/slime/utils/async_utils.py
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
# Huawei copyright applies to modifications; original notices are retained.

import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError

__all__ = ["get_async_loop", "run"]


# Create a background event loop thread
class AsyncLoopThread:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="vime-async-loop", daemon=True
        )
        self._thread.start()

    def is_closed(self):
        return self._loop.is_closed()

    def close(self):
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot close a background event loop from its own thread")
        with self._close_lock:
            if self._loop.is_closed():
                return
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._loop.close()

    def run(self, coro, timeout=None):
        """Submit a coroutine; rejection leaves its lifecycle with the caller."""
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot synchronously run a coroutine from its own background event loop")
        with self._close_lock:
            if self._loop.is_closed():
                raise RuntimeError("Cannot run a coroutine on a closed background event loop")
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            raise


# Create one global instance
_async_loop = None
_async_loop_lock = threading.Lock()


def get_async_loop():
    global _async_loop
    with _async_loop_lock:
        if _async_loop is not None and not _async_loop.is_closed():
            return _async_loop
        _async_loop = AsyncLoopThread()
        return _async_loop


def run(coro, timeout=None):
    """Run in the background loop; an explicit timeout bounds the wait in seconds."""
    return get_async_loop().run(coro, timeout=timeout)
