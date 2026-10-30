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

"""Opt-in structured performance events for offline timeline rendering.

The disabled path intentionally avoids clocks, UUID generation, JSON
serialization, background threads, and resource discovery.
"""

from __future__ import annotations

import functools
import atexit
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

PERF_TRACE_PREFIX = "VIME_PERF_TRACE "

# Process-local configuration initialized once by the driver. Existing APIs
# still accept args while callers are migrated to the Timer-style decorator.
_ACTIVE_ARGS: Any | None = None
_CURRENT_ROLLOUT_STEP: int | None = None


def _current_perf_args() -> Any | None:
    return _ACTIVE_ARGS


def set_perf_timeline_step(step: int) -> None:
    global _CURRENT_ROLLOUT_STEP
    _CURRENT_ROLLOUT_STEP = step


def perf_timeline_enabled(args: Any) -> bool:
    if not getattr(args, "enable_perf_timeline", False):
        return False
    step_start = getattr(args, "perf_timeline_step_start", None)
    step_end = getattr(args, "perf_timeline_step_end", None)
    if step_start is not None or step_end is not None:
        step = _CURRENT_ROLLOUT_STEP
        if step is None:
            step = 0
        if step_start is not None and step < step_start:
            return False
        if step_end is not None and step > step_end:
            return False
    return True


def _safe_rank() -> int | None:
    rank = os.environ.get("RANK")
    return int(rank) if rank is not None else None


def _get_node_ip() -> str:
    """Get the primary IP address of this node for multi-machine identification."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostname()


@dataclass(frozen=True)
class _Source:
    source_id: str
    node: str
    pid: int
    rank: int | None


_SOURCE_LOCK = threading.Lock()
_SOURCE: _Source | None = None
_SOURCE_SEQ = 0


def _get_source() -> _Source:
    global _SOURCE
    if _SOURCE is not None:
        return _SOURCE
    with _SOURCE_LOCK:
        if _SOURCE is None:
            rank = _safe_rank()
            node = _get_node_ip()
            pid = os.getpid()
            source_id = f"{node}:{pid}"
            _SOURCE = _Source(
                source_id=source_id,
                node=node,
                pid=pid,
                rank=rank,
            )
    return _SOURCE


def _next_source_seq() -> int:
    global _SOURCE_SEQ
    with _SOURCE_LOCK:
        value = _SOURCE_SEQ
        _SOURCE_SEQ += 1
        return value


class _PerfWriter:
    def __init__(self, flush_events: int, run_id: str):
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._flush_events = max(1, flush_events)
        self._stop = threading.Event()
        self._high_watermark = 0
        self._total_events = 0
        self._perf_run_id = run_id
        self._thread = threading.Thread(target=self._run, name="vime-perf-timeline-writer", daemon=True)
        self._thread.start()

    def emit(self, event: dict[str, Any]) -> None:
        self._queue.put_nowait(event)
        self._high_watermark = max(self._high_watermark, self._queue.qsize())
        self._total_events += 1

    def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        while not self._stop.is_set() or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                if batch:
                    self._write(batch)
                    batch.clear()
                continue
            batch.append(event)
            if len(batch) >= self._flush_events:
                self._write(batch)
                batch.clear()
        if batch:
            self._write(batch)

    @staticmethod
    def _write(events: list[dict[str, Any]]) -> None:
        for event in events:
            logger.info(
                "%s%s",
                PERF_TRACE_PREFIX,
                json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str),
            )

    def close(self) -> None:
        source = _get_source()
        self.emit(
            {
                "schema_version": 1,
                "type": "counter",
                "event_id": uuid.uuid4().hex,
                "perf_run_id": self._perf_run_id,
                "source_id": source.source_id,
                "source_seq": _next_source_seq(),
                "name": "perf_writer",
                "wall_ns": time.time_ns(),
                "values": {
                    "total_event_count": self._total_events,
                    "buffer_high_watermark": self._high_watermark,
                },
            }
        )
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning(
                "Perf timeline writer did not drain within 2 seconds; buffered events may be lost (queue_size=%d)",
                self._queue.qsize(),
            )


_WRITER_LOCK = threading.Lock()
_WRITER: _PerfWriter | None = None


def init_perf_timeline(args: Any) -> None:
    global _ACTIVE_ARGS
    _ACTIVE_ARGS = args
    global _WRITER
    if not getattr(args, "enable_perf_timeline", False):
        return
    run_id = args.perf_run_id
    _get_source()
    with _WRITER_LOCK:
        if _WRITER is None:
            _WRITER = _PerfWriter(
                flush_events=args.perf_timeline_flush_events,
                run_id=run_id,
            )


def shutdown_perf_timeline() -> None:
    global _WRITER
    with _WRITER_LOCK:
        writer = _WRITER
        _WRITER = None
    if writer is not None:
        writer.close()
    global _ACTIVE_ARGS
    _ACTIVE_ARGS = None


atexit.register(shutdown_perf_timeline)


def _base_event(args: Any, event_type: str, name: str) -> dict[str, Any]:
    source = _get_source()
    return {
        "schema_version": 1,
        "type": event_type,
        "event_id": uuid.uuid4().hex,
        "perf_run_id": args.perf_run_id,
        "source_id": source.source_id,
        "source_seq": _next_source_seq(),
        "node": source.node,
        "pid": source.pid,
        "rank": source.rank,
        "name": name,
    }


def _writer_for(args: Any) -> _PerfWriter:
    if _WRITER is None:
        init_perf_timeline(args)
    if _WRITER is None:
        raise RuntimeError("perf timeline writer was not initialized")
    return _WRITER


def _emit_policy_version(
    args: Any,
    version: Any,
    *,
    wall_ns: int,
) -> None:
    """Emit the version baseline after a successful rollout weight update."""
    if version is None:
        return
    event = _base_event(args, "instant", "policy_version")
    event.update(
        {
            "wall_ns": wall_ns,
            "attrs": {"version": str(version)},
        }
    )
    _writer_for(args).emit(event)


def perf_policy_version(func):
    """Emit the previous and published versions after a successful update."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        perf_args = getattr(self, "args", None)
        if not getattr(perf_args, "enable_perf_timeline", False):
            return func(self, *args, **kwargs)
        current_version = getattr(self, "published_policy_version", None)
        if current_version is None:
            current_version = getattr(
                getattr(self, "weight_updater", None), "weight_version", None
            )
        before_wall_ns = time.time_ns()

        result = func(self, *args, **kwargs)

        after_wall_ns = time.time_ns()
        _emit_policy_version(
            perf_args,
            current_version,
            wall_ns=before_wall_ns,
        )
        _emit_policy_version(
            perf_args,
            getattr(self, "published_policy_version", None),
            wall_ns=after_wall_ns,
        )
        return result
    return wrapper


def emit_request_started(args: Any, samples: Any) -> None:
    """Emit the processing-start event for one sample or a sample group."""
    if not perf_timeline_enabled(args):
        return
    samples = samples if isinstance(samples, list) else [samples]
    cycle = _CURRENT_ROLLOUT_STEP
    event = _base_event(args, "instant", "request_processing_started")
    event.update(
        {
            "wall_ns": time.time_ns(),
            **({"rl_cycle_id": cycle} if cycle is not None else {}),
            "attrs": {
                "sample_indices": [sample.index for sample in samples],
                "group_ids": [sample.group_index for sample in samples],
            },
        },
    )
    _writer_for(args).emit(event)


def emit_sample_traces(args: Any, samples: list[Any]) -> None:
    """Export completed in-memory Sample.trace spans for request rendering."""
    if not perf_timeline_enabled(args):
        return
    for sample in samples:
        trace = getattr(sample, "trace", None)
        if not trace or not trace.get("events"):
            continue
        events = trace["events"]
        exported_count = int(trace.get("_perf_exported_event_count", 0))
        if exported_count >= len(events):
            continue
        response_length, prompt_length = _sample_token_lengths(sample)
        starts = {}
        for event_index, item in enumerate(events):
            span_id = item.get("span_id")
            if not span_id:
                continue
            if item.get("type") == "span_start":
                starts[str(span_id)] = item
                continue
            if item.get("type") != "span_end" or str(span_id) not in starts:
                continue
            start = starts.pop(str(span_id))
            if event_index < exported_count:
                continue
            start_ns = int(float(start.get("ts", 0)) * 1_000_000_000)
            end_ns = int(float(item.get("ts", 0)) * 1_000_000_000)
            event = _base_event(args, "span", start.get("name", "trace_span"))
            event.update({
                "span_id": span_id,
                "sample_id": trace.get("sample_id", getattr(sample, "index", None)),
                "group_id": trace.get("group_id"),
                "trace_id": trace.get("trace_id"),
                "parent_span_id": start.get("parent_span_id"),
                "start_wall_ns": start_ns,
                "end_wall_ns": end_ns,
                "duration_ns": max(0, end_ns - start_ns),
                "status": "error" if (item.get("attrs") or {}).get("error_type") else "ok",
                "attrs": {
                    **(start.get("attrs") or {}),
                    **(item.get("attrs") or {}),
                    **({"response_length": response_length} if response_length is not None else {}),
                    **({"prompt_length": prompt_length} if prompt_length is not None else {}),
                },
            })
            _writer_for(args).emit(event)
        trace["_perf_exported_event_count"] = len(events)


def _sample_token_lengths(sample: Any) -> tuple[Any, int | None]:
    """Return the response and prompt token counts for span metadata."""
    response_length = getattr(sample, "effective_response_length", None)
    if response_length is None:
        response_length = getattr(sample, "response_length", None)
    tokens = getattr(sample, "tokens", None)
    prompt_length = None
    if tokens is not None and response_length is not None:
        prompt_length = len(tokens) - int(response_length)
    return response_length, prompt_length


def emit_perf_counter(
    args: Any,
    *,
    name: str,
    values: dict[str, int | float],
) -> None:
    if not perf_timeline_enabled(args):
        return
    event = _base_event(args, "counter", name)
    event.update(
        {
            "wall_ns": time.time_ns(),
            "values": values,
        }
    )
    _writer_for(args).emit(event)


def _emit_train_consume_relation(
    args: Any,
    *,
    rollout_id: int,
    rollout_data: dict[str, Any],
    published_policy_version: Any,
    dp_rank: int | None,
) -> None:
    """Emit a raw train_consumes_rollout relation event.

    Only records raw lineage data — staleness is computed offline by the renderer.
    """
    event = _base_event(args, "relation", "train_consumes_rollout")
    event.update(
        {
            "wall_ns": time.time_ns(),
            "source": {
                "rollout_ids": list(rollout_data.get("rollout_ids", [])),
                "sample_indices": list(rollout_data.get("sample_indices", [])),
            },
            "target": {
                "rl_cycle_id": rollout_id,
                "actor_policy_version": published_policy_version,
                "dp_rank": dp_rank,
            },
        }
    )
    _writer_for(args).emit(event)


def perf_consume(func):
    """Decorate rollout-data fetches and emit their consume relation."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        data = func(self, *args, **kwargs)
        owner_args = getattr(self, "args", None)
        if perf_timeline_enabled(owner_args):
            dp_rank = getattr(self, "_perf_dp_rank", None)
            if dp_rank is None:
                dp_rank = getattr(self, "_dp_rank", lambda: None)()
            _emit_train_consume_relation(
                owner_args,
                rollout_id=_CURRENT_ROLLOUT_STEP,
                rollout_data=data,
                published_policy_version=getattr(self, "published_policy_version", None),
                dp_rank=dp_rank,
            )
        return data
    return wrapper


@contextmanager
def perf_span(
    args: Any,
    *,
    name: str,
) -> Iterator[None]:
    if not perf_timeline_enabled(args):
        yield
        return

    start_wall_ns = time.time_ns()
    start_mono_ns = time.perf_counter_ns()
    yield
    end_mono_ns = time.perf_counter_ns()
    end_wall_ns = time.time_ns()
    event = _base_event(args, "span", name)
    cycle = _CURRENT_ROLLOUT_STEP
    event.update(
        {
            "start_wall_ns": start_wall_ns,
            "end_wall_ns": end_wall_ns,
            "duration_ns": end_mono_ns - start_mono_ns,
            **({"rl_cycle_id": cycle} if cycle is not None else {}),
        }
    )
    _writer_for(args).emit(event)


# -- decorators & helpers for low-invasion instrumentation --


class _PerfTimerDecorator:
    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            owner_args = _current_perf_args()
            if owner_args is None and args:
                owner_args = getattr(args[0], "args", None)
            if owner_args is None:
                return func(*args, **kwargs)
            if not getattr(owner_args, "enable_perf_timeline", False):
                return func(*args, **kwargs)
            with perf_span(owner_args, name=self.name):
                return func(*args, **kwargs)
        return wrapper


def perf_timer_prefix(name):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            owner_args = _current_perf_args()
            if owner_args is None and args:
                owner_args = getattr(args[0], "args", None)
            if owner_args is None:
                return func(*args, **kwargs)
            if not getattr(owner_args, "enable_perf_timeline", False):
                return func(*args, **kwargs)
            prefix = kwargs.get("store_prefix", "")
            if len(args) >= 4:
                prefix = args[3]
            with perf_span(owner_args, name=f"{prefix}{name}"):
                return func(*args, **kwargs)
        return wrapper
    return decorate


def start_perf_span(name: str, args: Any = None) -> Any:
    """Start a perf span, return an opaque handle. Call end_perf_span(handle) to close."""
    args = args if args is not None else _current_perf_args()
    if args is None:
        return None
    if not perf_timeline_enabled(args):
        return None
    cm = perf_span(args, name=name)
    cm.__enter__()
    return cm


def end_perf_span(span: Any) -> None:
    if span is not None:
        span.__exit__(None, None, None)


def perf_step(func):
    """Set the process-local perf step before entering a step-level method."""

    @functools.wraps(func)
    def wrapper(self, rollout_id, *args, **kwargs):
        set_perf_timeline_step(rollout_id)
        return func(self, rollout_id, *args, **kwargs)

    return wrapper


def perf_timer(name_or_func):
    """Timer-style performance span decorator."""
    if isinstance(name_or_func, str):
        return _PerfTimerDecorator(name_or_func)
    return _PerfTimerDecorator(name_or_func.__name__)(name_or_func)
