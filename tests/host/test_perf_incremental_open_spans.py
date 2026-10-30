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

import ast
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "vime-forge/vime/utils"


def definitions(name, names, **namespace):
    """Execute the real definitions, isolating unrelated Ray/model imports."""
    path = UTILS / name
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) in names
    ]
    assert {node.name for node in nodes} == set(names)
    namespace["__name__"] = __name__
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def module(name):
    """Import a complete standalone utility module, without replacing imports."""
    path = UTILS / name
    identifier = "host_" + path.stem
    spec = importlib.util.spec_from_file_location(identifier, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[identifier] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def trace_module(caplog):
    loaded = module("perf_timeline.py")
    # Explicit local provenance avoids the utility's unrelated IP discovery.
    setattr(loaded, "_SOURCE", getattr(loaded, "_Source")("host:1", "host", 1, None))
    args = SimpleNamespace(
        enable_perf_timeline=True, perf_run_id="host", perf_timeline_flush_events=1
    )
    caplog.set_level(logging.INFO, logger=loaded.logger.name)
    loaded.init_perf_timeline(args)
    return loaded, args


def recorded_spans(loaded, caplog):
    loaded.shutdown_perf_timeline()
    spans = []
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith(loaded.PERF_TRACE_PREFIX):
            continue
        event = json.loads(message.removeprefix(loaded.PERF_TRACE_PREFIX))
        if event.get("type") == "span":
            spans.append(event)
    return spans


def test_open_span_survives_incremental_export_and_stays_idempotent(caplog):
    loaded, args = trace_module(caplog)
    start = {"type": "span_start", "span_id": "s1", "name": "operation", "ts": 1.0}
    sample = SimpleNamespace(
        index=0, tokens=[1, 2], response_length=1, trace={"events": [start]}
    )
    loaded.emit_sample_traces(args, [sample])
    sample.trace["events"].append({"type": "span_end", "span_id": "s1", "ts": 2.0})
    loaded.emit_sample_traces(args, [sample])
    loaded.emit_sample_traces(args, [sample])
    spans = recorded_spans(loaded, caplog)
    assert len(spans) == 1
    assert spans[0]["duration_ns"] == 1_000_000_000


def test_complete_first_export_remains_single(caplog):
    loaded, args = trace_module(caplog)
    sample = SimpleNamespace(
        index=0,
        tokens=[],
        response_length=0,
        trace={
            "events": [
                {"type": "span_start", "span_id": "s", "name": "operation", "ts": 1.0},
                {"type": "span_end", "span_id": "s", "ts": 2.0},
            ]
        },
    )
    loaded.emit_sample_traces(args, [sample])
    loaded.emit_sample_traces(args, [sample])
    assert len(recorded_spans(loaded, caplog)) == 1
