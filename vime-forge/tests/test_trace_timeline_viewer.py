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

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from vime.observability.trace_utils import trace_event, trace_span
from vime.utils.types import Sample


def load_viewer():
    path = Path(__file__).parents[1] / "tools" / "trace_timeline_viewer.py"
    spec = importlib.util.spec_from_file_location("trace_timeline_viewer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_cache(path):
    viewer = load_viewer()
    paths = viewer.TimelinePaths(
        pt_path=path,
        cache_path=path.with_suffix(".cache.json"),
        html_path=path.with_suffix(".html"),
    )
    return viewer.ensure_cache(paths)


class TraceTimelineViewerTests(unittest.TestCase):
    def test_viewer_rebuilds_span_and_event(self):
        sample = Sample(index=1, group_index=2)
        with trace_span(sample, "agent_run") as span:
            span.set("exit_code", 0)
        trace_event(sample, "agentic.sandbox_ready", attrs={"sandbox_role": "agent"})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.pt"
            torch.save({"samples": [sample]}, path)
            cache = build_cache(path)

        self.assertEqual(cache["trial_count"], 1)
        self.assertEqual(cache["rows"][0]["closed_span_count"], 1)
        self.assertEqual(cache["rows"][0]["point_event_count"], 1)
        item_names = {item["name"] for item in cache["rows"][0]["items"]}
        self.assertEqual(item_names, {"agent_run", "agentic.sandbox_ready"})

    def test_viewer_reads_trace_only_jsonl(self):
        sample = Sample(index=3, group_index=4)
        with trace_span(sample, "evaluation"):
            pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            path.write_text(json.dumps({"index": 3, "trace": sample.trace}) + "\n", encoding="utf-8")
            cache = build_cache(path)

        self.assertEqual(cache["trial_count"], 1)
        self.assertEqual(cache["rows"][0]["closed_span_count"], 1)

    def test_viewer_collapses_samples_from_one_trial(self):
        first = Sample(index=1, group_index=2, rollout_id=7)
        with trace_span(first, "agent_run"):
            pass
        second = Sample(index=2, group_index=2, rollout_id=7)
        second.trace = first.trace

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.pt"
            torch.save({"samples": [first, second]}, path)
            cache = build_cache(path)

        self.assertEqual(cache["trial_count"], 1)
        self.assertEqual(len(cache["rows"]), 1)
        self.assertEqual(cache["rows"][0]["trial_id"], first.trace["trace_id"])
        self.assertEqual(cache["rows"][0]["rollout_id"], 7)
        self.assertEqual(cache["rows"][0]["trace_id"], first.trace["trace_id"])

    def test_viewer_preserves_zero_rollout_id_with_trace_id(self):
        sample = Sample(index=1, group_index=2, rollout_id=0)
        with trace_span(sample, "agent_run"):
            pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.pt"
            torch.save({"samples": [sample]}, path)
            cache = build_cache(path)

        row = cache["rows"][0]
        self.assertEqual(row["rollout_id"], 0)
        self.assertEqual(row["trace_id"], sample.trace["trace_id"])
        self.assertEqual(row["trial_id"], sample.trace["trace_id"])


if __name__ == "__main__":
    unittest.main()
