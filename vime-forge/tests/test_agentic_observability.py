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

"""Focused tests for the Agent additions to VIME's Trace carrier.

Run with ``python -m unittest tests.test_agentic_observability``.
"""

import unittest
from unittest.mock import AsyncMock, patch

from vime.agent.sandbox import exec_and_wait
from vime.agent.trajectory import TrajectoryManager, TurnRecord
from vime.observability.trace_utils import TraceHandle, bind_trace, trace_event, trace_span
from vime.utils.types import Sample


class _FakeSandbox:
    sandbox_id = "sandbox-1"
    observation_role = "verifier"

    def __init__(self, trace):
        self.trace = trace

    async def write_file(self, path, content, *, user="root"):
        pass

    async def exec(self, cmd, **kwargs):
        if "done" in cmd:
            result = (0, "7", "")
        elif cmd.startswith("tail -c"):
            result = (0, "failure output", "")
        else:
            result = (0, "", "")
        return result

    async def read_file(self, path, *, user="root"):
        return "failure output"


class AgentTraceTests(unittest.IsolatedAsyncioTestCase):
    def test_agent_events_use_upstream_carrier_and_sample_trace(self):
        base = Sample(index=7, group_index=3, rollout_id=7)
        trace = bind_trace(base)
        manager = TrajectoryManager()

        with trace_span(base, "coding_rollout", record_error=False):
            trace_event(trace, "agentic.sandbox_ready", attrs={"sandbox_role": "agent"})
            manager.record_turn(
                "session-7",
                turn=TurnRecord(
                    prompt_ids=[1, 2],
                    output_ids=[3, 4],
                    finish_reason="stop",
                    output_log_probs=[-0.1, -0.2],
                ),
                prompt_messages=[{"role": "user", "content": "fix it"}],
                response_message={"role": "assistant", "content": "done"},
                trace=trace,
            )
            samples = manager.get_trajectory("session-7", base_sample=base, trace=trace)

        self.assertEqual(len(samples), 1)
        self.assertIs(samples[0].trace, base.trace)
        names = [event["name"] for event in base.trace["events"]]
        self.assertIn("coding_rollout", names)
        self.assertIn("agentic.sandbox_ready", names)
        self.assertIn("agentic.turn_recorded", names)
        self.assertIn("agentic.trajectory_summary", names)
        self.assertIn("trajectory_segment", samples[0].metadata)
        self.assertIn("trajectory_summary", samples[0].metadata)

    def test_multiple_samples_share_one_trace_carrier(self):
        base = Sample(index=8, group_index=4, rollout_id=8)
        trace = bind_trace(base)
        manager = TrajectoryManager(fork_threshold_tokens=1)

        for index in range(2):
            manager.record_turn(
                "session-8",
                turn=TurnRecord(prompt_ids=[index + 1], output_ids=[index + 2], finish_reason="stop"),
                prompt_messages=[{"role": "user", "content": str(index)}],
                response_message={"role": "assistant", "content": str(index)},
                trace=trace,
            )

        samples = manager.get_trajectory("session-8", base_sample=base, trace=trace)
        self.assertGreaterEqual(len(samples), 1)
        self.assertTrue(all(sample.trace is base.trace for sample in samples))

    async def test_exec_and_wait_records_real_command_once(self):
        trace = TraceHandle("trace-2", {"events": []})
        sandbox = _FakeSandbox(trace)
        with patch("vime.agent.sandbox.asyncio.sleep", new=AsyncMock()):
            with trace_span(trace, "agent_run", record_error=False):
                result = await exec_and_wait(
                    sandbox,
                    cmd="pytest tests/test_agent.py",
                    time_budget_sec=10,
                    tag="tests",
                )

        self.assertEqual(result, (7, "failure output"))
        events = trace.carrier["events"]
        command_spans = [event for event in events if event.get("name") == "sandbox_command"]
        self.assertEqual(len(command_spans), 2)
        agent_start = next(event for event in events if event.get("name") == "agent_run")
        span_start = next(event for event in command_spans if event["type"] == "span_start")
        span_end = next(event for event in command_spans if event["type"] == "span_end")
        self.assertEqual(span_start["parent_span_id"], agent_start["span_id"])
        self.assertEqual(span_start["attrs"]["command_prefix"], "pytest tests/test_agent.py")
        self.assertEqual(span_end["attrs"]["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
