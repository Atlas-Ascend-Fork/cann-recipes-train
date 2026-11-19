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

"""Tests for GRPO reward counts reported by rollout metrics."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vime.observability.rollout_metrics import get_grpo_max_cg, log_rollout_data, record_grpo_group_reward_events
from vime.utils.types import Sample


ARGS = SimpleNamespace(
    advantage_estimator="grpo",
    n_samples_per_prompt=4,
    reward_key=None,
    log_reward_category=None,
    vllm_speculative_algorithm=None,
)


def sample(group_index, rollout_id, reward, *, index=None):
    return Sample(
        group_index=group_index,
        rollout_id=rollout_id,
        index=index,
        reward=reward,
        response_length=1,
        response="ok",
    )


class GrpoGroupRewardMetricsTests(unittest.TestCase):
    def test_records_unique_rollout_counts_for_each_group(self):
        group_rewards = {10: [1, 0, 0, 0], 20: [1, 1, 0, 0], 30: [1, 1, 1, 1], 40: [0, 0, 0, 0]}
        samples = [
            sample(group_index, group_index * 10 + index, reward)
            for group_index, rewards in group_rewards.items()
            for index, reward in enumerate(rewards)
        ]
        # One rollout split into two segments must count once.
        samples.append(sample(10, 100, 1))

        metrics = record_grpo_group_reward_events(ARGS, samples)

        for group_index, rewards in group_rewards.items():
            representative = next(item for item in samples if item.group_index == group_index)
            events = [
                event
                for event in representative.trace["events"]
                if event["name"] == "agentic.grpo_group_reward_counts"
            ]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["attrs"], {
                "group_index": group_index,
                "reward_1_count": rewards.count(1),
                "reward_0_count": rewards.count(0),
            })
        self.assertEqual(metrics, {"grpo/mixed_group_count": 2})

    def test_ignores_samples_without_rollout_id(self):
        samples = [
            sample(10, None, 1, index=100),
            sample(10, 101, 0),
        ]
        self.assertEqual(record_grpo_group_reward_events(ARGS, samples), {"grpo/mixed_group_count": 0})
        self.assertIsNone(getattr(samples[0], "trace", None))

    def test_partial_groups_need_both_reward_values_to_count_as_mixed(self):
        cases = (
            ([1, 1], 0),
            ([0, 0], 0),
            ([1, 0], 1),
            ([1, 0.5], 0),
        )
        for rewards, expected_count in cases:
            with self.subTest(rewards=rewards):
                samples = [sample(10, 100 + index, reward) for index, reward in enumerate(rewards)]

                metrics = record_grpo_group_reward_events(ARGS, samples)

                self.assertEqual(metrics, {"grpo/mixed_group_count": expected_count})
                event = samples[0].trace["events"][-1]
                self.assertEqual(event["attrs"]["reward_1_count"], rewards.count(1))
                self.assertEqual(event["attrs"]["reward_0_count"], rewards.count(0))

    def test_only_grpo_reports_group_reward_metrics(self):
        args = SimpleNamespace(**{**ARGS.__dict__, "advantage_estimator": "gspo"})
        samples = [sample(10, 100, 1)]
        self.assertEqual(record_grpo_group_reward_events(args, samples), {})
        self.assertIsNone(getattr(samples[0], "trace", None))

    def test_mixed_group_upper_bound_can_be_configured(self):
        samples = [sample(group, group * 10 + index, reward)
                   for group, rewards in ((10, [1, 0, 0, 0]), (20, [1, 1, 0, 0]))
                   for index, reward in enumerate(rewards)]

        with patch.dict(os.environ, {"VIME_GRPO_MAX_CG": "1"}):
            metrics = record_grpo_group_reward_events(ARGS, samples)

        self.assertEqual(metrics, {"grpo/mixed_group_count": 1})

    def test_group_count_is_forwarded_to_rollout_logger(self):
        args = SimpleNamespace(
            custom_rollout_log_function_path=None,
            load_debug_rollout_data=False,
            wandb_always_use_train_step=False,
        )
        with (
            patch("vime.observability.rollout_metrics.compute_metrics_from_samples", return_value={}),
            patch("vime.observability.rollout_metrics.compute_perf_metrics_from_samples", return_value={}),
            patch("vime.observability.rollout_metrics.logging_utils.log") as log,
        ):
            log_rollout_data(7, args, [], {"grpo/mixed_group_count": 2}, 1.0)

        logged_metrics = log.call_args.args[1]
        self.assertEqual(logged_metrics["grpo/mixed_group_count"], 2)

    def test_invalid_max_cg_is_rejected(self):
        with patch.dict(os.environ, {"VIME_GRPO_MAX_CG": "invalid"}):
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                get_grpo_max_cg(ARGS)

        with patch.dict(os.environ, {"VIME_GRPO_MAX_CG": "4"}):
            with self.assertRaisesRegex(ValueError, r"must be in \[1, 3\]"):
                get_grpo_max_cg(ARGS)


if __name__ == "__main__":
    unittest.main()
