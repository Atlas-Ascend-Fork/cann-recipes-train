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

"""Task metrics from explicit evaluation fields, not from reward > 0."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


def reward_summary(samples):
    """Final raw reward mean and per-question best-of-five reward, not pass rate."""
    if not samples or any(s.metadata.get("single_turn_evaluation_pending") for s in samples):
        return {}
    groups = defaultdict(list)
    rewards = []
    for sample in samples:
        reward = sample.metadata.get("single_turn_reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            return {}  # Missing/unfinalized rewards are not placeholder zeroes.
        rewards.append(reward)
        group_index = getattr(sample, "group_index", None)
        identity = (
            ("group_index", group_index)
            if group_index is not None
            else ("prompt", sample.metadata["op_name"], json.dumps(sample.prompt, ensure_ascii=False))
        )
        groups[identity].append(reward)
    values = {"reward_mean": sum(rewards) / len(rewards)}
    if all(len(group) == 5 for group in groups.values()):
        values["reward_pass5"] = sum(max(group) for group in groups.values()) / len(groups)
    return values


def summarize(samples):
    """Only the six CANN business metrics; native rollout counters stay native."""
    valid = [s for s in samples if not s.remove_sample and not s.metadata.get("evaluator_infrastructure_failure")]
    results = [s.metadata.get("evaluation", {}) for s in valid]

    def full(result):
        return (
            result.get("verdict") == "passed"
            and result.get("case_total", 0) > 0
            and result.get("case_pass") == result["case_total"]
        )

    metrics = reward_summary(samples)
    if valid:
        metrics.update(
            compile_rate=sum(r.get("compiled") is True for r in results) / len(valid),
            accuracy_rate=sum(full(r) for r in results) / len(valid),
        )
    total_cases = sum(r.get("case_total", 0) for r in results)
    if total_cases:
        metrics["case_accuracy_rate"] = sum(r.get("case_pass", 0) for r in results) / total_cases
    speedups = [r["speedup"] for r in results if r.get("speedup") is not None]
    if speedups:
        metrics["mean_speedup"] = sum(speedups) / len(speedups)
    return metrics


def log_train(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    del rollout_extra_metrics, rollout_time
    from vime.observability import logging_utils
    from vime.observability.metric_utils import compute_rollout_step

    values = {f"cann/train/{name}": value for name, value in summarize(samples).items()}
    for key in ("compile_queue", "run_queue", "compile", "run", "total"):
        times = [s.metadata.get("evaluation", {}).get("timing", {}).get(f"{key}_elapsed_sec") for s in samples]
        times = [t for t in times if t is not None]
        if times:
            values[f"cann/train/mean_{key}_seconds"] = sum(times) / len(times)
    times = [
        s.metadata["single_turn_generation_elapsed_sec"]
        for s in samples
        if "single_turn_generation_elapsed_sec" in s.metadata
    ]
    if times:
        values["cann/train/mean_generation_seconds"] = sum(times) / len(times)
    values["rollout/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, values, step_key="rollout/step")
    return False  # Native throughput/rollout_time statistics remain unchanged.


def _eval_generation_metrics(args, samples):
    from vime.observability.rollout_metrics import compute_metrics_from_samples

    if not samples:
        return {}
    metrics = compute_metrics_from_samples(args, samples)
    valid = [s for s in samples if not s.remove_sample and not s.metadata.get("evaluator_infrastructure_failure")]
    if len(valid) != len(samples):
        # Length/repetition/cache statistics cover all generated responses, but
        # service failures must not create artificial zero-reward groups.
        reward_prefixes = ("zero_std/", "error_cat/")
        metrics = {key: value for key, value in metrics.items() if not key.startswith(reward_prefixes)}
        if valid:
            metrics.update(
                (key, value)
                for key, value in compute_metrics_from_samples(args, valid).items()
                if key.startswith(reward_prefixes)
            )
    return metrics


def _eval_timing_metrics(samples):
    from vime.observability.metric_utils import compute_statistics

    timings = []
    for sample in samples:
        result = sample.metadata.get("evaluation", {})
        timing = dict(result.get("timing", {}))
        # Stage durations cover all resubmission attempts; total already spans
        # the whole evaluator call and must not be added to the attempts again.
        for key in ("compile_queue", "run_queue", "queue", "compile", "run", "total"):
            field = f"{key}_elapsed_sec"
            if key == "total" and field in timing:
                continue
            times = [attempt[field] for attempt in result.get("attempts", []) if attempt.get(field) is not None]
            if times:
                timing[field] = sum(times)
        timings.append(timing)
    distributions = {
        f"evaluator/{key}_time": [t[f"{key}_elapsed_sec"] for t in timings if t.get(f"{key}_elapsed_sec") is not None]
        for key in ("compile_queue", "run_queue", "queue", "compile", "run", "total")
    }
    distributions["evaluator/compile_run_time"] = [
        t.get("compile_elapsed_sec", 0.0) + t.get("run_elapsed_sec", 0.0)
        for t in timings
        if "compile_elapsed_sec" in t or "run_elapsed_sec" in t
    ]
    # These are per-request durations, not a sum/max claimed as batch wall time.
    distributions["generation_time"] = [
        s.metadata["single_turn_generation_elapsed_sec"]
        for s in samples
        if s.metadata.get("single_turn_generation_elapsed_sec") is not None
    ]
    values = {}
    for name, times in distributions.items():
        if times:
            for stat, value in compute_statistics(times).items():
                values[f"{name}/{stat}"] = value
    return values


def log_eval(rollout_id, args, data, extra_metrics):
    from vime.observability import logging_utils
    from vime.observability.metric_utils import compute_rollout_step

    values = dict(extra_metrics or {})
    for name, dataset in data.items():
        samples = dataset["samples"]
        dataset_values = {}
        for metrics in (
            _eval_generation_metrics(args, samples),
            _eval_timing_metrics(samples),
            summarize(samples),
        ):
            dataset_values.update(metrics)
        values.update({f"eval/{name}/{key}": value for key, value in dataset_values.items()})
        headline = {key: value for key, value in dataset_values.items() if key in ("reward_mean", "reward_pass5")}
        logger.info("eval %s %s reward summary: %s", rollout_id, name, headline)
    values["eval/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, values, step_key="eval/step")
    return True  # Do not let the default logger count infrastructure failures as model zeroes.
