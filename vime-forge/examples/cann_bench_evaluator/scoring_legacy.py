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

"""Legacy non-custom policy, ported from e50f804d:examples/cann_bench_mcp/scoring.py.

Equivalent to the experiment's formal_scoring adapter with stage_reward=False.
Only stable EvaluationResult fields are used; model diagnostics are not transport
errors. No environment switch, format scoring, or legacy client is imported.
"""

from __future__ import annotations

import math

from .protocol import EvaluationResult
from .scoring import Score, validate_score

_HARD_ZERO_VERDICTS = {"anti_hack_fail", "eval_aborted", "parse_fail"}
_ZERO_REWARD_VERDICTS = _HARD_ZERO_VERDICTS | {"compile_fail", "queue_timeout", "timeout_fail"}
_PERFORMANCE_WEIGHT = 0.5
_PERF_ALIASES = {
    "candidate_us": "elapsed_us",
    "baseline_us": "baseline_perf_us",
    "hardware_bound_us": "t_hw_us",
    "hap": "perf_score",
}


def _as_float(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return None


def _case_metric(case, key):
    for source in (case, case["perf"]):
        value = _as_float(source.get(key))
        if value is not None and math.isfinite(value):
            return value
    return None


def _case_sol_score(case):
    elapsed = _case_metric(case, "elapsed_us")
    baseline = _case_metric(case, "baseline_perf_us")
    bound = _case_metric(case, "t_hw_us")
    if any(value is None or value <= 0 for value in (elapsed, baseline, bound)):
        return None
    denominator = (elapsed - bound) + (baseline - bound)
    if denominator <= 0:
        return None
    return (baseline - bound) / denominator


def _speedup_score(speedup):
    if not math.isfinite(speedup) or speedup <= 0:
        return 0.0
    return speedup / (1.0 + speedup)


def _performance_cases(details):
    cases = []
    for detail in details:
        perf = dict(detail.get("perf") or {})
        for current, old in _PERF_ALIASES.items():
            if perf.get(current) is not None:
                perf[old] = perf[current]
        cases.append({**detail, "success": detail.get("status") == "success", "perf": perf})
    return cases


def _case_performance_score(case, speedup):
    """Keep the legacy priority: explicit performance score, SOL, then speedup."""
    perf_score = _case_metric(case, "perf_score")
    if perf_score is not None:
        return perf_score
    eq3_score = _case_sol_score(case)
    if eq3_score is not None:
        return eq3_score
    if speedup is not None and speedup > 0:
        return _speedup_score(speedup)
    return None


def _performance_reward(result):
    cases = _performance_cases(result.case_details)
    total = result.case_total or len(cases)
    score_sum = 0.0
    scored_cases = 0
    speedups = []
    if total > 0:
        for case in cases:
            if not case["success"]:
                continue
            speedup = _case_metric(case, "speedup")
            if speedup is not None and speedup > 0:
                speedups.append(speedup)
            perf_score = _case_performance_score(case, speedup)
            if perf_score is not None:
                score_sum += perf_score
                scored_cases += 1

    average = _as_float(result.speedup)
    if average is not None and not math.isfinite(average):
        average = None
    if average is None and speedups:
        average = sum(speedups) / len(speedups)
    # Historical fallback only when no individual case could be scored. Failed
    # cases still contribute zero; do not replace partially measured evidence.
    if scored_cases == 0 and total > 0 and average is not None:
        if average > 0:
            score_sum = _speedup_score(average) * min(result.case_pass, total)
            scored_cases = min(result.case_pass, total)
    return (_PERFORMANCE_WEIGHT * score_sum / total if total > 0 else 0.0), scored_cases


def score_result(result: EvaluationResult) -> Score:
    """Apply the fixed legacy online policy, without stage or task-level bonuses."""
    if result.infrastructure_failure or result.retryable:
        raise ValueError("infrastructure/retryable results must not be scored")
    base = (result.score or 0.0) / 100.0
    if result.verdict in _HARD_ZERO_VERDICTS:
        reward = 0.0
    elif result.score is not None:
        reward = base
        performance, scored_cases = _performance_reward(result)
        pass_rate = min(1.0, max(0.0, result.case_pass / result.case_total)) if result.case_total > 0 else 0.0
        no_performance_reward = 0.2 + 0.3 * pass_rate
        # Preserve the original numerical test, including partial correctness
        # and its 1e-4 tolerance. Do not substitute the default scorer's HAP gate.
        if (
            result.case_total > 0
            and scored_cases > 0
            and math.isclose(base, no_performance_reward, rel_tol=0.0, abs_tol=1e-4)
        ):
            reward += performance
    elif result.verdict in _ZERO_REWARD_VERDICTS or result.case_total <= 0:
        reward = 0.0
    else:
        reward = min(1.0, max(0.0, 0.2 + 0.3 * min(1.0, max(0.0, result.case_pass / result.case_total))))
    # Keep the experiment adapter's component names, including its score=None
    # fallback accounting. Never modify the evaluator's original score/cases.
    return validate_score(Score(reward, {"base_reward": base, "performance_reward": reward - base}))
