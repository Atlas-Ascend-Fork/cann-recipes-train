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

"""Pure operator scoring; independent of evaluator requests and training runtimes."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .protocol import EvaluationResult, normalize_speedup

logger = logging.getLogger(__name__)

BOX_COX_RHO = 0.5
BOX_COX_BETA = 0.05
FULL_ACCURACY_BONUS = 0.2


@dataclass(frozen=True)
class Score:
    reward: float
    components: dict[str, float]


def validate_score(score: Score) -> Score:
    """Reject broken policies without clipping their negative or unbounded rewards."""
    if not isinstance(score, Score) or not isinstance(score.components, dict):
        raise TypeError("scorer must return Score(reward, components)")
    if any(not isinstance(name, str) or not name for name in score.components):
        raise ValueError("score component names must be nonempty strings")
    for value in (score.reward, *score.components.values()):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("score reward and components must be finite numbers")
    if not math.isclose(math.fsum(score.components.values()), score.reward, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("score components must sum to reward")
    return score


def box_cox_reward(speedup, *, op="", job_id=""):
    speedup = normalize_speedup(speedup, op=op, job_id=job_id)
    if speedup is None:
        return 0.0
    return BOX_COX_BETA * (speedup**BOX_COX_RHO - 1) / (2**BOX_COX_RHO - 1)


def _has_full_accuracy(result: EvaluationResult) -> bool:
    if result.verdict not in {"passed", "accuracy_fail"} or result.compiled is not True:
        return False
    if result.case_total <= 0 or result.case_pass != result.case_total:
        return False
    return True


def score_result(result: EvaluationResult) -> Score:
    """Default CANN policy. Infrastructure failures must be handled by the caller."""
    if result.infrastructure_failure or result.retryable:
        raise ValueError("infrastructure/retryable results must not be scored")
    base = (result.score or 0.0) / 100.0
    components = {"base_reward": base, "full_accuracy_bonus": 0.0, "performance_reward": 0.0}
    if not _has_full_accuracy(result):
        return validate_score(Score(base, components))
    components["full_accuracy_bonus"] = FULL_ACCURACY_BONUS
    successful = [d for d in result.case_details if d.get("status") == "success"]
    # Service HAP already contributes to the base score, but is not our full-accuracy bonus.
    if any(isinstance(d.get("perf"), dict) and d["perf"].get("hap") is not None for d in successful):
        return validate_score(Score(base + FULL_ACCURACY_BONUS, components))
    increments = []
    has_case_perf = False
    for detail in successful:
        perf = detail.get("perf") or {}
        if not isinstance(perf, dict):
            logger.warning(
                "invalid case perf ignored: op=%s, job_id=%s, case_id=%s, value=%r, type=%s",
                result.op,
                result.job_id,
                detail.get("case_id"),
                perf,
                type(perf).__name__,
            )
            perf = {}
        has_case_perf |= any(
            perf.get(k) is not None for k in ("candidate_us", "baseline_us", "hardware_bound_us", "speedup")
        )
        candidate, baseline, bound = (perf.get(k) for k in ("candidate_us", "baseline_us", "hardware_bound_us"))
        if all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (candidate, baseline, bound)):
            denominator = candidate + baseline - 2 * bound
            if denominator > 0:
                increments.append(0.5 * (baseline - bound) / denominator)
                continue
        increments.append(box_cox_reward(perf.get("speedup"), op=result.op, job_id=result.job_id))
    performance = sum(increments) / result.case_total
    if not has_case_perf and result.speedup is not None:
        performance = box_cox_reward(result.speedup, op=result.op, job_id=result.job_id)
    components["performance_reward"] = performance
    return validate_score(Score(base + FULL_ACCURACY_BONUS + performance, components))
