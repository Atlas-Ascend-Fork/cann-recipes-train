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

"""Stable /api/v1 contract. Raw harness output is diagnostic, never authoritative."""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ProtocolError(ValueError):
    """An incompatible service response; retrying another legacy route cannot fix it."""


@dataclass(frozen=True)
class EvaluatorConfig:
    url: str
    poll_interval: float = 5.0
    queue_timeout: float = 1800.0
    execution_timeout: float = 900.0
    request_timeout: float = 30.0
    request_retries: int = 2
    delete_retries: int = 5
    queue_retries: int = 1

    def __post_init__(self):
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("cann_evaluator.url must be an explicit HTTP(S) service URL")
        if parsed.query or parsed.fragment:
            raise ValueError("cann_evaluator.url must be an explicit HTTP(S) service URL")
        for name in ("poll_interval", "queue_timeout", "execution_timeout", "request_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"cann_evaluator.{name} must be finite and positive")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"cann_evaluator.{name} must be finite and positive")
        for name in ("request_retries", "delete_retries", "queue_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"cann_evaluator.{name} must be a nonnegative integer")


@dataclass
class EvaluationResult:
    op: str
    verdict: str
    job_id: str = ""
    score: float | None = None
    compiled: bool | None = None
    case_pass: int = 0
    case_total: int = 0
    speedup: float | None = None
    case_details: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    links: dict[str, Any] = field(default_factory=dict)
    infrastructure_failure: bool = False
    retryable: bool = False
    resubmit_safe: bool = False
    error: str | None = None
    compile_log: str | None = None
    evaluate_log: str | None = None
    log_fetch_errors: dict[str, str] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    cleanup: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    envelope: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Archive stable fields once, retaining all extra service diagnostics."""
        values = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "envelope"}
        if self.envelope:
            payload = self.envelope.get("result")
            ops = payload.get("ops") if isinstance(payload, dict) else None
            if not isinstance(ops, list) or len(ops) != 1 or not isinstance(ops[0], dict):
                values["diagnostics"] = self.envelope
                return deepcopy(values)
            op = ops[0]
            extras = {key: value for key, value in op.items() if key not in NORMALIZED_OP_FIELDS}
            if op.get("op") != self.op:
                extras["reported_op"] = op.get("op")
                extras["reported_result"] = op
            if op.get("speedup") != self.speedup:
                extras["speedup_raw"] = {"value": op.get("speedup"), "type": type(op.get("speedup")).__name__}
            if op.get("score") != self.score:
                extras["score_raw"] = {"value": op.get("score"), "type": type(op.get("score")).__name__}
            counts = op.get("cases")
            case_extras = (
                {key: value for key, value in counts.items() if key not in {"passed", "total"}}
                if isinstance(counts, dict)
                else {"raw": counts}
            )
            if isinstance(counts, dict) and (counts.get("passed"), counts.get("total")) != (
                self.case_pass,
                self.case_total,
            ):
                case_extras["raw_counts"] = {key: counts.get(key) for key in ("passed", "total")}
            if case_extras:
                extras["cases"] = case_extras
            values["diagnostics"] = {
                **{key: value for key, value in self.envelope.items() if key != "result"},
                "result": {key: value for key, value in payload.items() if key != "ops"},
                "op": extras,
            }
        # Preserve snapshot isolation without deep-copying the duplicate envelope.
        return deepcopy(values)


NORMALIZED_OP_FIELDS = {"op", "verdict", "score", "compiled", "cases", "speedup", "case_details", "summary", "links"}


VERDICTS = {
    "passed",
    "accuracy_fail",
    "runtime_fail",
    "timeout_fail",
    "compile_fail",
    "anti_hack_fail",
    "parse_fail",
    "unsupported_task",
    "eval_aborted",
    "env_fail",
    "unknown",
}


def submitted_job(payload: dict[str, Any]) -> dict[str, Any]:
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1 or not isinstance(jobs[0], dict):
        raise ProtocolError("v1 submit must return exactly one jobs[] entry")
    job = jobs[0]
    if not isinstance(job.get("job_id"), str) or not job["job_id"]:
        raise ProtocolError("v1 submit returned no job_id")
    return job


def normalize_speedup(value: Any, *, op: str, job_id: str) -> float | None:
    """Optional performance is diagnostic, not a reason to reject a scored result."""
    if value is None:
        return None
    speedup = None
    if not isinstance(value, bool) and isinstance(value, (int, float, str)):
        try:
            speedup = float(value)
        except (ValueError, OverflowError):
            pass
    if speedup is None or not math.isfinite(speedup) or speedup <= 0:
        logger.warning(
            "invalid speedup ignored: op=%s, job_id=%s, value=%r, type=%s",
            op,
            job_id,
            value,
            type(value).__name__,
        )
        return None
    return speedup


def _normalize_score(raw_score):
    score = raw_score
    if isinstance(score, str):
        try:
            score = float(score)
        except ValueError:
            pass
    if score is None:
        return None
    message = f"score must be null or a finite number in [0,100]: value={raw_score!r}, type={type(raw_score).__name__}"
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ProtocolError(message)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ProtocolError(message)
    return score


def _result_item(envelope, *, job_id):
    payload = envelope.get("result")
    if not isinstance(envelope.get("job"), dict) or not isinstance(payload, dict):
        raise ProtocolError("v1 result must contain {job, result}")
    version = payload.get("schema_version")
    if not isinstance(version, str) or not version.startswith("1."):
        raise ProtocolError(f"unsupported evaluator schema_version: {version!r}")
    if payload.get("job_id") != job_id or envelope["job"].get("job_id") != job_id:
        raise ProtocolError(
            f"result job_id does not match submitted job: expected={job_id!r}, "
            f"job.job_id={envelope['job'].get('job_id')!r}, result.job_id={payload.get('job_id')!r}"
        )
    ops = payload.get("ops")
    if not isinstance(ops, list) or len(ops) != 1 or not isinstance(ops[0], dict):
        raise ProtocolError("one submitted operator must return exactly one result.ops entry")
    item = ops[0]
    if not isinstance(item.get("op"), str) or not item["op"]:
        raise ProtocolError("result.ops entry requires a nonempty operator name")
    return item


def _case_counts(item, verdict):
    counts = item.get("cases") or {}
    if not isinstance(counts, dict):
        raise ProtocolError("missing cases counts")
    passed, total = counts.get("passed"), counts.get("total")
    # Failed/precheck jobs may have no case measurements; do not reject a valid failure.
    if verdict != "passed":
        passed = 0 if passed is None else passed
        total = 0 if total is None else total
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (passed, total)) or not 0 <= passed <= total:
        raise ProtocolError(f"invalid cases.passed/total: passed={passed!r}, total={total!r}")
    return passed, total


def normalize_result(envelope: dict[str, Any], *, job_id: str, op: str) -> EvaluationResult:
    item = _result_item(envelope, job_id=job_id)
    if item["op"] != op:
        # A valid result for a different generated plugin is a model submission
        # error, not an infrastructure error to retry or mask out of training.
        return EvaluationResult(
            op=op,
            job_id=job_id,
            verdict="parse_fail",
            score=0.0,
            error=f"operator_mismatch: expected={op!r}, reported={item['op']!r}",
            envelope=envelope,
        )
    verdict = item.get("verdict")
    if verdict not in VERDICTS:
        raise ProtocolError(f"unknown v1 verdict: {verdict!r}")
    if "compiled" not in item or item["compiled"] is not None and not isinstance(item["compiled"], bool):
        raise ProtocolError(f"missing/invalid compiled field: value={item.get('compiled')!r}")
    if "score" not in item:
        raise ProtocolError("missing score field")
    raw_score = item["score"]
    score = _normalize_score(raw_score)
    passed, total = _case_counts(item, verdict)
    if verdict == "passed" and (score is None or item["compiled"] is not True):
        raise ProtocolError("scored verdict requires score and compiled=true")
    if verdict == "compile_fail" and item["compiled"] not in {False, None}:
        raise ProtocolError(f"compile_fail requires compiled=false/null: value={item['compiled']!r}")
    details = item.get("case_details")
    if details is None:
        details = []
    if not isinstance(details, list) or any(not isinstance(d, dict) for d in details):
        raise ProtocolError(f"invalid case_details: value={details!r}, type={type(details).__name__}")
    speedup = normalize_speedup(item.get("speedup"), op=op, job_id=job_id)
    infrastructure = verdict in {"env_fail", "unknown"}
    return EvaluationResult(
        op=op,
        job_id=job_id,
        verdict=verdict,
        score=score,
        compiled=item["compiled"],
        case_pass=passed,
        case_total=total,
        speedup=speedup,
        case_details=details,
        summary=str(item.get("summary") or ""),
        links=item.get("links") or {},
        infrastructure_failure=infrastructure,
        retryable=infrastructure,
        error=item.get("first_error") or item.get("root_cause"),
        envelope=envelope,
    )


def active_stage(status: dict[str, Any]) -> str:
    if not isinstance(status.get("terminal"), bool):
        raise ProtocolError(f"v1 status requires a boolean terminal field: value={status.get('terminal')!r}")
    phase = status.get("phase")
    if phase in {"queued", "precheck"}:
        return "compile_queue"
    if phase == "compiling":
        return "compiling"
    if phase == "waiting_device":
        return "run_queue"
    if phase in {"evaluating", "collecting"}:
        return "running"
    if status["terminal"]:
        return "done"
    raise ProtocolError(f"unsupported v1 job phase: {phase!r}")
