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

"""Submit one operator and await a terminal v1 result, with phase-aware limits."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from .client import EvaluatorClient, HTTPError, ServiceUnavailable, SubmissionUnknown
from .protocol import EvaluationResult, EvaluatorConfig, ProtocolError, active_stage, normalize_result, submitted_job
from .submission import validate_submission

logger = logging.getLogger(__name__)


@dataclass
class JobTiming:
    started_at: float
    stage_started_at: float
    stage: str = "compile_queue"
    elapsed: dict[str, float] = field(default_factory=dict)

    def transition(self, stage: str, now: float):
        if stage != self.stage:
            self.elapsed[self.stage] = self.elapsed.get(self.stage, 0.0) + max(0.0, now - self.stage_started_at)
            self.stage, self.stage_started_at = stage, now

    def snapshot(self, now: float) -> dict[str, float]:
        elapsed = {**self.elapsed}
        elapsed[self.stage] = elapsed.get(self.stage, 0.0) + max(0.0, now - self.stage_started_at)
        result = {f"{key}_elapsed_sec": elapsed.get(key, 0.0) for key in ("compile_queue", "run_queue")}
        result["compile_elapsed_sec"] = elapsed.get("compiling", 0.0)
        result["run_elapsed_sec"] = elapsed.get("running", 0.0)
        result["result_elapsed_sec"] = elapsed.get("done", 0.0)
        result["queue_elapsed_sec"] = result["compile_queue_elapsed_sec"] + result["run_queue_elapsed_sec"]
        result["execution_elapsed_sec"] = (
            result["compile_elapsed_sec"] + result["run_elapsed_sec"] + result["result_elapsed_sec"]
        )
        result["total_elapsed_sec"] = max(0.0, now - self.started_at)
        return result


async def _collect_failure_logs(client, result):
    """Archive complete service logs without changing the scored result."""
    if result.verdict in {"passed", "parse_fail", "unsupported_task"}:
        return
    if result.verdict == "compile_fail" or result.compiled is False:
        stages = ("compile",)
    elif result.compiled is True:
        stages = ("evaluate",)
    elif result.verdict in {"env_fail", "unknown", "eval_aborted", "timeout_fail", "runtime_fail", "accuracy_fail"}:
        stages = ("compile", "evaluate")
    else:
        return  # A precheck failure need not have compile/execute logs.

    for stage in stages:
        try:
            setattr(result, f"{stage}_log", await client.log(result.job_id, stage))
        except Exception as exc:
            # Diagnostic 404/network/decoding failures must not replace a valid result.
            result.log_fetch_errors[f"{stage}_log"] = f"{type(exc).__name__}: {exc}"
            logger.warning("CANN %s log unavailable: job_id=%s, error=%s", stage, result.job_id, exc)


@dataclass
class _PollingState:
    """Accepted-job state shared by polling and the single cleanup path."""

    timing: JobTiming
    result: EvaluationResult | None = None
    cancel_job: bool = False
    terminal: bool = False
    last_unavailable: Exception | None = None
    query_failures: int = 0
    consecutive_failures: int = 0
    request_stage: str = "poll"


def _deadline_result(op, job_id, state):
    stage = state.timing.stage
    queued = stage in {"compile_queue", "run_queue"}
    if state.last_unavailable is not None:
        # An unreachable status/result endpoint is not evidence of a model timeout.
        return EvaluationResult(
            op=op,
            job_id=job_id,
            verdict=f"{state.request_stage}_failed",
            error=(
                f"query unavailable until phase deadline: "
                f"{type(state.last_unavailable).__name__}: {state.last_unavailable}"
            ),
            infrastructure_failure=True,
            retryable=True,
            timing={"timeout_phase": stage, "timeout_scope": "infrastructure"},
        )
    return EvaluationResult(
        op=op,
        job_id=job_id,
        verdict="queue_timeout" if queued else "timeout_fail",
        score=None if queued else 0.0,
        compiled=True if stage == "running" else None,
        infrastructure_failure=queued,
        retryable=queued,
        timing={"timeout_phase": stage, "timeout_scope": "queue" if queued else "evaluation"},
    )


async def _query_job(client, op, job_id, state) -> bool:
    """Poll and, when terminal, fetch the result in the same iteration."""
    response = None
    try:
        if not state.terminal:
            state.request_stage = "poll"
            status = await client.status(job_id)
            response = status
            state.timing.transition(active_stage(status), asyncio.get_running_loop().time())
            state.terminal = status["terminal"]
        if state.terminal:
            state.request_stage = "result"
            response = await client.result(job_id)
            state.result = normalize_result(response, job_id=job_id, op=op)
            await _collect_failure_logs(client, state.result)
            return True
        state.last_unavailable = None
        state.consecutive_failures = 0
    except (ServiceUnavailable, asyncio.TimeoutError) as exc:
        state.last_unavailable = exc
        state.query_failures += 1
        state.consecutive_failures += 1
        logger.warning(
            "CANN %s temporarily unavailable; retaining job: op=%s, job_id=%s, failures=%s, error=%s",
            state.request_stage,
            op,
            job_id,
            state.consecutive_failures,
            exc,
        )
    except ProtocolError as exc:
        if isinstance(exc, HTTPError) and exc.status in {401, 403}:
            raise  # Authentication needs an explicit deployment/configuration fix.
        state.cancel_job = True
        logger.warning("CANN %s failed: op=%s, job_id=%s, error=%s", state.request_stage, op, job_id, exc)
        state.result = EvaluationResult(
            op=op,
            job_id=job_id,
            verdict=f"{state.request_stage}_failed",
            error=str(exc),
            infrastructure_failure=True,
            retryable=True,
            envelope={"response": response} if response is not None else {},
        )
        return True
    return False


async def _poll_job(client, config, op, job_id, state):
    loop = asyncio.get_running_loop()
    while True:
        snapshot = state.timing.snapshot(loop.time())
        queued = state.timing.stage in {"compile_queue", "run_queue"}
        elapsed = snapshot["queue_elapsed_sec"] if queued else snapshot["execution_elapsed_sec"]
        limit = config.queue_timeout if queued else config.execution_timeout
        if elapsed >= limit:
            state.cancel_job = True
            state.result = _deadline_result(op, job_id, state)
            return state.result
        # normalize_result binds the score to the trusted dataset operator.
        if await _query_job(client, op, job_id, state):
            return state.result
        snapshot = state.timing.snapshot(loop.time())
        queued = state.timing.stage in {"compile_queue", "run_queue"}
        elapsed = snapshot["queue_elapsed_sec"] if queued else snapshot["execution_elapsed_sec"]
        limit = config.queue_timeout if queued else config.execution_timeout
        delay = config.poll_interval
        if state.consecutive_failures:
            delay = min(config.poll_interval * 2 ** min(state.consecutive_failures - 1, 4), 8.0)
        await asyncio.sleep(min(delay, max(0.0, limit - elapsed)))


async def _finish_attempt(client, job_id, state):
    # One cleanup path; do not restart DELETE when its own policy fails.
    result = state.result
    if state.cancel_job and result is None:
        # Preserve the original protocol/cancellation exception.
        try:
            await asyncio.shield(client.delete(job_id))
        except Exception:
            logger.exception("Failed to cancel accepted CANN job %s while handling an error", job_id)
    elif state.cancel_job:
        result.cleanup = await client.delete(job_id)
        result.resubmit_safe = result.verdict == "queue_timeout" and result.cleanup["delete_succeeded"]
    if result is not None:
        result.timing = {**state.timing.snapshot(asyncio.get_running_loop().time()), **result.timing}
        if state.query_failures:
            result.timing["query_failure_count"] = state.query_failures


async def _attempt(client, config, *, op, submission, alias, filename, on_submitted=None):
    loop = asyncio.get_running_loop()
    start = loop.time()
    response = None
    try:
        response = await client.submit(op=op, submission=submission, alias=alias, filename=filename)
        job = submitted_job(response)
    except HTTPError:
        raise  # Bad address/authentication/configuration is not a model failure.
    except (SubmissionUnknown, ProtocolError) as exc:
        # No known job id to poll/cancel. The caller filters this service
        # failure; it is not a scored model failure or another queue retry.
        logger.warning("CANN submit failed: op=%s, error=%s", op, exc)
        return EvaluationResult(
            op=op,
            verdict="submit_failed",
            error=str(exc),
            infrastructure_failure=True,
            retryable=True,
            timing={
                "total_elapsed_sec": loop.time() - start,
                "submit_attempt_count": getattr(exc, "attempt_count", 1),
            },
            envelope={"response": response} if response is not None else {},
        )
    job_id = job["job_id"]
    state = _PollingState(JobTiming(start, loop.time()))
    try:
        if on_submitted is not None:
            on_submitted(job_id)
        return await _poll_job(client, config, op, job_id, state)
    except BaseException:
        state.cancel_job = True
        raise
    finally:
        await _finish_attempt(client, job_id, state)


async def evaluate_submission(
    config: EvaluatorConfig, *, op: str, submission: str, alias: str, filename: str | None = None, on_submitted=None
) -> EvaluationResult:
    """Await one result; optionally notify locally when POST returns a known job id."""
    valid, info = validate_submission(submission)
    if not valid:
        error = f"invalid four-file submission: {info['format_parse_error']}"
        logger.warning("CANN submission rejected locally: op=%s, error=%s", op, error)
        return EvaluationResult(op=op, verdict="parse_fail", score=0.0, error=error)
    if not op.strip() or not alias.strip():
        raise ValueError("op and alias must be nonempty")
    attempts = []
    start = asyncio.get_running_loop().time()
    async with EvaluatorClient(config) as client:
        for index in range(config.queue_retries + 1):
            result = await _attempt(
                client,
                config,
                op=op,
                submission=submission,
                alias=alias,
                filename=filename,
                on_submitted=on_submitted,
            )
            attempts.append(
                {
                    "attempt": index + 1,
                    "job_id": result.job_id,
                    "verdict": result.verdict,
                    **result.timing,
                    **result.cleanup,
                }
            )
            if not result.resubmit_safe:
                break
    result.attempts = attempts
    result.timing = {**result.timing, "total_elapsed_sec": asyncio.get_running_loop().time() - start}
    return result
