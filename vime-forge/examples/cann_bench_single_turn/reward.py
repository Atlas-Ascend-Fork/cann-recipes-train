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

"""Single-turn reward orchestration; operator scoring is independently replaceable."""

from __future__ import annotations

import asyncio
import logging
import math
import time

from examples.cann_bench_evaluator import EvaluationResult, evaluate_submission
from examples.cann_bench_evaluator.client import HTTPError
from examples.cann_bench_evaluator.scoring import validate_score
from vime.utils.types import Sample

from .config import scoring_settings, settings
from .rollout_jsonl import archive_final

logger = logging.getLogger(__name__)

FORMAT_RECOVERY_PENALTY = 0.05
POLICY_FAILURE_REWARD = -0.1
REPETITIVE_OUTPUT_REWARD = -1.0
THINKING_CLOSE_BONUS = 0.05


def _thinking_close_token_id(args):
    # Reuse the tokenizer already loaded for native generation, without loading
    # a second model/tokenizer or adding fields to the shared Sample contract.
    from vime.rollout.vllm_rollout import GenerateState

    ids = GenerateState(args).tokenizer.encode("</think>", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError("thinking-close reward requires a single-token </think>")
    return int(ids[0])


def _thinking_close_bonus(args, sample, result, *, enabled, token_id=None):
    metadata = sample.metadata
    metadata.update(thinking_close_bonus=0.0, thinking_close_bonus_applied=False)

    def skip(reason):
        metadata["thinking_close_bonus_skip_reason"] = reason
        return 0.0

    if not enabled:
        return skip("disabled")
    if result is None:
        return skip("model_policy_failure")
    if result.infrastructure_failure or result.retryable or sample.remove_sample:
        return skip("evaluator_infrastructure_failure")
    if result.compiled is not True:
        return skip("compile_failed")
    length = sample.response_length
    logprobs = sample.rollout_log_probs
    if length <= 0 or len(sample.tokens) < length:
        return skip("missing_or_unaligned_rollout_logprob")
    if logprobs is None or len(logprobs) != length:
        return skip("missing_or_unaligned_rollout_logprob")
    try:
        tag_id = _thinking_close_token_id(args) if token_id is None else token_id
    except Exception as exc:
        metadata["thinking_close_tokenizer_error"] = f"{type(exc).__name__}: {exc}"
        return skip("tokenizer_error")
    tokens = list(sample.tokens[-length:])
    if tag_id not in tokens:
        return skip("think_close_tag_missing")
    index = length - 1 - tokens[::-1].index(tag_id)
    metadata.update(thinking_close_token_id=tag_id, thinking_close_token_index=index)
    try:
        logprob = float(logprobs[index])
    except (TypeError, ValueError):
        return skip("invalid_logprob")
    if not math.isfinite(logprob):
        return skip("invalid_logprob")
    metadata["thinking_close_logprob"] = logprob
    # Replay gives forced singleton supports exactly zero. Follow the explicit
    # nonzero rule, not a length heuristic or a requirement for full accuracy.
    if logprob == 0.0:
        return skip("zero_logprob")
    metadata.update(
        thinking_close_bonus=THINKING_CLOSE_BONUS,
        thinking_close_bonus_applied=True,
        thinking_close_bonus_skip_reason=None,
    )
    return THINKING_CLOSE_BONUS


async def _evaluate_sample(config, metadata, *, on_submitted=None):
    """Record evaluation diagnostics without turning deployment errors into rewards."""
    started = time.perf_counter()
    try:
        result = await evaluate_submission(
            config,
            op=metadata["op_name"],
            submission=metadata["single_turn_submission"],
            alias=metadata["eval_alias"],
            **({"on_submitted": on_submitted} if on_submitted is not None else {}),
        )
    except HTTPError:
        raise  # Wrong endpoint/authentication needs a configuration fix, not model-zero training.
    except Exception as exc:
        # Transport failures remain untrainable; policy errors are handled separately below.
        logger.exception("CANN evaluation failed: op=%s, error=%s", metadata["op_name"], exc)
        error = f"{type(exc).__name__}: {exc}"
        result = EvaluationResult(
            op=metadata["op_name"],
            verdict="evaluation_failed",
            error=error,
            infrastructure_failure=True,
            retryable=True,
        )
    metadata["single_turn_evaluation_started_at"] = started
    metadata["single_turn_evaluation_finished_at"] = time.perf_counter()
    metadata["evaluation"] = result.to_dict()
    return result


async def _reward_one(args, sample, *, thinking_close_token_id=None, on_submitted=None):
    metadata = sample.metadata
    if "single_turn_submission" not in metadata:
        raise ValueError("CANN reward requires prepare_sample via --rollout-sample-hook-path")
    config, task = settings(args, evaluation=metadata["single_turn_evaluation"])
    scorer, scoring_info = scoring_settings(args)
    metadata["scoring"] = scoring_info
    failures = metadata["single_turn_policy_failures"]
    result = None
    if failures:
        reward = REPETITIVE_OUTPUT_REWARD if "repetitive_output" in failures else POLICY_FAILURE_REWARD
        metadata["evaluation"] = {"skipped": True, "reason": failures, "infrastructure_failure": False}
        breakdown = {"policy_failure_reward": reward}
        scoring_info["skipped"] = "model_policy_failure"
    else:
        result = await _evaluate_sample(config, metadata, on_submitted=on_submitted)
        if result.infrastructure_failure or result.retryable:
            sample.remove_sample = True
            sample.status = Sample.Status.ABORTED
            metadata["evaluator_infrastructure_failure"] = True
            scoring_info["skipped"] = "evaluator_infrastructure_failure"
            reward, breakdown = 0.0, {}
        else:
            try:
                score = validate_score(scorer(result))
            except Exception as exc:
                scoring_info["error"] = f"{type(exc).__name__}: {exc}"
                metadata["single_turn_reward"] = None
                metadata["single_turn_reward_breakdown"] = {}
                archive_final(sample, None)
                raise RuntimeError(
                    f"CANN scoring failed: {scoring_info['function_path']}, op={result.op}, job_id={result.job_id}"
                ) from exc
            scoring_info.update(reward=score.reward, components=dict(score.components))
            reward, breakdown = score.reward, dict(score.components)
    bonus = _thinking_close_bonus(
        args, sample, result, enabled=task.thinking_close_reward, token_id=thinking_close_token_id
    )
    breakdown["thinking_close_bonus"] = bonus
    if result is not None and not (result.infrastructure_failure or result.retryable):
        penalty = FORMAT_RECOVERY_PENALTY if metadata["single_turn_format_info"].get("format_recovered") else 0.0
        reward = reward + bonus - penalty
        breakdown["format_recovery_penalty"] = penalty
    metadata["single_turn_reward"] = reward
    metadata["single_turn_reward_breakdown"] = breakdown
    archive_final(sample, reward)
    return reward


async def reward_func(args, sample: Sample | list[Sample], **kwargs):
    del kwargs
    if isinstance(sample, list):
        return list(await asyncio.gather(*(_reward_one(args, item) for item in sample)))
    return await _reward_one(args, sample)
