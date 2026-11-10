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

"""Overlap online evaluation with the next generation using ordinary Ray data refs.

There is no local evaluation queue: each sample starts the existing evaluator
coroutine immediately. Native Vime still converts, schedules and splits data.
Only the final per-DP data refs wait for rewards; generate() itself never does.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import ray
import torch

from vime.rollout.reward_normalization import normalize_rewards_by_trajectory
from vime.utils.misc import Box, load_function

from .config import settings
from .reward import _reward_one, _thinking_close_token_id
from .rollout_jsonl import archive_final

logger = logging.getLogger(__name__)


@dataclass
class RewardPatch:
    sample_indices: list[int]
    raw_rewards: list[float]
    rewards: list[float]
    masked: list[bool]
    rollout_mask_sums: list[float]


def build_reward_patch(args, samples):
    """Normalize only complete valid groups; retain the original DP schedule."""
    groups = defaultdict(list)
    for position, sample in enumerate(samples):
        groups[sample.group_index].append(position)
    masked = [False] * len(samples)
    for positions in groups.values():
        failed = any(
            samples[i].remove_sample or samples[i].metadata.get("evaluator_infrastructure_failure") for i in positions
        )
        for i in positions:
            masked[i] = failed
            samples[i].metadata["single_turn_training_group_masked"] = failed
            if failed:
                samples[i].loss_mask = [0] * samples[i].response_length

    raw = [float(s.get_reward_value(args)) for s in samples]
    valid_positions = [i for i in range(len(samples)) if not masked[i]]
    valid_samples = [samples[i] for i in valid_positions]
    valid_raw = [raw[i] for i in valid_positions]
    if (
        args.advantage_estimator in {"grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"}
        and args.rewards_normalization
    ):
        valid_rewards = normalize_rewards_by_trajectory(args, valid_samples, valid_raw)
    else:
        valid_rewards = valid_raw
    rewards = [0.0] * len(samples)
    for i, value in zip(valid_positions, valid_rewards, strict=True):
        rewards[i] = value

    mask_sums = defaultdict(float)
    rollout_ids = [s.rollout_id if s.rollout_id is not None else s.index for s in samples]
    for i, sample in enumerate(samples):
        if not masked[i]:
            mask_sums[rollout_ids[i]] += (
                sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length
            )
    return RewardPatch([s.index for s in samples], raw, rewards, masked, [mask_sums[rid] for rid in rollout_ids])


def apply_reward_patch(data, patch):
    """Copy small outcome fields, never mutate Ray's shared/read-only tensors."""
    partition = data["partition"]
    if data["sample_indices"] != [patch.sample_indices[i] for i in partition]:
        raise ValueError("CANN reward patch does not match the DP sample order")
    updated = dict(data)
    updated["rewards"] = [patch.rewards[i] for i in partition]
    updated["raw_reward"] = list(patch.raw_rewards)  # Native pass-rate metrics need the whole batch.
    updated["loss_masks"] = [
        torch.zeros_like(mask) if patch.masked[i] else mask
        for i, mask in zip(partition, data["loss_masks"], strict=True)
    ]
    updated["rollout_mask_sums"] = torch.tensor([patch.rollout_mask_sums[i] for i in partition], dtype=torch.float32)
    return updated


_patch_rank = ray.remote(num_cpus=0)(apply_reward_patch)


class OnlineEvaluation:
    """An async actor body: independent evaluator tasks, keyed by step/sample id.

    Waiting for one batch yields the event loop, so another batch can submit.
    HTTP polling, timeouts, DELETE and safe resubmission stay in evaluate_submission.
    """

    def __init__(self, args):
        from vime.observability.logging_utils import configure_logger

        configure_logger()
        self.args = args
        self.tracking_ready = False
        self.batches = {}
        self.started_at = {}
        self.completed_at = {}

    async def submit(self, step, sample, thinking_close_token_id=None, context=None):
        if not self.tracking_ready:
            from vime.observability.logging_utils import init_tracking

            # The actor is created before native train() initializes tracking.
            # First generation carries its finalized run id / worker settings.
            if context is not None:
                self.args = context
            init_tracking(self.args, primary=False)
            self.tracking_ready = True
        tasks = self.batches.setdefault(step, {})
        if sample.index in tasks:
            raise ValueError(f"duplicate CANN evaluation: step={step}, sample={sample.index}")
        self.started_at.setdefault(step, time.time())
        accepted = asyncio.Event()

        def on_submitted(job_id):
            sample.metadata["single_turn_submitted_job_id"] = job_id
            accepted.set()

        async def evaluate():
            try:
                return await self._evaluate(step, sample, thinking_close_token_id, on_submitted=on_submitted)
            finally:
                accepted.set()  # Local/model/transport failure before a job id must also unblock admission.

        task = tasks[sample.index] = asyncio.create_task(evaluate())
        await accepted.wait()  # POST acknowledgement only, never the remote terminal result.
        if task.done():
            task.result()  # Propagate explicit configuration/authentication errors as in inline mode.

    async def finish(self, step, sample_indices, generation):
        tasks = self.batches[step]
        if set(tasks) != set(sample_indices) or len(sample_indices) != len(tasks):
            raise ValueError(f"CANN evaluation manifest mismatch: step={step}")
        samples = list(await asyncio.gather(*(tasks[i] for i in sample_indices)))
        patch = build_reward_patch(self.args, samples)
        for sample in samples:
            archive_final(sample, sample.reward)  # Include the final training-group mask, not just the scored outcome.
        from vime.observability.rollout_data_utils import save_debug_rollout_data
        from vime.observability.rollout_metrics import log_rollout_data

        elapsed, finished_at, extra = generation
        timing = {
            **extra,
            "perf/generation_time": elapsed,
            "perf/evaluator_pipeline_time": max(0.0, self.completed_at[step] - self.started_at[step]),
            "perf/evaluator_tail_time": max(0.0, self.completed_at[step] - finished_at),
        }
        save_debug_rollout_data(self.args.save_debug_rollout_data, samples, rollout_id=step, evaluation=False)
        # Reward-dependent metrics are emitted only here, with final rewards.
        log_rollout_data(step, self.args, samples, timing, max(elapsed, 1e-6))
        del self.batches[step]
        del self.started_at[step]
        del self.completed_at[step]
        return patch

    async def close(self):
        tasks = [task for batch in self.batches.values() for task in batch.values()]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.batches.clear()
        self.started_at.clear()
        self.completed_at.clear()

    async def _evaluate(self, step, sample, token_id, *, on_submitted=None):
        sample.metadata["single_turn_evaluation_pending"] = False
        sample.reward = await _reward_one(
            self.args, sample, thinking_close_token_id=token_id, on_submitted=on_submitted
        )
        self.completed_at[step] = max(self.completed_at.get(step, 0.0), time.time())
        return sample


async def submit_reward(args, sample, **kwargs):
    """Native custom-RM adapter: admit one sample, return an internal placeholder."""
    if isinstance(sample, list):
        return list(await asyncio.gather(*(submit_reward(args, item, **kwargs) for item in sample)))
    if sample.metadata["single_turn_evaluation"]:
        return await _reward_one(args, sample)  # Periodic validation remains blocking.
    if sample.index is None or sample.group_index is None:
        raise ValueError("deferred CANN evaluation requires native sample/group indices")
    _, task = settings(args)
    token_id = None
    if task.thinking_close_reward:
        try:
            token_id = _thinking_close_token_id(args)
        except Exception as exc:
            # Preserve inline reward's tokenizer-error behavior (skip bonus).
            sample.metadata["thinking_close_tokenizer_error"] = f"{type(exc).__name__}: {exc}"
    sample.metadata["single_turn_evaluation_pending"] = True
    context = None if getattr(args, "cann_evaluation_configured", False) else args
    await args.cann_evaluation_actor.submit.remote(
        sample.metadata["single_turn_rollout_step"], sample, token_id, context
    )
    args.cann_evaluation_configured = True
    return 0.0  # Never train/log this placeholder: the postprocess hook replaces its data ref.


def log_rollout(rollout_id, args, samples, extra_metrics, rollout_time):
    if any(s.metadata.get("single_turn_evaluation_pending") for s in samples):
        pending = getattr(args, "cann_generation", None)
        if pending is None:
            pending = args.cann_generation = {}
        pending[rollout_id] = (rollout_time, time.time(), dict(extra_metrics or {}))
        return True  # Suppress both task and native statistics for placeholder rewards.
    path = args.cann_inline_log_path
    return load_function(path)(rollout_id, args, samples, extra_metrics, rollout_time) if path else False


def postprocess_train_data(args, rollout_id, samples, data_refs):
    """Return pending DP refs immediately, leaving the native async loop intact."""
    if not all(s.metadata.get("single_turn_evaluation_pending") for s in samples):
        raise ValueError("async CANN evaluation requires every training sample to use submit_reward")
    generation = args.cann_generation.pop(rollout_id)
    patch_ref = args.cann_evaluation_actor.finish.remote(rollout_id, [s.index for s in samples], generation)
    return [Box(_patch_rank.remote(ref.inner, patch_ref)) for ref in data_refs]


def start(args, entrypoint):
    """Install example adapters before serializing args to native Ray workers."""
    if entrypoint != "train_async.py" or args.debug_rollout_only or args.colocate:
        raise ValueError("async CANN evaluation requires non-colocated train_async.py training")
    if args.update_weights_interval != 1:
        raise ValueError("async CANN evaluation requires update_weights_interval=1")
    if args.group_rm or args.partial_rollout or args.over_sampling_batch_size != args.rollout_batch_size:
        raise ValueError("async CANN evaluation requires per-sample RM, no partial rollout, and no oversampling")
    if getattr(args, "dynamic_sampling_filter_path", None) not in {
        None,
        "examples.cann_bench_single_turn.hooks.keep_trainable_group",
    }:
        raise ValueError("async CANN evaluation cannot dynamically resample using pending rewards")
    if args.rollout_data_transport != "object-store":
        raise ValueError("async CANN evaluation currently requires object-store data transport")
    if args.custom_reward_post_process_path or args.custom_convert_samples_to_train_data_path:
        raise ValueError("async CANN evaluation currently uses native reward normalization/data conversion")
    if args.custom_train_data_postprocess_path:
        raise ValueError("async CANN evaluation owns custom_train_data_postprocess_path")
    if args.custom_rm_path not in {None, "examples.cann_bench_single_turn.reward.reward_func"}:
        raise ValueError("async CANN evaluation requires the single-turn online reward")
    args.cann_inline_log_path = args.custom_rollout_log_function_path
    args.custom_rollout_log_function_path = __name__ + ".log_rollout"
    args.custom_rm_path = __name__ + ".submit_reward"
    args.custom_train_data_postprocess_path = __name__ + ".postprocess_train_data"
    actor = ray.remote(num_cpus=0)(OnlineEvaluation).remote(args)
    args.cann_evaluation_actor = actor
    return actor


def stop(actor):
    try:
        ray.get(actor.close.remote())
    except Exception:
        logger.exception("Failed to close background CANN evaluations")
    finally:
        try:
            ray.kill(actor)
        except Exception:
            logger.exception("Failed to stop the CANN evaluation actor")
