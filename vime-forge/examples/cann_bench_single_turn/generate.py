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

"""Task generation limits and loss masks; native Vime owns token/logprob handling."""

from __future__ import annotations

import time

from .config import settings, task_identity
from .thinking_budget import budget_boundary, mask_budget_end


async def generate(args, sample, sampling_params, evaluation=False):
    # Validate before spending tokens or accepting a job on the remote evaluator.
    settings(args, evaluation=evaluation)
    task_identity(sample)
    from vime.rollout.vllm_rollout import GenerateState, _prepare_prompt_ids
    from vime.rollout.vllm_rollout import generate as native_generate

    params = dict(sampling_params)
    budget = getattr(args, "rollout_thinking_token_budget", None)
    if budget is not None:
        params["thinking_token_budget"] = budget
    state = GenerateState(args)
    if any(value is not None for value in (sample.multimodal_inputs or {}).values()):
        raise ValueError("CANN single-turn expects text-only prompts")
    prefix_ids = list(_prepare_prompt_ids(sample, state.tokenizer, state.processor))
    prefix_length = len(prefix_ids)
    sample.tokens = prefix_ids
    prompt_length = prefix_length - sample.response_length
    if prompt_length <= 0:
        raise ValueError("empty/invalid prompt token prefix")
    response_start = sample.response_length
    boundary = budget_boundary(args, state.tokenizer, prefix_ids, params.get("thinking_token_budget"))
    limit_name = "eval_max_context_len" if evaluation else "rollout_max_context_len"
    context_limit = getattr(args, limit_name, None)
    if context_limit is not None:
        # native_generate subtracts existing response_length exactly once.
        params["max_new_tokens"] = min(params["max_new_tokens"], int(context_limit) - prompt_length)
        if params["max_new_tokens"] <= sample.response_length:
            raise ValueError("prompt/continuation leaves no response space in the configured context")
    started = time.perf_counter()
    sample = await native_generate(args, sample, params)
    mask_budget_end(sample, boundary, prefix_length=prefix_length, response_start=response_start)
    sample.metadata = {
        **(sample.metadata or {}),
        "single_turn_generation_started_at": started,
        "single_turn_generation_finished_at": time.perf_counter(),
    }
    sample.metadata["single_turn_generation_elapsed_sec"] = (
        sample.metadata["single_turn_generation_finished_at"] - started
    )
    return sample
