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

"""Mask budget-forced closing tokens without changing generation or logprobs."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)


def _find(tokens, marker, *, last=False):
    indices = range(len(tokens) - len(marker), -1, -1) if last else range(len(tokens) - len(marker) + 1)
    return next((index for index in indices if tokens[index: index + len(marker)] == marker), -1)


def budget_boundary(args, tokenizer, prompt_ids, budget):
    """Match vLLM's budget count for this recipe's prefilled thinking block."""
    if budget is None:
        return None
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("thinking token budget must be a nonnegative integer")
    if getattr(args, "vllm_speculative_config", None):
        raise ValueError("thinking-budget loss masking does not support speculative decoding")
    config = getattr(args, "vllm_reasoning_config", None) or {}
    if isinstance(config, str):
        config = json.loads(config)

    # vLLM CLI returns ReasoningConfig; also accept mappings and JSON.
    def field(name):
        return config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)

    start = field("reasoning_start_token_ids") or tokenizer.encode(
        field("reasoning_start_str") or "", add_special_tokens=False
    )
    end = field("reasoning_end_token_ids") or tokenizer.encode(
        field("reasoning_end_str") or "", add_special_tokens=False
    )
    start, end = list(start), list(end)
    if not start or not end:
        raise ValueError("thinking-budget loss masking requires explicit start/end token boundaries")
    start_index = _find(prompt_ids, start, last=True)
    if start_index < 0 or start_index < _find(prompt_ids, end, last=True):
        return None  # The request does not continue an open thinking block.
    prefilled = len(prompt_ids) - start_index - len(start)
    return max(0, budget - prefilled), end


def mask_budget_end(sample, boundary, *, prefix_length, response_start):
    """Mask only matching forced tokens; preserve all tokens and rollout logprobs."""
    if boundary is None:
        return
    offset, end = boundary
    response = list(sample.tokens[prefix_length:])
    if len(response) != sample.response_length - response_start:
        raise ValueError("thinking-budget mask requires aligned response token IDs")
    if _find(response[:offset], end) >= 0 or len(response) <= offset:
        return  # Natural closing before the boundary, or not enough output yet.
    count = min(len(end), len(response) - offset)
    if response[offset: offset + count] != end[:count]:
        sample.metadata = {
            **(sample.metadata or {}),
            "thinking_budget_mask_skip": {
                "reason": "end_token_mismatch",
                "response_index": response_start + offset,
            },
        }
        logger.warning(
            "Skipping thinking-budget loss masking for sample %s at response index %s: "
            "end token mismatch; keeping the original loss_mask",
            getattr(sample, "index", None),
            response_start + offset,
        )
        return
    mask = list(sample.loss_mask) if sample.loss_mask is not None else [1] * sample.response_length
    if len(mask) != sample.response_length:
        raise ValueError("thinking-budget mask requires an aligned loss_mask")
    indices = list(range(response_start + offset, response_start + offset + count))
    for index in indices:
        mask[index] = 0
    sample.loss_mask = mask
    sample.metadata = {**(sample.metadata or {})}
    sample.metadata["thinking_budget_masked_indices"] = sorted(
        set(sample.metadata.get("thinking_budget_masked_indices", ())) | set(indices)
    )
