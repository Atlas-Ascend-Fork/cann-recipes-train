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

"""Native sample/filter hooks; no custom RolloutManager or actor needed."""

from __future__ import annotations

from vime.observability.metric_utils import has_repetition
from vime.rollout.filter_hub.base_types import DynamicFilterOutput

from .config import settings, task_identity
from .formatting import parse_response
from .rollout_jsonl import archive_generated


def prepare_sample(args, sample, *, rollout_id=None, evaluation=False):
    _, config = settings(args, evaluation=evaluation)
    if rollout_id is None:
        raise ValueError("sample hook requires the native rollout iteration id")
    op = task_identity(sample)
    parsed = parse_response(sample.response)
    metadata = sample.metadata = {**(sample.metadata or {})}
    metadata.update(
        {
            "op_name": op,
            "single_turn_rollout_step": rollout_id,
            "single_turn_evaluation": evaluation,
            "eval_alias": f"{config.alias}_eval_iter_{rollout_id + 1}" if evaluation else config.alias,
            "single_turn_submission": parsed.submission,
            "single_turn_format_valid": parsed.format_valid,
            "single_turn_format_info": parsed.format_info,
        }
    )
    failures = []
    if not parsed.format_valid:
        failures.append("invalid_submission_format")
    if has_repetition(sample.response):
        failures.append("repetitive_output")
    metadata["single_turn_policy_failures"] = failures
    archive_generated(sample, config.archive_dir)
    return sample


def keep_trainable_group(args, samples, **kwargs):
    del args, kwargs
    if any(sample.remove_sample or sample.metadata.get("evaluator_infrastructure_failure") for sample in samples):
        return DynamicFilterOutput(keep=False, reason="evaluator_infrastructure_failure", keep_when_insufficient=False)
    return DynamicFilterOutput(keep=True)
