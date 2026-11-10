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

"""Task-specific arguments, configuration, and operator identity resolution."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

from examples.cann_bench_evaluator import EvaluatorConfig

DEFAULT_SCORER = "examples.cann_bench_evaluator.scoring.score_result"


def add_example_arguments(parser):
    parser.add_argument(
        "--rollout-thinking-token-budget",
        type=int,
        default=None,
        help="Per-request reasoning token budget; requires configured vLLM reasoning boundaries.",
    )
    return parser


@lru_cache(maxsize=16)
def _load_scorer(path):
    from vime.utils.misc import load_function

    scorer = load_function(path)
    if not callable(scorer) or inspect.iscoroutinefunction(scorer):
        raise ValueError("cann_scoring.function_path must refer to a synchronous callable")
    inspect.signature(scorer).bind(None)  # Check the one-result call without executing the policy.
    try:
        source = inspect.getsourcefile(scorer)
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None
    except (OSError, TypeError):
        digest = None
    return scorer, digest


def scoring_settings(args):
    values = dict(getattr(args, "cann_scoring", None) or {})
    if values.keys() - {"function_path"}:
        raise ValueError("cann_scoring only supports function_path")
    path = values.get("function_path", DEFAULT_SCORER)
    if not isinstance(path, str) or not path.strip() or path != path.strip():
        raise ValueError("cann_scoring.function_path must be a nonempty import path")
    scorer, digest = _load_scorer(path)
    return scorer, {"function_path": path, "source_file_sha256": digest}


@dataclass(frozen=True)
class TaskConfig:
    alias: str = "vime-cann-single-turn"
    archive_dir: str | None = None
    eval_url: str | None = None
    thinking_close_reward: bool = False
    async_evaluation: bool = False

    def __post_init__(self):
        if not isinstance(self.alias, str) or not self.alias.strip():
            raise ValueError("cann_single_turn.alias must be nonempty")
        if not isinstance(self.thinking_close_reward, bool):
            raise ValueError("cann_single_turn.thinking_close_reward must be a boolean")
        if not isinstance(self.async_evaluation, bool):
            raise ValueError("cann_single_turn.async_evaluation must be a boolean")


@lru_cache(maxsize=16)
def _resolved_settings(evaluator_items, task_items):
    """Immutable settings reused per worker, keyed only by the driver snapshot."""
    evaluator = EvaluatorConfig(**dict(evaluator_items))
    task = TaskConfig(**dict(task_items))
    eval_url = task.eval_url or evaluator.url
    eval_evaluator = replace(evaluator, url=eval_url) if eval_url != evaluator.url else evaluator
    return (evaluator, task), (eval_evaluator, task)


def settings(args, *, evaluation=False) -> tuple[EvaluatorConfig, TaskConfig]:
    if getattr(args, "cann_settings_resolved", False):
        return _resolved_settings(
            tuple(sorted(args.cann_evaluator.items())), tuple(sorted(args.cann_single_turn.items()))
        )[bool(evaluation)]
    environment = os.environ
    task_values = dict(getattr(args, "cann_single_turn", None) or {})
    if environment.get("CANN_EVAL_ALIAS"):
        task_values["alias"] = environment["CANN_EVAL_ALIAS"]
    if environment.get("CANN_ARCHIVE_DIR"):
        task_values["archive_dir"] = environment["CANN_ARCHIVE_DIR"]
    thinking_close = environment.get("VIME_CANN_THINKING_CLOSE_REWARD")
    if thinking_close is not None:
        if thinking_close not in {"0", "1"}:
            raise ValueError("VIME_CANN_THINKING_CLOSE_REWARD must be 0 or 1")
        task_values["thinking_close_reward"] = thinking_close == "1"
    async_evaluation = environment.get("CANN_ASYNC_EVALUATION")
    if async_evaluation is not None:
        if async_evaluation not in {"0", "1"}:
            raise ValueError("CANN_ASYNC_EVALUATION must be 0 or 1")
        task_values["async_evaluation"] = async_evaluation == "1"
    task = TaskConfig(**task_values)
    evaluator_values = dict(getattr(args, "cann_evaluator", None) or {})
    evaluator_values["url"] = environment.get("CANN_EVAL_URL") or evaluator_values.get("url") or ""
    if evaluation:
        evaluator_values["url"] = environment.get("CANN_EVAL_EVAL_URL") or task.eval_url or evaluator_values["url"]
    return EvaluatorConfig(**evaluator_values), task


def resolve_run_settings(args):
    """Snapshot driver overrides into serialized args, not the Ray head environment."""
    budget = getattr(args, "rollout_thinking_token_budget", None)
    if budget is not None and budget < 0:
        raise ValueError("--rollout-thinking-token-budget must be a nonnegative integer")
    evaluator, task = settings(args)
    eval_evaluator, _ = settings(args, evaluation=True)
    _, scoring = scoring_settings(args)
    args.cann_evaluator = asdict(evaluator)
    args.cann_single_turn = asdict(replace(task, eval_url=eval_evaluator.url))
    args.cann_scoring = {"function_path": scoring["function_path"]}
    args.cann_settings_resolved = True
    return args


def task_identity(sample) -> str:
    label = sample.label or {}
    if isinstance(label, str):
        try:
            label = json.loads(label)
        except json.JSONDecodeError as exc:
            raise ValueError("CANN label must be a JSON object with op_name") from exc
    if not isinstance(label, dict):
        raise ValueError("CANN label must be an object")
    metadata = sample.metadata or {}
    names = {str(value).strip() for value in (metadata.get("op_name"), label.get("op_name"), label.get("op")) if value}
    if len(names) != 1:
        raise ValueError(f"missing/conflicting operator identity: {sorted(names)}")
    for value in (metadata.get("submission_format"), label.get("format")):
        if value and value != "4string":
            raise ValueError("single-turn CANN requires four-file submissions")
    return names.pop()
