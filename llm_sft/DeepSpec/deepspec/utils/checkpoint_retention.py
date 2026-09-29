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

import json
import math
import os
import re
import shutil

from .io import safe_symlink


BEST_CHECKPOINT_LINK = "step_best"
BEST_CHECKPOINT_STATE = "best_checkpoint.json"
_STEP_PATTERN = re.compile(r"^step_(\d+)$")


def _step_checkpoints(checkpoint_root):
    checkpoints = []
    for name in os.listdir(checkpoint_root):
        match = _STEP_PATTERN.fullmatch(name)
        path = os.path.join(checkpoint_root, name)
        if match and os.path.isdir(path) and not os.path.islink(path):
            checkpoints.append((int(match.group(1)), os.path.realpath(path)))
    return sorted(checkpoints, reverse=True)


def _best_checkpoint_dir(checkpoint_root):
    link = os.path.join(checkpoint_root, BEST_CHECKPOINT_LINK)
    if not os.path.islink(link):
        return None
    target = os.path.realpath(link)
    return target if os.path.isdir(target) else None


def validate_checkpoint_retention_config(
    *,
    keep_last,
    keep_best,
    metric_name,
    greater_is_better,
):
    _validate_retention_limits(keep_last=keep_last, keep_best=keep_best)
    if not isinstance(metric_name, str) or not metric_name:
        raise TypeError("logging.best_checkpoint_metric must be a non-empty string")
    if not isinstance(greater_is_better, bool):
        raise TypeError("logging.greater_is_better must be a boolean")


def _validate_retention_limits(*, keep_last, keep_best):
    if isinstance(keep_last, bool) or not isinstance(keep_last, int):
        raise TypeError("logging.keep_last_checkpoints must be an integer")
    if keep_last < 1:
        raise ValueError("logging.keep_last_checkpoints must be at least 1")
    if not isinstance(keep_best, bool):
        raise TypeError("logging.keep_best_checkpoint must be a boolean")


def retain_checkpoints(checkpoint_root, *, keep_last, keep_best=True):
    """Keep the newest checkpoints plus the checkpoint selected by validation."""
    _validate_retention_limits(keep_last=keep_last, keep_best=keep_best)

    checkpoints = _step_checkpoints(checkpoint_root)
    preserved = {path for _, path in checkpoints[:keep_last]}
    if keep_best:
        best_checkpoint = _best_checkpoint_dir(checkpoint_root)
        if best_checkpoint is not None:
            preserved.add(best_checkpoint)
    else:
        for name in (BEST_CHECKPOINT_LINK, BEST_CHECKPOINT_STATE):
            path = os.path.join(checkpoint_root, name)
            if os.path.lexists(path):
                os.unlink(path)

    removed = []
    for _, path in checkpoints:
        if path not in preserved:
            shutil.rmtree(path)
            removed.append(path)
    return removed


def update_best_checkpoint(
    checkpoint_root,
    checkpoint_dir,
    *,
    metric_name,
    metric_value,
    greater_is_better,
):
    """Update the persistent best-checkpoint state and symlink."""
    if not isinstance(greater_is_better, bool):
        raise TypeError("logging.greater_is_better must be a boolean")
    metric_value = float(metric_value)
    if not math.isfinite(metric_value):
        return False

    state_path = os.path.join(checkpoint_root, BEST_CHECKPOINT_STATE)
    previous = None
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as handle:
            previous = json.load(handle)

    previous_checkpoint = (
        str(previous.get("checkpoint", "")) if previous is not None else ""
    )
    previous_checkpoint_exists = bool(
        _STEP_PATTERN.fullmatch(previous_checkpoint)
        and os.path.isdir(os.path.join(checkpoint_root, previous_checkpoint))
    )
    same_metric = (
        previous_checkpoint_exists
        and previous.get("metric_name") == metric_name
        and previous.get("greater_is_better") == greater_is_better
    )
    if same_metric:
        previous_value = float(previous["metric_value"])
        improved = (
            metric_value > previous_value
            if greater_is_better
            else metric_value < previous_value
        )
        if not improved:
            return False

    checkpoint_dir = os.path.realpath(checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )
    checkpoint_root = os.path.realpath(checkpoint_root)
    if (
        os.path.dirname(checkpoint_dir) != checkpoint_root
        or not _STEP_PATTERN.fullmatch(os.path.basename(checkpoint_dir))
    ):
        raise ValueError(
            f"Checkpoint must be a step_N directory inside {checkpoint_root}: "
            f"{checkpoint_dir}"
        )
    state = {
        "checkpoint": os.path.basename(checkpoint_dir),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "greater_is_better": greater_is_better,
    }
    temporary_path = state_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, state_path)
    safe_symlink(
        checkpoint_dir,
        os.path.join(checkpoint_root, BEST_CHECKPOINT_LINK),
    )
    return True
