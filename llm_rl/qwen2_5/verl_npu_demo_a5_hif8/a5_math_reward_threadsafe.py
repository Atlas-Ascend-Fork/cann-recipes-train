# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Run the sample's new_math_reward under verl 0.9, which scores in worker threads.

math_verify's signal.alarm() timeout only works on the main thread, so every
parse() raises there. Only parse/verify are rebound; scoring is unchanged.
Set A5_OFFICIAL_REWARD_PATH if new_math_reward.py is not beside this file.
"""

import hashlib
import importlib.util
import os
from functools import partial
from pathlib import Path

import math_verify

OFFICIAL_REWARD_SHA256 = "1c9f68ac7f3ebe387bf7f44fb3134031cb50375b2643a8952f2afe76d8625124"


def _locate_official_reward() -> Path:
    candidates = []
    if os.environ.get("A5_OFFICIAL_REWARD_PATH"):
        candidates.append(Path(os.environ["A5_OFFICIAL_REWARD_PATH"]))
    here = Path(__file__).resolve().parent
    candidates.append(here / "new_math_reward.py")
    candidates.append(here / "verl" / "utils" / "reward_score" / "new_math_reward.py")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "new_math_reward.py not found; tried: " + ", ".join(str(c) for c in candidates)
    )


def _load_official_module():
    path = _locate_official_reward()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OFFICIAL_REWARD_SHA256:
        raise RuntimeError(
            f"new_math_reward.py changed: {path} sha256={digest}, expected {OFFICIAL_REWARD_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("a5_official_new_math_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.parse = partial(math_verify.parse, parsing_timeout=None)
    module.verify = partial(math_verify.verify, timeout_seconds=None)
    return module, path


_OFFICIAL, OFFICIAL_REWARD_PATH = _load_official_module()

compute_score = _OFFICIAL.compute_score
compute_accuracy_score = _OFFICIAL.compute_accuracy_score
compute_format_score = _OFFICIAL.compute_format_score
compute_cot_score = _OFFICIAL.compute_cot_score
