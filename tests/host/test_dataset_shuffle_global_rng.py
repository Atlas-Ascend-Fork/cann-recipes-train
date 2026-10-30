# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import ast
import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "vime-forge/vime/utils"


def definitions(name, names, **namespace):
    """Execute the real definitions, isolating unrelated Ray/model imports."""
    path = UTILS / name
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) in names
    ]
    assert {node.name for node in nodes} == set(names)
    namespace["__name__"] = __name__
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def module(name):
    """Import a complete standalone utility module, without replacing imports."""
    path = UTILS / name
    identifier = "host_" + path.stem
    spec = importlib.util.spec_from_file_location(identifier, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[identifier] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def dataset():
    cls = definitions("data.py", ["Dataset"], random=random)["Dataset"]
    value = cls.__new__(cls)
    value.epoch_id, value.seed = -1, 42
    value.origin_samples = list(range(20))
    value.samples = value.origin_samples
    return value


def test_shuffle_preserves_unrelated_rng_and_is_repeatable():
    value = dataset()
    original = random.getstate()
    try:
        random.seed(37)
        state = random.getstate()
        value.shuffle(3)
        assert random.getstate() == state
        control = dataset()
        control.shuffle(3)
        assert value.samples == control.samples
        assert sorted(value.samples) == value.origin_samples
    finally:
        random.setstate(original)


def test_repeated_epoch_is_an_identity():
    value = dataset()
    value.shuffle(2)
    old = value.samples
    value.shuffle(2)
    assert value.samples is old
