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


"""Vision input seam. A multimodal recipe registers its translator factory from its own
``config_registry``; nothing model-specific is named here.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

# Keyed by recipe function name, matching the launcher's --config value.
VisionInputBuilder = Callable[..., Callable[..., dict[str, Any]]]

_VISION_INPUT_BUILDERS: dict[str, VisionInputBuilder] = {}


def vision_recipe(builder: VisionInputBuilder):
    """Register a vision translator factory under the decorated recipe's name."""

    def declare(recipe):
        _VISION_INPUT_BUILDERS[recipe.__name__] = builder
        return recipe

    return declare


def vision_input_builder(recipe: str | None) -> VisionInputBuilder | None:
    """That recipe's builder: None for a text recipe, or for a name that resolved to nothing."""
    if not recipe:
        return None
    return _VISION_INPUT_BUILDERS.get(recipe)


def assert_vision_parallelism(
    *,
    recipe: str,
    cp_enabled: bool,
    cp: int,
    spmd_backend: str,
    multimodal_keys: dict | None = None,
) -> None:
    """Reject CP/SPMD modes unsupported by vision kwargs merged after input preparation.

    Applies when a recipe registers a builder or ``multimodal_keys`` is nonempty.
    """
    if not _VISION_INPUT_BUILDERS.get(recipe) and not multimodal_keys:
        return
    if cp_enabled:
        raise ValueError(
            f"{recipe} needs --parallelism.context_parallel_degree 1: token-indexed vision "
            f"kwargs cannot be sharded like positions; got cp={cp}."
        )
    if spmd_backend == "spmd_types":
        raise ValueError(
            f"{recipe} needs spmd_backend=partial_dtensor: vision kwargs are merged after the "
            f"input seam; got {spmd_backend}."
        )


@dataclass(frozen=True)
class VisionModelKwargs:
    """Forward kwargs by index space: ``item_indexed`` escapes CP sharding and SPMD
    annotation, ``token_indexed`` would have to be sharded with ``positions``.
    """

    item_indexed: dict[str, Any]
    token_indexed: dict[str, Any] = field(default_factory=dict)


def iter_sample_inputs(samples):
    """``(index, processor output)`` for the samples carrying one, in packing order."""
    for index, mm_inputs in enumerate(samples or ()):
        if mm_inputs:
            yield index, mm_inputs


def config_field(config, name: str):
    """Read ``name`` off an HF config node that may be an object or a plain dict."""
    value = getattr(config, name, None)
    if value is None and isinstance(config, dict):
        value = config.get(name)
    return value
