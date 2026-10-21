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

"""The single seam between precision-guard collection and the training code.

Collection needs handles only the actor owns (megatron's HF weight iterator, torchtitan's
state-dict adapter and TP group). Threading those through ``train()`` / ``train_one_step()``
would put diagnostic parameters into the production API, so the actor registers itself once
and collectors fetch what they need.

One training actor per process (Ray actor), so a single module-level reference suffices.
The reference is weak: diagnostics must not extend the actor's lifetime.
"""

from __future__ import annotations

import weakref

_ref: weakref.ReferenceType | None = None


def attach(actor) -> None:
    """Register the training actor and install injected hooks.

    Call after the actor is fully built: hook installation reads parallelism info off the
    actor to name per-rank quantities (see ``collect_torchtitan.install_hooks``).
    """
    global _ref
    _ref = weakref.ref(actor)

    backend = getattr(getattr(actor, "args", None), "train_backend", None)
    if backend == "torchtitan":
        from . import collect_torchtitan as backend_mod  # noqa: PLC0415
    elif backend == "megatron":
        from . import collect_megatron as backend_mod  # noqa: PLC0415
    else:
        return
    backend_mod.install_hooks(actor)


def actor():
    """The registered actor, or None. Callers must skip or raise explicitly on None."""
    return _ref() if _ref is not None else None


def require_actor(what: str):
    """The registered actor; raise if absent. For collection points that cannot proceed without it."""
    a = actor()
    assert a is not None, (
        f"precision guard needs an actor to collect {what}; "
        "check that precision_guard.attach(self) is still in the actor's init"
    )
    return a
