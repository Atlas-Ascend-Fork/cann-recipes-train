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

"""Collection entry points for the precision-validation harness.

``record`` and its variants are the public surface. When collection is off they return on
the first line, at zero cost.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from .fingerprint import tensor_stats
from .registry import attach
from .writer import read_records, records_path, write_records

__all__ = [
    "ALL_SITES",
    "OPT_IN_SITES",
    "WIRED_SITES",
    "attach",
    "cast_stream",
    "export_dtype",
    "wired_sites",
    "enabled",
    "merge_rank_records",
    "read_records",
    "record",
    "record_from_batch",
    "record_stream",
    "records_path",
    "validate_sites",
    "tensor_stats",
    "write_records",
]

logger = logging.getLogger(__name__)

# Every collection point defined in design section 3.
ALL_SITES = (
    "logprob",
    "loss",
    "grad",
    "grad_norm",
    "optim_hparam",
    "weight",
    "optim_state",
    "hidden",
)

# Sites wired on both backends; collected by default when --precision-guard-sites is
# omitted. weight reuses each backend's existing weight-export path (the same one that
# feeds vLLM), so quantity names are HF parameter names.
# optim_state is wired on neither backend and stays in ALL_SITES to keep the design intent
# visible. Requesting it raises: passing validation while producing nothing is the hardest
# failure to diagnose.
WIRED_SITES = ("logprob", "loss", "grad", "grad_norm", "optim_hparam", "weight")

# Wired but kept out of the default set: hidden fingerprints every layer for every sample
# (~30 s/step) and only answers "which layer does a forward difference start at", so
# enabling it by default would slow every other experiment.
OPT_IN_SITES = ("hidden",)

def wired_sites() -> tuple[str, ...]:
    """Sites that can actually collect, opt-in ones included."""
    return WIRED_SITES + OPT_IN_SITES


def enabled(args, site: str) -> bool:
    """Whether this site collects on this run."""
    if site not in ALL_SITES:
        raise ValueError(f"unknown site {site!r}; valid values: {ALL_SITES}")
    available = wired_sites()
    if site not in available:
        raise NotImplementedError(
            f"site {site!r} is not wired; wired sites: {available}"
        )
    if not getattr(args, "precision_guard_dir", None):
        return False
    selected = getattr(args, "precision_guard_sites", None)
    if selected:
        return site in selected
    return site not in OPT_IN_SITES  # unnamed runs collect the default set only


def validate_sites(args) -> None:
    """Validate --precision-guard-sites at startup so an unusable site fails before training.

    Without it, an unwired site only surfaces as zero artifacts after a full run.
    """
    selected = getattr(args, "precision_guard_sites", None)
    if not selected:
        return
    available = wired_sites()
    for site in selected:
        if site not in ALL_SITES:
            raise ValueError(f"--precision-guard-sites has unknown site {site!r}; valid: {ALL_SITES}")
        if site not in available:
            raise NotImplementedError(
                f"--precision-guard-sites has {site!r}, which is not wired; "
                f"wired sites: {available}"
            )


def export_dtype(args):
    """Precision the per-parameter sites cast to before fingerprinting; None keeps the original.

    None by default: torchtitan parameters are fp32, and fingerprinting a bf16 view would
    quantize away perturbations smaller than bf16's ~4e-3 relative resolution, which reads
    as a dead metric rather than insufficient sampling precision. Cross-backend comparison
    needs the opposite: pass bfloat16 explicitly so both sides match megatron's bf16
    model weights.
    """
    name = getattr(args, "precision_guard_dtype", None)
    if name is None:
        return None

    return getattr(torch, name)


def cast_stream(named_tensors, dtype):
    """Cast floating tensors in a ``(name, tensor)`` stream to ``dtype``; None passes through.

    For backends whose export path takes no target dtype (megatron's HF weight iterator).
    Non-floating buffers are left alone: floating an integer buffer would destroy it.
    """
    if dtype is None:
        yield from named_tensors
        return
    for name, tensor in named_tensors:
        cast = tensor.is_floating_point() and tensor.dtype != dtype
        yield name, (tensor.to(dtype) if cast else tensor)


def record(args, *, site, role, rollout_id, step_id, tensors=None, scalars=None, merge=False) -> None:
    """Fingerprint named tensors and scalars and write them.

    Must be called on all ranks: an object-level all_gather merges the part each rank can
    see (design section 4.1), so a rank filter would strand the others in the collective.

    tensors:  {name: tensor}, each reduced to distribution statistics.
    scalars:  {name: scalar}, stored as {"value": ...}.
    merge:    union into an existing file at the same path instead of overwriting, for sites
              called several times per step. Requires a clean output directory per run,
              otherwise a previous run's records merge in.
    """
    if not enabled(args, site):
        return

    local: dict = {}
    for name, tensor in (tensors or {}).items():
        local[name] = tensor_stats(tensor)
    for name, value in (scalars or {}).items():
        local[name] = {"value": _as_json_scalar(value)}

    _finish(args, local, site=site, role=role, rollout_id=rollout_id, step_id=step_id, merge=merge)


def _finish(args, local, *, site, role, rollout_id, step_id, merge) -> None:
    """Merge across ranks and write. Shared by record and record_stream."""
    merged, _ = _gather_and_merge(local)
    if merged is None:
        return  # not the writing rank

    path = records_path(
        args.precision_guard_dir,
        backend=getattr(args, "train_backend", "unknown"),
        role=role,
        rollout_id=rollout_id,
        step_id=step_id,
        site=site,
    )
    if merge and path.exists():
        # Read back and union. The old _conflicts is metadata, not a quantity, and is recomputed.
        previous = {k: v for k, v in read_records(path).items() if k != "_conflicts"}
        merged = merge_rank_records([previous, merged])
    write_records(path, merged)


def record_stream(args, *, site, role, rollout_id, step_id, named_tensors, progress=None) -> None:
    """Stream fingerprints, never holding all full tensors at once.

    Per-parameter sites are hundreds of full tensors on a 4B model, so building a dict first
    would OOM. ``named_tensors`` is an iterable of ``(name, tensor)``, usually a generator
    from the backend export path.

    All ranks must iterate to completion: both export paths contain collectives
    (full_tensor / all_gather / bucket gathering), so breaking early or iterating on only
    some ranks hangs the rest. Paths like EP>1 yield tensors only on the source rank; the
    per-name union in design section 4.1 fills the rest in.
    """
    if not enabled(args, site):
        return

    local: dict = {}
    try:
        for name, tensor in named_tensors:
            local[name] = tensor_stats(tensor)
            if progress is not None:
                progress.update(elements=tensor.numel())
            del tensor  # release early; the next full tensor is about to be allocated

        _finish(
            args, local, site=site, role=role,
            rollout_id=rollout_id, step_id=step_id, merge=False,
        )
    finally:
        if progress is not None:
            progress.close()


def record_from_batch(args, batch, *, site, per_sample=None, scalars=None) -> None:
    """Collect per-sample quantities from a micro-batch, called inside ``loss_function``.

    ``loss_function`` cannot reach role / rollout_id / step_id (on megatron it is a
    ``forward_backward_func`` callback), so the coordinates travel in ``batch["_pg_coord"]``
    and sample identity comes from ``batch["sample_indices"]`` (global indices, not
    rank-local positions).

    ``per_sample`` values must be per-sample lists (list[Tensor[R_i]]), not one concatenated
    block, otherwise samples cannot be separated and named by global index.

    Called once per micro-batch, so it accumulates with merge semantics.
    """
    if not enabled(args, site):
        return

    coord = (batch or {}).get("_pg_coord")
    if coord is None:
        # Call paths without coordinates (the forward_only log_prob pass) are skipped.
        return

    indices = batch.get("sample_indices")
    if indices is None:
        logger.warning(
            "precision_guard %s: batch has no sample_indices, naming by rank-local position; "
            "these names do not line up across backends or parallel layouts", site,
        )
    tensors: dict = {}
    for name, values in (per_sample or {}).items():
        if not values:
            continue
        for pos, tensor in enumerate(values):
            sample_id = indices[pos] if indices is not None else pos
            tensors[f"{name}/sample_{int(sample_id):06d}"] = tensor

    record(args, site=site, tensors=tensors, scalars=scalars, merge=True, **coord)


def merge_rank_records(parts) -> dict:
    """Union each rank's records by name.

    A name colliding with a different value means a quantity that should be globally
    consistent differs per rank. The first value wins and the name is listed in
    ``_conflicts``, so the anomaly leaves a trace in the artifact instead of vanishing.
    """
    merged: dict = {}
    conflicts: list[str] = []
    for part in parts:
        for name, rec in (part or {}).items():
            if name in merged:
                if merged[name] != rec and name not in conflicts:
                    conflicts.append(name)
                continue
            merged[name] = rec
    if conflicts:
        logger.warning("precision_guard: these quantities differ across ranks, keeping the first: %s", conflicts)
        merged["_conflicts"] = conflicts
    return merged


def _gather_and_merge(local: dict):
    """Gather and merge records from all ranks.

    Returns ``(merged, any_rank_recorded)``; merged is None on non-writing ranks, while the
    flag is the same on every rank so they can assert together.
    """

    if not dist.is_available() or not dist.is_initialized():
        return local, bool(local)

    from vime.utils.distributed_utils import get_gloo_group  # noqa: PLC0415

    # A few KB of plain Python dicts, not tensors: a small object-level collective.
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local, group=get_gloo_group())
    any_recorded = any(bool(part) for part in gathered)
    if dist.get_rank() != 0:
        return None, any_recorded
    return merge_rank_records(gathered), any_recorded


def _as_json_scalar(value):
    """Normalize scalars: torch scalar tensors, numpy scalars and Python numbers to JSON types.

    The two backends can report the same quantity with different types (one a Tensor, one
    already ``.item()``-ed), which would make artifacts incomparable.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value  # a few quantities are identifiers, not numbers (e.g. sample composition)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return value
    return float(value)
