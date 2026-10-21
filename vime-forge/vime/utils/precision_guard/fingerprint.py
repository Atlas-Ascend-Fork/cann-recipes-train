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

"""Tensor distribution statistics: the only fingerprint form the harness stores.

Reduces each tensor to a set of scalars; raw tensors are never written. Field definitions
are defined by FIELDS below.

Pure functions, with no torch.distributed and no filesystem, so the whole module is testable
without accelerators.
"""

from __future__ import annotations

import math

import torch

# Chunk size (elements) for the reduction. Casting a whole tensor to float64 multiplies
# its footprint: the 4B model's embedding is ~3.9e8 elements, 778 MB in bf16 and 3.1 GB in
# fp64, while chunked reduction stays at a constant 16M * 8B = 128 MB.
_CHUNK_NUMEL = 1 << 24

# Above this size quantiles switch to deterministic sampling (see _sampled_quantiles):
# torch.quantile rejects very large inputs, and the sort itself is expensive.
_QUANTILE_MAX_NUMEL = 1 << 20

_QUANTILE_LEVELS = (0.5, 0.95, 0.99)

# Blocks for the piecewise linear functional, the only part of the fingerprint that sees
# position. Every scalar field below is permutation-invariant, so a value that lands in the
# wrong place (shard reassembly, name/layout mapping, gather ordering) shows up here or
# nowhere. 8 blocks locate such a move to 1/8 of a tensor and cost ~170 bytes per tensor.
# For detecting a change in the values themselves the scalars are the more sensitive fields.
_NUM_BLOCKS = 8

# Numeric fields. All-non-finite or empty tensors fill None rather than 0: zero is a valid
# value and would read as "genuinely zero".
_VALUE_FIELDS = ("mean", "absmean", "std", "l2norm", "min", "max", "p50", "p95", "p99")
_EXTRA_FIELDS = ("block_sums",)

FIELDS = ("numel", "shape", "dtype", "nan_count", "inf_count", *_VALUE_FIELDS, *_EXTRA_FIELDS)


def tensor_stats(t: torch.Tensor) -> dict:
    """Reduce a tensor to scalars. Non-finite values are counted separately and excluded elsewhere."""
    flat = t.detach().reshape(-1)
    out: dict = {
        "numel": int(flat.numel()),
        "shape": list(t.shape),
        "dtype": str(t.dtype).removeprefix("torch."),
    }

    if flat.numel() == 0:
        out["nan_count"] = 0
        out["inf_count"] = 0
        out.update({k: None for k in _VALUE_FIELDS})
        out["block_sums"] = []
        return out

    acc = _scan_in_chunks(flat)
    out["nan_count"] = acc["nan_count"]
    out["inf_count"] = acc["inf_count"]

    n = acc["n_finite"]
    if n == 0:
        # All non-finite: distribution stats are meaningless so they stay None, but block
        # sums are still computed (non-finite counted as 0) to keep the field set identical
        # on every path, so offline comparison never has to test for key presence.
        out.update({k: None for k in _VALUE_FIELDS})
        out["block_sums"] = acc["block_sums"]
        return out

    mean = acc["sum"] / n
    # Variance from sum(x^2). In float64 this is safe for weights and gradients, where
    # |mean| is far below std. Quantities with |mean| >> std would need a two-pass algorithm.
    var = max(acc["sumsq"] / n - mean * mean, 0.0)

    out["mean"] = mean
    out["absmean"] = acc["sumabs"] / n
    out["std"] = math.sqrt(var)
    out["l2norm"] = math.sqrt(acc["sumsq"])
    out["min"] = acc["min"]
    out["max"] = acc["max"]
    quantiles = acc["quantiles"]
    if quantiles is None:
        quantiles = _sampled_quantiles(flat)
    out["p50"], out["p95"], out["p99"] = quantiles
    out["block_sums"] = acc["block_sums"]
    return out


def _scan_in_chunks(flat: torch.Tensor) -> dict:
    """Compute all fingerprint accumulators during one CPU copy per input chunk.

    Scalar reductions and block sums consume the same CPU chunk. Small-tensor quantiles
    reuse those chunks; large tensors use source-device sampling to avoid expensive CPU
    indexing. This removes the duplicate full-tensor traversal while preserving fields.
    """
    n = flat.numel()
    num_blocks = min(_NUM_BLOCKS, n)
    bounds = [round(i * n / num_blocks) for i in range(num_blocks + 1)]
    quantile_values = [] if n <= _QUANTILE_MAX_NUMEL else None
    block_sums = [0.0] * num_blocks
    n_finite = 0
    total = total_sq = total_abs = 0.0
    lo, hi = math.inf, -math.inf
    nan_count = inf_count = 0

    for start in range(0, n, _CHUNK_NUMEL):
        end = min(start + _CHUNK_NUMEL, n)
        # Move to CPU before widening. On NPU float64 is not supported on device, and
        # this is the only full chunk copy performed by the fingerprint scan.
        chunk = flat[start:end].to(device="cpu", dtype=torch.float64)

        is_nan = torch.isnan(chunk)
        is_inf = torch.isinf(chunk)
        n_nan = int(is_nan.sum().item())
        n_inf = int(is_inf.sum().item())
        nan_count += n_nan
        inf_count += n_inf

        has_nonfinite = bool(n_nan or n_inf)
        values = chunk[~(is_nan | is_inf)] if has_nonfinite else chunk
        if values.numel():
            n_finite += int(values.numel())
            total += float(values.sum().item())
            total_sq += float(values.square().sum().item())
            total_abs += float(values.abs().sum().item())
            lo = min(lo, float(values.min().item()))
            hi = max(hi, float(values.max().item()))

        # Use the original chunk with non-finite values mapped to zero, matching the old
        # block-sum behavior, and accumulate only the overlapping block slices.
        block_chunk = (
            torch.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            if has_nonfinite else chunk
        )
        for block in range(num_blocks):
            lo_overlap = max(start, bounds[block]) - start
            hi_overlap = min(end, bounds[block + 1]) - start
            if hi_overlap > lo_overlap:
                block_sums[block] += float(block_chunk[lo_overlap:hi_overlap].sum().item())

        # Small tensors reuse the full CPU chunks. Large tensors are sampled on the source
        # device after this scan, avoiding million-element CPU indexing for every expert.
        if quantile_values is not None:
            quantile_values.append(chunk)

    if not n_finite:
        quantiles = (None, None, None)
    else:
        if quantile_values is None:
            quantiles = None
        else:
            sample = torch.cat(quantile_values)
            quantiles = _cpu_quantiles(sample)

    return {
        "n_finite": n_finite,
        "sum": total,
        "sumsq": total_sq,
        "sumabs": total_abs,
        "min": lo if n_finite else None,
        "max": hi if n_finite else None,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "quantiles": quantiles,
        "block_sums": block_sums,
    }


def _sampled_quantiles(flat: torch.Tensor) -> tuple:
    """Sample large tensors on their source device, then reduce the sample on CPU."""
    idx = torch.linspace(
        0, flat.numel() - 1,
        steps=_QUANTILE_MAX_NUMEL,
        dtype=torch.float64,
        device=flat.device,
    ).long()
    sample = flat[idx].to(device="cpu", dtype=torch.float64)
    return _cpu_quantiles(sample)


def _cpu_quantiles(sample: torch.Tensor) -> tuple:
    sample = sample[torch.isfinite(sample)]
    if sample.numel() == 0:
        return None, None, None
    levels = torch.tensor(_QUANTILE_LEVELS, dtype=torch.float64)
    return tuple(float(v) for v in torch.quantile(sample, levels))
