"""Per-layer forward hidden states, backend-independent half.

Requires no changes to the training loop: hooks are installed from outside by
``attach(actor)``, and each micro-batch's coordinates ride along with ``stamp_coord``.

Records every transformer layer's output, sliced per sample by ``cu_seqlens`` and named
``layer_{NN}/sample_{global index}``. Slicing per sample makes the comparison independent
of micro-batch composition and padding, and keeps token order identical across backends,
which is what ``block_sums`` (an order-sensitive metric) needs.

Optionally, off by default:

* ``--precision-guard-hidden-extra act_grad`` also records the activation gradient flowing
  back into each layer (``agrad_{NN}``). Activation gradients are bf16 like the forward
  activations; parameter gradients are fp32 and are each layer's endpoint, so only the
  former shows where backward error enters.

Cost: one fingerprint per layer per sample, roughly 30 s/step. ``hidden`` therefore stays
out of the default site set (see ``OPT_IN_SITES``) and must be named explicitly.

Scope: pipeline_model_parallel_size == 1. Layer numbers are chunk-local, so PP stages would
collide, and the forward hook reads the module-level context, which 1F1B would mispair.
"""

from __future__ import annotations

import logging
import re

from . import enabled, records_path, tensor_stats
from .writer import read_records, write_records

logger = logging.getLogger(__name__)

# Current micro-batch's collection context; hooks cannot reach the batch otherwise.
# None means not armed: the hooks are outside a training forward.
_ctx: dict | None = None

# Fingerprints accumulate across micro-batches and are written once per optimizer step.
# Writing per layer would cost one gloo collective plus a read-modify-write each time.
_pending: dict = {}
_coord: dict | None = None


def arm(args, *, coord, cu_seqlens, sample_indices) -> None:
    """Arm the hooks for one micro-batch. Called from ``stamp_coord`` before the forward.

    cu_seqlens: sample boundaries in the packed micro-batch (length = samples + 1).
    sample_indices: per-sample global indices, which become the quantity names.
    """
    global _ctx, _coord
    if not enabled(args, "hidden") or coord is None:
        _ctx = None
        return
    if cu_seqlens is None or sample_indices is None:
        # Padded (non-packed) batches slice differently; skip rather than guess.
        logger.warning("precision_guard hidden: micro-batch has no cu_seqlens/sample_indices, skipped")
        _ctx = None
        return
    _ctx = {
        "args": args,
        "bounds": [int(x) for x in cu_seqlens],
        "samples": [int(x) for x in sample_indices],
        "fired": {},   # {group: times triggered this micro-batch}; reaches 2 under recompute
        "agrad_hooked": set(),   # layer groups whose backward hook is already registered
    }
    _coord = dict(coord)


def disarm() -> None:
    global _ctx
    _ctx = None


def expected_tokens():
    """Total packed token count of this micro-batch (padding included); None when disarmed.

    Normalizers use it to decide whether a module output is sharded along seq, rather than
    inferring from the parallel config: layer outputs are seq-sharded under sequence
    parallelism, but the final norm has already been gathered back. Deciding by shape holds
    for any deployment, and the decision is identical on every TP rank, so it cannot make
    the collective asymmetric.
    """
    return _ctx["bounds"][-1] if _ctx is not None else None


# Under activation checkpointing a layer appears twice in the module tree:
#   layers.7                             the wrapper
#   layers.7._checkpoint_wrapped_module  the layer itself
# Recompute re-runs the inner module only, so hooks must go on the inner one to see both
# the first forward and the recompute.
_LAYER_NAME = re.compile(r"(?:^|\.)layers\.(\d+)(?:\._checkpoint_wrapped_module)?$")


def _want_act_grad(args) -> bool:
    return "act_grad" in getattr(args, "precision_guard_hidden_extra", ())


def install_on_layers(model, to_token_major, tail=()) -> int:
    """Install hooks on every transformer layer, plus the modules named in ``tail``.

    Both backends hold layers at ``...layers.<N>``, so matching module names covers both;
    wrappers (DDP / Float16Module / FSDP / checkpointing) only change the prefix.

    ``tail`` is a sequence of ``(name regex, group, normalizer)`` for modules after the layer
    stack (final norm, lm_head). Those matter because forward nondeterminism, if any, can sit
    outside the layer stack, and hooking only layers would miss it.
    """
    # Collect candidates first: under checkpointing both the wrapper and the inner module
    # match, and hooking both would fire twice on the first forward. Prefer the inner one,
    # which fires once on the forward and once on the recompute.
    layers: dict[int, tuple[bool, object]] = {}
    count = 0
    for name, module in model.named_modules():
        m = _LAYER_NAME.search(name)
        if m is not None:
            idx = int(m.group(1))
            inner = name.endswith("._checkpoint_wrapped_module")
            if idx not in layers or (inner and not layers[idx][0]):
                layers[idx] = (inner, module)
            continue
        for pattern, group, fn in tail:
            if re.search(pattern, name):
                module.register_forward_hook(make_hook(group, fn))
                count += 1
                break

    for _idx, (_inner, module) in sorted(layers.items()):
        module.register_forward_hook(make_hook(f"layer_{_idx:02d}", to_token_major))
        count += 1
    return count


def _make_act_grad_hook(group: str, to_token_major, ctx):
    """Tensor hook on a module output; receives the gradient w.r.t. that output.

    A tensor hook rather than ``register_full_backward_hook``: the latter is fragile for
    modules returning tuples (megatron layers return ``(output, context)``), while a tensor
    hook treats both backends alike.

    ``ctx`` is captured explicitly so the record always lands on the micro-batch that
    produced the forward, regardless of where micro-batch boundaries move.
    """

    def hook(grad):
        if grad is None or group in ctx["fired"]:
            return
        ctx["fired"][group] = 1
        h = to_token_major(grad)
        if h is not None:
            _record_layer(group, h, ctx)

    return hook


def make_hook(group: str, to_token_major):
    """Forward hook recording this module's output under ``group``.

    ``to_token_major(output)`` is backend-supplied: it normalizes the output to a 2-D
    ``[tokens, hidden]`` tensor, gathering across TP along seq when needed, and returns None
    when this rank should not fingerprint.

    Under full activation checkpointing each layer's forward runs twice. Both are recorded;
    the recompute pass gets a ``#recompute`` suffix, which is what makes "the values fed to
    backward" comparable against "the values first computed".
    """

    def hook(_module, _inputs, output):
        ctx = _ctx
        if ctx is None:
            return  # not armed: outside a training forward, e.g. the log_prob forward_only pass
        seen = ctx["fired"].get(group, 0)
        if seen >= 2:
            return  # forward plus one recompute is the maximum a micro-batch can produce
        ctx["fired"][group] = seen + 1

        # Register the backward tensor hook before normalizing: to_token_major returns None
        # on non-zero TP ranks, and registering after it would leave only rank 0 hooked,
        # making the collective inside the backward asymmetric.
        # It goes on whichever pass carries grad: the first forward under non-reentrant
        # checkpointing, the recompute under reentrant, where the first runs in no_grad.
        if _want_act_grad(ctx["args"]) and group.startswith("layer_") and group not in ctx["agrad_hooked"]:
            t = output[0] if isinstance(output, tuple) else output
            if getattr(t, "requires_grad", False):
                ctx["agrad_hooked"].add(group)
                t.register_hook(_make_act_grad_hook(
                    group.replace("layer_", "agrad_", 1), to_token_major, ctx))

        # Exceptions propagate: a collection point that cannot collect must fail the run
        # rather than leave an empty artifact that passes existence checks.
        h = to_token_major(output)
        if h is None:
            return
        _record_layer(group, h, ctx, suffix="#recompute" if seen else "")

    return hook


def _record_layer(group: str, h, ctx, suffix: str = "") -> None:
    """Slice one module output per sample and fingerprint each slice."""
    bounds, samples = ctx["bounds"], ctx["samples"]
    n = len(samples)
    assert h.dim() == 2, f"hidden: expected [tokens, hidden] after normalization, got {tuple(h.shape)}"
    # Packing may append one padding segment as a dummy sequence (aligned to
    # pad_to_multiple, excluded from loss), which also appears in cu_seqlens. The loop below
    # only walks real samples, so that segment is skipped, but "one extra segment" must stay
    # distinguishable from "the counts genuinely disagree".
    assert len(bounds) in (n + 1, n + 2), (
        f"cu_seqlens has {len(bounds)} boundaries for {n} samples; "
        "at most one trailing padding segment is allowed"
    )
    assert bounds[-1] == h.shape[0], (
        f"{group}: cu_seqlens ends at {bounds[-1]} but the tensor has {h.shape[0]} tokens. "
        "Under sequence parallelism this usually means the cross-TP gather was incomplete, "
        "or the two sides shard differently"
    )

    for pos, sample_id in enumerate(samples):
        seg = h[bounds[pos] : bounds[pos + 1]]
        if seg.numel() == 0:
            continue
        _pending[f"{group}/sample_{sample_id:06d}{suffix}"] = tensor_stats(seg)
        del seg


def flush(args) -> None:
    """Write the accumulated fingerprints. Called once per optimizer step.

    Must run on every rank: merging is cross-rank, since TP/PP ranks each see only part of
    the layers and tokens.
    """
    global _pending, _coord
    if not enabled(args, "hidden"):
        return

    from . import _gather_and_merge  # noqa: PLC0415 - avoids a circular import at module load

    local, coord = _pending, _coord
    _pending, _coord = {}, None
    disarm()

    merged, any_rank_recorded = _gather_and_merge(local)
    # Asserted on every rank: a raise on the writer alone would leave the others waiting in
    # the next collective, which presents as a hang. Individual ranks may legitimately hold
    # nothing (see the TP dedup in to_token_major), so the condition is the gathered one.
    assert any_rank_recorded, (
        "precision_guard hidden: no records from any rank this step. Check that attach() "
        "installed the hooks and that stamp_coord received cu_seqlens/sample_indices"
    )
    if merged is None:
        return  # not the writing rank
    if coord is None:
        return
    path = records_path(
        args.precision_guard_dir,
        backend=getattr(args, "train_backend", "unknown"),
        site="hidden",
        **coord,
    )
    if path.exists():
        previous = {k: v for k, v in read_records(path).items() if k != "_conflicts"}
        previous.update(merged)
        merged = previous
    write_records(path, merged)
