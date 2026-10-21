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

"""Precision-guard collection for the torchtitan backend.

Production code under ``vime/backends/torchtitan_utils/`` holds only single-line calls into
this module, so removing the diagnostics never means picking statements out of the training
loop.

Collection timing, which must match the megatron side for artifacts to be comparable:

* grad: before ``optimizer.step()``; the step consumes gradients and ``zero_grad`` clears
  them. Values are post-``clip_grad_norm_`` (clipping rewrites p.grad in place) while
  megatron collects pre-clip. Clipping is a uniform scale, so multiplying this side back by
  ``grad_norm`` aligns them; the ``grad_clipped`` scalar records which side an artifact is on.
* weight: after ``optimizer.step()``, i.e. this step's update result.
* loss / grad_norm / optim_hparam: after ``lr_scheduler.step()``, matching megatron's
  position after ``opt_param_scheduler.step()``.

Every collection point runs on all ranks (they contain collectives) and must never sit
inside a rank filter.
"""

from __future__ import annotations

import logging
import re
import torch
import torch.distributed as dist

from . import hidden, registry
from . import enabled, export_dtype, record, record_from_batch, record_stream
from .progress import FingerprintProgress

logger = logging.getLogger(__name__)
_LOGPROB_CP_WARNING_EMITTED = False


def _device():
    """Current accelerator device index."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.current_device()
    return torch.cuda.current_device()


def _tp_group():
    """The TP process group, or None when tp == 1.

    Reuses the group the engine already built, so there is one source of truth for how the
    sub-mesh is obtained.
    """
    a = registry.actor()
    return getattr(a.engine, "_tp_group", None) if a is not None else None


def stamp_coord(args, batch, *, role, rollout_id, step_id) -> None:
    """Attach collection coordinates to the batch and arm the hidden hooks.

    ``loss_function`` is where per-token log probs exist but its signature carries no
    role / rollout_id / step_id, so the coordinates ride along in the batch.

    This is also the only point every micro-batch passes before its forward, so the
    per-sample boundaries hidden needs (``cu_seqlens`` / ``sample_indices``, which hooks
    cannot reach) are handed over here.
    """
    coord = {"role": role, "rollout_id": rollout_id, "step_id": step_id}
    if enabled(args, "logprob"):
        batch["_pg_coord"] = coord
    hidden.arm(
        args, coord=coord,
        cu_seqlens=batch.get("cu_seqlens"),
        sample_indices=batch.get("sample_indices"),
    )


_EXPERT_NAME = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")


def _expert_id(name):
    match = _EXPERT_NAME.search(name)
    return None if match is None else int(match.group(1))


def _is_expert_name(name):
    return ".experts." in name


def _is_selected_expert(name, expert_limit):
    """Return whether an expert parameter is within the configured expert limit."""
    expert_id = _expert_id(name)
    return (
        expert_limit is None
        or expert_limit < 0
        or expert_id is None
        or expert_id < expert_limit
    )


def _is_selected_param(name, expert_limit):
    """Keep non-expert parameters and apply the limit only to expert parameters."""
    return not _is_expert_name(name) or _is_selected_expert(name, expert_limit)


def _selected_expert_positions(ep_names, expert_limit):
    """Select local positions containing at least one requested global expert."""
    return [
        position
        for position, names in enumerate(zip(*ep_names, strict=True))
        if any(_is_selected_expert(name, expert_limit) for name in names)
    ]

# ---------------------------------------------------------------------------
# hidden: per-layer forward states
# ---------------------------------------------------------------------------


def _needs_seq_gather(h, tp) -> bool:
    """Whether this output is sharded along seq and must be gathered across TP.

    Decided by actual shape, not by the parallel config: layer outputs are seq-sharded under
    sequence parallelism, while the final norm has already been gathered back. The check
    depends only on the token count and this micro-batch's expected total, so it is identical
    on every TP rank and cannot make the collective asymmetric. A shape that is neither full
    nor 1/tp raises.
    """

    want = hidden.expected_tokens()
    if want is None or h.shape[0] == want:
        return False
    size = dist.get_world_size(tp)
    assert h.shape[0] * size == want, (
        f"hidden: {h.shape[0]} tokens is neither the expected {want} nor 1/{size} of it; "
        "the sharding is not an even split along seq"
    )
    return True


def _gather_seq(h, tp):
    """Gather along seq across TP; only TP rank 0 returns a tensor.

    After the gather every rank in the group holds the same full tensor. Fingerprinting is a
    float64 CPU reduction, so having four ranks on one host repeat it would contend for host
    CPU. The other ranks still take part in the collective, then return None.
    """

    parts = [torch.empty_like(h) for _ in range(dist.get_world_size(tp))]
    dist.all_gather(parts, h.contiguous(), group=tp)
    if dist.get_rank(tp) != 0:
        return None
    return torch.cat(parts, dim=0)


def _to_token_major(out):
    """Normalize a layer output to ``[tokens, hidden]``.

    A layer output is ``[1, T, H]`` before upstream #4121 and ``[T, H]`` after it; vime packs one
    row, so dropping a leading singleton covers both. Under sequence parallelism T is split across
    TP ranks: a DTensor is gathered by ``full_tensor()``, while the local-tensor form (torchtitan
    converts at the embedding with ``use_local_output=True``) is gathered here. Either way the
    gather runs on the TP group, symmetric across its ranks.
    """
    h = out[0] if isinstance(out, tuple) else out
    if hasattr(h, "full_tensor"):
        h = h.full_tensor()
    if h.dim() == 3 and h.shape[0] == 1:
        h = h[0]
    assert h.dim() == 2, f"hidden: expected a [T, H] layer output, got {tuple(h.shape)}"
    h = h.detach()

    tp = _tp_group()
    if tp is None or not _needs_seq_gather(h, tp):
        return h
    return _gather_seq(h, tp)


def _logits_full(out):
    """lm_head output with the vocab dimension gathered back to full width.

    Shard width follows the parallel config (37984 at TP=4, the full vocab under FSDP4), so
    only the gathered form is comparable across deployments.

    Costs roughly 2.4 GB for a micro-batch's logits plus gather buffers. Only TP rank 0
    fingerprints.
    """
    h = out[0] if isinstance(out, tuple) else out
    # DTensor.full_tensor() already gathers the sharded dimension; a plain tensor is this
    # rank's vocab shard and must be gathered here. The flag distinguishes the two paths.
    gathered_by_dtensor = hasattr(h, "full_tensor")
    if gathered_by_dtensor:
        h = h.full_tensor()
    if h.dim() == 3 and h.shape[0] == 1:
        h = h[0]
    assert h.dim() == 2, f"logits: expected [T, V], got {tuple(h.shape)}"
    h = h.detach()

    tp = _tp_group()
    if tp is None:
        return h                            # tp == 1 deployments already hold the full vocab

    want = hidden.expected_tokens()
    assert want is None or h.shape[0] == want, (
        f"logits: {h.shape[0]} tokens against an expected {want}; seq should not be sharded here"
    )
    if gathered_by_dtensor:
        return h if dist.get_rank(tp) == 0 else None

    parts = [torch.empty_like(h) for _ in range(dist.get_world_size(tp))]
    dist.all_gather(parts, h.contiguous(), group=tp)
    if dist.get_rank(tp) != 0:
        return None
    return torch.cat(parts, dim=-1)         # concatenate along vocab


def install_hooks(actor) -> None:
    """Hook every transformer layer plus the final norm and lm_head.

    The two tail modules matter: forward nondeterminism, if any, can sit outside the layer
    stack, and hooking only the layers would miss it.
    """
    if not enabled(actor.args, "hidden"):
        return
    # torchtitan's Qwen3 places self.norm (RMSNorm) then self.lm_head after the stack.
    # tail_norm carries no layer number, so it holds for any model depth.
    tail = [(r"(^|\.)norm$", "tail_norm", _to_token_major)]
    if "logits" in getattr(actor.args, "precision_guard_hidden_extra", ()):
        tail.append((r"(^|\.)lm_head$", "logits", _logits_full))

    total = 0
    for part in actor.engine.model_parts:
        total += hidden.install_on_layers(part, _to_token_major, tail=tail)
    assert total, "hidden: no transformer layer found, hooks were not installed"
    logger.info("precision_guard hidden: installed %d hooks", total)


# ---------------------------------------------------------------------------
# logprob / loss
# ---------------------------------------------------------------------------


def record_logprob(args, batch, *, log_probs, old_log_probs, entropy) -> None:
    """Record the loss function's per-sample inputs. Called once per micro-batch.

    Per-token log probs exist only inside ``loss_function``; the rollout data does not carry
    them. Values stay per-sample lists rather than one concatenated block so samples remain
    separable.

    ``advantages`` and ``loss_masks`` are recorded alongside because they are also loss
    inputs: without them a comparison can conclude "inputs identical, outputs differ" while
    an unobserved input was in fact the one that changed.
    """
    global _LOGPROB_CP_WARNING_EMITTED
    actor = registry.actor()
    parallel_dims = getattr(getattr(actor, "engine", None), "parallel_dims", None)
    if parallel_dims is not None and int(getattr(parallel_dims, "cp", 1)) > 1:
        if not _LOGPROB_CP_WARNING_EMITTED:
            logger.warning(
                "precision_guard logprob collection is disabled for CP>1: "
                "per-sample values are CP-local shards but sample_indices are global"
            )
            _LOGPROB_CP_WARNING_EMITTED = True
        return

    record_from_batch(
        args, batch, site="logprob",
        per_sample={
            "train_log_probs": log_probs,
            "old_log_probs": old_log_probs,
            "entropy": entropy,
            "rollout_log_probs": batch.get("rollout_log_probs"),
            "ref_log_probs": batch.get("ref_log_probs"),
            "advantages": batch.get("advantages"),
            "loss_masks": batch.get("loss_masks"),
        },
    )


def record_loss(args, keys, sums, divisor, *, role, rollout_id, step_id, grad_norm) -> None:
    """Record this step's loss metrics, matching megatron's per-step position.

    Reduction mirrors ``actor._log_train_metrics``: sum across micro-batches, all-reduce
    across DP x CP, divide by the already-global divisor.

    Runs on all ranks (it contains a collective) and merges, since the per-micro-batch
    records land in the same artifact.
    """
    if keys is None or not enabled(args, "loss"):
        return

    a = registry.require_actor("loss")
    loss_mesh = a.engine.parallel_dims.get_optional_mesh("loss")
    grp = loss_mesh.get_group() if loss_mesh is not None else None
    vals = sums.clone()
    if grp is not None:
        dist.all_reduce(vals, group=grp)
    div = divisor if torch.is_tensor(divisor) else float(divisor)
    reduced = {k: (vals[i] / div).item() for i, k in enumerate(keys)}
    record(
        args, site="loss", role=role, rollout_id=rollout_id, step_id=step_id,
        scalars={**reduced, "grad_norm": grad_norm}, merge=True,
    )


# ---------------------------------------------------------------------------
# grad / weight / optim
# ---------------------------------------------------------------------------

def get_hf_per_tensor_selected_experts(
    model_parts,
    sd_adapter,
    device,
    *,
    param_dtype,
    parallel_dims=None,
    raw_params=None,
    yield_experts_on_all_ranks=False,
    expert_limit=None,
    progress=None,
):
    """Filter the shared HF export stream for precision-guard fingerprints.

    ``get_hf_per_tensor`` owns DTensor/FSDP/EP materialization and its collectives.  This
    wrapper only adds the precision-guard policy of limiting numbered experts and reports the
    resulting progress total before the stream starts.
    """
    from vime.backends.torchtitan_utils.update_weight import common

    params = common._hf_params_dict(model_parts, sd_adapter, raw_params)
    ep_group, ep_size = common._get_ep_group(parallel_dims)

    if progress is not None:
        non_expert_params = [(name, param) for name, param in params.items() if ".experts." not in name]
        if ep_group is None:
            selected = [
                (name, param)
                for name, param in params.items()
                if _is_selected_param(name, expert_limit)
            ]
            progress.set_total(
                tensors=len(selected),
                elements=sum(param.numel() for _, param in selected),
            )
        else:
            expert_params = [(name, param) for name, param in params.items() if ".experts." in name]
            ep_names = [None] * ep_size
            dist.all_gather_object(ep_names, [name for name, _ in expert_params], group=ep_group)
            assert all(len(names) == len(expert_params) for names in ep_names), (
                f"EP ranks hold different expert counts {[len(names) for names in ep_names]}; "
                "names and tensors would be paired wrongly"
            )
            selected_positions = _selected_expert_positions(ep_names, expert_limit)
            is_src = not dist.is_initialized() or dist.get_rank() == 0
            selected_expert_names = [
                name
                for position in selected_positions
                for names in ep_names
                for name in [names[position]]
                if _is_selected_expert(name, expert_limit)
            ]
            include_experts = is_src or yield_experts_on_all_ranks
            expert_tensors = len(selected_expert_names) if include_experts else 0
            expert_elements = (
                sum(
                    expert_params[position][1].numel()
                    for position in selected_positions
                    for names in ep_names
                    if _is_selected_expert(names[position], expert_limit)
                )
                if include_experts
                else 0
            )
            progress.set_total(
                tensors=len(non_expert_params) + expert_tensors,
                elements=sum(param.numel() for _, param in non_expert_params)
                + expert_elements,
            )

    tensors = common.get_hf_per_tensor(
        model_parts,
        sd_adapter,
        device,
        param_dtype=param_dtype,
        parallel_dims=parallel_dims,
        raw_params=raw_params,
        yield_experts_on_all_ranks=yield_experts_on_all_ranks,
    )()
    for name, tensor in tensors:
        if _is_selected_param(name, expert_limit):
            yield name, tensor


def _fingerprint_progress(site):
    main_rank = not dist.is_initialized() or dist.get_rank() == 0
    return FingerprintProgress(f"precision_guard {site}", enabled=main_rank)


def _run_state_dict_hooks(model, grads: dict) -> None:
    """Rewrite fused-parameter gradients into the keys the model's own state_dict reports.

    ``FusedQKVLinear`` publishes its fused ``wqkv`` as ``wq`` / ``wk`` / ``wv`` from a state-dict
    post hook, so a gradient keyed by the parameter name has no counterpart in ``state_dict`` and
    ``sd_adapter.to_hf`` would drop it. The split is taken from the hooks the module registered
    (torch stores them in ``_state_dict_hooks``) rather than re-derived here, so gradients and
    weights are reshaped by exactly the same code.
    """
    from torchtitan.models.common.attention import FusedQKVLinear

    for name, module in model.named_modules():
        if not isinstance(module, FusedQKVLinear):
            continue
        prefix = f"{name}.".replace("._checkpoint_wrapped_module.", ".") if name else ""
        fused = {k: grads.pop(k) for k in list(grads) if k.startswith(f"{prefix}wqkv.")}
        if not fused:
            continue
        for hook in module._state_dict_hooks.values():
            hook(module, fused, prefix, None)
        grads.update(fused)


def _hf_named_grads(args, model_parts, parallel_dims, progress=None):
    """Yield ``(HF parameter name, full gradient tensor)``.

    Feeds ``{state_dict key: p.grad}`` through the weight-export pipeline, so gradients and
    weights share one set of names. Keys must line up with the model's ``state_dict``, since a
    mismatch would make ``sd_adapter.to_hf`` drop tensors silently.
    """

    # Local import: the backend's engine.py imports this module at load time, so reaching
    # back into that package at module level would close an import cycle.
    from vime.backends.torchtitan_utils.update_weight.common import _state_dict_chunks

    grads = {}
    expected = set()
    for m in model_parts:
        # Chunked: only the key set is needed, and a whole-model state_dict() would gather every
        # fused QKV to Replicate in fp32 at once.
        for chunk in _state_dict_chunks(m):
            expected |= set(chunk)
        for name, p in m.named_parameters():
            if p.grad is None:
                continue
            # Activation checkpointing inserts "_checkpoint_wrapped_module." into
            # named_parameters() names, which state_dict keys do not carry.
            grads[name.replace("._checkpoint_wrapped_module.", ".")] = p.grad
        _run_state_dict_hooks(m, grads)

    missing = set(grads) - expected
    assert not missing, (
        f"gradient keys do not match the model state_dict, to_hf would drop tensors: "
        f"{sorted(missing)[:5]}"
    )

    a = registry.require_actor("grad")
    return get_hf_per_tensor_selected_experts(
        model_parts, a.engine.sd_adapter, _device(),
        param_dtype=export_dtype(args), parallel_dims=parallel_dims, raw_params=grads,
        expert_limit=getattr(args, "precision_guard_expert_limit", -1),
        progress=progress,
    )


def _optim_hparam_scalars(optimizer, *, update_applied: bool) -> dict:
    """Optimizer hyper-parameters and state. Field names match megatron's so artifacts align."""
    scalars: dict = {
        "update_applied": update_applied,
        # torchtitan does not produce this, but the key must exist for the two backends'
        # artifacts to have the same key set.
        "num_zeros_in_grad": None,
    }
    groups = [g for opt in getattr(optimizer, "optimizers", [optimizer]) for g in opt.param_groups]
    for i, group in enumerate(groups):
        scalars[f"lr_pg{i}"] = float(group["lr"])
        scalars[f"weight_decay_pg{i}"] = float(group.get("weight_decay", 0.0))
        scalars[f"eps_pg{i}"] = float(group.get("eps", 0.0))
        betas = group.get("betas", (None, None))
        scalars[f"beta1_pg{i}"] = None if betas[0] is None else float(betas[0])
        scalars[f"beta2_pg{i}"] = None if betas[1] is None else float(betas[1])
    scalars["num_param_groups"] = float(len(groups))
    return scalars


def record_grad(args, model_parts, parallel_dims, *, role, rollout_id, step_id) -> None:
    """Called once before ``optimizer.step()``."""
    # Check enabled before building the iterator: _hf_named_grads is an ordinary call and
    # would run even when grad is not requested. ``enabled`` reads only args, so every rank
    # agrees and the collectives stay symmetric.
    if not enabled(args, "grad"):
        return
    progress = _fingerprint_progress("grad")
    record_stream(
        args, site="grad", role=role, rollout_id=rollout_id, step_id=step_id,
        named_tensors=_hf_named_grads(args, model_parts, parallel_dims, progress),
        progress=progress,
    )
    record(
        args, site="grad", role=role, rollout_id=rollout_id, step_id=step_id,
        scalars={"grad_clipped": True}, merge=True,
    )


def record_weight(args, model_parts, parallel_dims, *, role, rollout_id, step_id) -> None:
    """Called once after ``optimizer.step()``.

    Costs one full all-gather of every tensor per step, only when weight is requested.
    """
    if not enabled(args, "weight"):
        return

    a = registry.require_actor("weight")
    progress = _fingerprint_progress("weight")
    record_stream(
        args, site="weight", role=role, rollout_id=rollout_id, step_id=step_id,
        named_tensors=get_hf_per_tensor_selected_experts(
            model_parts, a.engine.sd_adapter, _device(),
            param_dtype=export_dtype(args), parallel_dims=parallel_dims,
            expert_limit=getattr(args, "precision_guard_expert_limit", -1),
            progress=progress,
        ),
        progress=progress,
    )


def record_optim(args, optimizer, *, role, rollout_id, step_id, grad_norm, update_applied) -> None:
    """Called once after ``lr_scheduler.step()``; records grad_norm and optimizer hyper-parameters.

    Two collection points share one call to keep the production side to a single line; they
    still write separate artifacts.
    """
    record(
        args, site="grad_norm", role=role, rollout_id=rollout_id, step_id=step_id,
        scalars={"grad_norm": grad_norm},
    )
    record(
        args, site="optim_hparam", role=role, rollout_id=rollout_id, step_id=step_id,
        scalars=_optim_hparam_scalars(optimizer, update_applied=update_applied),
    )
    # hidden accumulates across micro-batches and is written here, at the end of the step.
    hidden.flush(args)
