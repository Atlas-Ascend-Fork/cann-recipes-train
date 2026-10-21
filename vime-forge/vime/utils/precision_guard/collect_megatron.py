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

"""Precision-guard collection for the megatron backend.

Production code under ``vime/backends/megatron_utils/`` holds only single-line calls into
this module, so removing the diagnostics never means picking statements out of the training
loop.

Collection timing, which must match the torchtitan side for artifacts to be comparable:

* grad: before ``optimizer.step()``; the step clips and consumes gradients and the
  following ``zero_grad_buffer()`` clears the buffer. Values are pre-clip while torchtitan
  collects post-clip. Clipping is a uniform scale, so multiplying the torchtitan side back
  by ``grad_norm`` aligns them; the ``grad_clipped`` scalar records which side an artifact
  is on.
* loss / grad_norm / optim_hparam: after ``opt_param_scheduler.step()``, matching
  torchtitan's position after ``lr_scheduler.step()``.
* weight: after ``optimizer.step()``, i.e. this step's update result. A rollout holds as many
  optimizer steps as the batch splits into, so this is per step, under its real step_id.

Every collection point runs on all ranks (they contain collectives) and must never sit
inside a rank filter.

Scope: pipeline_model_parallel_size == 1. Under PP the logprob site hangs, since it merges
over the world group while ``loss_func`` runs only on the last stage.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from megatron.core import mpu

from . import hidden, registry
from . import cast_stream, enabled, export_dtype, record, record_from_batch, record_stream

logger = logging.getLogger(__name__)


def _named_params_and_buffers(args, model):
    """This rank's ``(name, parameter)`` pairs.

    ``convert_to_global_name`` must match the weight_updater's mode: raw takes global names,
    bridge takes ``vp_stages.N.``-prefixed ones. The condition mirrors the one in
    ``actor.init()`` where weights_backuper is built, keeping a single source of truth.
    """
    # Local import: the backend's model.py imports this module at load time, so reaching
    # back into that package at module level would close an import cycle.
    from vime.backends.megatron_utils.update_weight.common import named_params_and_buffers

    return named_params_and_buffers(
        args, model, convert_to_global_name=args.megatron_to_hf_mode == "raw"
    )


_iterator = None


def _hf_weight_iterator():
    """The HF weight iterator, from the weight_updater when it has one, else built here.

    Only colocate and bridge mode build one, so raw without colocate leaves it None.
    """
    global _iterator
    updater = getattr(registry.require_actor("grad/weight"), "weight_updater", None)
    assert updater is not None, (
        "precision guard needs actor.weight_updater; attach(self) must run after it is built"
    )
    if getattr(updater, "_hf_weight_iterator", None) is not None:
        return updater._hf_weight_iterator
    if _iterator is None:
        from vime.backends.megatron_utils.update_weight import create_hf_weight_iterator

        _iterator = create_hf_weight_iterator(
            updater.args, updater.model,
            model_name=updater.model_name, quantization_config=updater.quantization_config,
        )
    return _iterator


# ---------------------------------------------------------------------------
# coordinates / loss
# ---------------------------------------------------------------------------


def stamp_coord(args, batch, model, rollout_id, step_id) -> None:
    """Attach collection coordinates to the batch and arm the hidden hooks.

    ``loss_function`` is a ``forward_backward_func`` callback and cannot reach the enclosing
    role / rollout_id / step_id, yet per-token log probs exist only there, so the
    coordinates ride along in the batch.

    This is also the only point every micro-batch passes before its forward, so the
    per-sample boundaries hidden needs are handed over here.
    """
    coord = {
        "role": getattr(model, "role", "actor"),
        "rollout_id": rollout_id,
        "step_id": step_id,
    }
    if enabled(args, "logprob"):
        batch["_pg_coord"] = coord
    hidden.arm(
        args, coord=coord,
        cu_seqlens=_cu_seqlens(batch),
        sample_indices=batch.get("sample_indices"),
    )


def _cu_seqlens(batch):
    """Sample boundaries for this micro-batch; megatron wraps them in PackedSeqParams."""
    psp = batch.get("packed_seq_params")
    if psp is not None and getattr(psp, "cu_seqlens_q", None) is not None:
        return psp.cu_seqlens_q
    return batch.get("cu_seqlens")


# ---------------------------------------------------------------------------
# hidden: per-layer forward states
# ---------------------------------------------------------------------------


def _to_token_major(out):
    """Normalize a layer output to ``[tokens, hidden]``.

    megatron layers return ``(hidden_states, context)`` with hidden_states as ``[s, b, h]``
    (sequence-first). Under sequence parallelism s is split across TP ranks and gathered
    here along dim 0, on the TP group and symmetric across its ranks.
    """
    h = out[0] if isinstance(out, tuple) else out
    if hasattr(h, "unwrap"):  # WrappedTensor
        h = h.unwrap()
    assert h.dim() == 3 and h.shape[1] == 1, (
        f"hidden: expected an [s, 1, h] layer output, got {tuple(h.shape)}; "
        "non-packed paths and b>1 slice differently"
    )
    h = h.detach()

    tp = _tp_group()
    if tp is None or not _needs_seq_gather(h, tp):
        return h[:, 0]

    parts = [torch.empty_like(h) for _ in range(dist.get_world_size(tp))]
    dist.all_gather(parts, h.contiguous(), group=tp)
    # As on the torchtitan side: after the gather every rank in the group holds the same
    # tensor, so only TP rank 0 fingerprints. The others take part in the collective and
    # return None, avoiding four processes contending for host CPU.
    if dist.get_rank(tp) != 0:
        return None
    return torch.cat(parts, dim=0)[:, 0]


def _needs_seq_gather(h, tp) -> bool:
    """Whether this output is sharded along seq and must be gathered across TP.

    Decided by actual shape, not by the parallel config: layer outputs are seq-sharded under
    sequence parallelism, while the final norm may already be gathered. The check depends
    only on the token count and this micro-batch's expected total, so it is identical on
    every TP rank.
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


def _tp_group():
    """The TP process group when a seq gather may be needed, else None.

    Layer outputs are seq-sharded only with sequence parallelism and tp > 1; without it each
    TP rank holds the full sequence, since the hidden dimension is what gets split and the
    layer output has already been all-reduced.
    """
    a = registry.actor()
    if a is None or not getattr(a.args, "sequence_parallel", False):
        return None
    if mpu.get_tensor_model_parallel_world_size() <= 1:
        return None
    return mpu.get_tensor_model_parallel_group()


def _logits_full(out):
    """output_layer's logits with the vocab dimension gathered back to full width.

    megatron's output_layer returns ``(logits, bias)`` with logits as ``[s, b, V/tp]``.
    Shard width follows the parallel config, so only the gathered form is comparable across
    deployments. Only TP rank 0 fingerprints.
    """
    h = out[0] if isinstance(out, tuple) else out
    if hasattr(h, "unwrap"):
        h = h.unwrap()
    assert h.dim() == 3 and h.shape[1] == 1, f"logits: expected [s, 1, V], got {tuple(h.shape)}"
    h = h.detach()[:, 0]

    if mpu.get_tensor_model_parallel_world_size() <= 1:
        return h
    tp = mpu.get_tensor_model_parallel_group()
    want = hidden.expected_tokens()
    assert want is None or h.shape[0] == want, (
        f"logits: {h.shape[0]} tokens against an expected {want}; seq should not be sharded here"
    )
    parts = [torch.empty_like(h) for _ in range(dist.get_world_size(tp))]
    dist.all_gather(parts, h.contiguous(), group=tp)
    if dist.get_rank(tp) != 0:
        return None
    return torch.cat(parts, dim=1)


def install_hooks(actor) -> None:
    """Hook every transformer layer plus the final_layernorm after the stack.

    The tail module matters: forward nondeterminism, if any, can sit outside the layer
    stack, and hooking only the layers would miss it.
    """
    if not enabled(actor.args, "hidden"):
        return
    # megatron's TransformerBlock ends with final_layernorm; GPTModel.output_layer follows.
    # Group names match the torchtitan side so artifacts line up across backends.
    tail = [(r"(^|\.)final_layernorm$", "tail_norm", _to_token_major)]
    if "logits" in getattr(actor.args, "precision_guard_hidden_extra", ()):
        tail.append((r"(^|\.)output_layer$", "logits", _logits_full))
    total = sum(hidden.install_on_layers(chunk, _to_token_major, tail=tail) for chunk in actor.model)
    assert total, "hidden: no transformer layer found, hooks were not installed"
    logger.info("precision_guard hidden: installed %d hooks", total)


def record_logprob(args, batch, *, log_probs, old_log_probs, entropy) -> None:
    """Record the loss function's per-sample inputs. Called once per micro-batch.

    Per-token log probs exist only inside ``loss_function``; the rollout data does not carry
    them. Values stay per-sample lists rather than one concatenated block so samples remain
    separable.

    ``advantages`` and ``loss_masks`` are recorded alongside because they are also loss
    inputs: without them a comparison can conclude "inputs identical, outputs differ" while
    an unobserved input was in fact the one that changed.
    """
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


def record_loss(args, model, *, rollout_id, step_id, loss_dict, grad_norm) -> None:
    """Called once after each optimizer step.

    Must sit outside any rank filter so every step is captured under its real step_id.
    Non-final pipeline stages pass an empty loss_dict; the per-name union inside ``record``
    fills the artifact in.
    """
    record(
        args, site="loss", role=getattr(model[0], "role", "actor"),
        rollout_id=rollout_id, step_id=step_id,
        scalars={**dict(loss_dict), "grad_norm": grad_norm},
    )


# ---------------------------------------------------------------------------
# grad
# ---------------------------------------------------------------------------


def _hf_named_grads(args, model):
    """Yield ``(HF parameter name, full gradient tensor)`` through the weight-export pipeline.

    ``param.main_grad`` is a view into the fp32 gradient buffer with the same shape as the
    parameter, so it feeds ``get_hf_weight_chunks`` directly and gradients share the weight
    names. A shape mismatch asserts rather than being assumed away.

    Parameters without main_grad (frozen, or not owned by this rank) are skipped rather than
    zero-filled: zeros would merge "no gradient" with "gradient is exactly zero".
    """
    local_grads = {}
    for name, param in _named_params_and_buffers(args, model):
        grad = getattr(param, "main_grad", None)
        if grad is None:
            continue
        assert grad.shape == param.shape, (
            f"{name}: main_grad shape {tuple(grad.shape)} differs from the parameter's "
            f"{tuple(param.shape)}, it cannot feed the weight conversion pipeline"
        )
        local_grads[name] = grad

    for hf_named_tensors in _hf_weight_iterator().get_hf_weight_chunks(
        local_grads, progress_desc="precision_guard grad"
    ):
        yield from hf_named_tensors


def record_grad(args, model, *, role=None, rollout_id, step_id) -> None:
    """Called once before ``optimizer.step()``."""
    if not enabled(args, "grad") or registry.actor() is None:
        return
    role = role or getattr(model[0], "role", "actor")
    record_stream(
        args, site="grad", role=role, rollout_id=rollout_id, step_id=step_id,
        named_tensors=cast_stream(_hf_named_grads(args, model), export_dtype(args)),
    )
    record(
        args, site="grad", role=role, rollout_id=rollout_id, step_id=step_id,
        scalars={"grad_clipped": False}, merge=True,
    )


# ---------------------------------------------------------------------------
# optim_hparam / grad_norm
# ---------------------------------------------------------------------------


def _optim_hparam_scalars(optimizer, opt_param_scheduler, *, update_applied, num_zeros_in_grad) -> dict:
    """Optimizer hyper-parameters and state. Field names match torchtitan's so artifacts align."""
    scalars: dict = {
        "update_applied": bool(update_applied),
        "num_zeros_in_grad": None if num_zeros_in_grad is None else float(num_zeros_in_grad),
    }
    groups = list(optimizer.param_groups)
    for i, group in enumerate(groups):
        scalars[f"lr_pg{i}"] = float(opt_param_scheduler.get_lr(group))
        scalars[f"weight_decay_pg{i}"] = float(group.get("weight_decay", 0.0))
        scalars[f"eps_pg{i}"] = float(group.get("eps", 0.0))
        betas = group.get("betas", (None, None))
        scalars[f"beta1_pg{i}"] = None if betas[0] is None else float(betas[0])
        scalars[f"beta2_pg{i}"] = None if betas[1] is None else float(betas[1])
    scalars["num_param_groups"] = float(len(groups))
    return scalars


def record_optim(
    args, model, optimizer, opt_param_scheduler, *,
    rollout_id, step_id, grad_norm, update_applied, num_zeros_in_grad,
) -> None:
    """Called once after ``opt_param_scheduler.step()``; records grad_norm and hyper-parameters.

    Two collection points share one call to keep the production side to a single line; they
    still write separate artifacts.
    """
    role = getattr(model[0], "role", "actor")
    record(
        args, site="grad_norm", role=role, rollout_id=rollout_id, step_id=step_id,
        scalars={"grad_norm": grad_norm},
    )
    record(
        args, site="optim_hparam", role=role, rollout_id=rollout_id, step_id=step_id,
        scalars=_optim_hparam_scalars(
            optimizer, opt_param_scheduler,
            update_applied=update_applied,
            # num_zeros_in_grad is only assigned on the valid_step branch.
            num_zeros_in_grad=num_zeros_in_grad,
        ),
    )
    # hidden accumulates across micro-batches and is written here, at the end of the step.
    hidden.flush(args)


# ---------------------------------------------------------------------------
# weight
# ---------------------------------------------------------------------------


def _hf_named_weights(args, model):
    """Yield ``(HF parameter name, full tensor)`` for the live weights.

    Reads the live parameters rather than a weights_backuper snapshot: snapshots exist for
    ref / old_actor and collection should not depend on them.
    """
    local_weights = dict(_named_params_and_buffers(args, model))
    for hf_named_tensors in _hf_weight_iterator().get_hf_weight_chunks(
        local_weights, progress_desc="precision_guard weight"
    ):
        yield from hf_named_tensors


def _master_weights_by_name(args, model, optimizer) -> dict:
    """``{parameter name: fp32 master weight}`` from the distributed optimizer.

    megatron's *model* parameters are the bf16 compute copy while the fp32 master lives in
    the optimizer (``main_params_dtype=torch.float32``); torchtitan's ``named_parameters``
    gives the fp32 master directly. Comparing a bf16 copy against an fp32 master would
    measure bf16 rounding rather than a backend difference: the two precisions of the same
    weights differ by a median 3.24e-05, the same order as the 1.43e-05 cross-backend value.

    The mapping is positional: ``model_float16_groups[g][i]`` and
    ``shard_fp32_from_float16_groups[g][i]`` are appended in the same order by one loop in
    DistributedOptimizer (distrib_optimizer.py from line 349). Natively fp32 parameters use
    model_fp32_groups / shard_fp32_groups the same way.

    A shard is a slice of the flattened parameter taken at param_range. At DP=1 it covers
    the whole parameter and only needs a reshape; at DP>1 it is a real shard and must be
    gathered across DP.
    """
    param_to_name = {id(p): name for name, p in _named_params_and_buffers(args, model)}

    out: dict = {}
    optimizers = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    for opt in optimizers:
        pairs = []
        for attr_model, attr_shard in (
            ("model_float16_groups", "shard_fp32_from_float16_groups"),
            ("model_fp32_groups", "shard_fp32_groups"),
        ):
            mgroups, sgroups = getattr(opt, attr_model, None), getattr(opt, attr_shard, None)
            if mgroups is None or sgroups is None:
                continue
            for mg, sg in zip(mgroups, sgroups, strict=True):
                pairs.extend(zip(mg, sg, strict=True))

        for model_param, shard in pairs:
            if shard is None:  # upstream sets None for types that cannot view(-1), e.g. MXFP8Tensor
                continue
            name = param_to_name.get(id(model_param))
            if name is None:
                continue
            # param_range is this rank's slice within the flattened parameter, read from
            # the same place upstream builds the shard (distrib_optimizer.py:355):
            #   gbuf_index, dtype, bucket_index = model_param_gbuf_map[model_param]
            #   gbuf_ranges[gbuf_index][dtype][bucket_index]["param_map"][model_param]["param"]
            gi, dt, bi = opt.model_param_gbuf_map[model_param]
            rng = opt.gbuf_ranges[gi][dt][bi]["param_map"][model_param]["param"]
            out[name] = (model_param, shard, rng.start, rng.end)

    assert out, "no fp32 master weights found; is the optimizer a DistributedOptimizer?"

    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    if dp_size == 1:
        # At DP=1 param_range covers the whole parameter, so the shard only needs reshaping.
        return {n: sh.view_as(p) for n, (p, sh, _s, _e) in out.items()}
    return _gather_master_weights_over_dp(out, dp_size)


def _gather_master_weights_over_dp(local: dict, dp_size: int) -> dict:
    """Reassemble full master-weight tensors from the per-rank shards at DP>1.

    The distributed optimizer shards optimizer state across DP (ZeRO-1), so each rank holds
    one slice of every parameter, its range given by
    ``gbuf_range["param_map"][model_param]["param"]``.

    The gather runs per parameter rather than moving all state at once: the 4B model's fp32
    masters total ~16 GB, while per-parameter peaks at the largest single tensor (~600 MB
    for the embedding).

    Ranks can hold different slice lengths for the same parameter (bucket boundaries do not
    align), so slices are padded to the maximum before gathering and written back by their
    own (start, end). Parameter iteration order must be identical on every rank, which the
    sorted name list guarantees.
    """
    dp_group = mpu.get_data_parallel_group(with_context_parallel=False)
    names = sorted(local)
    # The name set must match too: a rank holding none of a parameter has no local entry.
    all_names = [None] * dp_size
    dist.all_gather_object(all_names, names, group=dp_group)
    names = sorted(set().union(*(set(x) for x in all_names)))

    # The collective must run on device: dp_group is an HCCL group and CPU tensors raise
    # "No backend type associated with device type cpu". Results move to CPU immediately
    # afterwards, since reassembly holds a full parameter (~1.5 GB for the embedding) that
    # would otherwise compete with training for device memory.
    dev = torch.npu.current_device() if hasattr(torch, "npu") else torch.cuda.current_device()

    out: dict = {}
    for name in names:
        entry = local.get(name)
        # A rank holding none of this parameter still joins with an empty range; skipping
        # would hang the others.
        if entry is None:
            start, end, shard = 0, 0, torch.empty(0, dtype=torch.float32, device=dev)
        else:
            _p, sh, start, end = entry
            shard = sh.detach().to(torch.float32)

        metas = [None] * dp_size
        dist.all_gather_object(metas, (start, end, int(shard.numel())), group=dp_group)
        width = max(m[2] for m in metas)
        buf = torch.zeros(width, dtype=torch.float32, device=dev)
        buf[: shard.numel()] = shard.reshape(-1)
        gathered = [torch.empty(width, dtype=torch.float32, device=dev) for _ in range(dp_size)]
        dist.all_gather(gathered, buf, group=dp_group)
        gathered = [g.cpu() for g in gathered]
        del buf

        shape = tuple(local[name][0].shape) if name in local else None
        shapes = [None] * dp_size
        dist.all_gather_object(shapes, shape, group=dp_group)
        shape = next((s for s in shapes if s is not None), None)
        if shape is None:
            continue

        numel = 1
        for d in shape:
            numel *= d
        full = torch.zeros(numel, dtype=torch.float32)
        covered = 0
        for (s, e, n), g in zip(metas, gathered, strict=True):
            if n == 0:
                continue
            full[s:e] = g[:n]
            covered += n
        assert covered == numel, (
            f"{name}: DP shards cover {covered} elements but the parameter has {numel}; "
            "the param_range assumption does not hold"
        )
        out[name] = full.view(*shape)
    return out


def _hf_named_master_weights(args, model, optimizer):
    """Yield ``(HF parameter name, fp32 master weight)`` through the weight-export pipeline."""
    for hf_named_tensors in _hf_weight_iterator().get_hf_weight_chunks(
        _master_weights_by_name(args, model, optimizer), progress_desc="precision_guard master weight"
    ):
        yield from hf_named_tensors


def record_weight(args, model, optimizer, *, rollout_id, step_id, role=None) -> None:
    """Called once after each ``optimizer.step()``, matching the torchtitan side.

    Reuses the production weight-export path (the same one that feeds vLLM:
    named_params_and_buffers -> bucket all-gather -> convert_to_hf), so quantity names are
    HF parameter names matching the torchtitan side. Parameters are sharded at TP>1 and must
    go through this pipeline rather than being converted one by one.

    megatron's HF weight iterator performs no dtype conversion, so any precision alignment
    happens here. The default collects the model's bf16 compute copy;
    ``--precision-guard-master-weight`` collects the optimizer's fp32 master instead, which
    is the quantity at the same level as torchtitan's ``named_parameters``.
    """
    if not enabled(args, "weight") or registry.actor() is None:
        return
    role = role or getattr(model[0], "role", "actor")
    use_master = getattr(args, "precision_guard_master_weight", False)
    record_stream(
        args, site="weight", role=role, rollout_id=rollout_id, step_id=step_id,
        named_tensors=cast_stream(
            _hf_named_master_weights(args, model, optimizer) if use_master
            else _hf_named_weights(args, model),
            export_dtype(args),
        ),
    )
