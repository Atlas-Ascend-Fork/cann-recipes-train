"""Vocab-parallel log_prob / entropy leaves, reducer, get_log_probs_and_entropy.

The leaves carry the validated dense and vocab-parallel math; the extract functions above them do
the indexing for the one layout vime feeds: packed [T,V] or [1,T,V] logits with flat [T] token ids,
addressed through cu_seqlens offsets. Response extraction can split the token dimension before
calling the same leaves, to bound their temporary activation footprint.

With CP enabled, cp_response_index_spans aligns local logits with global response labels.
Loss and stats retain CP-local contributions; detached metrics are summed over
the DP x CP loss mesh when reporting. forward_only restores full response outputs before rollout write-back.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from vime.utils.distributed_utils import distributed_masked_whiten
from vime.utils.misc import load_function
from vime.utils.ppo_utils import (
    compute_approx_kl,
    compute_entropy_from_logits,
    compute_policy_loss,
    get_grpo_returns,
)
from vime.utils.precision_guard import collect_torchtitan as pg_titan  # precision-guard


# ---------------------------------------------------------------------------
# Vocab-parallel leaves (category 2 reuse)
# ---------------------------------------------------------------------------


def _scale_logits_by_temperature(logits, temperature):
    """Scale logits by temperature when rollout_temperature != 1.0, so training reads the same
    distribution sampling drew from. Callers cast to float32 first, so the division does not
    round in the compute dtype."""
    assert logits.dtype == torch.float32, f"{logits.dtype}"
    if temperature is not None and float(temperature) != 1.0:
        return logits.div(float(temperature))
    return logits


_VOCAB_PARALLEL_CE = None  # lazily-built autograd.Function (avoids torch import-time dependency)


def _build_vocab_parallel_ce():
    """Autograd.Function for vocab-parallel cross entropy (= -logprob).

    Takes local [N, V_shard] logits, global vocab ids [N] and the TP group; returns [N]. Log-sum-exp
    and the target logit come from explicit all_reduce, with a hand-written backward, so no
    collective has to be differentiable.
    """
    class _VocabParallelCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, vocab_parallel_logits, target, process_group):
            # Numerical stability: global max is used only for shifting (softmax is shift-invariant
            # -> gradient does not pass through this, no need to back-prop).
            logits_max = vocab_parallel_logits.max(dim=-1).values
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
            vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(-1)

            partition_size = vocab_parallel_logits.size(-1)
            rank = dist.get_rank(group=process_group)
            vocab_start = rank * partition_size
            vocab_end = vocab_start + partition_size

            # Targets not owned by this rank are zeroed as placeholders; all_reduce(SUM) collects
            # contributions (only the owning rank contributes non-zero values).
            target_mask = (target < vocab_start) | (target >= vocab_end)  # [N]
            masked_target = (target - vocab_start).clone()
            masked_target[target_mask] = 0

            logits_2d = vocab_parallel_logits.view(-1, partition_size)
            masked_target_1d = masked_target.view(-1)
            arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)
            predicted_logits = logits_2d[arange_1d, masked_target_1d].clone().view_as(target)
            predicted_logits[target_mask] = 0.0
            dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=process_group)

            exp_logits = vocab_parallel_logits.exp()
            sum_exp_logits = exp_logits.sum(dim=-1)
            dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=process_group)

            loss = torch.log(sum_exp_logits) - predicted_logits  # [N] = -logprob

            exp_logits.div_(sum_exp_logits.unsqueeze(-1))  # reuse as softmax (for backward)
            ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            softmax, target_mask, masked_target_1d = ctx.saved_tensors
            partition_size = softmax.size(-1)
            grad_2d = softmax.view(-1, partition_size)
            arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)
            # grad = softmax - onehot(target) (subtract 1 only at target positions owned by this rank).
            grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).to(grad_2d.dtype)
            grad_input = grad_2d.mul_(grad_output.unsqueeze(-1))
            return grad_input, None, None

    return _VocabParallelCrossEntropy


def _vocab_parallel_log_probs(local_logits, tokens, tp_group):
    """Manual vocab-parallel log_prob (local_logits [N,V_shard] local + explicit all_reduce)."""
    global _VOCAB_PARALLEL_CE
    if _VOCAB_PARALLEL_CE is None:
        _VOCAB_PARALLEL_CE = _build_vocab_parallel_ce()
    return -_VOCAB_PARALLEL_CE.apply(local_logits.contiguous(), tokens, tp_group)


def _to_local_logits(logits):
    """Convert vocab-sharded DTensor logits to local [..., V_shard] (only vocab dim is sharded; batch/seq dims are intact and indexable)."""
    return logits.to_local() if isinstance(logits, DTensor) else logits


def _packed_logits_2d(logits):
    """Normalize compact packed logits from [T,V] or [1,T,V] to [T,V]."""
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3 and logits.size(0) == 1:
        return logits[0]
    raise ValueError(f"packed logits must have shape [T,V] or [1,T,V], got {tuple(logits.shape)}")


def _dense_log_probs(logits, tokens):
    return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def _dense_entropy(logits):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def _dense_log_probs_and_entropy(logits, tokens):
    log_probs = torch.log_softmax(logits, dim=-1)
    selected = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return selected, entropy


def _chunked_log_probs(logits, tokens, chunk_size, operation):
    """Apply a token-wise log-prob operation without changing the vocab partition."""
    if logits.size(0) != tokens.numel():
        raise ValueError("logits and labels must have identical token counts")
    if logits.size(0) == 0:
        return logits.new_empty((0,))
    if chunk_size is None or chunk_size <= 0 or chunk_size >= logits.size(0):
        return operation(logits, tokens)
    logits_chunks = logits.split(chunk_size, dim=0)
    token_chunks = tokens.split(chunk_size, dim=0)
    return torch.cat(
        [
            operation(logits_chunk, tokens_chunk)
            for logits_chunk, tokens_chunk in zip(logits_chunks, token_chunks, strict=True)
        ],
        dim=0,
    )


def _chunked_entropy(logits, chunk_size, operation):
    """Apply a token-wise entropy operation without changing the vocab partition."""
    if logits.size(0) == 0:
        return logits.new_empty((0,))
    if chunk_size is None or chunk_size <= 0 or chunk_size >= logits.size(0):
        return operation(logits)
    return torch.cat([operation(chunk) for chunk in logits.split(chunk_size, dim=0)], dim=0)


def _chunked_log_probs_and_entropy(logits, tokens, chunk_size, operation):
    """Compute both outputs chunk by chunk so they can share the same softmax."""
    if logits.size(0) != tokens.numel():
        raise ValueError("logits and labels must have identical token counts")
    if logits.size(0) == 0:
        empty = logits.new_empty((0,))
        return empty, empty
    if chunk_size is None or chunk_size <= 0 or chunk_size >= logits.size(0):
        return operation(logits, tokens)

    log_probs = []
    entropies = []
    for logits_chunk, tokens_chunk in zip(
        logits.split(chunk_size, dim=0), tokens.split(chunk_size, dim=0), strict=True
    ):
        chunk_log_probs, chunk_entropy = operation(logits_chunk, tokens_chunk)
        log_probs.append(chunk_log_probs)
        entropies.append(chunk_entropy)
    return torch.cat(log_probs, dim=0), torch.cat(entropies, dim=0)


def _vocab_parallel_log_probs_and_entropy(logits, tokens, tp_group):
    """Keep both vocab-parallel leaves in the same token-chunk schedule."""
    return (
        _vocab_parallel_log_probs(logits, tokens, tp_group),
        compute_entropy_from_logits(logits, tp_group),
    )


def _extract_response_rows(
    logits,
    packed_ids,
    cu,
    total_lengths,
    response_lengths,
    cp_response_index_spans,
):
    for i, (length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=True)):
        if cp_response_index_spans is not None:
            # Spans locate CP rows directly; sample lengths only align sample counts.
            spec = cp_response_index_spans[i]
            span_length = spec.length
            logit_start = spec.local_logit_start
            target_start = spec.global_target_start
            rows = logits[logit_start : logit_start + span_length]
            tok = (
                packed_ids[target_start : target_start + span_length]
                if packed_ids is not None
                else None
            )
        else:
            base = int(cu[i])
            start = base + int(length) - int(response_length)
            rows = logits[start - 1 : base + int(length) - 1]
            tok = (
                packed_ids[start : base + int(length)]
                if packed_ids is not None
                else None
            )
        if tok is not None and rows.size(0) != tok.numel():
            raise ValueError("logits and labels must have identical token counts")
        yield rows, tok


def extract_packed_response_logprobs(
    logits,
    packed_ids,
    cu_seqlens,
    total_lengths,
    response_lengths,
    temperature=1.0,
    vocab_parallel=False,
    tp_group=None,
    cp_response_index_spans=None,
    chunk_size=-1,
):
    """Per-token log_prob of each sequence's response, from packed logits [T,V] or [1,T,V] and flat
    packed_ids [T]; returns list[Tensor[R_i]].

    Sequence i sits at packed positions [cu[i], cu[i+1]); its response tokens [L-R:L] are predicted
    by logits[L-R-1:L-1], since a causal LM's logit at p predicts token[p+1]. Temperature scaling
    is on the logits, before the log_prob.

    vocab_parallel (tp>1): the [R, V_shard] slices go through the manual vocab-parallel log_prob
    (all_reduce over tp_group) on plain tensors, never full_tensor.
    """
    cu = cu_seqlens.tolist()
    out = []
    if vocab_parallel:
        local = _to_local_logits(logits)
        local = _packed_logits_2d(_scale_logits_by_temperature(local.float(), temperature))
        for rows, tokens in _extract_response_rows(
            local, packed_ids, cu, total_lengths, response_lengths, cp_response_index_spans
        ):
            out.append(_chunked_log_probs(
                rows, tokens, chunk_size,
                lambda a, b: _vocab_parallel_log_probs(a, b, tp_group),
            ))
        return out

    logits = _packed_logits_2d(_scale_logits_by_temperature(logits.float(), temperature))
    for rows, tokens in _extract_response_rows(
        logits, packed_ids, cu, total_lengths, response_lengths, cp_response_index_spans
    ):
        out.append(_chunked_log_probs(rows, tokens, chunk_size, _dense_log_probs))
    return out


def extract_packed_response_entropy(
    logits,
    cu_seqlens,
    total_lengths,
    response_lengths,
    temperature=1.0,
    vocab_parallel=False,
    tp_group=None,
    cp_response_index_spans=None,
    chunk_size=-1,
):
    """Per-token entropy H=-Σp·logp of each sequence's response, from packed logits [T,V] or
    [1,T,V]; returns list[Tensor[R_i]]. Under tp>1 the [R, V_shard] slices go through the manual
    vocab-parallel entropy in ppo_utils."""
    cu = cu_seqlens.tolist()
    out = []
    if vocab_parallel:
        local = _packed_logits_2d(_scale_logits_by_temperature(_to_local_logits(logits).float(), temperature))
        for rows, _ in _extract_response_rows(
            local, None, cu, total_lengths, response_lengths, cp_response_index_spans
        ):
            out.append(_chunked_entropy(
                rows, chunk_size,
                lambda chunk: compute_entropy_from_logits(chunk, tp_group),
            ))
        return out

    logits = _packed_logits_2d(_scale_logits_by_temperature(logits.float(), temperature))
    for rows, _ in _extract_response_rows(
        logits, None, cu, total_lengths, response_lengths, cp_response_index_spans
    ):
        out.append(_chunked_entropy(rows, chunk_size, _dense_entropy))
    return out


def extract_packed_response_logprobs_and_entropy(
    logits,
    packed_ids,
    cu_seqlens,
    total_lengths,
    response_lengths,
    temperature=1.0,
    vocab_parallel=False,
    tp_group=None,
    cp_response_index_spans=None,
    chunk_size=-1,
):
    """Both packed response outputs in one token-chunk traversal. Same layouts as
    extract_packed_response_logprobs: logits [T,V] or [1,T,V], packed_ids flat [T]."""
    logits = (
        _scale_logits_by_temperature(_to_local_logits(logits).float(), temperature)
        if vocab_parallel
        else _scale_logits_by_temperature(logits.float(), temperature)
    )
    logits = _packed_logits_2d(logits)
    log_probs = []
    entropies = []
    if vocab_parallel:
        def op(a, b):
            return _vocab_parallel_log_probs_and_entropy(a, b, tp_group)
    else:
        op = _dense_log_probs_and_entropy
    for rows, tokens in _extract_response_rows(logits, packed_ids, cu_seqlens.tolist(), total_lengths, response_lengths, cp_response_index_spans):
        lp, ent = _chunked_log_probs_and_entropy(rows, tokens, chunk_size, op)
        log_probs.append(lp)
        entropies.append(ent)
    return log_probs, entropies


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


def _sum_of_sample_mean(x, response_lengths, loss_masks, sample_denoms):
    """Cumulative sum of (x_i * mask_i).sum() / denom_i per sample."""
    total = 0.0
    for x_i, m_i, denom in zip(x.split([int(r) for r in response_lengths], dim=0), loss_masks, sample_denoms):
        m_i = m_i.float()
        total = total + (x_i * m_i).sum() / torch.clamp_min(denom, 1.0)
    return total


def _sum_of_token(x, response_lengths, loss_masks):
    """Cumulative sum of (x_i * mask_i).sum() per sample (reducer for the calculate_per_token_loss path)."""
    total = 0.0
    for x_i, m_i in zip(x.split([int(r) for r in response_lengths], dim=0), loss_masks):
        total = total + (x_i * m_i.float()).sum()
    return total

# ---------------------------------------------------------------------------
# get_log_probs_and_entropy (packed response extraction, including CP-local layouts)
# ---------------------------------------------------------------------------


def get_log_probs_and_entropy(args, logits, batch, tp_group, *, with_entropy=False, vocab_parallel=False, entropy_requires_grad=True):
    """Split per-token log_prob (and optional entropy) by sample; returns dict[str, list[Tensor]].

    The micro-batch is always packed (THD): get_batch has no other layout. Without CP each sequence's
    response segment is addressed through the cu_seqlens offsets. With CP, cp_response_index_spans
    maps local logit rows to global target positions and outputs retain local_response_lengths.
    entropy_requires_grad=False runs entropy extraction under no_grad;
    log_probs always keep the caller's grad context.
    positive args.log_probs_chunk_size bounds each leaf to that many response-token rows, leaving
    the vocab shard dimension whole.
    """
    total_lengths = batch["total_lengths"]
    response_lengths = batch.get("local_response_lengths", batch["response_lengths"])
    temperature = float(getattr(args, "rollout_temperature", 1.0) or 1.0)
    chunk_size = int(getattr(args, "log_probs_chunk_size", -1) or -1)
    out = {}

    cu = batch["cu_seqlens"]
    packed_ids = batch["packed_ids"]
    if with_entropy and entropy_requires_grad:
        out["log_probs"], out["entropy"] = extract_packed_response_logprobs_and_entropy(
            logits, packed_ids, cu, total_lengths, response_lengths,
            temperature=temperature, vocab_parallel=vocab_parallel, tp_group=tp_group,
            cp_response_index_spans=batch.get("cp_response_index_spans"),
            chunk_size=chunk_size,
        )
        return out
    out["log_probs"] = extract_packed_response_logprobs(
        logits, packed_ids, cu, total_lengths, response_lengths,
        temperature=temperature, vocab_parallel=vocab_parallel, tp_group=tp_group,
        cp_response_index_spans=batch.get("cp_response_index_spans"),
        chunk_size=chunk_size,
    )
    if with_entropy:
        with torch.set_grad_enabled(entropy_requires_grad):
            out["entropy"] = extract_packed_response_entropy(
                logits, cu, total_lengths, response_lengths,
                temperature=temperature, vocab_parallel=vocab_parallel, tp_group=tp_group,
                cp_response_index_spans=batch.get("cp_response_index_spans"),
                chunk_size=chunk_size,
            )

    return out


# ---------------------------------------------------------------------------
# compute_advantages_and_returns
# ---------------------------------------------------------------------------


def compute_advantages_and_returns(args, rollout_data, dp_group):
    """Compute advantages / returns / kl in-place and write them into rollout_data.

    KL -> custom_advantage -> grpo/gspo -> ppo (todo) -> reinforce++ (todo) -> opd (todo) -> normalize.
    """
    rollout_log_probs = rollout_data.get("rollout_log_probs")
    log_probs = rollout_log_probs if getattr(args, "use_rollout_logprobs", False) else rollout_data.get("log_probs")
    ref_log_probs = rollout_data.get("ref_log_probs")
    rewards = rollout_data["rewards"]
    values = rollout_data.get("values")
    response_lengths = rollout_data["response_lengths"]
    loss_masks = rollout_data["loss_masks"]

    # kl enters returns only when kl_coef != 0; otherwise it is a zero placeholder shaped like the
    # available per-token tensor (log_probs / rollout_log_probs / values).
    if (getattr(args, "kl_coef", 0) or 0) == 0 or not log_probs:
        xs = log_probs or rollout_log_probs or values
        kl = [torch.zeros_like(x, dtype=torch.float32) for x in xs]
    else:
        kl = [compute_approx_kl(log_probs[i], ref_log_probs[i], args.kl_loss_type) for i in range(len(log_probs))]
    rollout_data["kl"] = kl

    if args.custom_advantage_function_path is not None:
        load_function(args.custom_advantage_function_path)(args, rollout_data)
        advantages, returns = rollout_data["advantages"], rollout_data["returns"]
    elif args.advantage_estimator in ["grpo", "gspo"]:
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards_t, kl)
        advantages = [r for r in returns]
    elif args.advantage_estimator == "ppo":
        raise NotImplementedError  # todo: ppo/critic not yet supported
    elif args.advantage_estimator == "reinforce_plus_plus":
        raise NotImplementedError  # todo: reinforce++ not yet supported
    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        raise NotImplementedError  # todo: reinforce++ not yet supported
    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported.")

    # todo: opd not yet supported

    if args.normalize_advantages:
        all_adv = torch.cat(advantages)
        all_mask = torch.cat(loss_masks).float()
        # Full response advantages and masks align here. Reduce over DP only, since
        # the same full response fields are replicated within each CP group.
        whitened = distributed_masked_whiten(all_adv, all_mask, process_group=dp_group, shift_mean=True)
        advantages = list(whitened.split([int(r) for r in response_lengths]))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


# ---------------------------------------------------------------------------
# TIS (Truncated Importance Sampling)
# ---------------------------------------------------------------------------


def vanilla_tis_function(args, *, pg_loss, train_log_probs, rollout_log_probs, loss_masks, **kwargs):
    """Truncated Importance Sampling: reweight each token's policy-gradient loss by the clamped
    ratio exp(train_log_prob - rollout_log_prob), correcting the train/rollout policy mismatch.
    Returns the reweighted pg_loss, unchanged loss_masks, and per-token tis metrics. **kwargs keeps
    the signature interchangeable with a custom_tis_function_path function.
    """
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def _apply_tis_and_mismatch_metrics(args, *, pg, ppo_kl, train_log_probs, rollout_log_probs,
                                    loss_masks, response_lengths, total_lengths, reducer):
    """Apply TIS reweighting to pg and compute ois + reduced tis mismatch metrics. No-op unless
    use_tis or get_mismatch_metrics is set.

    A custom_tis_function_path function may zero rejected tokens and return modified masks, which
    the caller rebuilds its pg reducer on. ``reducer`` stays on the original masks: mismatch
    metrics are defined over pre-rejection valid tokens.
    """
    if not (getattr(args, "use_tis", False) or getattr(args, "get_mismatch_metrics", False)):
        return pg, {}, loss_masks

    assert rollout_log_probs, "rollout_log_probs must be provided for TIS / mismatch metrics."

    ois = (-ppo_kl).exp()
    if getattr(args, "custom_tis_function_path", None):
        tis_func = load_function(args.custom_tis_function_path)
    else:
        tis_func = vanilla_tis_function
    pg, modified_masks, tis_metrics = tis_func(
        args=args,
        pg_loss=pg,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    )

    extra = {"ois": reducer(ois).detach()}
    for key, value in tis_metrics.items():
        extra[key] = reducer(value).detach()
    return pg, extra, modified_masks


# ---------------------------------------------------------------------------
# policy_loss_function
# ---------------------------------------------------------------------------


def policy_loss_function(args, logits, batch, tp_group, *, vocab_parallel):
    """GRPO policy loss + metrics: loss / pg_loss / entropy_loss / pg_clipfrac / ppo_kl, plus
    kl_loss under use_kl_loss. TIS, mismatch and OIS are implemented; GSPO / OPSM /
    custom_pg_loss_reducer are structure with todo bodies. Entropy is always reported; when its
    coefficient is 0, it is extracted without building an autograd graph.
    """
    response_lengths = batch.get("local_response_lengths", batch["response_lengths"])
    loss_masks = batch["loss_masks"]
    denoms = batch["rollout_mask_sums"]
    per_token = bool(getattr(args, "calculate_per_token_loss", False))
    if per_token:
        def reducer(v):
            return _sum_of_token(v, response_lengths, loss_masks)
    else:
        def reducer(v):
            return _sum_of_sample_mean(v, response_lengths, loss_masks, denoms)

    ent_coef = float(getattr(args, "entropy_coef", 0.0) or 0.0)
    # Keep the metric available at coef 0, but avoid retaining its autograd graph.
    lp = get_log_probs_and_entropy(args, logits, batch, tp_group,
                                   with_entropy=True, vocab_parallel=vocab_parallel,
                                   entropy_requires_grad=ent_coef != 0.0)
    new = torch.cat(lp["log_probs"])
    # old policy log_probs: rollout-recorded (off-policy) or the separately computed pass; when
    # neither is present (the on-policy single-update path that skips the extra forward), reuse the
    # freshly computed log_probs detached, so the first/only update has ratio 1.
    use_rollout_logprobs = getattr(args, "use_rollout_logprobs", False)
    old_list = batch.get("rollout_log_probs") if use_rollout_logprobs else batch.get("log_probs")
    if not use_rollout_logprobs and not old_list:
        old_list = [t.detach() for t in lp["log_probs"]]
    old = torch.cat(old_list).to(new.device)

    pg_titan.record_logprob(  # precision-guard
        args, batch, log_probs=lp["log_probs"], old_log_probs=old_list, entropy=lp.get("entropy"),
    )

    adv = torch.cat(batch["advantages"]).to(new.device)

    # todo: gspo not yet supported (it needs gspo_kl + full_log_probs here)
    ppo_kl = old - new
    pg, clipfrac = compute_policy_loss(
        ppo_kl, adv, args.eps_clip, getattr(args, "eps_clip_high", args.eps_clip))

    # todo: opsm not yet supported

    # TIS / mismatch train log_probs = the train-recomputed old log_probs, INDEPENDENT of
    # use_rollout_logprobs; falls back to detached new when the extra forward was skipped. Distinct
    # from old_list, which becomes rollout-recorded under use_rollout_logprobs; using old_list here
    # would make the TIS ratio exp(rollout-rollout)=1 and silently zero the mismatch metric.
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [t.detach() for t in lp["log_probs"]]

    # TIS / mismatch metrics (async off-policy): reweight pg by the clamped train/rollout
    # importance ratio and collect ois/tis metrics. No-op unless use_tis/get_mismatch_metrics.
    pg, tis_extra_metrics, modified_masks = _apply_tis_and_mismatch_metrics(
        args, pg=pg, ppo_kl=ppo_kl, train_log_probs=train_log_probs_for_tis,
        rollout_log_probs=batch.get("rollout_log_probs"), loss_masks=loss_masks,
        response_lengths=response_lengths, total_lengths=batch["total_lengths"], reducer=reducer,
    )
    # todo: custom_pg_loss_reducer_function_path not yet supported

    # Rebuilt on the masks a custom tis_func returned, so every downstream per-rollout mean drops
    # the rejected tokens from its numerator while the denominators stay rollout_mask_sums. On the
    # vanilla and no-TIS paths those masks are loss_masks, so this reducer is the base one.
    if per_token:
        def reducer(v):
            return _sum_of_token(v, response_lengths, modified_masks)
    else:
        def reducer(v):
            return _sum_of_sample_mean(v, response_lengths, modified_masks, denoms)
    pg_loss = reducer(pg)
    pg_clipfrac = reducer(clipfrac)
    ppo_kl_reduced = reducer(ppo_kl)
    entropy_loss = reducer(torch.cat(lp["entropy"]))
    loss = pg_loss - ent_coef * entropy_loss

    if getattr(args, "use_kl_loss", False) and batch.get("ref_log_probs") is not None:
        ref = torch.cat(batch["ref_log_probs"]).to(new.device)
        # Unbiased KL applies the importance ratio pi_theta/pi_old to the KL estimate.
        importance_ratio = None
        if getattr(args, "use_unbiased_kl", False):
            importance_ratio = torch.exp(new - old)
        kl = compute_approx_kl(new, ref, args.kl_loss_type, importance_ratio=importance_ratio)
        kl_loss = reducer(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    # Mean abs diff between the train (old) log_probs and the rollout-recorded log_probs;
    # surfaces train/rollout numerical divergence. Only when rollout_log_probs is present.
    train_rollout_logprob_abs_diff = None
    if batch.get("rollout_log_probs"):
        rollout_lp = torch.cat(batch["rollout_log_probs"]).to(new.device)
        train_rollout_logprob_abs_diff = reducer((old - rollout_lp).abs())

    # Guard against empty gradients: when sequence length is all-zero, logits has no grad_fn;
    # adding zero ensures backward does not break.
    if new.numel() == 0:
        loss = loss + 0 * logits.sum()

    metrics = {
        "loss": loss.detach(),
        "pg_loss": pg_loss.detach(),
        "entropy_loss": entropy_loss.detach(),
        "pg_clipfrac": pg_clipfrac.detach(),
        "ppo_kl": ppo_kl_reduced.detach(),
    }
    if train_rollout_logprob_abs_diff is not None:
        metrics["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.detach()
    if getattr(args, "use_kl_loss", False) and batch.get("ref_log_probs") is not None:
        metrics["kl_loss"] = kl_loss.detach()

    # ois / tis / tis_clipfrac / tis_abs (when use_tis or get_mismatch_metrics).
    if tis_extra_metrics:
        metrics.update(tis_extra_metrics)

    # todo: opsm_clipfrac / opd_reverse_kl metrics not yet supported

    return loss, metrics


# ---------------------------------------------------------------------------
# sft_loss_function
# ---------------------------------------------------------------------------


def sft_loss_function(args, logits, batch, tp_group, *, vocab_parallel):
    """SFT negative log-likelihood loss.

    Fetches data via get_log_probs_and_entropy (with_entropy=False); loss = -reducer(log_probs).
    """
    response_lengths = batch.get("local_response_lengths", batch["response_lengths"])
    loss_masks = batch["loss_masks"]
    denoms = batch["rollout_mask_sums"]
    lp = get_log_probs_and_entropy(args, logits, batch, tp_group,
                                   with_entropy=False, vocab_parallel=vocab_parallel)
    new = torch.cat(lp["log_probs"])
    if bool(getattr(args, "calculate_per_token_loss", False)):
        loss = -_sum_of_token(new, response_lengths, loss_masks)
    else:
        loss = -_sum_of_sample_mean(new, response_lengths, loss_masks, denoms)

    # Guard against empty gradients.
    if new.numel() == 0:
        loss = loss + 0 * logits.sum()

    return loss, {"loss": loss.detach()}


# ---------------------------------------------------------------------------
# loss_function (dispatcher)
# ---------------------------------------------------------------------------


def loss_function(args, logits, batch, tp_group, *, vocab_parallel, divisor, dp_size):
    """Dispatch by args.loss_type and apply the step-level scaling ``loss * dp_size / divisor``:
    the dp_size factor cancels FSDP's gradient averaging, leaving a net gradient of
    ``(1/divisor) * Σ_ranks ∇(reducer_sum)``. The value_loss branch is structure with a todo body.
    """
    if args.loss_type == "policy_loss":
        func = policy_loss_function
    elif args.loss_type == "value_loss":
        raise NotImplementedError  # todo: critic/value_loss not yet supported
    elif args.loss_type == "sft_loss":
        func = sft_loss_function
    else:
        raise NotImplementedError(f"loss_type {args.loss_type!r} is not supported.")

    if getattr(args, "recompute_loss_function", False):
        # Recompute the loss head in backward instead of keeping the logsoftmax intermediates.
        # The metrics dict is detached inside, so checkpoint passes it through unchanged.
        loss, metrics = checkpoint(
            func, args, logits, batch, tp_group,
            vocab_parallel=vocab_parallel, use_reentrant=False,
        )
    else:
        loss, metrics = func(args, logits, batch, tp_group, vocab_parallel=vocab_parallel)

    div = divisor if torch.is_tensor(divisor) else float(divisor)
    scaled = loss * dp_size / div

    # todo: cp zero-loss guard not yet supported (it needs an all-gather over the cp group)

    return scaled, metrics
