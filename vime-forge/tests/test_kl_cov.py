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

"""CPU numerical regressions; no Ray, NPU, or Megatron installation required.

Backend functions are loaded from their AST to avoid importing distributed
runtime dependencies. Only log-prob extraction and instrumentation are stubbed;
losses, reducers, TIS, token selection, packing, and autograd use real PyTorch.
Run with TORCH_COMPILE_DISABLE=1 python -m pytest tests/test_kl_cov.py.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vime.utils.ppo_utils import (
    compute_cispo_loss,
    compute_kl_cov_loss,
    compute_kl_cov_penalty,
    compute_kl_cov_selection,
    compute_policy_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace=None):
    """Execute the actual source definitions, deferring dependency annotations."""
    path = ROOT / path
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    scope = {"torch": torch, "math": math, **(namespace or {})}
    exec(compile(module, str(path), "exec"), scope)  # noqa: S102 — trusted local definitions only
    return scope


def args_for_loss(**overrides):
    values = {
        "use_kl_cov": True,
        "kl_cov_ratio": 0.25,
        "kl_cov_coef": 0.7,
        "loss_type": "policy_loss",
        "use_kl_loss": False,
        "kl_coef": 0.0,
        "advantage_estimator": "grpo",
        "use_opsm": False,
        "context_parallel_size": 1,
        "eps_clip": 0.2,
        "eps_clip_high": 0.28,
        "eps_clip_c": None,
        "entropy_coef": 0.0,
        "use_rollout_logprobs": False,
        "use_tis": False,
        "get_mismatch_metrics": False,
        "custom_tis_function_path": None,
        "calculate_per_token_loss": True,
        "tis_clip_low": 0.5,
        "tis_clip": 2.0,
    }
    return SimpleNamespace(**(values | overrides))


def test_penalty_matches_equation_and_analytical_gradient():
    old = torch.tensor([-2.0, -2.0, -2.0, -2.0], dtype=torch.float64)
    delta = torch.tensor([0.4, -0.6, 0.0, 0.2], dtype=torch.float64)
    new = (old + delta).requires_grad_()
    advantage = torch.tensor([2.0, -3.0, 1.0, -1.0], dtype=torch.float64)
    selected = torch.tensor([1.0, 1.0, 1.0, 0.0])
    loss, fraction, penalty = compute_kl_cov_penalty(new, old, advantage, selected, 0.7)
    expected_penalty = 0.7 * delta.abs() * selected
    torch.testing.assert_close(penalty, expected_penalty)
    torch.testing.assert_close(loss, -advantage * delta.exp() + expected_penalty)
    torch.testing.assert_close(fraction, selected)
    loss.sum().backward()
    torch.testing.assert_close(
        new.grad, -advantage * delta.exp() + 0.7 * selected * delta.sign()
    )


def test_selection_ignores_masked_extremes_and_is_order_invariant():
    lp = [torch.tensor([-1.0, -9.0, 1e6]), torch.tensor([-4.0, -2.0])]
    adv = [torch.tensor([1.0, 1.0, -1e6]), torch.tensor([-1.0, -1.0])]
    masks = [torch.tensor([1.0, 1.0, 0.0]), torch.ones(2)]
    selected = compute_kl_cov_selection(lp, adv, masks, 0.25)
    # Valid mean logp=-4; covariance scores=[3, -5, 0, -2].
    assert [x.tolist() for x in selected] == [[1.0, 0.0, 0.0], [0.0, 0.0]]
    reordered = compute_kl_cov_selection(lp[::-1], adv[::-1], masks[::-1], 0.25)
    assert [x.tolist() for x in reordered] == [x.tolist() for x in selected[::-1]]
    assert all(not x.requires_grad for x in selected)


@pytest.mark.parametrize("missing_field", ["log_probs", "advantages"])
@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_missing_sample_does_not_change_selection_for_valid_samples(missing_field, missing_index, caplog):
    lp = [torch.tensor([-1., -9.]), torch.tensor([-4., -2.])]
    adv = [torch.ones(2), -torch.ones(2)]
    masks = [torch.ones(2), torch.ones(2)]
    expected = compute_kl_cov_selection(lp, adv, masks, 0.25)
    lp.insert(missing_index, None if missing_field == "log_probs" else torch.full((3,), 1e6))
    adv.insert(missing_index, None if missing_field == "advantages" else torch.full((3,), 1e6))
    masks.insert(missing_index, torch.ones(3))
    expected.insert(missing_index, torch.zeros(3))

    actual = compute_kl_cov_selection(lp, adv, masks, 0.25)
    torch.testing.assert_close(actual, expected)
    assert caplog.messages == ["KL-Cov selection skipped 1/3 samples with missing log_probs or advantages"]


def test_all_missing_samples_keep_their_mask_shapes_and_warn_once(caplog):
    masks = [torch.ones(3), torch.ones(1)]
    actual = compute_kl_cov_selection([None], [torch.ones(3), None], masks, 0.25)
    torch.testing.assert_close(actual, [torch.zeros_like(m) for m in masks])
    assert caplog.messages == ["KL-Cov selection skipped 2/2 samples with missing log_probs or advantages"]


@pytest.mark.parametrize("ratio,count", [(2e-4, 1), (0.49, 1), (0.5, 2), (1.0, 4)])
def test_selection_count(ratio, count):
    selected = compute_kl_cov_selection(
        [torch.tensor([-1.0, -9.0, -4.0, -2.0])],
        [torch.tensor([1.0, 1.0, -1.0, -1.0])],
        [torch.ones(4)],
        ratio,
    )
    assert selected[0].sum().item() == count


def test_empty_and_all_masked_selection():
    assert compute_kl_cov_selection([], [], [], 0.25) == []
    for size in (0, 3):
        values = torch.zeros(size)
        selected = compute_kl_cov_selection([values], [values], [values], 0.25)
        assert selected[0].shape == values.shape
        assert selected[0].sum().item() == 0
        loss, mask, penalty = compute_kl_cov_loss(
            values, values, values, values, 0.25, 1.0
        )
        assert loss.numel() == size
        assert mask.sum().item() == penalty.sum().item() == 0


def test_single_response_microbatches_reuse_batch_selection_and_gradients():
    old = torch.tensor([-1.0, -9.0, -4.0, -2.0])
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    selected = compute_kl_cov_selection(
        list(old.split(2)), list(adv.split(2)), [torch.ones(2)] * 2, 0.25
    )
    assert [x.tolist() for x in selected] == [[1.0, 0.0], [0.0, 0.0]]
    # Each response alone has constant advantage and zero covariance. Its local
    # fallback also selects at least one token, incorrectly doubling this budget.
    fallback_count = sum(
        compute_kl_cov_loss(lp, lp, a, torch.ones(2), 0.25, 1.0)[1].sum().item()
        for lp, a in zip(old.split(2), adv.split(2), strict=True)
    )
    assert fallback_count == 2
    new = (old + 0.1).requires_grad_()
    full = compute_kl_cov_penalty(new, old, adv, torch.cat(selected), 0.7)[0].sum()
    full_grad = torch.autograd.grad(full, new)[0]
    separate = sum(
        compute_kl_cov_penalty(n, o, a, s, 0.7)[0].sum()
        for n, o, a, s in zip(
            new.split(2), old.split(2), adv.split(2), selected, strict=True
        )
    )
    torch.testing.assert_close(separate, full)
    torch.testing.assert_close(torch.autograd.grad(separate, new)[0], full_grad)


def backend_loss(backend, new, batch, args):
    probabilities = {
        "log_probs": list(new.split(batch["response_lengths"])),
        "entropy": [torch.zeros_like(x) for x in batch["advantages"]],
    }
    noop = SimpleNamespace(record_logprob=lambda *a, **kw: None)
    scope = {
        "compute_policy_loss": compute_policy_loss,
        "compute_cispo_loss": compute_cispo_loss,
        "compute_kl_cov_penalty": compute_kl_cov_penalty,
        "compute_kl_cov_loss": compute_kl_cov_loss,
        "pg_titan": noop,
        "pg_megatron": noop,
    }
    if backend == "torchtitan":
        scope["get_log_probs_and_entropy"] = lambda *a, **kw: probabilities
        scope = load_definitions(
            "vime/backends/torchtitan_utils/loss.py",
            {
                "policy_loss_function",
                "vanilla_tis_function",
                "_apply_tis_and_mismatch_metrics",
                "_sum_of_token",
                "_sum_of_sample_mean",
            },
            scope,
        )
        return scope["policy_loss_function"](
            args, new, batch, None, vocab_parallel=False
        )
    scope.update(
        get_log_probs_and_entropy=lambda *a, **kw: (None, probabilities),
        get_rollout_top_p_logprob_kwargs=lambda *a: {},
        _maybe_capture_log_probs=lambda *a: None,
        mpu=SimpleNamespace(get_context_parallel_world_size=lambda: 1),
    )
    scope = load_definitions(
        "vime/backends/megatron_utils/cp_utils.py", {"get_sum_of_sample_mean"}, scope
    )
    scope = load_definitions(
        "vime/backends/megatron_utils/loss.py",
        {"policy_loss_function", "vanilla_tis_function"},
        scope,
    )
    reducer = scope["get_sum_of_sample_mean"](
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch["rollout_mask_sums"],
        args.calculate_per_token_loss,
    )
    return scope["policy_loss_function"](args, batch, new, reducer)


@pytest.mark.parametrize("backend", ["torchtitan", "megatron"])
@pytest.mark.parametrize("use_tis", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("preselected", [False, True])
def test_backend_loss_masks_tis_and_gradients(backend, use_tis, per_token, preselected):
    args = args_for_loss(use_tis=use_tis, calculate_per_token_loss=per_token)
    old = torch.tensor([-1.0, -9.0, -4.0, -2.0])
    delta = torch.tensor([0.1, -0.2, 0.3, -0.4])
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    selected = (
        torch.tensor([0.0, 0.0, 0.0, 1.0])
        if preselected
        else torch.tensor([1.0, 0.0, 0.0, 0.0])
    )
    batch = {
        "log_probs": list(old.split(2)),
        "advantages": list(adv.split(2)),
        "loss_masks": list(mask.split(2)),
        "response_lengths": [2, 2],
        "total_lengths": [3, 3],
        "unconcat_tokens": [],
        "rollout_mask_sums": [torch.tensor(3.0)] * 2,
        "rollout_log_probs": list((old - torch.tensor([1.0, 0.0, -2.0, 0.2])).split(2)),
    }
    if preselected:
        batch["kl_cov_selected"] = list(selected.split(2))
    new = (old + delta).requires_grad_()
    loss, metrics = backend_loss(backend, new, batch, args)
    delta = new.detach() - old
    weight = (
        (old - torch.cat(batch["rollout_log_probs"])).exp().clamp(0.5, 2.0)
        if use_tis
        else 1.0
    )
    denom = 1.0 if per_token else 3.0
    penalty = 0.7 * delta.abs() * selected
    expected = ((-adv * delta.exp() + penalty) * weight * mask).sum() / denom
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["kl_cov"], (penalty * mask).sum() / denom)
    torch.testing.assert_close(metrics["kl_cov_selected_ratio"], (selected * mask).sum() / denom)
    assert metrics["pg_clipfrac"].item() == 0
    loss.backward()
    expected_grad = (
        (-adv * delta.exp() + 0.7 * selected * delta.sign()) * weight * mask / denom
    )
    torch.testing.assert_close(new.grad, expected_grad)
    assert all(not x.requires_grad for x in metrics.values())


@pytest.mark.parametrize(
    "backend,estimator,dual_clip",
    [
        ("torchtitan", "grpo", None),
        ("megatron", "grpo", None),
        ("megatron", "grpo", 2.0),
        ("megatron", "cispo", None),
    ],
)
def test_disabled_kl_cov_preserves_existing_loss(backend, estimator, dual_clip):
    args = args_for_loss(
        use_kl_cov=False, advantage_estimator=estimator, eps_clip_c=dual_clip
    )
    old = torch.tensor([-5.0, -5.0])
    new = torch.tensor([-3.0, -3.0], requires_grad=True)
    adv = torch.tensor([1.0, -1.0])
    batch = {
        "log_probs": [old],
        "advantages": [adv],
        "loss_masks": [torch.ones(2)],
        "response_lengths": [2],
        "total_lengths": [3],
        "unconcat_tokens": [],
        "rollout_mask_sums": [torch.tensor(2.0)],
    }
    loss, metrics = backend_loss(backend, new, batch, args)
    if estimator == "cispo":
        pg, clipped = compute_cispo_loss(old - new, new, adv, 0.2, 0.28)
    else:
        pg, clipped = compute_policy_loss(old - new, adv, 0.2, 0.28, eps_clip_c=dual_clip)
    expected = pg.sum()
    torch.testing.assert_close(metrics["pg_clipfrac"], clipped.sum())
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, new)[0], torch.autograd.grad(expected, new)[0]
    )
    assert "kl_cov" not in metrics
    assert "kl_cov_selected_ratio" not in metrics


def test_torchtitan_packing_keeps_selection_aligned_after_reordering():
    scope = load_definitions(
        "vime/backends/torchtitan_utils/data.py",
        {
            "DataIterator",
            "build_packed_microbatch",
            "get_batch",
        },
    )
    data = {
        "tokens": [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])],
        "loss_masks": [torch.ones(2), torch.ones(2)],
        "total_lengths": [3, 3],
        "response_lengths": [2, 2],
        "kl_cov_selected": [torch.tensor([1.0, 0.0]), torch.zeros(2)],
    }
    iterator = scope["DataIterator"](data, [[1], [0]])
    batches = [
        scope["get_batch"](
            SimpleNamespace(), iterator, SimpleNamespace(cp=1, tp=1), "cpu"
        )
        for _ in range(2)
    ]
    assert [b["packed_ids"].tolist() for b in batches] == [[3, 4, 5], [0, 1, 2]]
    assert [b["kl_cov_selected"][0].tolist() for b in batches] == [
        [0.0, 0.0],
        [1.0, 0.0],
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"loss_type": "sft_loss"},
        {"use_kl_loss": True},
        {"kl_coef": 0.1},
        {"advantage_estimator": "gspo"},
        {"advantage_estimator": "cispo"},
        {"use_opsm": True},
        {"kl_cov_ratio": 0.0},
        {"kl_cov_ratio": -0.1},
        {"kl_cov_ratio": 1.01},
        {"kl_cov_ratio": math.nan},
        {"kl_cov_ratio": math.inf},
        {"kl_cov_coef": -1.0},
        {"kl_cov_coef": math.nan},
        {"kl_cov_coef": math.inf},
        {"context_parallel_size": 2},
    ],
)
def test_invalid_kl_cov_configuration_is_rejected_only_when_enabled(overrides):
    validate = load_definitions("vime/utils/arguments.py", {"validate_kl_cov_args"})[
        "validate_kl_cov_args"
    ]
    with pytest.raises(ValueError):
        validate(args_for_loss(**overrides))
    validate(args_for_loss(**(overrides | {"use_kl_cov": False})))


def test_valid_kl_cov_configuration_and_legacy_namespace():
    validate = load_definitions("vime/utils/arguments.py", {"validate_kl_cov_args"})[
        "validate_kl_cov_args"
    ]
    validate(args_for_loss())
    validate(args_for_loss(kl_cov_coef=0.0, kl_cov_ratio=1.0))
    validate(SimpleNamespace())
