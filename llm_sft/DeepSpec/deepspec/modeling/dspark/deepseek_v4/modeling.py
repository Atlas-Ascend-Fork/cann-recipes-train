# Adapted from
# https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/dspark/deepseek_v4/modeling.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""DeepSeek-V4 native DSpark draft model.

Reference: `DeepSeek-V4-Flash-DSpark <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark>`_

The official Ascend inference model embeds the DSpark speculative decoder as
``mtp.*`` stages inside the full 43-layer DeepSeek-V4 backbone.  That path uses
Shared-KV/MQA attention, MoE and hyper-connections, but requires the full
checkpoint and NPU custom kernels.

This module provides a **standalone** draft model that keeps the same DSpark
training contract while using PyTorch/SDPA-friendly approximations for the
kernel-heavy pieces.  It exposes the same ``forward()`` / ``_forward_backbone()``
interface as
:class:`~deepspec.modeling.dspark.qwen3.modeling.Qwen3DSparkModel` so it is
a drop-in replacement in the training pipeline.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from typing_extensions import Tuple, Unpack

from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.qwen3.modeling_qwen3 import FlashAttentionKwargs

from deepspec.modeling.dspark.common import (
    AcceptRatePredictor,
    DSparkForwardOutput,
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    log_sampler_stats,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.markov_head import build_markov_head
from deepspec.utils.sampling import sample_tokens

try:
    import torch_npu
except ImportError:  # CPU/CUDA development and tests.
    torch_npu = None


# ---------------------------------------------------------------------------
# RoPE — CANN DSpark C1A-compatible frequency and interleaved layout
# ---------------------------------------------------------------------------


class DeepSeekV4RotaryEmbedding(nn.Module):
    """Plain C1A RoPE with the existing DSpark position-id interface."""

    def __init__(self, config):
        super().__init__()
        self.rope_head_dim = int(getattr(config, "qk_rope_head_dim", 64))
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))

        dim = self.rope_head_dim
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        position_ids: torch.LongTensor,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Return ``(cos, sin)`` each of shape ``(total_len, rope_head_dim)``."""
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(position_ids.reshape(-1).float(), inv_freq)
        emb = freqs.repeat_interleave(2, dim=-1)
        return (
            emb.cos().to(dtype),
            emb.sin().to(dtype),
        )


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to *x*, shape ``(*, seq_len, head_dim)``.
    *cos* / *sin* shape ``(seq_len, rope_dim)``.  The seq_len dim in *x*
    is the second-to-last non-rope dim (position -2 counting from the
    right, excluding the rope_dim split)."""
    rope_dim = cos.shape[-1]
    x_rope, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    # Insert singleton dims so cos broadcasts over everything except seq_len.
    if x.ndim == 4:
        cos = cos.view(1, -1, 1, rope_dim)
        sin = sin.view(1, -1, 1, rope_dim)
    elif x.ndim == 3:
        cos = cos.view(1, -1, rope_dim)
        sin = sin.view(1, -1, rope_dim)
    x_pairs = x_rope.unflatten(-1, (-1, 2))
    rotated = torch.stack((-x_pairs[..., 1], x_pairs[..., 0]), dim=-1).flatten(-2)
    return torch.cat([x_rope * cos + rotated * sin, x_pass], dim=-1)


# ---------------------------------------------------------------------------
# RMSNorm (same as DeepseekV4RMSNorm)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV4RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.variance_epsilon = eps

    def ln_npu(self, hidden_states):
        result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return result

    def forward(self, hidden_states, *args):
        if len(args) == 0: # only hidden_states exists
            result = self.ln_npu(hidden_states)
            return result
        elif len(args) == 1 and args[0] is None: # residual is None
            result = self.ln_npu(hidden_states)
            residual = hidden_states
            return (result, residual)
        elif len(args) == 1: # residual is not None
            residual = args[0]
            y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
            return (y, x)
        else:
            raise NotImplementedError(
                f"insupportable DeepseekV3RMSNorm for input_args len as (include hid): {len(args) + 1}"
            )


# ---------------------------------------------------------------------------
# Attention — DeepSeek-V4 MLA projections with shared NPU RMSNorm
# ---------------------------------------------------------------------------

class DeepSeekV4DSparkSDPAAttention(nn.Module):
    """DeepSeek-V4 MLA projections followed by PyTorch SDPA.

    DeepSeek-V4 uses one shared KV head. It is expanded across query heads
    before SDPA for compatibility with torch_npu versions without GQA support.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(
            getattr(config, "head_dim", self.hidden_size // self.num_heads)
        )
        self.q_lora_rank = int(
            getattr(config, "q_lora_rank", self.hidden_size)
        )
        self.o_lora_rank = int(
            getattr(config, "o_lora_rank", self.hidden_size)
        )
        self.rope_head_dim = int(getattr(config, "qk_rope_head_dim", 64))
        self.qk_nope_head_dim = self.head_dim - self.rope_head_dim
        self.scaling = self.head_dim**-0.5
        self.eps = float(getattr(config, "rms_norm_eps", 1e-6))

        # One learnable sink logit per query head. The sink contributes to the
        # softmax denominator with a zero value, matching the inference kernel.
        self.attn_sink = nn.Parameter(
            torch.zeros(self.num_heads, dtype=torch.float32)
        )

        assert self.qk_nope_head_dim >= 0, (
            "qk_rope_head_dim must not exceed head_dim: "
            f"{self.rope_head_dim} > {self.head_dim}"
        )

        self.wq_a = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
        )

        # Shared K=V projection used by DeepSeek-V4's MQA layout.
        self.wkv = nn.Linear(
            self.hidden_size,
            self.head_dim,
            bias=False,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)

        self.wo_a = nn.Linear(
            self.num_heads * self.head_dim,
            self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            self.o_lora_rank,
            self.hidden_size,
            bias=False,
        )

    def _project_q(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        q = self.wq_b(self.q_norm(self.wq_a(hidden_states)))
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim :]
        q_rope = _apply_rotary(q_rope, cos, sin)
        return torch.cat([q_nope, q_rope], dim=-1)

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        kv = self.kv_norm(self.wkv(hidden_states))
        kv = kv.unflatten(-1, (1, self.head_dim))
        kv_nope = kv[..., : self.qk_nope_head_dim]
        kv_rope = kv[..., self.qk_nope_head_dim :]
        kv_rope = _apply_rotary(kv_rope, cos, sin)
        return torch.cat([kv_nope, kv_rope], dim=-1)

    def _validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        *,
        batch_size: int,
        q_len: int,
        kv_len: int,
    ) -> None:
        assert attention_mask.ndim == 4, (
            "DSpark SDPA attention_mask must have shape "
            "[batch, heads|1, q_len, kv_len], got "
            f"{tuple(attention_mask.shape)}"
        )
        assert attention_mask.shape[0] in (1, batch_size), (
            "DSpark SDPA attention_mask batch dimension must be 1 or match "
            f"the HC-expanded batch size: {attention_mask.shape[0]} vs "
            f"{batch_size}"
        )
        assert attention_mask.shape[1] in (1, self.num_heads), (
            "DSpark SDPA attention_mask head dimension must be 1 or match "
            f"num_heads: {attention_mask.shape[1]} vs {self.num_heads}"
        )
        assert attention_mask.shape[-2:] == (q_len, kv_len), (
            "DSpark SDPA attention_mask Q/KV dimensions do not match "
            f"attention inputs: {tuple(attention_mask.shape[-2:])} vs "
            f"{(q_len, kv_len)}"
        )

    def _attention_with_sink(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        """Run SDPA with an extra per-head sink logit and zero value."""
        batch_size, _, q_len, head_dim = q.shape
        kv_len = kv.shape[-2]
        sink_kv = kv.new_zeros((batch_size, self.num_heads, 1, head_dim))
        kv_with_sink = torch.cat([kv, sink_kv], dim=-2)

        if attention_mask is None:
            additive_mask = q.new_zeros((1, self.num_heads, q_len, kv_len + 1))
            if is_causal:
                causal_mask = torch.ones(
                    (q_len, kv_len), dtype=torch.bool, device=q.device
                ).tril()
                additive_mask[..., :kv_len].masked_fill_(
                    ~causal_mask.view(1, 1, q_len, kv_len),
                    float("-inf"),
                )
        elif attention_mask.dtype == torch.bool:
            additive_mask = torch.zeros(
                (*attention_mask.shape[:-1], kv_len + 1),
                dtype=q.dtype,
                device=q.device,
            )
            additive_mask[..., :kv_len].masked_fill_(
                ~attention_mask, float("-inf")
            )
        else:
            additive_mask = F.pad(attention_mask.to(dtype=q.dtype), (0, 1))

        additive_mask = additive_mask.expand(
            additive_mask.shape[0], self.num_heads, q_len, kv_len + 1
        ).clone()
        sink_bias = self.attn_sink.to(device=q.device, dtype=q.dtype)
        additive_mask[..., -1] = sink_bias.view(1, self.num_heads, 1)
        return F.scaled_dot_product_attention(
            q,
            kv_with_sink,
            kv_with_sink,
            attn_mask=additive_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        del cache_position

        is_hc = hidden_states.ndim == 4
        if is_hc:
            original_batch_size, q_len, hc_mult, _ = hidden_states.shape
            hidden_states = hidden_states.transpose(1, 2).reshape(
                original_batch_size * hc_mult,
                q_len,
                -1,
            )
            target_hidden_states = target_hidden_states.unsqueeze(1).expand(
                -1,
                hc_mult,
                -1,
                -1,
            ).reshape(
                original_batch_size * hc_mult,
                target_hidden_states.shape[1],
                -1,
            )
            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(
                    hc_mult,
                    dim=0,
                )
        else:
            hc_mult = 1

        batch_size, q_len = hidden_states.shape[:2]
        context_len = target_hidden_states.shape[1]
        current_kv_len = context_len + q_len
        cos, sin = position_embeddings

        q = self._project_q(
            hidden_states,
            cos[current_kv_len - q_len :],
            sin[current_kv_len - q_len :],
        )
        kv_context = self._project_kv(
            target_hidden_states,
            cos[:context_len],
            sin[:context_len],
        )
        kv_draft = self._project_kv(
            hidden_states,
            cos[context_len:],
            sin[context_len:],
        )
        kv = torch.cat([kv_context, kv_draft], dim=1)

        q = q.transpose(1, 2)
        kv = kv.expand(-1, -1, self.num_heads, -1).transpose(1, 2)

        if past_key_values is not None:
            kv, _ = past_key_values.update(
                kv,
                kv,
                self.layer_idx,
            )

        if attention_mask is not None:
            self._validate_attention_mask(
                attention_mask,
                batch_size=batch_size,
                q_len=q_len,
                kv_len=kv.shape[-2],
            )

        attn_output = self._attention_with_sink(
            q,
            kv,
            attention_mask,
            bool(kwargs.get("is_causal", False)),
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_nope = attn_output[..., : self.qk_nope_head_dim]
        attn_rope = _apply_rotary(
            attn_output[..., self.qk_nope_head_dim :],
            cos[current_kv_len - q_len :],
            -sin[current_kv_len - q_len :],
        )
        attn_output = torch.cat([attn_nope, attn_rope], dim=-1)
        attn_output = attn_output.reshape(
            batch_size,
            q_len,
            self.num_heads * self.head_dim,
        )
        output = self.wo_b(self.wo_a(attn_output))

        if is_hc:
            output = output.reshape(
                original_batch_size,
                hc_mult,
                q_len,
                -1,
            ).transpose(1, 2)
        return output, None


# ---------------------------------------------------------------------------
# MoE — training counterpart of official ``DeepseekV3MoE`` in cann-recipes
# ---------------------------------------------------------------------------

class DSparkSharedExpert(nn.Module):
    """Unquantized shared SwiGLU expert used by the DSpark MoE.

    The gate and up projections are stored as one ``gate_up_proj`` so the NPU
    path can feed their combined output directly to the fused SwiGLU kernel.
    Tensor parallelism, quantization, and stream management belong to the
    inference implementation and are intentionally omitted here.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit=None):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = (
            None if swiglu_limit is None else float(swiglu_limit)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        if self.swiglu_limit is None or self.swiglu_limit <= 0:
            if hidden_states.device.type == "npu":
                if torch_npu is None or not hasattr(torch_npu, "npu_swiglu"):
                    raise RuntimeError(
                        "The installed torch_npu build does not expose "
                        "npu_swiglu, which is required for the NPU shared expert."
                    )
                activated = torch_npu.npu_swiglu(gate_up)
            else:
                gate, up = gate_up.chunk(2, dim=-1)
                activated = F.silu(gate) * up
        else:
            # Preserve the configurable training clamp while still using the
            # fused NPU activation whenever it is available.
            gate, up = gate_up.chunk(2, dim=-1)
            limit = float(self.swiglu_limit)
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            if hidden_states.device.type == "npu":
                if torch_npu is None or not hasattr(torch_npu, "npu_swiglu"):
                    raise RuntimeError(
                        "The installed torch_npu build does not expose "
                        "npu_swiglu, which is required for the NPU shared expert."
                    )
                activated = torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1))
            else:
                activated = F.silu(gate) * up
        return self.down_proj(activated)

def _apply_fused_expert_activation(
    gate_up: torch.Tensor,
    swiglu_limit: Optional[float],
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit is None:
        return F.silu(gate) * up

    gate = F.silu(gate).clamp(max=float(swiglu_limit))
    up = up.clamp(
        min=-float(swiglu_limit),
        max=float(swiglu_limit),
    )
    return gate * up


def _grouped_moe_reference(
    sorted_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    expert_token_counts: torch.Tensor,
    swiglu_limit: Optional[float],
) -> torch.Tensor:
    """Differentiable grouped-MoE fallback used on CPU and CUDA."""
    outputs = []
    start = 0
    for expert_id, count in enumerate(expert_token_counts.tolist()):
        end = start + int(count)
        if end > start:
            gate_up = sorted_states[start:end].matmul(w13_weight[expert_id])
            activated = _apply_fused_expert_activation(
                gate_up,
                swiglu_limit,
            )
            outputs.append(activated.matmul(w2_weight[expert_id]))
        start = end
    if not outputs:
        return sorted_states.new_empty((0, w2_weight.shape[-1]))
    return torch.cat(outputs, dim=0)


class GMMFunction(torch.autograd.Function):
    """Autograd wrapper for non-quantized NPU grouped matmul."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
    ) -> torch.Tensor:
        assert torch_npu is not None
        ctx.save_for_backward(x, weight, group_list)
        return torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        assert torch_npu is not None
        input_tensor, weight, group_list = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight.transpose(1, 2)],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        grad_weight = torch_npu.npu_grouped_matmul(
            [input_tensor.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        return grad_input, grad_weight, None


class DSparkMoE(nn.Module):
    """Training implementation of the DeepSeek-V4 routed + shared MoE.

    This mirrors ``DeepseekV3MoE`` in cann-recipes-infer. Routed experts use
    fused 3D weights so the NPU path can execute both projections with
    autograd-wrapped ``torch_npu.npu_grouped_matmul``:

    - ``gate.weight`` and ``gate.e_score_correction_bias``
    - ``w13_weight`` with shape ``[experts, hidden, 2 * intermediate]``
    - ``w2_weight`` with shape ``[experts, intermediate, hidden]``
    - ``shared_experts.gate_up_proj/down_proj.weight``
    """

    def __init__(self, config, prefix: str = "", layer_idx: int = 0):
        super().__init__()
        _ = prefix  # retained for interface parity with the inference model
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(
            getattr(config, "moe_intermediate_size", config.hidden_size * 4)
        )
        self.n_routed_experts = int(getattr(config, "n_routed_experts", 256))
        self.num_experts = self.n_routed_experts
        self.top_k = int(getattr(config, "num_experts_per_tok", 6))
        self.num_experts_per_tok = self.top_k
        self.n_shared_experts = int(getattr(config, "n_shared_experts", 1))
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        self.scoring_func = str(getattr(config, "scoring_func", "softmax"))
        self.topk_method = str(getattr(config, "topk_method", "greedy"))
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))
        self.layer_idx = int(layer_idx)
        self.hash = self.layer_idx < int(getattr(config, "num_hash_layers", 0))
        swiglu_limit = getattr(config, "swiglu_limit", None)
        self.swiglu_limit = (
            None
            if swiglu_limit is None or float(swiglu_limit) <= 0
            else float(swiglu_limit)
        )

        assert 0 < self.top_k <= self.n_routed_experts, (
            "num_experts_per_tok must be in [1, n_routed_experts], got "
            f"{self.top_k} and {self.n_routed_experts}."
        )
        assert self.n_shared_experts in (0, 1), (
            "The DeepSeek-V4 DSpark training path currently supports zero or "
            f"one shared expert, got {self.n_shared_experts}."
        )

        self.gate = nn.Linear(
            self.hidden_size,
            self.n_routed_experts,
            bias=False,
        )
        # Match the reference checkpoint location. The correction bias affects
        # expert selection only; selected expert weights come from raw scores.
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.n_routed_experts, dtype=torch.float32)
        )

        self.w13_weight = nn.Parameter(
            torch.empty(
                self.n_routed_experts,
                self.hidden_size,
                2 * self.intermediate_size,
            )
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.n_routed_experts,
                self.intermediate_size,
                self.hidden_size,
            )
        )
        self._reset_fused_expert_parameters()
        self.shared_experts = (
            DSparkSharedExpert(
                self.hidden_size,
                self.intermediate_size,
                swiglu_limit=self.swiglu_limit,
            )
            if self.n_shared_experts > 0
            else None
        )

        # Hash routing is used only by the first target-model layers. DSpark
        # layers are numbered after the target backbone and therefore normally
        # use learned soft routing. Keep hash support for architectural parity.
        if self.hash:
            vocab_size = int(config.vocab_size)
            self.register_buffer(
                "tid2eid",
                torch.randint(
                    high=self.n_routed_experts,
                    size=(vocab_size, self.top_k),
                    dtype=torch.int32,
                ),
                persistent=True,
            )
        else:
            self.tid2eid = None

    def _reset_fused_expert_parameters(self) -> None:
        for expert_id in range(self.n_routed_experts):
            nn.init.kaiming_uniform_(
                self.w13_weight[expert_id].transpose(0, 1),
                a=5**0.5,
            )
            nn.init.kaiming_uniform_(
                self.w2_weight[expert_id].transpose(0, 1),
                a=5**0.5,
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        w13_key = f"{prefix}w13_weight"
        w2_key = f"{prefix}w2_weight"
        legacy_w1_key = f"{prefix}experts.0.w1.weight"
        if w13_key not in state_dict and legacy_w1_key in state_dict:
            fused_w13 = []
            fused_w2 = []
            legacy_keys = []
            for expert_id in range(self.n_routed_experts):
                base = f"{prefix}experts.{expert_id}"
                w1_name = f"{base}.w1.weight"
                w2_name = f"{base}.w2.weight"
                w3_name = f"{base}.w3.weight"
                fused_w13.append(
                    torch.cat(
                        [state_dict[w1_name], state_dict[w3_name]],
                        dim=0,
                    ).transpose(0, 1)
                )
                fused_w2.append(state_dict[w2_name].transpose(0, 1))
                legacy_keys.extend((w1_name, w2_name, w3_name))
            state_dict[w13_key] = torch.stack(fused_w13, dim=0)
            state_dict[w2_key] = torch.stack(fused_w2, dim=0)
            for key in legacy_keys:
                state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _compute_scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.scoring_func == "sigmoid":
            return logits.sigmoid()
        if self.scoring_func == "softmax":
            return logits.softmax(dim=-1, dtype=torch.float32)
        if self.scoring_func == "sqrtsoftplus":
            return F.softplus(logits).sqrt()
        raise NotImplementedError(
            f"Unsupported MoE scoring function: {self.scoring_func!r}"
        )

    def _select_experts(
        self,
        logits: torch.Tensor,
        input_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self._compute_scores(logits)
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores,
                k=self.top_k,
                dim=-1,
                sorted=False,
            )
        elif self.topk_method == "noaux_tc":
            if self.hash:
                assert input_ids is not None, (
                    "input_ids is required for hash-routed MoE layers."
                )
                topk_idx = self.tid2eid[input_ids.reshape(-1).long()].long()
            else:
                selection_scores = (
                    scores
                    + self.gate.e_score_correction_bias.to(
                        device=scores.device,
                        dtype=scores.dtype,
                    ).unsqueeze(0)
                )
                _, topk_idx = torch.topk(
                    selection_scores,
                    k=self.top_k,
                    dim=-1,
                    sorted=False,
                )
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(
                f"Unsupported MoE top-k method: {self.topk_method!r}"
            )

        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor

    def _forward_routed_experts(
        self,
        flat_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = flat_states.shape[0]
        flat_expert_idx = topk_idx.reshape(-1).to(torch.int64)
        flat_token_idx = torch.arange(
            num_tokens,
            device=flat_states.device,
            dtype=torch.int64,
        ).unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        # NPU ArgSort sends int32/int64 keys to AiCPU. Expert ids are small
        # exact integers in float32, so sorting a float key stays on AiCore.
        sorted_order = torch.argsort(flat_expert_idx.to(torch.float32))
        sorted_expert_idx = flat_expert_idx.index_select(0, sorted_order)
        sorted_token_idx = flat_token_idx.index_select(0, sorted_order)
        sorted_states = flat_states.index_select(0, sorted_token_idx).contiguous()
        expert_token_counts = torch.bincount(
            sorted_expert_idx,
            minlength=self.n_routed_experts,
        ).to(dtype=torch.int64)

        use_npu_gmm = flat_states.device.type == "npu"
        if use_npu_gmm:
            if torch_npu is None or not hasattr(
                torch_npu,
                "npu_grouped_matmul",
            ):
                raise RuntimeError(
                    "The installed torch_npu build does not expose "
                    "npu_grouped_matmul, which is required for NPU MoE training."
                )
            gate_up = GMMFunction.apply(
                sorted_states,
                self.w13_weight,
                expert_token_counts,
            )
            if self.swiglu_limit is None:
                activated = torch_npu.npu_swiglu(gate_up, dim=-1)
            else:
                activated = _apply_fused_expert_activation(
                    gate_up,
                    self.swiglu_limit,
                )
            sorted_output = GMMFunction.apply(
                activated,
                self.w2_weight,
                expert_token_counts,
            )
        else:
            sorted_output = _grouped_moe_reference(
                sorted_states,
                self.w13_weight,
                self.w2_weight,
                expert_token_counts,
                self.swiglu_limit,
            )

        sorted_weight = topk_weight.reshape(-1).index_select(
            0,
            sorted_order,
        ).to(dtype=sorted_output.dtype)
        routed_output = torch.zeros_like(flat_states).index_add(
            0,
            sorted_token_idx,
            sorted_output * sorted_weight.unsqueeze(-1),
        )
        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool = False,
        cur_topk_list=None,
        input_ids=None,
        shared_expert_stream=None,
    ):
        _ = (is_prefill, cur_topk_list, shared_expert_stream)
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        # The reference implementation evaluates gate matmul/scoring in FP32
        # on Atlas A3. Preserve that numerical behaviour during training.
        logits = F.linear(
            flat_states.float(),
            self.gate.weight.float(),
        )
        topk_idx, topk_weight = self._select_experts(logits, input_ids)
        output = self._forward_routed_experts(
            flat_states,
            topk_idx,
            topk_weight,
        )
        if self.shared_experts is not None:
            output = output + self.shared_experts(flat_states)
        return output.view(original_shape)


# ---------------------------------------------------------------------------
# Hyper-Connection — OpKernel dispatch (fused AscendC / PyPTO when available)
# ---------------------------------------------------------------------------


class _OpKernel:
    """Dispatcher matching ``cann-recipes-infer_dspark`` OpKernel HC API.

    Production: fused AscendC / PyPTO kernels (``npu_hc_pre``, ``npu_hc_post``).
    Fallback: pure-PyTorch native path (this file).  The signatures deliberately
    mirror ``models/deepseek-v4/models/modules/op_impls/mhc.py`` so that
    swapping in the real fused kernels requires only changing the backend.
    """

    @staticmethod
    def hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
        """Pre-sublayer HC: fold ``hc_mult``→1 via learned mixing + Sinkhorn.

        *x*: (bsz, seq_len, hc_mult, hidden_size).
        *hc_fn*: (mix_hc, hc_mult * hidden_size) where ``mix_hc = (2+hc)*hc``.
        *hc_scale*: (3,).  *hc_base*: (mix_hc,).
        Returns ``(y, post, comb)`` — *y* shape (bsz, seq_len, hidden_size).
        """
        return hc_pre_native(
            x, hc_fn, hc_scale, hc_base,
            hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps,
        )

    @staticmethod
    def hc_post(x, residual, post, comb):
        """Post-sublayer HC: expand 1→``hc_mult``.

        *x*: (bsz, seq_len, hidden_size).  *residual*: (bsz, seq_len, hc_mult, hidden_size).
        *post*: (bsz, seq_len, hc_mult).  *comb*: (bsz, seq_len, hc_mult, hc_mult).
        Returns (bsz, seq_len, hc_mult, hidden_size).
        """
        return hc_post_native(x, residual, post, comb)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Pure-PyTorch Sinkhorn decomposition — matches tilelang kernel
    ``hc_split_sinkhorn`` in cann-recipes.

    Reference: ``ops/tilelang/ds_v4/hc_split_sinkhorn.py`` and
    ``models/deepseek-v4/models/modules/op_impls/mhc.py``.
    """
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult)

    comb = comb.softmax(-1) + eps
    col_sum = comb.sum(-2, keepdim=True)
    comb = comb / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb


def hc_pre_native(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """``hc_pre`` native fallback — matches ``hc_pre_native`` in mhc.py."""
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(x_flat, hc_fn.float()) * rsqrt

    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps,
    )
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype), post, comb


def hc_post_native(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    """``hc_post`` native fallback — matches ``hc_post_native`` in mhc.py."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2,
    )
    return y.type_as(x)


def init_hc_parameters(
    fn_weight: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    mix_hc: int,
):
    """Initialize HC parameters with ``mix_hc``-sized tensors.

    *fn_weight*: (mix_hc, hc_mult * hidden) — mixing projection.
    *base*: (mix_hc,) — per-dim bias.
    *scale*: (3,) — per-group scale.
    """
    nn.init.normal_(fn_weight, mean=0.0, std=0.02)
    nn.init.zeros_(base)
    nn.init.ones_(scale)


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DeepSeekV4DSparkDecoderLayer(nn.Module):
    """One DSpark proposal layer with first/last-stage responsibilities.

    The first layer projects and normalizes the concatenated target hidden
    states. The last layer folds HC streams, normalizes the LM representation,
    and owns the Markov/confidence heads. This mirrors the native DSpark stage
    layout while keeping the implementation free of inference-only KV-cache,
    quantization, and tensor-parallel machinery.
    """

    def __init__(self, config, layer_idx: int, stage_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.stage_idx = layer_idx if stage_idx is None else stage_idx
        self.dspark_num_layers = getattr(config, "dspark_num_layers", config.num_hidden_layers)
        
        self.is_first = self.stage_idx == 0
        self.is_last = self.stage_idx == self.dspark_num_layers - 1
        self.hidden_size = config.hidden_size
        self.hc_mult = getattr(config, "hc_mult", 1)
        self.hc_sinkhorn_iters = getattr(config, "hc_sinkhorn_iters", 20)
        self.hc_eps = getattr(config, "hc_eps", 1e-6)
        self.norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.attn = DeepSeekV4DSparkSDPAAttention(config, layer_idx=layer_idx)
        global_layer_idx = int(
            getattr(config, "num_target_layers", config.num_hidden_layers)
        ) + int(layer_idx)
        self.mlp = DSparkMoE(
            config,
            prefix=f"mtp.{self.stage_idx}.ffn",
            layer_idx=global_layer_idx,
        )
        self.attn_norm = RMSNorm(self.hidden_size, eps=self.norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, eps=self.norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))
        torch.set_default_dtype(origin_dtype)
        init_hc_parameters(self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale, mix_hc)
        init_hc_parameters(self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale, mix_hc)

        if self.is_first:
            target_layer_num = len(config.target_layer_ids)
            self.main_proj = nn.Linear(
                target_layer_num * self.hidden_size,
                self.hidden_size,
                bias=False,
            )
            self.main_norm = RMSNorm(self.hidden_size, eps=self.norm_eps)

        if self.is_last:
            self.norm = RMSNorm(self.hidden_size, eps=self.norm_eps)
            self.markov_head = build_markov_head(config)
            self.enable_confidence_head = bool(
                getattr(config, "enable_confidence_head", False)
            )
            self.confidence_head_with_markov = self.enable_confidence_head and bool(
                getattr(config, "confidence_head_with_markov", False)
            )
            if self.enable_confidence_head and self.confidence_head_with_markov:
                assert self.markov_head is not None
            confidence_dim = self.hidden_size
            if self.confidence_head_with_markov:
                confidence_dim += config.markov_rank
            self.confidence_head = (
                AcceptRatePredictor(confidence_dim, bias=False)
                if self.enable_confidence_head
                else None
            )
            if self.hc_mult > 1:
                self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim))
                self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult))
                self.hc_head_scale = nn.Parameter(torch.ones(1))
                nn.init.normal_(self.hc_head_fn, mean=0.0, std=0.02)
                nn.init.zeros_(self.hc_head_base)
            else:
                self.hc_head_fn = self.hc_head_base = self.hc_head_scale = None

    def project_main_hidden(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.is_first:
            raise ValueError("Only the first DSpark layer owns main_proj/main_norm.")
        return self.main_norm(self.main_proj(target_hidden_states))

    def fold_hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.is_last:
            raise ValueError("Only the last DSpark layer owns the HC head mixer.")
        if self.hc_mult <= 1:
            return hidden_states
        original_dtype = hidden_states.dtype
        flat = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_head_fn.float()) * rsqrt
        pre = torch.sigmoid(
            mixes * self.hc_head_scale.float() + self.hc_head_base.float()
        ) + self.hc_eps
        folded = torch.sum(
            pre.unsqueeze(-1)
            * flat.view(flat.shape[0], flat.shape[1], self.hc_mult, self.hidden_size),
            dim=2,
        )
        return folded.to(original_dtype)

    def forward_head_hidden(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_last:
            raise ValueError("Only the last DSpark layer owns the output heads.")
        confidence_hidden = self.fold_hc_head(hidden_states)
        return confidence_hidden, self.norm(confidence_hidden)

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        lm_head: Optional[nn.Module] = None,
        head_shape: Optional[tuple[int, int, int]] = None,
        prev_token_ids: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # Stage-owned parameters must be consumed inside this method. FSDP
        # auto-wraps decoder layers and only exposes their full parameters for
        # the duration of the wrapped layer's forward call.
        projected_target_hidden = None
        if self.is_first:
            target_hidden_states = self.project_main_hidden(target_hidden_states)
            projected_target_hidden = target_hidden_states

        residual = hidden_states
        hidden_states, post, comb = _OpKernel.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps,
        )
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = _OpKernel.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = _OpKernel.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps,
        )
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _OpKernel.hc_post(hidden_states, residual, post, comb)

        confidence_hidden = None
        output_hidden = None
        draft_logits = None
        confidence_pred = None
        if self.is_last:
            confidence_hidden, output_hidden = self.forward_head_hidden(hidden_states)
            if head_shape is not None:
                if lm_head is None or prev_token_ids is None:
                    raise ValueError(
                        "lm_head and prev_token_ids are required with head_shape."
                    )
                output_hidden_4d = output_hidden.reshape(*head_shape, -1)
                confidence_hidden_4d = confidence_hidden.reshape(*head_shape, -1)
                draft_logits = lm_head(output_hidden_4d)
                if self.markov_head is not None:
                    draft_logits = self.markov_head.apply_block_logits(
                        draft_logits,
                        token_ids=prev_token_ids,
                        hidden_states=output_hidden_4d,
                    )

                if self.confidence_head is not None:
                    confidence_features = confidence_hidden_4d
                    if self.confidence_head_with_markov:
                        prev_embeddings = self.markov_head.get_prev_embeddings(
                            prev_token_ids
                        ).to(dtype=confidence_hidden_4d.dtype)
                        confidence_features = torch.cat(
                            [confidence_hidden_4d, prev_embeddings],
                            dim=-1,
                        )
                    confidence_pred = self.confidence_head(
                        confidence_features
                    ).float()

        if head_shape is not None:
            # Training only needs the predictions. Avoid retaining additional
            # references to the large final-layer activation graph.
            return (
                None,
                projected_target_hidden,
                None,
                None,
                draft_logits,
                confidence_pred,
            )
        return (
            hidden_states,
            projected_target_hidden,
            confidence_hidden,
            output_hidden,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# DSpark Model
# ---------------------------------------------------------------------------

class DeepSeekV4DSparkModel(PreTrainedModel):
    """Standalone DSpark draft model for DeepSeek-V4 Flash.

    Follows the Ascend DeepSeek-V4-Flash DSpark layout while keeping the
    training path in PyTorch/SDPA instead of NPU custom inference kernels.

    Interface-compatible with :class:`Qwen3DSparkModel` so the existing
    trainer and evaluator can use it without changes.
    """

    _no_split_modules = ["DeepSeekV4DSparkDecoderLayer"]
    config_class = DeepseekV4Config
    _supports_sdpa = True
    _supports_flash_attention_2 = False
    _supports_flex_attention = False
    _is_stateful = False

    def __init__(self, config):
        super().__init__(config)

        # Validate required fields.
        for field in (
            "target_layer_ids", "mask_token_id", "num_anchors",
            "enable_confidence_head", "markov_rank",
        ):
            assert hasattr(config, field), f"config.{field} must be provided."
        if int(config.markov_rank) > 0:
            assert hasattr(config, "markov_head_type")

        self.target_layer_ids = config.target_layer_ids
        self.hidden_size = config.hidden_size
        self.hc_mult = getattr(config, "hc_mult", 1)
        num_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList([
            DeepSeekV4DSparkDecoderLayer(config, layer_idx)
            for layer_idx in range(num_layers)
        ])
        self.rotary_emb = DeepSeekV4RotaryEmbedding(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)
        self.hc_eps = float(getattr(config, "hc_eps", 1e-6))

        self.post_init()

    @property
    def first_layer(self) -> DeepSeekV4DSparkDecoderLayer:
        return self.layers[0]

    @property
    def last_layer(self) -> DeepSeekV4DSparkDecoderLayer:
        return self.layers[-1]

    @property
    def markov_head(self):
        return self.last_layer.markov_head

    @property
    def confidence_head(self):
        return self.last_layer.confidence_head

    @property
    def confidence_head_with_markov(self) -> bool:
        return self.last_layer.confidence_head_with_markov

    def save_pretrained(self, save_directory, **kwargs):
        """Save with original architecture name, stripping quantization_config."""
        cfg = self.config
        saved_arch = getattr(cfg, "architectures", None)
        had_quant = hasattr(cfg, "quantization_config")
        saved_quant = cfg.quantization_config if had_quant else None
        try:
            cfg.architectures = ["DeepseekV4ForCausalLM"]
            if had_quant:
                del cfg.quantization_config
            super().save_pretrained(save_directory, **kwargs)
        finally:
            if saved_arch is not None:
                cfg.architectures = saved_arch
            if had_quant:
                cfg.quantization_config = saved_quant

    # -- weight init helpers (same interface as Qwen3DSparkModel) ----------

    def initialize_embeddings_and_head(self, *, embed_tokens, lm_head, freeze=True):
        if isinstance(embed_tokens, torch.Tensor):
            embed_weight = embed_tokens
        else:
            embed_weight = embed_tokens.weight
        if isinstance(lm_head, torch.Tensor):
            head_weight = lm_head
        else:
            head_weight = lm_head.weight
        assert self.embed_tokens.weight.shape == embed_weight.shape
        assert self.lm_head.weight.shape == head_weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_weight.detach())
            self.lm_head.weight.copy_(head_weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    # -- forward pass ------------------------------------------------------

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def fold_hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fold HC streams using the final DSpark layer's head mixer."""
        return self.last_layer.fold_hc_head(hidden_states)

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert base_logits.ndim == 2
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits, token_ids=prev_token_ids, hidden_states=hidden_states,
            )
        sampled = sample_tokens(step_logits.unsqueeze(1), temperature=temperature).squeeze(1)
        return sampled, step_logits

    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        return_confidence_hidden: bool = False,
        head_shape: Optional[tuple[int, int, int]] = None,
        prev_token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = noise_embedding
        position_embeddings = self.rotary_emb(
            position_ids, dtype=hidden_states.dtype, device=hidden_states.device,
        )

        # Expand to hyper-connection multiplicity.
        if self.hc_mult > 1:
            hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        confidence_hidden = None
        output_hidden = None
        draft_logits = None
        confidence_pred = None
        last_layer_idx = len(self.layers) - 1
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == last_layer_idx
            (
                hidden_states,
                projected_target_hidden,
                layer_confidence_hidden,
                layer_output_hidden,
                layer_draft_logits,
                layer_confidence_pred,
            ) = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                lm_head=(
                    self.lm_head
                    if is_last_layer and head_shape is not None
                    else None
                ),
                head_shape=head_shape if is_last_layer else None,
                prev_token_ids=prev_token_ids if is_last_layer else None,
                **kwargs,
            )
            if hidden_states is None and not is_last_layer:
                raise RuntimeError("A non-final DSpark layer returned no hidden states.")
            if projected_target_hidden is not None:
                target_hidden_states = projected_target_hidden
            if is_last_layer:
                confidence_hidden = layer_confidence_hidden
                output_hidden = layer_output_hidden
                draft_logits = layer_draft_logits
                confidence_pred = layer_confidence_pred

        if head_shape is not None:
            if draft_logits is None:
                raise RuntimeError("The final DSpark layer did not produce draft logits.")
            return draft_logits, confidence_pred
        if confidence_hidden is None or output_hidden is None:
            raise RuntimeError("The final DSpark layer did not produce head hidden states.")
        if return_confidence_hidden:
            return confidence_hidden, output_hidden
        return output_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> DSparkForwardOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )
        context_position_ids = torch.arange(
            seq_len, device=device, dtype=torch.long,
        ).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = create_position_ids(anchor_positions, self.block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        dspark_attn_mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            device=device,
            attn_implementation=self.config._attn_implementation,
            sliding_window=getattr(self.config, "sliding_window", None),
        )
        num_blocks = anchor_positions.size(1)
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(
            input_ids,
            1,
            anchor_positions,
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits, confidence_pred = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
            head_shape=(bsz, num_blocks, self.block_size),
            prev_token_ids=prev_token_ids,
        )

        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1, anchor_positions.size(1), -1, -1,
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1, -1, -1, target_last_hidden_states.shape[-1],
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)

        log_sampler_stats(
            seq_len=seq_len,
            loss_mask=loss_mask,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
        )

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )


__all__ = [
    "DeepSeekV4DSparkSDPAAttention",
    "RMSNorm",
    "DSparkSharedExpert",
    "DSparkMoE",
    "DeepSeekV4DSparkModel",
    "DeepSeekV4DSparkDecoderLayer",
    "DeepSeekV4MLAttention",
]

# Backward-compatible aliases.
DSparkAttention = DeepSeekV4DSparkSDPAAttention
DeepSeekV4MLAttention = DeepSeekV4DSparkSDPAAttention
