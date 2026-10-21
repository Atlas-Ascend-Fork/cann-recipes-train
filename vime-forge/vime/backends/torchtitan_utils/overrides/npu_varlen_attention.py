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

"""NPU varlen attention: an inner attention that calls npu_fusion_attention in TND layout.

Upstream's ``VarlenAttention`` goes through ``torch.nn.attention.varlen.varlen_attn`` ->
``aten::_flash_attention_forward``, for which torch_npu registers no kernel, so it raises
``NotImplementedError``. Named in the recipe's ``_VIME_OVERRIDES``.

torchtitan_npu has no general replacement to defer to: its master (2026-09-04) deleted the generic
``models/common/npu_varlen_attention.NPUVarlenAttention`` and keeps only ``override/qwen3_5``, which
is bound to context parallelism. Delete this module if a general one comes back.

- q/k/v arrive as [B,L,N,H] on a torchtitan before #4121 and as [T,H,K] after it; the kernel takes
  TND, so a 4-D input is flattened and the output restored to the shape that came in. The parameter
  names are the pre-#4121 ones and must follow upstream's, see the note on the @override below.
- ``actual_seq_*`` are cumulative END positions; ``VarlenMetadata.cu_seq_*`` carries a leading 0.
- Causal: sparse_mode=3 with a 2048x2048 upper-triangular mask, also correct past 2048 tokens.
"""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata

_MASK_SIZE = 2048
_SPARSE_MODE_CAUSAL = 3

_CAUSAL_MASK_CACHE: dict[str, torch.Tensor] = {}


def _causal_mask(device) -> torch.Tensor:
    key = str(device)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.triu(
            torch.ones(_MASK_SIZE, _MASK_SIZE, device=device, dtype=torch.bool), diagonal=1
        )
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


class NPUVarlenAttention(VarlenAttention):
    """Per-document causal attention through CANN's fused attention kernel, TND layout."""

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if tuple(self.window_size) != (-1, 0):
            raise NotImplementedError(
                f"NPU varlen attention implements causal attention only, got "
                f"window_size={self.window_size}. A sliding window needs a different sparse_mode "
                "and pre/next_tokens, which is not wired up."
            )

    def forward(
        self,
        # The names are load-bearing: parallelize reads them off this signature and looks them up in
        # the sharding config's in_dst_shardings (protocols/module.py:_cache_pos_arg_names).
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        out_transform=None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs  # enable_gqa: TND handles grouped queries natively, kv heads are not expanded
        assert isinstance(attention_masks, VarlenMetadata), (
            f"attention_masks must be a VarlenMetadata, got {type(attention_masks).__name__}"
        )
        assert out_transform is None, (
            "NPU varlen attention has no out_transform hook; a model that rescales attention "
            "outputs (gpt_oss sinks) would silently lose it."
        )
        assert q_BLNH.dim() in (3, 4), f"expected [B,L,N,H] or [T,N,H], got {tuple(q_BLNH.shape)}"

        out_shape = q_BLNH.shape
        head_dim = q_BLNH.shape[-1]
        n_heads = q_BLNH.shape[-2]
        input_dtype = q_BLNH.dtype
        # The kernel takes TND and bf16/fp16 only: wider inputs are downcast and the output cast
        # back, and a [B,L,N,H] input is flattened to [B*L,N,H].
        query, key, value = (
            t.reshape(-1, t.shape[-2], head_dim).to(torch.bfloat16).contiguous()
            for t in (q_BLNH, k_BLNH, v_BLNH)
        )

        cu_seq_q = attention_masks.cu_seq_q.to(torch.int64)
        cu_seq_k = attention_masks.cu_seq_k.to(torch.int64)
        actual_seq_qlen = cu_seq_q[1:].tolist() if int(cu_seq_q[0]) == 0 else cu_seq_q.tolist()
        actual_seq_kvlen = cu_seq_k[1:].tolist() if int(cu_seq_k[0]) == 0 else cu_seq_k.tolist()

        out = torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            n_heads,
            "TND",
            atten_mask=_causal_mask(query.device),
            scale=float(scale) if scale is not None else head_dim**-0.5,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            sparse_mode=_SPARSE_MODE_CAUSAL,
        )[0]  # a tuple; index 0 is the attention output [T, N, H]
        # reshape, not view: the kernel does not guarantee a contiguous output.
        return out.reshape(out_shape).to(input_dtype)


@override(
    target=VarlenAttention.Config,
    # Only the decoder's inner attention; a VarlenAttention.Config elsewhere is not covered. The
    # fqn is rooted at Trainer.Config, hence the model_spec.model prefix.
    fqns=["model_spec.model.layers.*.attention.inner_attention"],
    exact=True,
    description="NPU varlen attention (torch_npu.npu_fusion_attention, TND)",
)
def npu_varlen(cfg: VarlenAttention.Config) -> NPUVarlenAttention.Config:
    return derive(cfg, NPUVarlenAttention.Config)
