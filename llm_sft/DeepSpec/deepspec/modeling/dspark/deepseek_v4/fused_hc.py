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

"""Fused AscendC hyper-connection kernel replacements.

Hot-swaps :class:`_OpKernel` so that ``hc_pre`` / ``hc_post`` dispatch to
``cann_ops_transformer`` fused kernels instead of the pure-PyTorch fallback.

Reference
---------
* ``mhc_pre_sinkhorn`` — ``ops-transformer/torch_extension/cann_ops_transformer/docs/zh/mhc_pre_sinkhorn.md``
* ``mhc_post`` — ``ops-transformer/torch_extension/cann_ops_transformer/docs/zh/mhc_post.md``

Usage (in training / inference entry-point)::

    from deepspec.modeling.dspark.deepseek_v4.fused_hc import install_fused_hc_kernels
    install_fused_hc_kernels("ascendc")   # enable AscendC
    install_fused_hc_kernels("native")    # fall back to pure PyTorch
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# AscendC fused kernels (cann_ops_transformer)
# ---------------------------------------------------------------------------

def _make_hc_pre_ascendc(out_flag: bool = True):
    """Factory: import cann_ops_transformer and return ``hc_pre`` closure.

    Args:
        out_flag: ``True`` saves intermediate Sinkhorn results for backward
            (training); ``False`` skips them (inference).
    """
    from cann_ops_transformer.ops import mhc_pre_sinkhorn

    def _hc_pre_ascendc(
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        hc_sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AscendC ``mhc_pre_sinkhorn`` — matches ``_OpKernel.hc_pre`` signature.

        Native → AscendC name mapping::

            x → x        (B,S,N,C) bf16
            hc_fn → phi  (N²+2N, N·C) fp32
            hc_scale → alpha  (3,) fp32
            hc_base → bias   (N²+2N,) fp32
            hc_mult → hcMult  (int)
            hc_sinkhorn_iters → numIters  (int)
            norm_eps → normEps  (float)
            hc_eps → hcEps  (float)

        Returns are remapped to match the native ``(y, post, comb)`` layout.
        """
        hin, h_post, h_res = mhc_pre_sinkhorn(
            x,
            hc_fn,          # phi
            hc_scale,       # alpha
            hc_base,        # bias
            hc_mult,        # hcMult
            hc_sinkhorn_iters,  # numIters
            hc_eps,         # hcEps
            norm_eps,       # normEps
            out_flag,       # outFlag
        )
        # h_res is (B, S, N*N) — unflatten to match native comb (B, S, N, N).
        comb = h_res.unflatten(-1, (hc_mult, hc_mult))
        return hin, h_post, comb

    return _hc_pre_ascendc


def _make_hc_post_ascendc():
    """Factory: import cann_ops_transformer and return ``hc_post`` closure."""
    from cann_ops_transformer.ops import mhc_post

    def _hc_post_ascendc(
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """AscendC ``mhc_post`` — matches ``_OpKernel.hc_post`` signature.

        Native → AscendC name mapping::

            residual → x       (B,S,N,D) bf16   — residual flows
            comb → h_res       (B,S,N,N) fp32    — Sinkhorn matrix
            x → h_out          (B,S,D) bf16      — sublayer output
            post → h_post      (B,S,N) fp32      — post-scale vector
        """
        return mhc_post(
            residual,   # x      — (B,S,N,D)
            comb,       # h_res  — (B,S,N,N)
            x,          # h_out  — (B,S,D)
            post,       # h_post — (B,S,N)
        )

    return _hc_post_ascendc


# ---------------------------------------------------------------------------
# PyPTO compiled kernels
# ---------------------------------------------------------------------------

def _make_hc_pre_pypto():
    """Import and return the PyPTO ``hc_pre`` kernel (eager import at install-time)."""
    from ops.pypto_python.impl.hc_pre_pypto import hc_pre_pypto

    def _hc_pre_pypto(
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        hc_sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, hc, d = x.shape
        x_2d = x.reshape(b * s, hc, d)
        y, post, comb = hc_pre_pypto(
            x_2d, hc_fn, hc_scale, hc_base,
            hc_mult, hc_sinkhorn_iters, hc_eps,
        )
        y = y.reshape(b, s, d)
        post = post.reshape(b, s, hc)
        comb = comb.reshape(b, s, hc, hc)
        return y, post, comb

    return _hc_pre_pypto


# ---------------------------------------------------------------------------
# Install / restore
# ---------------------------------------------------------------------------

def install_fused_hc_kernels(backend: str = "ascendc", *, out_flag: bool = True) -> None:
    """Hot-swap ``_OpKernel.hc_pre`` and ``_OpKernel.hc_post``.

    Args:
        backend:
            * ``"ascendc"`` — ``cann_ops_transformer.mhc_pre_sinkhorn`` / ``mhc_post``
            * ``"pypto"``  — PyPTO tilelang kernels (hc_pre only; hc_post stays native)
            * ``"native"`` — restore the pure-PyTorch fallback
        out_flag:
            Only used with ``backend="ascendc"``.  ``True`` saves intermediate
            Sinkhorn results for backward (training); ``False`` skips them
            (inference).
    """
    from deepspec.modeling.dspark.deepseek_v4.modeling import (
        _OpKernel,
        hc_post_native,
        hc_pre_native,
    )

    backend = backend.lower()

    if backend == "ascendc":
        try:
            _hc_pre_fn = _make_hc_pre_ascendc(out_flag=out_flag)
            _hc_post_fn = _make_hc_post_ascendc()
        except ImportError as e:
            raise RuntimeError(
                f"AscendC HC kernels are not available — {e}"
            ) from e
        _OpKernel.hc_pre = staticmethod(_hc_pre_fn)
        _OpKernel.hc_post = staticmethod(_hc_post_fn)

    elif backend == "pypto":
        try:
            _hc_pre_pypto_fn = _make_hc_pre_pypto()
        except ImportError as e:
            raise RuntimeError(
                f"PyPTO HC kernel is not available — {e}"
            ) from e
        _OpKernel.hc_pre = staticmethod(_hc_pre_pypto_fn)
        _OpKernel.hc_post = staticmethod(hc_post_native)

    elif backend == "native":
        _OpKernel.hc_pre = staticmethod(hc_pre_native)
        _OpKernel.hc_post = staticmethod(hc_post_native)

    else:
        raise ValueError(
            f"Unknown HC backend {backend!r}; expected 'ascendc', 'pypto', or 'native'"
        )


__all__ = ["install_fused_hc_kernels"]
