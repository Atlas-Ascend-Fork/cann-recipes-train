"""DSV4 whitelist weight export with optional routed-expert MXFP4 QAT."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

import torch

# HF names emitted by DeepSeekV4StateDictAdapter.to_hf(). Keep this list in
# lockstep with interfaces/torchao_converter.py in torchtitan-npu. The mix
# recipe applies MXFP8 to attention/shared experts and Block FP8 to routed
# experts (MXFP4 when routed-expert QAT is enabled). MTP e_proj/h_proj remain
# absent from the TorchTitan quantization suffix allowlist; MTP attention/MoE projections are
# covered by the same suffix rules as the main layers.
_MXFP8_HF_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(?:model\.)?(?:layers|mtp)\.\d+\.attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)\.weight$",
        r"^(?:model\.)?(?:layers|mtp)\.\d+\.attn\.indexer\.wq_b\.weight$",
        r"^(?:model\.)?(?:layers|mtp)\.\d+\.ffn\.shared_experts\.(?:w1|w2|w3)\.weight$",
    )
)
_BLOCK_FP8_HF_PATTERNS = (
    re.compile(
        r"^(?:model\.)?(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.(?:w1|w2|w3)\.weight$"
    ),
)

# These names are emitted by DeepSeekV4StateDictAdapter.to_hf(). They cover
# the HF-facing form of _dsv4_fp32_overrides(), plus FP32 model state used by
# the router (expert bias and hash tid2eid). Keep them separate from the
# quantization allowlist: FP32 wins before recipe classification.
_DSV4_FP32_EXPORT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(?:model\.)?layers\.\d+\.hc_(?:attn|ffn)_(?:base|fn|scale)$",
        r"^(?:model\.)?layers\.\d+\.hc_head_(?:base|fn|scale)$",
        r"^(?:model\.)?mtp\.\d+\.hc_(?:attn|ffn)_(?:base|fn|scale)$",
        r"^(?:model\.)?mtp\.\d+\.hc_head_(?:base|fn|scale)$",
        r"^(?:model\.)?hc_head_(?:base|fn|scale)$",
        r"^(?:model\.)?(?:layers|mtp)\.\d+\.ffn\.gate\.(?:weight|bias)$",
        r"^(?:model\.)?(?:layers|mtp)\.\d+\.ffn\.gate\.tid2eid$",
        r"^(?:model\.)?(?:layers|mtp)\.\d+\.attn\.attn_sink$",
        r"^(?:model\.)?layers\.\d+\.attn\.compressor\.ape$",
        r"^(?:model\.)?layers\.\d+\.attn\.indexer\.compressor\.ape$",
        r"^(?:model\.)?mtp\.\d+\.attn\.compressor\.ape$",
        r"^(?:model\.)?mtp\.\d+\.attn\.indexer\.compressor\.ape$",
    )
)


def is_dsv4_fp32_export(name: str) -> bool:
    """Return whether a DSV4 HF tensor must remain FP32 for export."""
    return any(pattern.fullmatch(name) for pattern in _DSV4_FP32_EXPORT_PATTERNS)


def dsv4_export_dtype(
    name: str, default_dtype: torch.dtype, *, recipe: str = "mix", mxfp4_config=None,
) -> torch.dtype | None:
    """Resolve the temporary DSV4 export dtype from the HF parameter name."""
    if mxfp4_config is not None and quantization_kind(name, recipe, enable_mxfp4=True) == "mxfp4":
        # QAT quantizes the master weight; casting to BF16 first changes its bits.
        return None
    return torch.float32 if is_dsv4_fp32_export(name) else default_dtype


def get_mxfp4_export_config(quantization_config):
    """Use the same config class and resolved options as TorchTitan's converter."""
    if not getattr(quantization_config, "enable_mxfp4_qat", False) or quantization_config.recipe == "all_mxfp8":
        return None
    from torchao_npu.quantization.quant_configs import MXQuantizeConfig

    return MXQuantizeConfig(
        elem_dtype=torch.float4_e2m1fn_x2,
        dst_type_max=quantization_config.dst_type_max,
    )


def validate_dsv4_export_config(quantization_config) -> None:
    """Reject modes not implemented by the temporary whitelist exporter."""
    if getattr(quantization_config, "fsdp_prequantize", False):
        raise ValueError(
            "The temporary DSV4 whitelist weight exporter does not support "
            "extension.quantization.fsdp_prequantize=True; set it to False."
        )


def validate_rollout_quantization_config(quantization_config=None) -> None:
    """Validate the resolved training configuration used for rollout export."""
    if not getattr(quantization_config, "enable_quantized_training", False):
        return
    recipe = getattr(quantization_config, "recipe", None)
    if recipe not in {"mix", "all_mxfp8", "all_block_fp8"}:
        raise ValueError(f"Unsupported DSV4 rollout quantization recipe: {recipe!r}")
    validate_dsv4_export_config(quantization_config)


def quantization_kind(name: str, recipe: str, *, enable_mxfp4: bool = False) -> str | None:
    """Classify one final HF weight name using TorchAO-NPU's DSV4 allowlist."""
    if recipe not in {"mix", "all_mxfp8", "all_block_fp8"}:
        raise ValueError(f"Unsupported DSV4 rollout quantization recipe: {recipe!r}")
    if is_dsv4_fp32_export(name):
        return None
    is_dense = any(pattern.fullmatch(name) for pattern in _MXFP8_HF_PATTERNS)
    is_routed = any(pattern.fullmatch(name) for pattern in _BLOCK_FP8_HF_PATTERNS)
    if not is_dense and not is_routed:
        return None
    if recipe == "all_mxfp8":
        return "mxfp8"
    if is_routed and enable_mxfp4:
        return "mxfp4"
    if recipe == "all_block_fp8":
        return "block_fp8"
    if is_dense:
        return "mxfp8"
    if is_routed:
        return "block_fp8"
    return None


def _mx_quant(tensor: torch.Tensor):
    """Produce vLLM's HF [N,K] MXFP8 weight/scale from a Linear weight."""
    import torch_npu

    # TorchAO quantizes B=[K,N] with the dual-axis primitive and consumes
    # B_q2/B_s2. Transposing those results back gives vLLM's pre-transform
    # checkpoint layout: weight [N,K], scale [N,ceil(K/32)].
    op = getattr(torch_npu, "npu_dynamic_mx_quant_with_dual_axis", None)
    if op is None:
        raise RuntimeError("TorchAO-NPU export requires npu_dynamic_mx_quant_with_dual_axis")
    result = op(
        tensor.transpose(-2, -1).contiguous(),
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        scale_alg=1,
        dst_type_max=0.0,
    )
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise RuntimeError("npu_dynamic_mx_quant_with_dual_axis returned an unexpected value")
    qweight_kn, scale_kn = result[2], result[3]
    qweight_nk = qweight_kn.transpose(-2, -1).contiguous()
    scale_nk = scale_kn.transpose(0, 1).reshape(tensor.shape[0], -1)
    scale_nk = scale_nk[:, : (tensor.shape[1] + 31) // 32].contiguous()
    return qweight_nk, scale_nk


def _block_quant(tensor: torch.Tensor):
    """Produce the Block-FP8 representation consumed by vLLM routed experts."""
    import torch_npu

    op = getattr(torch_npu, "npu_dynamic_block_mx_quant", None)
    if op is None:
        raise RuntimeError("TorchAO-NPU export requires npu_dynamic_block_mx_quant")
    result = op(
        tensor.transpose(-2, -1).contiguous(),
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
        scale_alg=0,
        dst_type_max=0.0,
    )
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise RuntimeError("npu_dynamic_block_mx_quant returned an unexpected value")
    # TorchAO consumes B_q/B_s2 in [K,N] matmul layout. Convert them back to
    # vLLM's per-expert HF loading layout [N,K] / [N,ceil(K/32)].
    qweight_kn, scale_kn = result[0], result[2]
    qweight_nk = qweight_kn.transpose(-2, -1).contiguous()
    scale_nk = scale_kn.transpose(0, 1).reshape(tensor.shape[0], -1)
    scale_nk = scale_nk[:, : (tensor.shape[1] + 31) // 32].contiguous()
    return qweight_nk, scale_nk


def _scale_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"Expected an HF weight name, got {weight_name!r}")
    return weight_name.removesuffix(".weight") + ".weight_scale"


def _mxfp4_quant(tensor: torch.Tensor, config):
    from torchao_npu.quantization.quant_primitives.mx import mx_quantize

    qweight, scale = mx_quantize(tensor, axis=-1, config=config)
    # Preserve packed FP4 and E8M0 bits in vLLM's HF storage layout.
    qweight = qweight.contiguous().view(torch.uint8)
    scale = scale.contiguous().view(torch.uint8).reshape(tensor.shape[0], -1)
    return qweight, scale[:, : (tensor.shape[1] + 31) // 32].contiguous()


def export_rollout_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    recipe: str,
    mxfp4_config=None,
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        kind = quantization_kind(name, recipe, enable_mxfp4=mxfp4_config is not None)
        if kind is None:
            yield name, tensor
            continue
        if mxfp4_config is not None and tensor.ndim == 3 and name.endswith(".attn.wo_a.weight"):
            tensor = tensor.reshape(-1, tensor.shape[-1])
        if tensor.ndim != 2:
            raise RuntimeError(f"{name}: expected a 2-D HF Linear weight, got shape={tuple(tensor.shape)}")
        if kind == "mxfp4":
            with torch.no_grad():
                qweight, scale = _mxfp4_quant(tensor, mxfp4_config)
        else:
            qweight, scale = _block_quant(tensor) if kind == "block_fp8" else _mx_quant(tensor)
        expected_scale_shape = (tensor.shape[0], (tensor.shape[1] + 31) // 32)
        expected_dtype = torch.uint8 if kind == "mxfp4" else torch.float8_e4m3fn
        expected_shape = (tensor.shape[0], tensor.shape[1] // 2) if kind == "mxfp4" else tensor.shape
        if qweight.dtype != expected_dtype or qweight.shape != expected_shape:
            raise RuntimeError(
                f"{name}: invalid {kind} weight dtype/shape: "
                f"dtype={qweight.dtype}, shape={tuple(qweight.shape)}"
            )
        if scale.shape != expected_scale_shape:
            raise RuntimeError(
                f"{name}: invalid {kind} scale shape={tuple(scale.shape)}, "
                f"expected={expected_scale_shape}"
            )
        yield name, qweight
        yield _scale_name(name), scale
