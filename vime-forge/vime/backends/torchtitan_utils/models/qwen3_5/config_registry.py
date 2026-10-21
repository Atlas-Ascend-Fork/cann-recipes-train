"""Dense text-only Qwen3.5 recipes for the prepared NPU stack and Ascend CP.

``vime_qwen35_4b`` and ``vime_qwen35_9b`` retain the prepared CP=1 setup.
``vime_qwen35_4b_cp`` selects the Ascend CP kernels, defaulting to CP=4.
"""

from dataclasses import replace

from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.parallelize import parallelize_qwen3_5
from torchtitan.trainer import Trainer

from ..qwen3.config_registry import _vime_rl_config


def _text_inputs(_model, args, kwargs):
    # TorchTitan requires these IDs even when no vision inputs are supplied.
    if kwargs.get("pixel_values") is not None or kwargs.get("pixel_values_videos") is not None:
        raise ValueError("This Qwen3.5 RL recipe accepts text-only inputs.")
    kwargs.setdefault("special_tokens", {"image_id": -1, "video_id": -1})
    return args, kwargs


def _parallelize_text(model, **kwargs):
    model.register_forward_pre_hook(_text_inputs, with_kwargs=True)
    return parallelize_qwen3_5(model, **kwargs)


def _parallelize_text_cp(model, **kwargs):
    parallel_dims = kwargs["parallel_dims"]
    if not parallel_dims.cp_enabled:
        raise ValueError("The Qwen3.5 CP recipe requires CP>1; use vime_qwen35_4b for CP=1.")
    if parallel_dims.tp != 1 or parallel_dims.dp_replicate != 1:
        raise ValueError(
            "Qwen3.5 Ascend CP requires tensor_parallel_degree=1 and "
            "data_parallel_replicate_degree=1: the NPU CP parallelizer does not "
            "implement TP or DP replicate gradient synchronization "
            f"(got tp={parallel_dims.tp}, dp_replicate={parallel_dims.dp_replicate})."
        )
    if kwargs["parallelism"].context_parallel_load_balancer is not None:
        raise ValueError(
            "Qwen3.5 Ascend CP requires contiguous sequence sharding; "
            "set --parallelism.context_parallel_load_balancer None."
        )
    from torchtitan_npu.override.qwen3_5.parallelize import parallelize_qwen3_5_cp

    model.register_forward_pre_hook(_text_inputs, with_kwargs=True)
    # The CP adapter operates on the language model only.
    model.vision_encoder = None
    return parallelize_qwen3_5_cp(model, **kwargs)


def _qwen35_config(
    flavor: str, *, tie_word_embeddings: bool, context_parallel: bool = False,
) -> Trainer.Config:
    if context_parallel:
        from torchtitan_npu.models.qwen3_5 import model_registry as npu_model_registry

        spec = npu_model_registry(flavor, attn_backend="varlen")
        parallelize_fn = _parallelize_text_cp
    else:
        spec = model_registry(flavor, attn_backend="varlen")
        parallelize_fn = _parallelize_text
    spec = replace(spec, parallelize_fn=parallelize_fn)
    spec.model.enable_weight_tying = tie_word_embeddings
    config = _vime_rl_config(
        spec,
        lr=1e-6 if context_parallel else 3e-6,
        # TorchTitan supports exactly three values here:
        # "partial_dtensor", "full_dtensor", and "spmd_types".  This recipe
        # uses partial-DTensor for both CP=1 and context-parallel training.
        spmd_backend="partial_dtensor",
    )
    if context_parallel:
        # Preserve the incoming CP recipe's optimizer and checkpoint behavior.
        config.optimizer = replace(config.optimizer, implementation="foreach")
        config.parallelism.context_parallel_degree = 4
        config.parallelism.context_parallel_load_balancer = None
        config.override.imports = [
            "torchtitan_npu.override.qwen3_5.varlen_attention.asc_cp",
            "torchtitan_npu.override.qwen3_5.gated_delta.context_parallel",
        ]
        config.checkpoint.initial_load_in_hf = True
        config.checkpoint.initial_load_model_only = True
    else:
        config.override.imports.extend(
            [
                "torchtitan_npu.override.qwen3_5.gated_delta.triton",
                # NovaSwap optimizer-state swapping with checkpoint support.
                "torchtitan_npu.override.common.optimizer.swap_optimizer",
            ]
        )
    return config


def vime_qwen35_4b() -> Trainer.Config:
    """Dense 4B GRPO with tied embeddings and the prepared NPU kernel overrides."""
    # The pinned TorchTitan factory leaves tying off. The HF 4B checkpoint has
    # one shared input/output embedding, so both gradients must update it.
    return _qwen35_config("4B", tie_word_embeddings=True)


def vime_qwen35_9b() -> Trainer.Config:
    """Dense 9B GRPO with separate input/output embeddings and the same NPU runtime."""
    return _qwen35_config("9B", tie_word_embeddings=False)


def vime_qwen35_4b_cp() -> Trainer.Config:
    """Dense 4B with Ascend CP kernels, tied embeddings, and CP=4 by default."""
    return _qwen35_config("4B", tie_word_embeddings=True, context_parallel=True)
