"""VIME-owned DeepSeek-V4 recipes."""

from dataclasses import replace

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.config import OverrideConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.trainer import Trainer
from torchtitan_npu.models.deepseek_v4 import model_registry
from torchtitan_npu.models.deepseek_v4 import _deepseek_v4_flash
from torchtitan_npu.models.deepseek_v4.config_registry import _dsv4_optimizer_config


_VIME_OVERRIDES = [
    "torchtitan_npu.override.common.rms_norm.asc",
    "torchtitan_npu.override.common.rope.asc_partial",
    "torchtitan_npu.override.common.swiglu_group.asc",
    "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
    "torchtitan_npu.override.deepseek_v4.compressor.asc",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
    "torchtitan_npu.override.common.token_dispatcher.asc_deepep",
    "torchtitan_npu.override.common.optimizer.swap_optimizer",
]


def _vime_rl_config(model_spec, *, lr: float = 2e-7) -> Trainer.Config:
    return Trainer.Config(
        model_spec=model_spec,
        optimizer=_dsv4_optimizer_config(model_spec, lr=lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            decay_type="linear",
            min_lr_factor=1.0,
        ),
        training=TrainingConfig(
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            seq_len=4096,
            local_batch_size=1,
            disable_cuda_graphs=True,
            enable_cpu_offload=False,
        ),
        parallelism=ParallelismConfig(
            spmd_backend="partial_dtensor",
            context_parallel_load_balancer=None,
        ),
        activation_checkpoint=FullAC.Config(),
        override=OverrideConfig(imports=list(_VIME_OVERRIDES)),
        checkpoint=CheckpointManager.Config(
            enable=True,
            # VIME owns save scheduling and only calls the manager on selected
            # rollouts. Avoid filtering those calls a second time.
            interval=1,
            initial_load_in_hf=True,
            initial_load_model_only=True,
        ),
    )


def _derived_spec(*, n_layers: int, num_experts: int, compress_ratios: tuple[int, ...]):
    """Derive a small RL model while preserving upstream ModelSpec hooks.

    The private upstream builder is used only as a config factory: it creates
    every nested router/expert config consistently for the requested expert
    count. The resulting ModelSpec hooks still come from the public registry.
    """
    if len(compress_ratios) != n_layers:
        raise ValueError("compress_ratios must contain one entry per DSV4 layer")
    spec = model_registry("deepseek_v4_flash", num_mtp_layers=0, moe_comm_backend="deepep")
    model = _deepseek_v4_flash(num_experts=num_experts, num_mtp_layers=0, moe_comm_backend="deepep")
    model = replace(
        model,
        n_layers=n_layers,
        # Keep the upstream list container: TorchTitan's config traversal
        # descends into lists, so per-layer NPU overrides remain discoverable.
        layers=model.layers[:n_layers],
        compress_ratios=compress_ratios,
    )
    return replace(spec, model=model, flavor=f"vime_dsv4_flash_{n_layers}layers_{num_experts}experts")


def vime_dsv4_flash_4layers_256experts() -> Trainer.Config:
    return _vime_rl_config(_derived_spec(n_layers=4, num_experts=256, compress_ratios=(1, 1, 4, 128)))


def vime_dsv4_flash_43layers_256experts() -> Trainer.Config:
    return _vime_rl_config(_derived_spec(n_layers=43, num_experts=256, compress_ratios=(1, 1, 4) + 20 * (128, 4)))


def vime_dsv4_flash() -> Trainer.Config:
    return _vime_rl_config(model_registry("deepseek_v4_flash", num_mtp_layers=0))
