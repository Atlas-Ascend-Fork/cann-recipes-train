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

"""vime RL recipes for the qwen3 family, in torchtitan's own config_registry shape::

    --module vime.backends.torchtitan_utils.models.qwen3 --config vime_qwen3_4b

``ConfigManager`` calls the named function for the base config; ``TitanEngine``'s runtime values and
then the script's dotted flags override it, in that order. What lives here is what tyro cannot
express: the ``model_registry`` arguments, the optimizer hyper-parameters, the activation
checkpointing policy and the override list.

A checkpoint matching no built-in flavor gets a recipe of its own, reusing the family's
``parallelize_fn`` and ``state_dict_adapter``::

    model = qwen3_configs["4B"](attn_backend="varlen")
    for layer in model.layers:
        layer.attention.rope = replace(layer.attention.rope, theta=500000.0)
    spec = replace(model_registry("4B", attn_backend="varlen"), model=model, flavor="4B-rope5e5")

``TitanEngine`` checks whatever the recipe built against the checkpoint's ``config.json``.
"""

from dataclasses import replace

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import OverrideConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.qwen3 import model_registry
from torchtitan.trainer import Trainer

try:
    from torchtitan.components.lr_scheduler import LRSchedulersContainer
except ImportError:
    # Newer TorchTitan versions export the scheduler from the optimizer package.
    from torchtitan.components.optimizer import LRSchedulersContainer

# Implementations upstream does not select on its own. ⚠ ``--override.imports`` on the command line
# REPLACES this list rather than appending to it.
_VIME_OVERRIDES = [
    # torch_npu has no kernel for aten::_flash_attention_forward, where upstream's varlen lands.
    "vime.backends.torchtitan_utils.overrides.npu_varlen_attention.npu_varlen",
]



def _vime_rl_config(model_spec, *, lr: float = 1e-6, spmd_backend: str = "partial_dtensor") -> Trainer.Config:
    """Common RL base. The dataloader, tokenizer, metrics, validator, profiler and loss sections
    stay at their defaults: vime owns all six and never builds torchtitan's.

    ⚠ marks values that break training when changed.
    """
    return Trainer.Config(
        model_spec=model_spec,
        optimizer=replace(
            default_adamw(lr=lr, betas=(0.9, 0.98), weight_decay=0.1),
            # ⚠ NPU has no CUDA-only fused optimizer kernels; "fused" (the upstream default) fails.
            # implementation="foreach",
        ),
        # Constant LR: linear decay to min_lr_factor=1.0 never moves.
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0, decay_type="linear", min_lr_factor=1.0
        ),
        training=TrainingConfig(
            max_norm=1.0,
            dtype="float32",  # ⚠
            mixed_precision_param="bfloat16",  # ⚠
            mixed_precision_reduce="float32",  # ⚠
            # Defaults: TitanEngine passes --rollout-max-context-len and --micro-batch-size through
            # as the real values.
            seq_len=4096,
            local_batch_size=1,
            # CUDA graphs capture the upstream forward/backward, which vime never calls.
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            # vime reads vocab-sharded DTensor logits and syncs weights through full_tensor();
            # spmd_types (upstream's default) and full_dtensor replace both types. The degrees stay
            # on the command line (--parallelism.tensor_parallel_degree / .expert_parallel_degree);
            # dp_shard keeps its -1 default, which takes whatever ranks the others leave.
            # The prepared Qwen3.5 stack calls this backend "default". Keep the
            # existing Qwen3 recipes' setting unless the recipe opts into it.
            spmd_backend=spmd_backend,
        ),
        activation_checkpoint=FullAC.Config(),
        override=OverrideConfig(imports=list(_VIME_OVERRIDES)),
        checkpoint=CheckpointManager.Config(
            # interval=1 because vime decides when to save and calls the checkpointer only then.
            # The seed and the load policy come from the command line.
            enable=True,
            interval=1,
        ),
    )


def vime_qwen3_4b() -> Trainer.Config:
    """Qwen3-4B, dense. varlen is the only backend that expresses the per-document boundaries a
    packed micro-batch needs."""
    return _vime_rl_config(model_registry("4B", attn_backend="varlen"))


def vime_qwen3_4b_instruct_2507() -> Trainer.Config:
    """Qwen3-4B-Instruct-2507 with its checkpoint's RoPE theta."""
    spec = model_registry("4B", attn_backend="varlen")
    for layer in spec.model.layers:
        layer.attention.rope = replace(layer.attention.rope, theta=5_000_000.0)
    return _vime_rl_config(replace(spec, flavor="4B-Instruct-2507"))


def vime_qwen3_30b_a3b() -> Trainer.Config:
    """Qwen3-30B-A3B, MoE. The token dispatcher stays at ``model_registry``'s default "standard";
    deepep / hybridep / minimal_async_ep need their own kernels. The AdamW moments go to host
    memory, which is what makes the model fit."""
    config = _vime_rl_config(model_registry("30B-A3B", attn_backend="varlen"))
    # Adam moments live in host memory and are swapped onto the device bucket by bucket around each
    # optimizer step.
    config.override.imports.append("torchtitan_npu.override.common.optimizer.swap_optimizer")
    # The swap container fills each bucket's state per parameter and rejects a foreach group
    config.optimizer = replace(config.optimizer, implementation="fused")
    return config
