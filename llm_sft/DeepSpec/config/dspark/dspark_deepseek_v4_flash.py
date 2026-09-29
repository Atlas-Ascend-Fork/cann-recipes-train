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

"""DSpark training config for DeepSeek-V4 Flash as target model.

Data preparation::

    python scripts/data/prepare_target_cache.py \\
        --config config/dspark/dspark_deepseek_v4_flash.py \\
        --train-data-path <train_data.jsonl> \\
        --output-dir <target_cache_dir> \\
        --local-batch-size 16

Training::

    bash scripts/train/train.sh  # after setting config_path / target_cache_dir

The draft model follows the Ascend DSpark inference recipe for
DeepSeek-V4-Flash-DSpark: block size 5, 3 draft stages, Shared-KV/MQA
attention (num_key_value_heads=1), q_lora_rank=1024, and hc_mult=4.

Target layer mapping
--------------------
Following the Ascend DSpark PR, ``target_layer_ids = [40, 41, 42]`` captures
the final three target hidden states used by ``mtp.0.main_proj`` on the
inference side.  The model's final ``last_hidden_state`` serves as the
verifier target for L1 loss and confidence head supervision.
"""

import os

from deepspec.trainer import DeepSeekV4DSparkTrainer

BASE_TB_DIR = os.path.expanduser(
    os.environ.get("BASE_TB_DIR", "~/tensorboard")
)
BASE_CKPT_DIR = os.path.expanduser(
    os.environ.get("BASE_CKPT_DIR", "~/checkpoints")
)

project_name = "deepspec"
exp_name = "dspark_block5_deepseek_v4_flash"
seed = 42

model = dict(
    # ---- Target model ----
    target_model_name_or_path=os.environ.get(
        "TARGET_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash"
    ),

    # ---- DSpark block drafting ----
    block_size=5,
    num_draft_layers=3,
    # Match the CANN DSpark SlidingWindow cache used by the Flash recipe.
    sliding_window=128,
    # None inherits the target model expert count; a positive value overrides it.
    n_routed_experts=64,
    # ---- Target layers captured into the cache ----
    # Layers 40, 41, 42 match dspark_target_layer_ids in the Ascend recipe and
    # feed the training equivalent of mtp.0.main_proj.
    # The last_hidden_state (layer 43 hc_head + norm output) serves as the
    # verifier target for L1 loss and confidence head.
    target_layer_ids=[40, 41, 42],

    # ---- Noise token for DSpark input embedding ----
    # Matches dspark_noise_token_id in the Ascend recipe.
    mask_token_id=128799,
    num_anchors=128,

    # ---- Markov head ----
    markov_rank=256,
    markov_head_type="vanilla",

    # ---- Confidence head ----
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    # ---- Loss ----
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
    # Auxiliary objective that maximizes normalized expected accepted length.
    tau_loss_alpha=0.05,
)

train = dict(
    trainer_cls=DeepSeekV4DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    # Keep FP32 master weights and AdamW states in host memory.
    optimizer_cpu_offload=False,
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    validation_fraction=0.1,
    chat_template="deepseek",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name_str = str(cfg["project_name"])
    exp_name_str = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        BASE_CKPT_DIR, project_name_str, exp_name_str
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        BASE_TB_DIR, project_name_str, exp_name_str
    )
    cfg["logging"] = logging_cfg
    return cfg
