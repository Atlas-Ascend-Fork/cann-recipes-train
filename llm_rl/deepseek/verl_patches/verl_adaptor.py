# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import sys
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

from verl_patches.tools import insert_patch
from megatron.core.transformer import TransformerConfig
from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy


def get_gpt_decoder_block_spec(tfconfig: TransformerConfig, use_transformer_engine: bool = False):
    return get_gpt_layer_local_spec(
        num_experts=tfconfig.num_moe_experts,
        moe_grouped_gemm=tfconfig.moe_grouped_gemm,
        qk_layernorm=tfconfig.qk_layernorm
    )


def mcore_models_adaptation():
    from verl_patches import bert_padding
    sys.modules['flash_attn.bert_padding'] = bert_padding

    from verl_patches.models.mcore.config_converter import MLATransformerConfig, hf_to_mcore_config_dpskv3
    setattr(sys.modules['megatron.core.transformer'], 'MLATransformerConfig', MLATransformerConfig)

    from megatron_patches.core.tensor_parallel.cross_entropy import calculate_logits_max
    VocabParallelCrossEntropy.calculate_logits_max = calculate_logits_max

    import megatron.core.models.gpt.gpt_layer_specs as original_specs
    original_specs.get_gpt_decoder_block_spec = get_gpt_decoder_block_spec
    sys.modules['megatron.core.models.gpt.gpt_layer_specs.get_gpt_decoder_block_spec'] = get_gpt_decoder_block_spec

    from megatron_patches.core.packed_seq_params import PackedSeqParams
    from megatron.core import packed_seq_params
    packed_seq_params.PackedSeqParams = PackedSeqParams

    from mindspeed_patches.core.tensor_parallel.cross_entropy import calculate_predicted_logits
    import mindspeed.core.tensor_parallel.cross_entropy as original_cross_entropy
    original_cross_entropy.calculate_predicted_logits = calculate_predicted_logits
    sys.modules['mindspeed.core.tensor_parallel.cross_entropy.calculate_predicted_logits'] = calculate_predicted_logits

    # config_converter
    from verl.models.mcore import config_converter
    config_converter.hf_to_mcore_config_dpskv3 = hf_to_mcore_config_dpskv3

    from verl.models.mcore import registry
    SupportedModel = registry.SupportedModel
    registry.MODEL_CONFIG_CONVERTER_REGISTRY[SupportedModel.DEEPSEEK_V3] = config_converter.hf_to_mcore_config_dpskv3

    from verl.models.mcore import util
    from verl_patches.models.mcore.util import postprocess_packed_seqs
    util.postprocess_packed_seqs = postprocess_packed_seqs

    # model_forward
    from verl.models.mcore import model_forward
    from verl_patches.models.mcore.model_forward import gptmodel_forward
    model_forward.gptmodel_forward = gptmodel_forward

    # model_initializer
    from verl_patches.models.mcore.model_initializer import (get_rope_scaling_args, DeepseekV3Model)
    from verl.models.mcore import model_initializer
    model_initializer.BaseModelInitializer.get_rope_scaling_args = get_rope_scaling_args
    setattr(model_initializer, "DeepseekV3Model", DeepseekV3Model)

    # registry
    from verl_patches.models.mcore.weight_converter import McoreToHFWeightConverterDeepseekV3
    from verl.models.mcore.registry import (
        MODEL_INITIALIZER_REGISTRY,
        MODEL_WEIGHT_CONVERTER_REGISTRY,
        SupportedModel,
        MODEL_FORWARD_REGISTRY,
    )

    MODEL_INITIALIZER_REGISTRY[SupportedModel.DEEPSEEK_V3] = DeepseekV3Model
    MODEL_WEIGHT_CONVERTER_REGISTRY[SupportedModel.DEEPSEEK_V3] = McoreToHFWeightConverterDeepseekV3
    MODEL_FORWARD_REGISTRY[SupportedModel.DEEPSEEK_V3] = gptmodel_forward

    # weight_loader_registry
    from verl.models import weight_loader_registry
    from verl_patches.models.mcore.weight_loader_registry import get_weight_saver
    weight_loader_registry.get_weight_saver = get_weight_saver

    import verl.utils.checkpoint.megatron_checkpoint_manager as megatron_checkpoint_manager_old
    megatron_checkpoint_manager_old.get_weight_saver = get_weight_saver


def verl_utils_adaptation():
    from verl_patches.utils.megatron.tensor_parallel import _VocabParallelEntropy
    from verl.utils.megatron import tensor_parallel
    tensor_parallel._VocabParallelEntropy = _VocabParallelEntropy

    from verl_patches.utils import model as model_patch
    import verl.utils.model as model_original
    insert_patch(model_patch, model_original)

    from verl_patches.utils import megatron_utils as megatron_utils_patch
    import verl.utils.megatron_utils as megatron_utils_original
    insert_patch(megatron_utils_patch, megatron_utils_original)


def verl_workers_adaptation():
    from verl_patches.workers.vllm_rollout import vllm_rollout_spmd
    from verl_patches.workers.actor import megatron_actor
    from verl_patches.single_controller.base.megatron import worker

    from verl_patches.workers.sharding_manager import megatron_vllm as megatron_vllm_patch
    import verl.workers.sharding_manager.megatron_vllm as megatron_vllm_original
    insert_patch(megatron_vllm_patch, megatron_vllm_original)


def verl_protocol_adaptation():
    from verl_patches.protocol import union_tensor_dict
    from verl import protocol
    protocol.union_tensor_dict = union_tensor_dict


def exe_adaptation():
    mcore_models_adaptation()
    verl_utils_adaptation()
    verl_workers_adaptation()
    verl_protocol_adaptation()

exe_adaptation()
