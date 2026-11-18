# Adapted from
# https://gitcode.com/Ascend/MindSpeed-RL/blob/2.1.0/cli/train_grpo.py
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


import os
import sys
import random
import logging
import omegaconf
import numpy as np
import torch
import torch_npu

from verl_patches.train_engine.megatron_config import MegatronConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def seed_all(seed=1234):
    """Make an NPU run deterministic.

    Adapted from ``mindspeed_rl.utils.seed_all`` (MindSpeed-RL 2.1.0,
    ``mindspeed_rl/utils/utils.py``). The upstream ``cli/train_grpo.py`` this file is
    adapted from imports it from that package, but this sample's environment ships
    MindSpeed and MindSpeed-LLM rather than MindSpeed-RL (see the sample Dockerfile),
    so the behaviour lives here instead of importing an absent package.

    Seeding the three PRNGs is not enough for the ``use_deter_comp`` path: the ATB and
    HCCL variables below are what make Ascend operator and collective ordering
    reproducible, which is what the caller's log line announces.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['HCCL_DETERMINISTIC'] = str(True)
    os.environ['LCCL_DETERMINISTIC'] = str(1)
    os.environ['CLOSE_MATMUL_K_SHIFT'] = str(1)
    os.environ['ATB_MATMUL_SHUFFLE_K_ENABLE'] = "0"
    os.environ['ATB_LLM_LCOC_ENABLE'] = "0"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)


def parse_args_from_config(config):
    """
    Parse the config into command-line arguments
    Mainly from Mindspeed-RL
    """
    for key, value in config.items():
        if isinstance(value, omegaconf.listconfig.ListConfig):
            sys.argv.append(f"--{key.replace('_', '-')}")
            for i in value:
                sys.argv.append(f"{i}")
        elif isinstance(value, bool):
            if value:
                sys.argv.append(f"--{key.replace('_', '-')}")
        elif value is None:
            continue
        else:
            sys.argv.append(f"--{key.replace('_', '-')}={value}")


def get_base_training_configs(config, model_configs, config_dict):
    config_dict['seed'] = config.actor_rollout_ref.actor.megatron.seed
    config_dict['bf16'] = True
    config_dict['global_batch_size'] = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
    config_dict['micro_batch_size'] = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
    config_dict['seq_length'] = config.data.max_prompt_length + config.data.max_response_length
    config_dict['variable_seq_lengths'] = True
    config_dict['attention_dropout'] = 0.0
    config_dict['init_method_std'] = 0.01
    config_dict['hidden_dropout'] = 0.0
    config_dict['lr'] = config.actor_rollout_ref.actor.optim.lr
    config_dict['lr_decay_style'] = 'constant'
    config_dict['min_lr'] = 0.0
    config_dict['weight_decay'] = config.actor_rollout_ref.actor.optim.weight_decay
    config_dict['lr_warmup_iters'] = max(0, config.actor_rollout_ref.actor.optim.lr_warmup_steps)
    config_dict['lr_warmup_fraction'] = config.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio
    config_dict['clip_grad'] = config.actor_rollout_ref.actor.optim.clip_grad
    config_dict['adam_beta1'] = 0.9
    config_dict['adam_beta2'] = 0.95
    config_dict['initial_loss_scale'] = 4096
    config_dict['moe_router_bias_update_rate'] = 0.001
    config_dict['moe_aux_loss_coeff'] = model_configs['aux_loss_alpha']
    config_dict['finetune'] = True
    config_dict['train_iters'] = config.trainer.total_training_steps
    config_dict['save_interval'] = config.trainer.save_freq
    config_dict['no_shared_storage'] = True

    # parallel strategy
    config_dict['tensor_model_parallel_size'] = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
    config_dict['pipeline_model_parallel_size'] = config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
    if config.actor_rollout_ref.actor.megatron.pipeline_parallel_num_layer_list:
        config_dict["num_layer_list"] = [
            int(item) for item in config.actor_rollout_ref.actor.megatron.pipeline_parallel_num_layer_list
        ]
    else:
        config_dict['num_layer_list'] = None
    config_dict['sequence_parallel'] = config.actor_rollout_ref.actor.megatron.sequence_parallel
    config_dict['expert_model_parallel_size'] = config.actor_rollout_ref.actor.megatron.expert_model_parallel_size
    config_dict['context_parallel_size'] = config.actor_rollout_ref.actor.megatron.context_parallel_size
    if config_dict['context_parallel_size'] > 1:
        config_dict['use_cp_send_recv_overlap'] = True
        config_dict['context_parallel_algo'] = 'megatron_cp_algo'
        config_dict['use_fused_ring_attention_update'] = True
        config_dict['cp_window_size'] = 1
    config_dict['use_distributed_optimizer'] = config.actor_rollout_ref.actor.megatron.use_distributed_optimizer
    config_dict['distributed_backend'] = 'nccl'


def get_optimization_configs(config, config_dict):
    config_dict['shape_order'] = 'BNSD'
    config_dict['use_fused_rmsnorm'] = True
    config_dict['use_flash_attn'] = True
    config_dict['no_masked_softmax_fusion'] = True
    config_dict['attention_softmax_in_fp32'] = True
    config_dict['no_gradient_accumulation_fusion'] = True
    config_dict['use_fused_swiglu'] = True
    config_dict['use_fused_rotary_pos_emb'] = True
    config_dict['overlap_grad_reduce'] = True
    config_dict['use_rotary_position_embeddings'] = True
    config_dict['overlap_param_gather'] = True
    config_dict['use_fused_mlp'] = True
    if config.actor_rollout_ref.model.enable_gradient_checkpointing:
        config_dict["recompute_granularity"] = (
            config.actor_rollout_ref.model.gradient_checkpointing_kwargs.activations_checkpoint_granularity
        )
        config_dict["recompute_method"] = (
            config.actor_rollout_ref.model.gradient_checkpointing_kwargs.activations_checkpoint_method
        )
        config_dict["recompute_num_layers"] = (
            config.actor_rollout_ref.model.gradient_checkpointing_kwargs.activations_checkpoint_num_layers
        )
    if config.actor_rollout_ref.actor.megatron.swap_optimizer:
        assert not config.actor_rollout_ref.actor.megatron.optimizer_offload, (
            "When using swap_optimizer, no need to use optimizer_offload."
        )
        config_dict['swap_optimizer'] = True


def get_regular_model_configs(config, model_configs, config_dict):
    """
    Get regular model configs from config.json
    """
    config_dict['use_mcore_models'] = True
    config_dict['tokenizer_type'] = 'PretrainedFromHF'
    config_dict['tokenizer_name_or_path'] = config.actor_rollout_ref.model.path
    config_dict['num_layers'] = model_configs['num_hidden_layers']
    config_dict['hidden_size'] = model_configs['hidden_size']
    config_dict['ffn_hidden_size'] = model_configs['intermediate_size']
    config_dict['num_attention_heads'] = model_configs['num_attention_heads']
    config_dict['rotary_base'] = int(model_configs['rope_theta'])
    config_dict['max_position_embeddings'] = model_configs['max_position_embeddings']
    config_dict['make_vocab_size_divisible_by'] = 1
    config_dict['padded_vocab_size'] = model_configs['vocab_size']
    config_dict['untie_embeddings_and_output_weights'] = True
    config_dict['add_qkv_bias'] = False
    config_dict['disable_bias_linear'] = True
    config_dict['num_query_groups'] = model_configs['num_attention_heads'] // model_configs['num_key_value_heads']
    config_dict['group_query_attention'] = config_dict['num_query_groups'] > 1
    config_dict['position_embedding_type'] = 'rope'
    config_dict['normalization'] = 'RMSNorm'
    config_dict['norm_epsilon'] = model_configs['rms_norm_eps']
    config_dict['swiglu'] = True

    if model_configs.get("rope_scaling"):
        config_dict['rope_scaling_beta_fast'] = model_configs['rope_scaling']['beta_fast']
        config_dict['rope_scaling_beta_slow'] = model_configs['rope_scaling']['beta_slow']
        config_dict['rope_scaling_mscale'] = model_configs['rope_scaling']['mscale']
        config_dict['rope_scaling_mscale_all_dim'] = model_configs['rope_scaling']['mscale_all_dim']
        config_dict['rope_scaling_original_max_position_embeddings'] = (
            model_configs['rope_scaling']['original_max_position_embeddings']
        )
        config_dict['rope_scaling_type'] = model_configs['rope_scaling']['type']
        config_dict['rope_scaling_factor'] = model_configs['rope_scaling']['factor']


def get_dsv3_configs(config, model_configs, config_dict):
    """
    Get other configs for DeepseekV3
    """
    config_dict['moe_router_topk'] = model_configs['num_experts_per_tok']
    config_dict['multi_head_latent_attention'] = True
    config_dict['q_lora_rank'] = model_configs['q_lora_rank']
    config_dict['kv_lora_rank'] = model_configs['kv_lora_rank']
    config_dict['qk_rope_head_dim'] = model_configs['qk_rope_head_dim']
    config_dict['qk_nope_head_dim'] = model_configs['qk_nope_head_dim']
    config_dict['v_head_dim'] = model_configs['v_head_dim']
    config_dict['qk_layernorm'] = True
    config_dict['first_k_dense_replace'] = model_configs['first_k_dense_replace']
    config_dict['moe_intermediate_size'] = model_configs['moe_intermediate_size']
    config_dict['moe_layer_freq'] = model_configs['moe_layer_freq']
    config_dict['num_experts'] = model_configs['n_routed_experts']
    config_dict['n_shared_experts'] = model_configs['n_shared_experts']
    config_dict['moe_router_load_balancing_type'] = model_configs['topk_method']
    config_dict['topk_group'] = model_configs['topk_group']
    config_dict['n_group'] = model_configs['n_group']
    config_dict['routed_scaling_factor'] = model_configs['routed_scaling_factor']
    config_dict['moe_router_score_function'] = model_configs['scoring_func']
    config_dict['seq_aux'] = model_configs['seq_aux']
    config_dict['moe_token_dispatcher_type'] = 'alltoall'
    config_dict['moe_alltoall_overlap_comm'] = True
    config_dict['moe_grouped_gemm'] = True
    config_dict['moe_router_enable_expert_bias'] = True
    config_dict['moe_tp_extend_ep'] = True
    config_dict['fix_router'] = config.actor_rollout_ref.actor.megatron.fix_router
    config_dict['moe_permutation_async_comm'] = True
    config_dict['gemm_gradient_accumulation_fusion'] = True
    config_dict['use_fused_moe_token_permute_and_unpermute'] = True
    config_dict['no_load_optim'] = True
    config_dict['no_load_rng'] = True
    config_dict['norm_topk_prob'] = True


def translate_verl_train_configs_to_megatron(config):
    """
    Translate the veRL training-related configs into corresponding Megatron (Mindspeed) configs using a dict
    Return a instance of MegatronConfig
    """
    config_dict = {}

    import json
    config_json_path = os.path.join(config.actor_rollout_ref.model.path, 'config.json')
    with open(config_json_path, 'r', encoding='utf-8') as f:
        model_configs = json.load(f)

    stage_mapping = {'grpo': 'ray_grpo'}
    if config.algorithm.adv_estimator not in stage_mapping:
        raise NotImplementedError(f"RL algorithm {config.algorithm.adv_estimator} is not supported yet.")
    config_dict['stage'] = stage_mapping[config.algorithm.adv_estimator]

    # hyper params for training
    get_base_training_configs(config, model_configs, config_dict)

    # optimizations of the train engine
    get_optimization_configs(config, config_dict)

    # model configs
    get_regular_model_configs(config, model_configs, config_dict)

    # get special configs for dsV3
    if model_configs['architectures'][0] == 'DeepseekV3ForCausalLM':
        get_dsv3_configs(config, model_configs, config_dict)

    # validate
    if model_configs['num_hidden_layers'] % config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size > 0:
        assert 'num_layer_list' in config_dict, (
            "num_layer_list must be set if num_hidden_layers is not divisible by PP size!"
        )
        assert sum(config_dict.get('num_layer_list')) == model_configs.get('num_hidden_layers')

    return MegatronConfig(training_config=config_dict, model_config=None)


def initialize_megatron(
        extra_args_provider=None,
        args_defaults=None,
        ignore_unknown_args=False,
        allow_no_cuda=False,
        skip_mpu_initialization=False,
        config=None,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)

    Mainly from Mindspeed-RL
    """
    _original_compile = torch.compile   # backup original torch.compile

    origin_sys_argv = sys.argv
    sys.argv = [sys.argv[0]]
    parse_args_from_config(config)

    # NOTE: Importing this line activates the megatron_adapter.
    from mindspeed_llm.training.arguments import parse_args_decorator
    import megatron

    # use torch.compile, not jit.script patched by mindspeed
    torch.compile = _original_compile
    # patch get_num_layers_to_build to process num_layer_list correctly
    from verl_patches.train_engine import mindspeed_patch
    # patch _communicate_shapes (refer to Mindspeed-RL)
    from verl_patches.train_engine import megatron_patch

    parse_args = parse_args_decorator(megatron.training.arguments.parse_args)
    args = parse_args(extra_args_provider, ignore_unknown_args)
    sys.argv = origin_sys_argv

    if not allow_no_cuda:
        if not torch.cuda.is_available():
            raise ValueError("Megatron requires CUDA.")

    from megatron.core import parallel_state
    from megatron.training import get_args
    from megatron.training.arguments import validate_args
    from megatron.training.checkpointing import load_args_from_checkpoint
    from megatron.training.global_vars import set_global_variables
    from megatron.training.initialize import _set_random_seed, \
        _init_autoresume, _compile_dependencies, \
        _initialize_tp_communicators

    if args_defaults is None:
        args_defaults = {}
    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        if args.load is None:
            raise ValueError("--use-checkpoints-args requires --load argument.")
        load_args_from_checkpoint(args)

    validate_args(args, args_defaults)

    set_global_variables(args)

    if args.use_deter_comp:
        seed_all(args.seed)
        logger.info("deterministic computing is applied for npu.")

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            logger.info("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        parallel_state.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        parallel_state.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            _initialize_tp_communicators()

        # No continuation function
        return None


def _initialize_distributed():
    """
    Initialize torch.distributed and core model parallel.
    Mainly from Mindspeed-RL
    """
    from megatron.core import parallel_state
    from megatron.training import get_args
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            logger.info("torch distributed is already initialized, skipping initialization...")
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        allocated_device = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(allocated_device)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            logger.info("model parallel is already initialized")
        else:
            parallel_state.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                args.pipeline_model_parallel_split_rank,
                context_parallel_size=args.context_parallel_size,
                expert_model_parallel_size=args.expert_model_parallel_size,
                distributed_timeout_minutes=args.distributed_timeout_minutes,
                nccl_communicator_config_path=args.nccl_communicator_config_path,
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',
            )
            if args.rank == 0:
                logger.info(
                    f"> initialized tensor model parallel with size "
                    f"{parallel_state.get_tensor_model_parallel_world_size()}"
                )
                logger.info(
                    f"> initialized pipeline model parallel with size "
                    f"{parallel_state.get_pipeline_model_parallel_world_size()}"
                )
