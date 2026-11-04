# Adapted from
# https://github.com/volcengine/verl/blob/v0.4.0/verl/utils/megatron_utils.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Pretrain utilities."""
__all__ = [
    "offload_megatron_model_to_cpu",
    "load_megatron_model_to_gpu",
    "offload_megatron_optimizer",
    "load_megatron_optimizer",
    "default_tp_concat_fn",
    "per_tensor_generator",
    "get_transformer_layer_offset",
]

import gc
import os
import re
import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.transformer import TransformerConfig

import verl.utils.megatron.tensor_parallel as tp_utils
from verl.utils.model import normalize_model_name
from verl.utils.megatron_utils import (
    unwrap_model,
    broadcast_from_megatron_pp,
    broadcast_str_from_megatron_pp,
    offload_megatron_copy_params,
    load_megatron_copy_params,
)
from verl_patches.utils.reshard import ep_param_reshard_by_alltoallv, get_rollout_expert_after_resharding
from verl_patches.tools import empty_cache


@torch.no_grad()
def offload_megatron_model_to_cpu(models, offload_param=True):
    """
    In megatron, the model and optimizer storage are:
    - bf16 parameter data chunked in model parallel group
    - fp32 grad chunked in model parallel group
    - fp32 main_parameter chunked in model and dp group
    - fp32 optimizer state chunked in model and dp group
    """
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # offload parameters
                    if offload_param:
                        if buffer.param_data.storage().size() > 0:
                            buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                            buffer.param_data_size = buffer.param_data.storage().size()
                            buffer.param_data.storage().resize_(0)

                        assert buffer.param_data_size == buffer.param_data.cpu_data.storage().size()

                    if buffer.grad_data.storage().size() > 0:
                        # if the grad_data size is already zero, we assume that it is already offloaded
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        buffer.grad_data.storage().resize_(0)
        else:
            # we need this for ref module
            for _, param in model_chunk.named_parameters():
                if offload_param:
                    param.data = param.data.to("cpu", non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to("cpu", non_blocking=True)
    gc.collect()
    empty_cache()


@torch.no_grad()
def load_megatron_model_to_gpu(models, load_grad=True):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # sometimes, we don't want to load grad for pure inference
                    if load_grad:
                        buffer.grad_data.storage().resize_(buffer.grad_data_size)
                        buffer.grad_data.zero_()

                    if buffer.param_data.storage().size() == 0:
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        # copy data from cpu to cuda
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)
        else:
            # we need this for ref module
            device_id = torch.cuda.current_device()
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to(device_id, non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to(device_id, non_blocking=True)
    gc.collect()
    empty_cache()


@torch.no_grad()
def offload_megatron_optimizer(optimizers):
    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        offload_megatron_copy_params(_opt)
        opt_state_dict_values = _opt.optimizer.state.values()
        for v in opt_state_dict_values:
            if "exp_avg" in v:
                v["exp_avg"] = v["exp_avg"].to("cpu", non_blocking=True)
            if "exp_avg_sq" in v:
                v["exp_avg_sq"] = v["exp_avg_sq"].to("cpu", non_blocking=True)
        gc.collect()
        empty_cache()


@torch.no_grad()
def load_megatron_optimizer(optimizers):
    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        load_megatron_copy_params(_opt)
        opt_state_dict_values = _opt.optimizer.state.values()
        for v in opt_state_dict_values:
            if "exp_avg" in v:
                v["exp_avg"] = v["exp_avg"].to(torch.cuda.current_device(), non_blocking=True)
            if "exp_avg_sq" in v:
                v["exp_avg_sq"] = v["exp_avg_sq"].to(torch.cuda.current_device(), non_blocking=True)
        gc.collect()
        empty_cache()


def default_tp_concat_fn(layer_name_mapping, name, train_params, infer_params, model_config, convert_qkv_gate_up_by_simple_split=False):
    """
    name: name of the parameter
    train_params: training parameters
    infer_params (Iterable[torch.Tensor]): a iterator towards list of parameters all-gathered from micro_dp_group
    model_config: huggingface model_config
    TODO(zhangchi.usc1992): currently, the implementation is adhoc. We can move this function to the model
    definition so that it is model-agnostic. If the model doesn't implement this function,
    we can throw an error to force user disable TP HybridEngine.
    """
    from megatron.core import mpu
    from megatron.training import get_args
    mindspeed_args = get_args()

    if layer_name_mapping.get("qkv_layer_name") in name and "layer_norm" not in name:
        # if the tensor is qkv, for each param on tp, split into q, k, v
        # concat q, k, v separately.
        q_lst = []
        k_lst = []
        v_lst = []
        assert model_config.num_attention_heads % model_config.num_key_value_heads == 0
        num_q_per_kv = model_config.num_attention_heads // model_config.num_key_value_heads
        assert infer_params[0].shape[0] % (num_q_per_kv + 2) == 0, f"param '{name}' shape '{infer_params[0].shape}' dim0 is not divisible by {num_q_per_kv + 2}"
        kv_size_per_tp = infer_params[0].shape[0] // (num_q_per_kv + 2)
        split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]
        for infer_param in infer_params:
            num_query_groups_per_partition = model_config.num_key_value_heads // mpu.get_tensor_model_parallel_world_size()
            for chunk in infer_param.chunk(num_query_groups_per_partition):
                split_size = [kv_size_per_tp * num_q_per_kv // num_query_groups_per_partition, kv_size_per_tp // num_query_groups_per_partition, kv_size_per_tp // num_query_groups_per_partition]
                q, k, v = chunk.split(split_size)
                q_lst.append(q)
                k_lst.append(k)
                v_lst.append(v)
        q = torch.cat(q_lst, dim=0)
        k = torch.cat(k_lst, dim=0)
        v = torch.cat(v_lst, dim=0)
        infer_params = torch.cat((q, k, v), dim=0) if not convert_qkv_gate_up_by_simple_split else [q, k, v]

    elif layer_name_mapping.get("gate_proj_layer_name") in name:
        # if the tensor is gate and proj
        gate_lst = []
        up_lst = []
        for infer_param in infer_params:
            gate, up = infer_param.chunk(2)
            gate_lst.append(gate)
            up_lst.append(up)
        gate = torch.cat(gate_lst, dim=0)
        up = torch.cat(up_lst, dim=0)
        infer_params = torch.cat((gate, up), dim=0) if not convert_qkv_gate_up_by_simple_split else [gate, up]

    elif "mlp.experts.linear_fc2.weight" in name:  # moe
        infer_params = torch.cat(infer_params, dim=1)

    elif "mlp.linear_fc2.weight" in name:  # dense
        infer_params = torch.cat(infer_params, dim=1)

    elif "mlp.experts.weight1" in name:  # dsv3 moe
        gate_pp_lst = []
        up_pp_lst = []

        if mindspeed_args.moe_tp_extend_ep:
            if os.getenv('NO_ALL_TO_ALL_RESHARD', '0') == '1':
                for infer_param in infer_params:
                    split_size = [
                        model_config.moe_intermediate_size,
                        model_config.moe_intermediate_size,
                    ] * (model_config.n_routed_experts // mpu.get_tensor_and_expert_parallel_world_size())
                    experts_weight = infer_param.split(split_size, dim=1)
                    gate_pp_lst.extend(experts_weight[::2])
                    up_pp_lst.extend(experts_weight[1::2])
                infer_params = [tensor.transpose(0, 1) for pair in zip(gate_pp_lst, up_pp_lst) for tensor in pair]
            else:
                # To optimize memory, only the params for the current rank are non-empty; others are empty tensors
                infer_params = get_rollout_expert_after_resharding(infer_params, model_config, is_weight1=True)
        else:
            raise NotImplementedError(f"The weight resharding of this mode still has problems and needs to be adapted!")

    elif "mlp.experts.weight2" in name:  # moe
        down_pp_lst = []
        if mindspeed_args.moe_tp_extend_ep:
            if os.getenv('NO_ALL_TO_ALL_RESHARD', '0') == '1':
                for infer_param in infer_params:
                    split_size = [
                        model_config.moe_intermediate_size
                    ] * (model_config.n_routed_experts // mpu.get_tensor_and_expert_parallel_world_size())
                    experts_weight = infer_param.split(split_size, dim=0)
                    down_pp_lst.extend(experts_weight)
                experts_down_pp = [downs.transpose(0, 1) for downs in down_pp_lst]
                infer_params = experts_down_pp
            else:
                # To optimize memory, only the params for the current rank are non-empty; others are empty tensors
                infer_params = get_rollout_expert_after_resharding(infer_params, model_config, is_weight1=False)
        else:
            raise NotImplementedError(f"The weight resharding of this mode still has problems and needs to be adapted!")

    else:
        # concat tensor
        infer_params = torch.cat(infer_params, dim=tp_utils.get_tensor_parallel_partition_dim(train_params))

    return infer_params


def per_tensor_generator(actor_module, model_config, weight_converter, transformer_config, layer_name_mapping, convert_qkv_gate_up_by_simple_split=True):
    from megatron.core import parallel_state as mpu
    from megatron.training import get_args

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_group = mpu.get_pipeline_model_parallel_group()
    ep_size = mpu.get_expert_model_parallel_world_size()
    etp_size = mpu.get_tensor_and_expert_parallel_world_size()
    ep_group = mpu.get_expert_model_parallel_group()
    etp_group = mpu.get_tensor_and_expert_parallel_group()
    vpp_size = len(actor_module)
    all_gather_group = mpu.get_tensor_model_parallel_group()
    all_gather_group_size = torch.distributed.get_world_size(group=all_gather_group)
    mindspeed_args = get_args()

    def tensor_generator():
        for scan_vpp_idx in range(vpp_size):
            existing_keys = set()
            model = unwrap_model(actor_module[scan_vpp_idx])
            for name, param in model.named_parameters():
                existing_keys.add(name)
                if not is_alltoall_optimization_supported(name, param, transformer_config):
                    yield name, param
            # note
            # there is a bug in megatron GPTModel
            # "decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in state_dict().
            # for now we patch it by adding those keys to extra_keys.
            extra_keys = [x for x in model.state_dict().keys() if "_extra_state" not in x and x not in existing_keys]
            for name in extra_keys:
                yield name, model.state_dict()[name].to(torch.cuda.current_device())

    # we need first make all rank get full model information
    meta_info = []
    for scan_vpp_idx in range(vpp_size):
        existing_keys = set()
        model = unwrap_model(actor_module[scan_vpp_idx])
        for idx, (name, param) in enumerate(model.named_parameters()):
            existing_keys.add(name)
            if not is_alltoall_optimization_supported(name, param, transformer_config):
                meta_info.append((pp_rank, scan_vpp_idx, idx, name))
        extra_keys = [x for x in model.state_dict().keys() if "_extra_state" not in x and x not in existing_keys]
        for name in extra_keys:
            meta_info.append((pp_rank, scan_vpp_idx, idx, name))

    obj_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(object_list=obj_spec_output, obj=meta_info, group=pp_group)
    layer_list_meta = [item for sublist in obj_spec_output for item in sublist]

    gen_func = tensor_generator()

    # lazy load tensor for full model
    for cur_pp_rank, scan_vpp_idx, _, name in layer_list_meta:
        if model_config.tie_word_embeddings and ("output_layers" in name):
            import warnings

            warnings.warn("Current model sharing word and embedding weights, skip output layer conversion", stacklevel=2)
            continue

        if cur_pp_rank == pp_rank:
            try:
                cur_name, cur_tensor = next(gen_func)
            except StopIteration:
                cur_name, cur_tensor = None, None
            cur_name = normalize_model_name(name, cur_pp_rank, scan_vpp_idx, transformer_config)
        else:
            cur_tensor, cur_name = None, None

        # pp broadcast model tensor and name
        cur_name = broadcast_str_from_megatron_pp(cur_name)
        broad_pp_tensor = broadcast_from_megatron_pp(cur_tensor)

        # (xya): this is a hack to fix the name of the parameters
        while cur_name.startswith("module."):
            cur_name = cur_name[len("module."):]

        if mindspeed_args.moe_tp_extend_ep:
            if ".mlp.experts.weight" in cur_name and etp_size > 1:
                if os.getenv('NO_ALL_TO_ALL_RESHARD', '0') == '1':
                    # Original allgather reshard for EP params, inefficient in both memory usage and performance
                    num_experts = weight_converter.mcore_config.num_moe_experts
                    etp_params = [torch.empty_like(broad_pp_tensor) for _ in range(etp_size)]
                    torch.distributed.all_gather(etp_params, broad_pp_tensor, group=etp_group)
                else:
                    # EP param reshard method based on AllToAllV, efficient in both memory usage and performance
                    etp_params = ep_param_reshard_by_alltoallv(
                        param_name=cur_name,
                        ep_param_train=broad_pp_tensor,
                        num_experts=weight_converter.mcore_config.num_moe_experts,
                        weight1_key_name="mlp.experts.weight1",
                        weight2_key_name="mlp.experts.weight2"
                    )
                merge_params = default_tp_concat_fn(layer_name_mapping, cur_name, broad_pp_tensor, etp_params, model_config, convert_qkv_gate_up_by_simple_split)
                converted_names, converted_params = weight_converter.convert_param(cur_name, merge_params)

                yield from zip(converted_names, converted_params)
                continue
        else:
            if ".mlp.experts.weight" in cur_name and ep_size > 1:
                raise NotImplementedError(
                    f"The weight resharding of this mode still has problems and needs to be adapted!"
                )

        # tp all gather
        if tp_utils.is_tensor_parallel_param(broad_pp_tensor):
            # allocate a new tensor with proper size
            if all_gather_group_size <= 1:
                infer_params = [broad_pp_tensor]
            else:
                infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(all_gather_group_size)]
                torch.distributed.all_gather(infer_params, broad_pp_tensor, group=mpu.get_tensor_model_parallel_group())
            infer_params = default_tp_concat_fn(layer_name_mapping, cur_name, broad_pp_tensor, infer_params, model_config, convert_qkv_gate_up_by_simple_split)
        else:
            if layer_name_mapping.get("qkv_layer_name") in name and "layer_norm" not in name:
                if model_config.q_lora_rank is None:
                    q_rank = model_config.num_attention_heads * (model_config.qk_rope_head_dim + model_config.qk_nope_head_dim)
                    kv_a_rank = model_config.kv_lora_rank + model_config.qk_rope_head_dim
                    split_size = [q_rank, kv_a_rank]
                    infer_params = list(broad_pp_tensor.split(split_size))
                else:
                    q_a_rank = model_config.q_lora_rank
                    kv_a_rank = model_config.kv_lora_rank + model_config.qk_rope_head_dim
                    split_size = [q_a_rank, kv_a_rank]
                    infer_params = list(broad_pp_tensor.split(split_size))
            else:
                infer_params = broad_pp_tensor

        if not isinstance(infer_params, list):
            infer_params = [infer_params]
        converted_names, converted_params = weight_converter.convert_param(cur_name, infer_params)

        yield from zip(converted_names, converted_params)

    params_dict = {
        'actor_module': actor_module,
        'vpp_size': vpp_size,
        'pp_rank': pp_rank,
        'pp_size': pp_size,
        'pp_group': pp_group,
        'all_gather_group': all_gather_group,
        'all_gather_group_size': all_gather_group_size,
        'model_config': model_config,
        'transformer_config': transformer_config,
        'layer_name_mapping': layer_name_mapping,
        'weight_converter': weight_converter,
        'convert_qkv_gate_up_by_simple_split': convert_qkv_gate_up_by_simple_split
    }
    # Replace the looped broadcast calls with alltoall
    yield from process_params_use_alltoall_optimize(params_dict)


def get_transformer_layer_offset(pipeline_rank, config: TransformerConfig):
    # Get the index offset of any pipeline stage.
    offset = 0
    if config.pipeline_model_parallel_size > 1:
        if config.num_layer_list is not None:
            for i in range(pipeline_rank):
                offset += config.num_layer_list[i]
        else:
            num_layers_per_pipeline_rank = config.num_layers // config.pipeline_model_parallel_size
            offset = pipeline_rank * num_layers_per_pipeline_rank

    return offset


def is_alltoall_optimization_supported(name, param, transformer_config):
    # Parameters related to input/output and experts should not use alltoall optimization
    excluded_keywords = {
        ".mlp.experts.weight", "embedding", "output_layer", "linear_qkv", "final_layernorm"
    }

    # MLP dense layers and the last 58 layers cannot use alltoall due to shape mismatches
    mlp_keywords = {
        "linear_fc1.", "linear_fc2.", "router.weight"
    }

    # Extract layer number from parameter name
    layer_match = re.search(r'\.layers\.(\d+)\.', name)
    layer_num = int(layer_match.group(1)) if layer_match else -1

    # Get the minimum number of layers from num_layer_list
    min_layers_per_pp_rank = transformer_config.num_layers // transformer_config.pipeline_model_parallel_size
    if transformer_config.num_layer_list is not None:
        min_layers_per_pp_rank = min(transformer_config.num_layer_list)

    # Extract the minimum number of layers processed by a single pipeline parallel rank
    is_beyond_pp_min_layers = layer_num >= min_layers_per_pp_rank
    is_dense_layer = layer_match and layer_num < transformer_config.first_k_dense_replace
    is_mlp_layer = any(keyword in name for keyword in mlp_keywords)
    contains_excluded_keyword = any(keyword in name for keyword in excluded_keywords)

    should_include = (not contains_excluded_keyword and
                      not is_beyond_pp_min_layers and
                      not (is_dense_layer and is_mlp_layer))

    return should_include


def process_params_use_alltoall_optimize(params_dict):
    actor_module = params_dict['actor_module']
    vpp_size = params_dict['vpp_size']
    pp_rank = params_dict['pp_rank']
    pp_size = params_dict['pp_size']
    pp_group = params_dict['pp_group']
    all_gather_group = params_dict['all_gather_group']
    all_gather_group_size = params_dict['all_gather_group_size']
    model_config = params_dict['model_config']
    transformer_config = params_dict['transformer_config']
    layer_name_mapping = params_dict['layer_name_mapping']
    weight_converter = params_dict['weight_converter']
    convert_qkv_gate_up_by_simple_split = params_dict['convert_qkv_gate_up_by_simple_split']

    def tensor_generator():
        for scan_vpp_idx in range(vpp_size):
            model = unwrap_model(actor_module[scan_vpp_idx])
            for name, param in model.named_parameters():
                if is_alltoall_optimization_supported(name, param, transformer_config):
                    yield name, param

    meta_info = []
    for scan_vpp_idx in range(vpp_size):
        model = unwrap_model(actor_module[scan_vpp_idx])
        for name, param in model.named_parameters():
            if is_alltoall_optimization_supported(name, param, transformer_config):
                meta_info.append((pp_rank, scan_vpp_idx, name))
    gen_func = tensor_generator()

    # lazy load tensor for full model
    for cur_pp_rank, scan_vpp_idx, name in meta_info:
        try:
            cur_name, cur_tensor = next(gen_func)
        except StopIteration:
            cur_name, cur_tensor = None, None
        cur_name = normalize_model_name(name, cur_pp_rank, scan_vpp_idx, transformer_config)

        while cur_name.startswith("module."):
            cur_name = cur_name[len("module."):]

        # tp all gather
        if tp_utils.is_tensor_parallel_param(cur_tensor):
            if all_gather_group_size <= 1:
                infer_params = [cur_tensor]
            else:
                infer_params = [torch.empty_like(cur_tensor) for _ in range(all_gather_group_size)]
                torch.distributed.all_gather(infer_params, cur_tensor, group=all_gather_group)
            infer_params = default_tp_concat_fn(layer_name_mapping, cur_name, cur_tensor, infer_params, model_config, convert_qkv_gate_up_by_simple_split)
        else:
            infer_params = cur_tensor

        # Perform allgather of names within the PP group
        recv_names = [None] * pp_size
        torch.distributed.all_gather_object(object_list=recv_names, obj=cur_name, group=pp_group)
        # Perform alltoall of tensors within the PP group
        recv_tensors = gather_pp_params_by_alltoall(infer_params, pp_size, pp_group)

        for pp_idx in range(pp_size):
            cur_name = recv_names[pp_idx]
            infer_params = recv_tensors[pp_idx]

            if not isinstance(infer_params, list):
                infer_params = [infer_params]
            converted_names, converted_params = weight_converter.convert_param(cur_name, infer_params)
            yield from zip(converted_names, converted_params)


def gather_pp_params_by_alltoall(infer_params, pp_size, group):
    # Ensure input is formatted as a list
    if not isinstance(infer_params, list):
        infer_params = [infer_params]

    num_tensors = len(infer_params)
    result = [[] for _ in range(pp_size)]

    # Perform alltoall communication for each tensor
    for tensor in infer_params:
        recv_tensors = [
            torch.empty_like(tensor, device=tensor.device)
            for _ in range(pp_size)
        ]
        send_tensors = [tensor] * pp_size

        torch.distributed.all_to_all(
            output_tensor_list=recv_tensors,
            input_tensor_list=send_tensors,
            group=group
        )
        # Restore the original storage order before the alltoall operation
        for pp_idx in range(pp_size):
            result[pp_idx].append(recv_tensors[pp_idx])

    return result
