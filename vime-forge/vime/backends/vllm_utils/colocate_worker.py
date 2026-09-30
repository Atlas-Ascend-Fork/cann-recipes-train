"""Backend-neutral NPU vLLM worker hook for colocated weight updates."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vime.utils import accelerator


def _post_process_colocate_moe_layers(layers: Sequence[torch.nn.Module]) -> None:
    for layer in layers:
        process_weights = getattr(getattr(layer, "quant_method", None), "process_weights_after_loading", None)
        if callable(process_weights):
            process_weights(layer)


def _remember_post_process_layer(layers: list[torch.nn.Module] | None, layer: torch.nn.Module) -> None:
    if layers is not None and not any(existing is layer for existing in layers):
        layers.append(layer)


def _load_colocate_weights_direct(
    model: torch.nn.Module,
    weights: list[tuple[str, torch.Tensor]],
    *,
    tp_rank: int,
    post_process_layers: list[torch.nn.Module] | None = None,
) -> None:
    """Load HF-named dense or MoE tensors into vLLM's packed parameters."""
    from vllm.model_executor.utils import get_packed_modules_mapping

    packed_map = get_packed_modules_mapping(model)
    sub_to_packed = {
        sub_name: (packed_name, index)
        for packed_name, sub_names in packed_map.items()
        for index, sub_name in enumerate(sub_names)
    }
    qkv_shard_ids = {"q_proj": "q", "k_proj": "k", "v_proj": "v"}
    params = dict(model.named_parameters())
    get_expert_mapping = getattr(model, "get_expert_mapping", None)
    expert_mapping = get_expert_mapping() if callable(get_expert_mapping) else []

    def load_expert(name: str, weight: torch.Tensor) -> bool:
        is_expert = False
        for param_name, weight_name, expert_id, shard_id in expert_mapping:
            if weight_name not in name:
                continue
            is_expert = True
            mapped_name = name.replace(weight_name, param_name)
            param = params.get(mapped_name)
            if param is None:
                continue
            module = model.get_submodule(mapped_name.rsplit(".", 1)[0])
            loader = getattr(param, "weight_loader", None) or getattr(module, "weight_loader", None)
            if loader is None:
                raise RuntimeError(f"Cannot load MoE expert weight without weight_loader: {mapped_name}")
            if loader(param, weight, mapped_name, shard_id=shard_id, expert_id=expert_id, return_success=True):
                if type(module).__name__ == "AscendFusedMoE":
                    _remember_post_process_layer(post_process_layers, module)
                return True
        return is_expert

    for name, weight in weights:
        if load_expert(name, weight):
            continue
        parts = name.split(".")
        shard_id = None
        for index, part in enumerate(parts):
            if part in sub_to_packed:
                packed_name, packed_index = sub_to_packed[part]
                parts[index] = packed_name
                shard_id = qkv_shard_ids.get(part, packed_index)
                break
        mapped_name = ".".join(parts)
        try:
            param = model.get_parameter(mapped_name)
        except AttributeError:
            continue
        loader = getattr(param, "weight_loader", None)
        if loader is not None and shard_id is not None:
            loader(param, weight, loaded_shard_id=shard_id)
        elif loader is not None:
            loader(param, weight)
        elif param.shape == weight.shape:
            param.data.copy_(weight)
        else:
            for dim, size in enumerate(param.shape):
                if size != weight.shape[dim]:
                    param.data.copy_(weight.narrow(dim, tp_rank * size, size))
                    break


def _patch_npu_colocate_worker() -> None:
    """Use vLLM-Ascend NPU IPC transport with VIME's direct parameter loader."""
    try:
        from vllm_ascend.worker.worker import NPUWorker
    except ImportError:
        return
    if getattr(NPUWorker, "_vime_colocate_patched", False):
        return

    original_start = NPUWorker.start_weight_update
    original_finish = NPUWorker.finish_weight_update

    def start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        original_start(self, is_checkpoint_format=is_checkpoint_format)
        self._vime_moe_layers_to_post_process = []

    def finish_weight_update(self) -> None:
        try:
            if not getattr(self, "_is_checkpoint_format", True):
                with torch.device(self.device):
                    _post_process_colocate_moe_layers(self._vime_moe_layers_to_post_process)
            original_finish(self)
        finally:
            self._vime_moe_layers_to_post_process = []

    def update_weights(self, update_info: dict) -> None:
        self._check_weight_transfer_engine()
        if not getattr(self, "_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before update_weights.")
        typed_info = self.weight_transfer_engine.parse_update_info(update_info)
        model = self.model_runner.model
        with torch.device(self.device):
            if self._is_checkpoint_format:
                self.weight_transfer_engine.receive_weights(typed_info, load_weights=model.load_weights)
            else:
                from vllm.config import set_current_vllm_config

                tp_rank = self.rank % self.parallel_config.tensor_parallel_size

                def load_direct(weights: list[tuple[str, torch.Tensor]]) -> None:
                    with torch.no_grad(), set_current_vllm_config(self.vllm_config):
                        _load_colocate_weights_direct(
                            model,
                            weights,
                            tp_rank=tp_rank,
                            post_process_layers=self._vime_moe_layers_to_post_process,
                        )

                self.weight_transfer_engine.receive_weights(typed_info, load_weights=load_direct)
        torch.npu.synchronize()

    NPUWorker.start_weight_update = start_weight_update
    NPUWorker.finish_weight_update = finish_weight_update
    NPUWorker.update_weights = update_weights
    NPUWorker._vime_colocate_patched = True


class _VLLMHijack:
    @staticmethod
    def patch_npu_rotary_emb() -> None:
        try:
            from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
        except ImportError:
            return
        if getattr(ApplyRotaryEmb, "_vime_npu_rotary_patched", False):
            return

        def init(self, enforce_enable=False, is_neox_style=True, enable_fp32_compute=False) -> None:
            super(ApplyRotaryEmb, self).__init__(enforce_enable=enforce_enable)
            self.is_neox_style = is_neox_style
            self.enable_fp32_compute = enable_fp32_compute
            self.apply_rotary_emb_flash_attn = None

        ApplyRotaryEmb.__init__ = init
        ApplyRotaryEmb._vime_npu_rotary_patched = True


class vLLMColocateWorkerExtension:
    """vLLM ``--worker-extension-cls`` entry point without a trainer dependency."""

    def __new__(cls, **kwargs):
        if accelerator.device_type() == "npu":
            _VLLMHijack.patch_npu_rotary_emb()
            _patch_npu_colocate_worker()
        return super().__new__(cls)
