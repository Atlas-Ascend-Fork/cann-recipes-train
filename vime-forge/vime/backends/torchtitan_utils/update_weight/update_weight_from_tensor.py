# Adapted from
# https://github.com/vllm-project/vime/blob/5ab8cc75/vime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
# which is derived from THUDM's slime at:
# https://github.com/THUDM/slime/blob/v0.3.1/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 Zhipu AI
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

"""NPU IPC weight sync for colocated TorchTitan and vLLM engines."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist

from vime.utils import accelerator
from vime.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


class UpdateWeightFromTensor:
    """Push TorchTitan's HF-named tensors to colocated NPU vLLM engines.

    ``weights_iter_factory`` must be invoked by every training rank because its
    DTensor materialisation can use collectives.  Every rank must also enter
    NPU IPC with identically ordered buckets: the transport all-gathers the
    local IPC handles and only its global rank zero issues the HTTP request.
    """

    def __init__(self, args, weights_iter_factory):
        if not accelerator.device_type() == "npu":
            raise RuntimeError("TorchTitan colocated weight sync is currently supported on NPU only.")
        self.args = args
        self._weights_iter_factory = weights_iter_factory
        self.weight_version = 0
        self.rollout_engines = []
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._ipc_initialized = False
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    def connect_rollout_engines(
        self,
        rollout_engines,
        rollout_engine_lock,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs=None,
    ) -> None:
        del rollout_engine_lock, engine_parallel_configs
        self.rollout_engines = list(rollout_engines)
        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(self.rollout_engines)
        if engine_gpu_offsets is None:
            engine_gpu_offsets = []
            offset = 0
            for count in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += count

        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        # Unlike Megatron's mixed colocated/distributed updater, TorchTitan's
        # NPU IPC path requires every rollout engine to be colocated.
        if any(offset + count > total_actor_gpus for offset, count in zip(
            engine_gpu_offsets, engine_gpu_counts, strict=True
        )):
            raise RuntimeError("TorchTitan IPC updater only supports fully colocated rollout engines.")

        rank = dist.get_rank()
        if self._ipc_gather_group is None:
            # Keep the group construction identical to the Megatron backend:
            # every rank creates every slot group, while only the local slot
            # retains its group handle and source rank.
            for offset, count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
                group_ranks = list(range(offset, offset + count))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if rank in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = offset

        self._ipc_engine = None
        for engine, start, count in zip(
            self.rollout_engines, engine_gpu_offsets, engine_gpu_counts, strict=True
        ):
            if start <= rank < start + count:
                self._ipc_engine = engine
                break
        if self._ipc_engine is None:
            raise RuntimeError(f"Training rank {rank} is not assigned to a colocated rollout engine.")

        if rank == 0 and self.rollout_engines and not self._ipc_initialized:
            ray.get([engine.init_weight_transfer_engine.remote({"init_info": {}}) for engine in self.rollout_engines])
            self._ipc_initialized = True

    @torch.no_grad()
    def update_weights(self) -> None:
        if self._ipc_engine is None or self._ipc_gather_src is None:
            raise RuntimeError("connect_rollout_engines must run before update_weights.")

        self.weight_version += 1
        rank = dist.get_rank()
        if rank == 0 and not self.args.offload_rollout:
            # sleep() already paused and cleared the offloaded engines. A DP
            # pause here would execute dummy forwards before KV restoration.
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        if rank == self._ipc_gather_src:
            # get_hf_per_tensor emits checkpoint names/layouts, including packed
            # experts and scales. Let the model loader shard and postprocess them.
            ray.get(self._ipc_engine.start_weight_update.remote())
        dist.barrier(group=get_gloo_group())

        bucket: list[tuple[str, torch.Tensor]] = []
        bucket_bytes = 0
        limit = int(getattr(self.args, "update_weight_buffer_size", 512 * 1024 * 1024))
        # Buckets are transport chunks and can split a layer. Keep one reload
        # session until all weights arrive: an intermediate finish/start can
        # materialize incomplete layers and overwrite already updated weights.
        for name, tensor in self._weights_iter_factory():
            tensor_bytes = tensor.numel() * tensor.element_size()
            bucket.append((name, tensor.contiguous()))
            bucket_bytes += tensor_bytes
            if bucket_bytes >= limit:
                self._send_bucket(bucket)
                bucket, bucket_bytes = [], 0

        if bucket:
            self._send_bucket(bucket)
        dist.barrier(group=get_gloo_group())
        torch.npu.synchronize()

        if rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.finish_weight_update.remote(weight_version=str(self.weight_version)))
        dist.barrier(group=get_gloo_group())

        # Split wake restores weights before this update and KV cache afterwards.
        # Keep scheduling paused until rollout_manager.onload_kv() completes.
        if rank == 0 and not self.args.offload_rollout:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def capture_train_forward_weights(self, rollout_id: int) -> None:
        """IPC transfer does not currently emit weight-consistency manifests."""
        del rollout_id

    def _send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        """Send one bucket of tensors via vLLM's trainer-side NPU IPC engine.

        ``NPUIPCWeightTransferEngine.trainer_send_weights`` and
        ``NPUIPCTrainerSendWeightsArgs`` were removed in vLLM 0.30 in favour of
        ``NPUIPCTrainerWeightTransferEngine``. VIME still owns the start/finish
        handshake and the bucketing that bounds memory, so here we only build a
        throwaway engine for this bucket and drive its non-packed send directly.
        IPC handles are all-gathered within this slot's group
        (``_ipc_gather_group``); packed mode is intentionally not used because it
        caps each tensor at the buffer size.
        """
        from vllm.distributed.weight_transfer.base import WeightSource
        from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerWeightTransferEngine

        url = ray.get(self._ipc_engine.get_url.remote())
        if url is None:
            raise RuntimeError("NPU IPC colocate requires a vLLM engine HTTP URL")

        process_group = self._ipc_gather_group
        is_sender = dist.get_rank() == self._ipc_gather_src

        class _BucketSource(WeightSource):
            def __iter__(self):
                return iter(bucket)

            def metadata(self) -> list:
                # The NPU IPC engine only iterates the source on the send path.
                return []

        class _SlotScopedNPUIPCTrainerEngine(NPUIPCTrainerWeightTransferEngine):
            """NPU IPC trainer engine scoped to one colocation slot's group.

            The stock engine all-gathers IPC handles over the default process
            group and treats global rank 0 as the sender. VIME's colocated layout
            groups each engine slot's co-located ranks (``_ipc_gather_group``) and
            lets that slot's source rank ship, so thread the slot group through
            the collectives.
            """

            def __init__(self, *, process_group=None, **kwargs):
                super().__init__(**kwargs)
                self._process_group = process_group

            @staticmethod
            def _merge_gathered_handles(handles, gathered):
                merged = []
                for param_idx in range(len(handles)):
                    merged_handle = {}
                    for rank_handles in gathered:
                        if rank_handles is not None:
                            merged_handle.update(rank_handles[param_idx])
                    merged.append(merged_handle)
                return merged

            def send(self, source):
                return self._send(source)

            def post_send_sync(self) -> None:
                self._post_send_sync()

            def _all_gather_and_merge_handles(self, handles):
                pg = self._process_group
                if not torch.distributed.is_initialized() or torch.distributed.get_world_size(group=pg) == 1:
                    return handles
                world_size = torch.distributed.get_world_size(group=pg)
                gathered = [None] * world_size
                torch.distributed.all_gather_object(gathered, handles, group=pg)
                torch.distributed.barrier(group=pg)
                torch.npu.synchronize()
                if self.is_sender:
                    return self._merge_gathered_handles(handles, gathered)
                return [{} for _ in handles]

            def _post_send_sync(self) -> None:
                pg = self._process_group
                if torch.distributed.is_initialized() and torch.distributed.get_world_size(group=pg) > 1:
                    torch.distributed.barrier(group=pg)
                torch.npu.synchronize()

        engine = _SlotScopedNPUIPCTrainerEngine(
            client=HTTPVLLMWeightSyncClient(url),
            source=_BucketSource(),
            is_sender=is_sender,
            packed=False,
            process_group=process_group,
        )
        logger.debug("Sending %d TorchTitan tensors to colocated vLLM engine", len(bucket))
        weight_refs = None
        try:
            weight_refs = engine.send(engine.source)
            engine.post_send_sync()
        finally:
            # IPC rebuild handles do not own the sender storage. Keep the
            # contiguous tensors alive through the slot barrier, then break
            # every source/closure reference before the next bucket/update.
            del weight_refs
            engine.source = None
            engine.client = None
            bucket.clear()
            del engine
