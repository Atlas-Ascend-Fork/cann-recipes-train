"""Broadcast training weights to sglang via NCCL/HCCL in decoupled deployment.

Source rank = training global rank 0 (no PP); all other ranks only participate in full_tensor
all-gathers and do not broadcast.
Broadcast primitives (_connect / _disconnect / _broadcast) are based on the vllm-ascend
WeightTransferEngine, with no top-level megatron dependency.
"""

import logging
import os
import socket
import time
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist

from vime.utils import accelerator
from vime.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


def _leader_relay_enabled() -> bool:
    return accelerator.device_type() == "npu" and os.environ.get("VIME_TORCHTITAN_WEIGHT_SYNC_LEADER_RELAY") == "1"


class UpdateWeightFromDistributed:
    """Broadcast training weights to sglang via NCCL/HCCL in decoupled deployment.

    Usage: construct in actor.init -> connect_rollout_engines(...) -> update_weights().
    Source rank = training global rank 0 (no PP); all other ranks only participate in full_tensor all-gathers.
    """

    def __init__(self, args, weights_iter_factory):
        self.args = args
        self._weights_iter_factory = weights_iter_factory  # callable[[], Iterator[(name, tensor)]]
        self.weight_version = 0
        self._group_name = "vime-tt"
        self._model_update_groups = None
        self._is_src = dist.get_rank() == 0

    def connect_rollout_engines(
        self, rollout_engines, rollout_engine_lock, engine_gpu_counts=None,
        engine_gpu_offsets=None, engine_parallel_configs=None,
    ):
        del engine_parallel_configs
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        if self._is_src:
            if self._model_update_groups is not None:
                _disconnect(self._group_name, rollout_engines)
            self._model_update_groups = _connect(self.args, self._group_name, rollout_engines, engine_gpu_counts)

    def disconnect_rollout_engines(self):
        if not self._is_src or self._model_update_groups is None:
            return
        _disconnect(self._group_name, self.rollout_engines)
        self._model_update_groups = None

    @torch.no_grad()
    def update_weights(self):
        self.weight_version += 1
        if self._is_src:
            ray.get([e.pause_generation.remote() for e in self.rollout_engines])
            ray.get([e.flush_cache.remote() for e in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        # Begin the weight update session: the PR#9152 worker is a state machine, so
        # start_weight_update must be called before update_weights.
        if self._is_src:
            ray.get([e.start_weight_update.remote(is_checkpoint_format=True) for e in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        try:
            bucket, bucket_bytes = [], 0
            _logged_expert = False
            limit = self.args.update_weight_buffer_size
            # ⚠ All ranks iterate and track the same bucket boundaries, so the per-bucket sync
            # stays in lockstep with the collectives; only rank0 accumulates and broadcasts. A
            # rank0-only sync desyncs the collective and ends in an HCCL timeout.
            for name, tensor in self._weights_iter_factory():
                if self._is_src and not _logged_expert and ".experts." in name:
                    # weight-sync self-check: expert dtype must be bf16 (not fp32), plus the transport in use.
                    logger.info(
                        "[weightsync-selfcheck] first expert name=%s dtype=%s contiguous=%s",
                        name, tensor.dtype, tensor.is_contiguous(),
                    )
                    _logged_expert = True
                bucket_bytes += tensor.numel() * tensor.element_size()
                if self._is_src:
                    bucket.append((name, tensor))
                if bucket_bytes > limit:
                    torch.cuda.synchronize()
                    if self._is_src:
                        self._flush(bucket)
                        bucket = []
                    bucket_bytes = 0
            # Final partial bucket: sync on all ranks, flush on rank0.
            if bucket_bytes > 0:
                torch.cuda.synchronize()
                if self._is_src and bucket:
                    self._flush(bucket)
        finally:
            # ⚠ Device sync before ending the session: the HCCL send is async, and a train or
            # backup that races it corrupts the weights vllm receives.
            if self._is_src:
                torch.cuda.synchronize()
                ray.get([e.finish_weight_update.remote() for e in self.rollout_engines])
            dist.barrier(group=get_gloo_group())

        if self._is_src:
            ray.get([e.continue_generation.remote() for e in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _flush(self, named_tensors):
        # Lock engines to prevent broadcast deadlock.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        refs = _broadcast(
            self._group_name, self._model_update_groups, self.weight_version,
            self.rollout_engines, named_tensors,
        )
        ray.get(refs)
        ray.get(self.rollout_engine_lock.release.remote())


# ---- broadcast primitives (vllm-ascend WeightTransferEngine transport, backend-agnostic) ----

def _connect(args, group_name, rollout_engines: Sequence, engine_gpu_counts=None):
    """Trainer (rank0) side: establish process group via vllm-ascend/vllm WeightTransferEngine.trainer_init.

    Cannot use torch native init_process_group + dist.broadcast: the rollout side (vllm-ascend PR#9152
    init_transfer_engine -> _stateless_init_process_group + broadcast_obj(unique_id)) expects the
    trainer_init stateless-PG handshake and must be paired with trainer_init / trainer_send_weights.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    master_address = ray._private.services.get_node_ip_address()
    backend = "hccl" if accelerator.device_type() == "npu" else "nccl"

    if _leader_relay_enabled():
        if not accelerator.device_type() == "npu":
            raise RuntimeError("leader relay weight sync is only supported on NPU/HCCL")
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

        groups = []
        for i, engine in enumerate(rollout_engines):
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            world_size = 2
            refs = [
                engine.init_weights_update_group.remote(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=1,
                    world_size=world_size,
                    group_name=f"{group_name}-engine-{i}",
                    backend=backend,
                    leader_relay=True,
                )
            ]
            init_info = {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
            accelerator.synchronize()
            accelerator.empty_cache()
            groups.append(HCCLWeightTransferEngine.trainer_init(init_info))
            ray.get(refs)
        return groups

    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1
    cumulative = [0]
    for c in engine_gpu_counts:
        cumulative.append(cumulative[-1] + c)
    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        for i, engine in enumerate(rollout_engines)
    ]
    init_info = {"master_address": master_address, "master_port": master_port, "world_size": world_size}
    accelerator.synchronize()
    accelerator.empty_cache()
    if accelerator.device_type() == "npu":
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

    groups = HCCLWeightTransferEngine.trainer_init(init_info)
    ray.get(refs)
    return groups


def _disconnect(group_name, rollout_engines):
    # The trainer holds a Py(H)cclCommunicator, not a torch c10d PG, so there is nothing to
    # destroy; destroy_process_group would raise. Notifying the engines is all that is left.
    refs = [e.destroy_weights_update_group.remote(group_name) for e in rollout_engines]
    ray.get(refs)


def _broadcast(group_name, group, weight_version, rollout_engines, named_tensors):
    if isinstance(group, list):
        refs = []
        for i, (engine, engine_group) in enumerate(zip(rollout_engines, group, strict=True)):
            refs.append(
                engine.update_weights_from_distributed.remote(
                    names=[n for n, _ in named_tensors],
                    dtypes=[p.dtype for _, p in named_tensors],
                    shapes=[p.shape for _, p in named_tensors],
                    weight_version=str(weight_version),
                    packed=True,
                )
            )
            _trainer_send_weights(engine_group, named_tensors)
        return refs

    refs = [
        e.update_weights_from_distributed.remote(
            names=[n for n, _ in named_tensors],
            dtypes=[p.dtype for _, p in named_tensors],
            shapes=[p.shape for _, p in named_tensors],
            weight_version=str(weight_version),
            packed=True,
        )
        for e in rollout_engines
    ]
    # ⚠ No sync here: it would run on rank0 only. The bucket-boundary sync in update_weights is
    # the one that guards this async send, and it runs on every rank.
    _trainer_send_weights(group, named_tensors)
    return refs

def _trainer_send_weights(group, named_tensors):
    named_gpu_iter = (
        (name, (p.data if hasattr(p, "data") else p).contiguous()) for name, p in named_tensors
    )
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerSendWeightsArgs,
        HCCLWeightTransferEngine,
    )

    HCCLWeightTransferEngine.trainer_send_weights(
        named_gpu_iter, HCCLTrainerSendWeightsArgs(group=group, packed=True)
    )
