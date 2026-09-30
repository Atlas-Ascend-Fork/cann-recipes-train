"""Multi-tag CPU weight snapshot/restore for FSDP2 model parts.

Mirrors the weights_backuper interface the actor uses (backup / restore / copy / backup_tags). A
snapshot holds a CPU copy of the local shard of every parameter and buffer, so it is per-rank and
takes no collective; restore writes those copies straight back into the live tensors.

Reading the tensors rather than ``state_dict`` avoids the ``FusedQKVLinear`` hooks, which
all-gather the fused weights and hand back fresh allocations only ``load_state_dict`` can write.
"""

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """The shard this rank owns, sharing storage with ``tensor`` so copy_ lands on the model."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _reshard(model) -> None:
    """Put every FSDP unit back on its sharded layout, so snapshot and restore always see the same
    one. The tied tok_embeddings / norm / lm_head group has reshard_after_forward False and stays
    all-gathered otherwise; the walk is because ``FSDPModule.reshard`` is not recursive.
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def _live_tensors(model) -> dict[str, torch.Tensor]:
    # Non-persistent buffers included on purpose: the MoE expert bias and token counts are updated
    # in place by every forward, and a ref or old_actor forward must not leave its counts behind.
    return {**dict(model.named_parameters()), **dict(model.named_buffers())}


class TitanWeightsBackuper:
    def __init__(self, model_parts):
        self._model_parts = model_parts
        self._backups: dict[str, list[dict]] = {}

    @property
    def backup_tags(self):
        return list(self._backups)

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        """Snapshot the current weights. Call between steps, never between a forward and its
        backward: the reshard would throw away an all-gather that backward still needs."""
        for m in self._model_parts:
            _reshard(m)
        self._backups[tag] = [
            {name: _local(t).detach().to("cpu", copy=True) for name, t in _live_tensors(m).items()}
            for m in self._model_parts
        ]

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        """Write a snapshot back into the live model. Same call-site rule as backup."""
        for m, snapshot in zip(self._model_parts, self._backups[tag], strict=True):
            _reshard(m)
            live = _live_tensors(m)
            assert live.keys() == snapshot.keys(), (
                f"weight snapshot '{tag}' does not match the model: "
                f"missing={sorted(live.keys() - snapshot.keys())[:5]}, "
                f"unexpected={sorted(snapshot.keys() - live.keys())[:5]}"
            )
            for name, saved in snapshot.items():
                target = _local(live[name])
                assert target.shape == saved.shape and target.dtype == saved.dtype, (
                    f"weight snapshot '{tag}' does not match {name}: model has "
                    f"{tuple(target.shape)}/{target.dtype}, snapshot has "
                    f"{tuple(saved.shape)}/{saved.dtype}"
                )
                target.copy_(saved)

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str) -> None:
        self._backups[dst_tag] = [
            {name: t.clone() for name, t in snapshot.items()} for snapshot in self._backups[src_tag]
        ]
