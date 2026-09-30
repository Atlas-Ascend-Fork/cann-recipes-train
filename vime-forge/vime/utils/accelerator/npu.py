"""Ascend NPU accelerator implementation."""

from __future__ import annotations

from typing import Any

import torch

from .torch_accelerator import TorchAccelerator


class NPUAccelerator(TorchAccelerator):
    name = "npu"
    device_type = "npu"
    communication_backend_name = "hccl"

    def _module(self) -> Any:
        return getattr(torch, "npu", None)

    def is_available(self) -> bool:
        module = self._module()
        checker = getattr(module, "is_available", None)
        try:
            return bool(module is not None and checker is not None and checker())
        except (ImportError, RuntimeError):
            return False

    @property
    def visible_devices_env(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    def supports(self, capability: str) -> bool:
        if capability in {"npu", "hccl"}:
            return True
        return super().supports(capability)

    def patch_device_compat(self) -> None:
        return None
