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

"""Translate HF processor outputs using TorchTitan's image layout and packed MRoPE builder."""

import inspect
from typing import Any

import torch

from ...multimodal import VisionModelKwargs, config_field, iter_sample_inputs

# HF processor keys. TorchTitan's forward takes "grid_thw"; the HF processor emits
# "image_grid_thw".
_HF_IMAGE_PATCHES_KEY = "pixel_values"
_HF_IMAGE_GRID_KEY = "image_grid_thw"
_HF_VIDEO_KEYS = ("pixel_values_videos", "video_grid_thw")

# Placeholders for the MultiModalCollator fields _build_mrope_positions never reads.
_MROPE_COLLATOR_PATCH_SIZE = 16
_MROPE_COLLATOR_TEMPORAL_PATCH_SIZE = 2


def bind(model, *, hf_config, args):
    """Resolve the per-checkpoint state once; return the per-micro-batch build.

    Registered by ``@vision_recipe`` and called by ``TitanEngine`` after parallelization.
    The engine checks CP/SPMD support through ``assert_vision_parallelism``.
    """
    special_token_ids = resolve_special_token_ids(hf_config)
    spatial_merge_size = int(model.spatial_merge_size)
    mrope_collator = _mrope_collator(spatial_merge_size)

    def build(samples, packed_ids, position_ids) -> dict:
        kwargs = build_vision_model_kwargs(
            samples,
            packed_ids,
            position_ids,
            special_token_ids=special_token_ids,
            spatial_merge_size=spatial_merge_size,
            mrope_collator=mrope_collator,
        )
        return {**kwargs.item_indexed, **kwargs.token_indexed}

    return build


def resolve_special_token_ids(hf_config: Any) -> dict[str, int]:
    """Resolve the ``special_tokens`` forward kwarg from config.json (nested nodes
    included). A missing id raises rather than defaulting to -1: a wrong id scatters
    vision embeddings onto the wrong tokens silently.
    """
    sources = [hf_config]
    for nested in ("text_config", "vision_config"):
        child = config_field(hf_config, nested)
        if child is not None:
            sources.append(child)

    special_token_ids: dict[str, int] = {}
    for kwarg_name, config_key in (("image_id", "image_token_id"), ("video_id", "video_token_id")):
        for source in sources:
            value = config_field(source, config_key)
            if value is not None:
                special_token_ids[kwarg_name] = int(value)
                break
        else:
            raise ValueError(
                f"The checkpoint config has no {config_key!r}, so the Qwen3.5 vision "
                "placeholder id cannot be resolved. A multimodal recipe needs both "
                "image_token_id and video_token_id; add them to config.json or use the "
                "text-only recipe (vime_qwen35_4b)."
            )
    return special_token_ids


def _as_grid(grid: torch.Tensor, sample_index: int) -> torch.Tensor:
    """Normalize one sample's grid to a CPU ``(n_items, 3)`` int64 tensor.

    Consumed on the host (patch split points, MRoPE), so it is moved once here
    instead of forcing a device sync per read.
    """
    grid = grid.detach().to(device="cpu", dtype=torch.long)
    if grid.ndim == 1:
        grid = grid.view(1, -1)
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError(
            f"Sample {sample_index}: image_grid_thw must be (n_items, 3) with [T, H, W] in "
            f"raw patches, got shape {tuple(grid.shape)}."
        )
    return grid


def _split_patches_by_grid(
    patches_flat: torch.Tensor, grid_thw_cpu: torch.Tensor, sample_index: int
) -> list[torch.Tensor]:
    """Cut the processor's packed patch stream back into one tensor per visual item.

    Split points are the cumulative sum of ``grid_thw.prod(-1)``, the same arithmetic
    the model uses to size each item (``vision_encoder.py``, ``model.py``).
    """
    if patches_flat.ndim != 2:
        raise ValueError(
            f"Sample {sample_index}: pixel_values must be (total_patches, patch_dim), "
            f"got shape {tuple(patches_flat.shape)}."
        )
    counts = [int(t * h * w) for t, h, w in grid_thw_cpu.tolist()]
    total = sum(counts)
    if total != patches_flat.shape[0]:
        raise ValueError(
            f"Sample {sample_index}: pixel_values has {patches_flat.shape[0]} patch rows but "
            f"image_grid_thw accounts for {total}. The processor output and its grid are "
            "inconsistent; the patch stream cannot be split into visual items."
        )
    return list(torch.split(patches_flat, counts, dim=0))


def _pad_patch_items(
    items: list[torch.Tensor], *, spatial_merge_size: int
) -> torch.Tensor:
    """Pad per-item patches into TorchTitan's ``(num_items, max_num_patch, patch_dim)``.

    ``max_num_patch`` rounds up to a multiple of ``spatial_merge_size ** 2``, the
    ``PatchMerger`` fold width (``mm_collator.py``). Pad rows stay zero and are
    masked out by the ViT block mask.
    """
    merge_unit = int(spatial_merge_size) ** 2
    max_num_patch = max(item.shape[0] for item in items)
    if max_num_patch % merge_unit != 0:
        max_num_patch = (max_num_patch // merge_unit + 1) * merge_unit

    # Items from all samples must share patch_dim.
    patch_dim = items[0].shape[1]
    padded = items[0].new_zeros(len(items), max_num_patch, patch_dim)
    for index, item in enumerate(items):
        padded[index, : item.shape[0]] = item
    return padded


def _assert_mrope_builder_contract(collator_cls) -> None:
    """Reject a TorchTitan whose private packed-MRoPE hook changed shape.

    ``_build_mrope_positions`` is not public API, and it is called positionally below, so a
    signature change must fail here instead of inside the training forward.
    """
    builder = getattr(collator_cls, "_build_mrope_positions", None)
    if builder is None:
        raise RuntimeError(
            f"{collator_cls.__module__}.{collator_cls.__name__} no longer exposes "
            "_build_mrope_positions, which builds the packed MRoPE positions."
        )
    expected = ("tokens", "grid_thw", "grid_thw_videos", "positions", "image_token_id", "video_token_id")
    parameters = inspect.signature(builder).parameters
    missing = [name for name in expected if name not in parameters]
    if missing:
        raise RuntimeError(
            f"MultiModalCollator._build_mrope_positions changed its parameters: missing "
            f"{missing}, got {tuple(parameters)}. Update _build_mrope_positions_via_titan."
        )


def _mrope_collator(spatial_merge_size: int):
    """Build a collator for packed MRoPE, retaining torchtitan-npu's patched method.

    Only ``spatial_merge_size`` and ``patch_order`` are used; other fields are placeholders.
    Import lazily to defer the image-processing dependencies.
    """
    from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator

    _assert_mrope_builder_contract(MultiModalCollator)
    return MultiModalCollator(
        batch_size=1,
        seq_len=0,
        max_images_per_batch=0,
        patch_size=_MROPE_COLLATOR_PATCH_SIZE,
        temporal_patch_size=_MROPE_COLLATOR_TEMPORAL_PATCH_SIZE,
        spatial_merge_size=int(spatial_merge_size),
        tokenizer=None,
        build_mrope_positions=True,
        patch_order="block",
    )


def _build_mrope_positions_via_titan(
    packed_ids: torch.Tensor,
    position_ids: torch.Tensor,
    grid_thw_cpu: torch.Tensor,
    *,
    image_id: int,
    video_id: int,
    spatial_merge_size: int,
    mrope_collator=None,
) -> torch.Tensor:
    """Build ``(1, T, 3)`` MRoPE positions with upstream's packed implementation.

    Build on CPU to avoid per-item device synchronization; MRoPE expects channels last.
    """
    collator = mrope_collator if mrope_collator is not None else _mrope_collator(spatial_merge_size)
    # 1-D and equal-length by construction: one torch.cat pair in build_packed_microbatch.
    tokens_cpu = packed_ids.detach().to(device="cpu", dtype=torch.long).unsqueeze(0)
    positions_cpu = position_ids.detach().to(device="cpu", dtype=torch.long)

    mrope_positions = collator._build_mrope_positions(
        tokens_cpu,
        grid_thw_cpu,
        None,  # videos are rejected before this point
        positions_cpu.unsqueeze(0),
        image_token_id=image_id,
        video_token_id=video_id,
    )
    return mrope_positions.to(device=packed_ids.device)


def build_vision_model_kwargs(
    samples,
    packed_ids: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    special_token_ids: dict[str, int],
    spatial_merge_size: int,
    mrope_collator=None,
) -> VisionModelKwargs:
    """Translate one micro-batch's processor outputs into Qwen3.5 forward kwargs.

    ``samples`` contains one dict (or None) per sample in ``packed_ids`` concatenation
    order. Always return ``special_tokens`` because the model reads it even without
    images; return patch, grid and MRoPE kwargs only when images are present.
    """
    items: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    for sample_index, mm_inputs in iter_sample_inputs(samples):
        if any(mm_inputs.get(key) is not None for key in _HF_VIDEO_KEYS):
            raise NotImplementedError(
                f"Sample {sample_index} carries video inputs ({_HF_VIDEO_KEYS}): the torchtitan "
                "backend's Qwen3.5 vision path handles images only. Video needs the collator's "
                "frame handling and the contiguous-video MRoPE semantics, neither of which is "
                "verified here."
            )
        patches = mm_inputs.get(_HF_IMAGE_PATCHES_KEY)
        grid = mm_inputs.get(_HF_IMAGE_GRID_KEY)
        if patches is None and grid is None:
            continue
        if patches is None or grid is None:
            raise ValueError(
                f"Sample {sample_index} has only one of {_HF_IMAGE_PATCHES_KEY!r} / "
                f"{_HF_IMAGE_GRID_KEY!r} (keys present: {sorted(mm_inputs)}). The patch "
                "stream cannot be split into visual items without its grid."
            )
        grid_cpu = _as_grid(grid, sample_index)
        items.extend(_split_patches_by_grid(patches, grid_cpu, sample_index))
        grids.append(grid_cpu)

    item_indexed: dict[str, Any] = {"special_tokens": dict(special_token_ids)}
    if not items:
        return VisionModelKwargs(item_indexed=item_indexed)

    pixel_values = _pad_patch_items(items, spatial_merge_size=spatial_merge_size)
    grid_thw_cpu = torch.cat(grids, dim=0)

    item_indexed["pixel_values"] = pixel_values
    # Keep the grid on the same device as the patches.
    item_indexed["grid_thw"] = grid_thw_cpu.to(device=pixel_values.device)
    return VisionModelKwargs(
        item_indexed=item_indexed,
        token_indexed={
            "mrope_positions": _build_mrope_positions_via_titan(
                packed_ids,
                position_ids,
                grid_thw_cpu,
                image_id=special_token_ids["image_id"],
                video_id=special_token_ids["video_id"],
                spatial_merge_size=spatial_merge_size,
                mrope_collator=mrope_collator,
            )
        },
    )
