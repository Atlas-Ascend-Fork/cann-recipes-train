# coding=utf-8
# Adapted from
# models/deepseek_v4/utils/convert_model.py (bfloat16 branch)
# models/deepseek_v4_1/models/modules/engram.py (Engram.fp8_weight_loader)
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Convert native DeepSeek-V4.1-Flash (Hybrid FP8-MXFP4) weights to BF16.

Dequantization math is reused verbatim from the V4 script:

* ``weight_dequant`` and ``unpack_mxfloat4_to_fp32`` are copied verbatim from
  models/deepseek_v4/utils/convert_model.py. The only call-site change is the
  FP8 block dequant now passing ``block_size=32`` (V4.1 uses 32x32 blocks,
  V4 used 128x128); the MXFP4 path already passed ``block_size=32, is_mx=True``.
* ``dequant_fp8_row_group`` is ported from ``Engram.fp8_weight_loader`` in
  models/deepseek_v4_1/models/modules/engram.py (sharding removed) — V4.1's
  engram embed stores one scale row per weight row, a layout the V4 script
  does not have.

Loop-level differences vs the V4 bfloat16 branch (required by the V4.1
checkpoint, everything else mirrors the V4 script):

1. Quantized tensors are routed like the V4 script (int8-packed -> MXFP4,
   other element-size-1 dtypes -> FP8), plus the V4.1 engram embed layout:
   e4m3 with one scale row per weight row -> per-row-group dequant; other e4m3
   -> 32x32 block. V4.1's config.json nests the model fields under
   ``text_config``, so nothing is read from it here.
2. A missing ``.scale`` sibling for a quantized-dtype tensor is a hard error
   (the V4 script printed a warning and wrote the raw quantized bytes).
3. The fp8 ``quantization_config`` is removed from the output config.json and
   the model index is rewritten without ``.scale`` entries, with
   ``total_size`` recomputed (the V4 script emptied the index metadata).
4. ``--dry_run`` only runs the scale-pairing check over the model index.

The script runs on CPU only.
"""

import json
import os
import shutil
from argparse import ArgumentParser
from collections import Counter
from glob import glob

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

BLOCK = 32


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128, is_mx: bool=False) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.

    Copied verbatim from models/deepseek_v4/utils/convert_model.py.

    Args:
        weight (torch.Tensor): The quantized weight tensor of shape(M, N).
        scale (torch.Tensor): The scale tensor of shape (M // block_size, N // block_size),
            or (M, N // block_size) when `is_mx` is True.
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.

    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after scaling.
    """

    # Get the original dimensions of weight
    M, N = weight.shape
    weight = weight.to(torch.float32)
    scale = scale.to(torch.float32)

    if is_mx:
        scale_expanded = scale.repeat_interleave(block_size, dim=1)
    else:
        # Compute the effective block dimensions for scale
        scale_m, scale_n = scale.shape
        assert scale_m == (
            M + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
        assert scale_n == (
            N + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."

        # Expand scale to match the weight tensor's shape
        scale_expanded = scale.repeat_interleave(
            block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:M, :N]

    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded

    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())

    return dequantized_weight


def unpack_mxfloat4_to_fp32(packed_tensor):
    """
    Copied verbatim from models/deepseek_v4/utils/convert_model.py.
    """
    e2m1_values = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32, device=packed_tensor.device)

    low_4bits = packed_tensor & 0x0F
    high_4bits = (packed_tensor // 16) & 0x0F

    unpacked = torch.stack([low_4bits, high_4bits], dim=-1)

    fp32_tensor = e2m1_values[unpacked.long()]
    new_shape = list(packed_tensor.shape)
    new_shape[-1] = new_shape[-1] * 2

    return fp32_tensor.view(*new_shape)


def dequant_fp8_row_group(weight, scale):
    """
    Dequantize per-row-group FP8 tensors (the engram embed layout: one scale
    row per weight row). Ported from Engram.fp8_weight_loader in
    models/deepseek_v4_1/models/modules/engram.py, sharding removed.
    """
    rows, cols = weight.shape
    scale_rows, scale_cols = scale.shape
    if scale_rows != rows or scale_cols <= 0 or cols % scale_cols != 0:
        raise ValueError(
            "Engram FP8 scale shape does not match the weight: "
            f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}, "
            "expected one scale row per embedding row and an integral "
            "number of values per scale")
    block_size = cols // scale_cols

    return (weight.to(torch.float32)
            .unflatten(-1, (-1, block_size))
            * scale.to(torch.float32).unsqueeze(-1)
            ).flatten(-2).to(torch.bfloat16)


def sibling_scale_name(weight_name):
    """Return the quantization-scale sibling for a weight name, or None.

    Only ``<prefix>.weight`` tensors can carry a ``<prefix>.scale`` sibling;
    the suffix is stripped once from the end, so ``weights_proj.weight`` maps
    to ``weights_proj.scale`` (never corrupted by a blind str.replace).
    """
    if not weight_name.endswith(".weight"):
        return None
    candidate = weight_name[:-len(".weight")] + ".scale"
    return candidate if candidate.endswith(".scale") else None


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def copy_py_json(src, target):
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith(('.py', '.json', '.jinja')):
                src_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, src)
                dst_dir = os.path.join(target, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, file)
                shutil.copy2(src_path, dst_path)


def check_scale_pairing(weight_map):
    """Ensure every .scale tensor has a paired .weight; return scale names."""
    scale_names = {n for n in weight_map if n.endswith(".scale")}
    orphan = sorted(
        s for s in scale_names
        if s[:-len(".scale")] + ".weight" not in weight_map)
    if orphan:
        raise ValueError(
            f"{len(orphan)} .scale tensors have no paired .weight, "
            f"e.g. {orphan[:5]}")
    return scale_names


def dry_run(input_path):
    """Pairing check + classification stats over the model index (no I/O of
    weight data)."""
    model_index = load_json(
        os.path.join(input_path, "model.safetensors.index.json"))
    weight_map = model_index["weight_map"]
    scale_names = check_scale_pairing(weight_map)

    quantized = sorted(
        n for n in weight_map
        if n.endswith(".weight")
        and n[:-len(".weight")] + ".scale" in scale_names)
    quantized_set = set(quantized)
    passthrough = [n for n in weight_map
                   if n not in scale_names and n not in quantized_set]

    print(f"index tensors      : {len(weight_map)}")
    print(f"quantized weights  : {len(quantized)} (will be dequantized)")
    print(f"scale tensors      : {len(scale_names)} (will be dropped)")
    print(f"passthrough tensors: {len(passthrough)} (copied unchanged)")
    suffixes = Counter(name.rsplit(".", 1)[-1] for name in passthrough)
    print("passthrough suffix breakdown:")
    for suffix, count in sorted(suffixes.items(), key=lambda kv: -kv[1]):
        print(f"  .{suffix:<16} {count}")


def convert(input_path, output_path):
    torch.set_default_dtype(torch.bfloat16)
    model_index = load_json(
        os.path.join(input_path, "model.safetensors.index.json"))
    config = load_json(os.path.join(input_path, "config.json"))
    weight_map = model_index["weight_map"]
    scale_names = check_scale_pairing(weight_map)

    os.makedirs(output_path, exist_ok=True)
    loaded_files = {}

    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(input_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    new_weight_map = {}
    total_size = 0
    stats = Counter()
    consumed_scales = set()
    seen_scales = set()
    processed_files = set()

    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"no .safetensors files under {input_path}")

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        processed_files.add(file_name)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith(".scale"):
                seen_scales.add(weight_name)
                continue
            scale_name = sibling_scale_name(weight_name)
            if scale_name is not None and scale_name in scale_names:
                # V4 bfloat16 routing, with V4.1 additions: block_size=32 and
                # the engram row-group layout.
                scale_inv = get_tensor(scale_name)
                if weight.dtype == torch.int8:
                    weight = unpack_mxfloat4_to_fp32(weight.view(torch.uint8))
                    weight = weight_dequant(weight, scale_inv,
                                            block_size=BLOCK, is_mx=True)
                    label = "fp4_group"
                elif (weight.dtype == torch.float8_e4m3fn
                        and scale_inv.shape[0] == weight.shape[0]):
                    weight = dequant_fp8_row_group(weight, scale_inv)
                    label = "fp8_row_group"
                else:
                    weight = weight_dequant(weight, scale_inv, block_size=BLOCK)
                    label = "fp8_block"
                new_state_dict[weight_name] = weight
                consumed_scales.add(scale_name)
                stats[label] += 1
            else:
                if (weight.dtype == torch.float8_e4m3fn
                        or (weight.dim() == 2
                            and weight.dtype == torch.int8)):
                    raise ValueError(
                        f"{weight_name}: quantized dtype {weight.dtype} has "
                        f"no sibling .scale tensor")
                new_state_dict[weight_name] = weight
                stats["passthrough"] += 1
            new_weight_map[weight_name] = file_name
            total_size += weight.numel() * weight.element_size()

        save_file(new_state_dict, os.path.join(output_path, file_name),
                  metadata={'format': 'pt'})

        # Memory management: keep only the 2 most recently used files.
        if len(loaded_files) > 2:
            del loaded_files[next(iter(loaded_files))]

    # Integrity check: every scale stored in a processed shard whose paired
    # weight was also processed must have been consumed by a dequant.
    orphan = sorted(
        s for s in seen_scales - consumed_scales
        if weight_map[s[:-len(".scale")] + ".weight"] in processed_files)
    if orphan:
        raise ValueError(
            f"{len(orphan)} scale tensors were never consumed, "
            f"e.g. {orphan[:5]}")
    deferred = seen_scales - consumed_scales - set(orphan)
    if deferred:
        print(f"Note: {len(deferred)} scales pair with weights in shards not "
              f"present under {input_path} (partial-directory run).")

    copy_py_json(input_path, output_path)

    config.pop("quantization_config", None)
    dump_json(config, os.path.join(output_path, "config.json"))

    metadata = dict(model_index.get("metadata") or {})
    metadata["total_size"] = total_size
    dump_json({"metadata": metadata, "weight_map": new_weight_map},
              os.path.join(output_path, "model.safetensors.index.json"))

    print(f"dequantized: fp8_block={stats['fp8_block']} "
          f"fp8_row_group={stats['fp8_row_group']} "
          f"fp4_group={stats['fp4_group']}")
    print(f"passthrough : {stats['passthrough']}")
    print(f"scales dropped: {len(consumed_scales)}, "
          f"total_size: {total_size} ({total_size / 2 ** 30:.3f} GiB)")
    print(f"output: {output_path}")


def main():
    parser = ArgumentParser(
        description="Convert native DeepSeek-V4.1-Flash Hybrid FP8-MXFP4 "
                    "weights to BF16 (CPU only).")
    parser.add_argument("--input_hf_path", type=str, required=True,
                        help="directory with the native checkpoint shards, "
                             "model.safetensors.index.json and config.json")
    parser.add_argument("--output_hf_path", type=str, default=None,
                        help="directory for the converted BF16 checkpoint "
                             "(required unless --dry_run)")
    parser.add_argument("--dry_run", action="store_true",
                        help="only run the scale-pairing check and print "
                             "classification stats from the model index")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args.input_hf_path)
        return
    if not args.output_hf_path:
        parser.error("--output_hf_path is required unless --dry_run")
    convert(args.input_hf_path, args.output_hf_path)


if __name__ == "__main__":
    main()
