# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def collect_parquet_files(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file() and path.suffix in (".parquet", ".pq"):
        return [path]
    if path.is_dir():
        return sorted(
            list(path.rglob("*.parquet")) + list(path.rglob("*.pq"))
        )
    return []


def _iter_texts(pf: pq.ParquetFile, batch_size: int):
    """Yield non-empty, stripped `text` column values from a ParquetFile."""
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if text is None:
                continue
            text_str = str(text).strip()
            if text_str:
                yield text_str


def _convert_file(pf, fout, batch_size, remaining, pbar) -> int:
    """Write cleaned texts from one parquet file as JSONL. Returns rows written."""
    written = 0
    for text in _iter_texts(pf, batch_size):
        if remaining is not None and written >= remaining:
            break
        json.dump({"text": text}, fout, ensure_ascii=False)
        fout.write("\n")
        written += 1
        pbar.update(1)
    return written


def _open_parquet(file: Path, idx: int, total: int):
    """Return ParquetFile or None (printing the reason) on failure."""
    try:
        return pq.ParquetFile(file)
    except Exception as e:
        print(f"[{idx}/{total}] skip {file.name}: {e}")
        return None


def _process_files(files, fout, batch_size, max_rows, pbar):
    """Convert all parquet files into JSONL. Returns (total_rows, skipped, processed)."""
    total_rows = 0
    skipped = 0
    processed = 0
    for idx, file in enumerate(files, 1):
        pf = _open_parquet(file, idx, len(files))
        if pf is None:
            skipped += 1
            continue
        pbar.set_postfix_str(f"{idx}/{len(files)} {file.name}")
        remaining = None if max_rows is None else max_rows - total_rows
        total_rows += _convert_file(pf, fout, batch_size, remaining, pbar)
        processed += 1
        if max_rows is not None and total_rows >= max_rows:
            break
    return total_rows, skipped, processed


def _report_summary(output_path: str, total_rows: int, files_done: int, skipped: int):
    size_mb = (
        os.path.getsize(output_path) / 1024 / 1024
        if os.path.exists(output_path)
        else 0.0
    )
    print(
        f"Done: wrote {total_rows} row(s) from "
        f"{files_done} file(s) -> {output_path} ({size_mb:.1f} MB)"
    )
    if skipped:
        print(f"  ({skipped} file(s) skipped due to errors)")


def parquet_to_json(
    input_path: str,
    output_path: str,
    max_rows: int | None = None,
    max_files: int | None = None,
    batch_size: int = 10000,
):
    files = collect_parquet_files(input_path)
    if not files:
        print(f"No parquet files found under {input_path}")
        return
    if max_files is not None:
        files = files[:max_files]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Found {len(files)} parquet file(s) under {input_path}")
    if max_rows:
        print(f"Will stop after {max_rows} row(s)")

    pbar = tqdm(total=max_rows, unit="row", desc="Converting", dynamic_ncols=True)
    try:
        with open(output_path, "w", encoding="utf-8") as fout:
            total_rows, skipped, processed = _process_files(
                files, fout, batch_size, max_rows, pbar
            )
    finally:
        pbar.close()

    _report_summary(output_path, total_rows, processed, skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="output.jsonl")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to write (default: no limit)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of parquet files to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Rows per parquet read batch (default: 10000)",
    )
    args = parser.parse_args()
    parquet_to_json(
        args.input,
        args.output,
        max_rows=args.max_rows,
        max_files=args.max_files,
        batch_size=args.batch_size,
    )
