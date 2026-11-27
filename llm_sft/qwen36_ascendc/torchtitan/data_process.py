# SPDX-License-Identifier: Apache-2.0
"""JSONL field mapping for the Qwen3.6 SFT sample."""

from __future__ import annotations

import argparse
import json
from os import SEEK_END
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO


def process_sample(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Map input/output or prompt/response to a single-turn chat."""

    prompt = sample.get("input")
    response = sample.get("output")
    if prompt is None and response is None:
        prompt = sample.get("prompt")
        response = sample.get("response")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("input or prompt must be a non-empty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("output or response must be a non-empty string")
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def build_manifest(sources: list[Path]) -> dict[str, Any]:
    """Validate source files and build one schema for the merged output."""

    from datasets import Features, load_dataset

    records = 0
    schemas: dict[str, int] = {}
    merged_features: dict[str, Any] = {}
    for source in sources:
        if source.is_file() and source.stat().st_size == 0:
            continue
        dataset = load_dataset("json", data_files=str(source), split="train")
        columns = set(dataset.column_names)
        if {"input", "output"} <= columns:
            schema = "instruction/input/output"
        elif {"prompt", "response"} <= columns:
            schema = "prompt/response"
        else:
            raise ValueError(f"unsupported fields in {source}")
        for sample in dataset:
            process_sample(sample)
        records += len(dataset)
        schemas[schema] = schemas.get(schema, 0) + len(dataset)
        for field, feature in dataset.features.items():
            if field in merged_features and merged_features[field] != feature:
                raise ValueError(f"incompatible field schema: {field}")
            merged_features[field] = feature

    if records == 0:
        raise ValueError("no training records found in the input JSONL files")

    return {
        "records": records,
        "schemas": schemas,
        "features": Features(merged_features).to_dict(),
        "template": "qwen3_6_nothink",
    }


def _same_file(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    try:
        return first.samefile(second)
    except FileNotFoundError:
        return False


def _append_jsonl(source: Path, dst: BinaryIO, needs_lf: bool) -> bool:
    with source.open("rb") as src:
        size = src.seek(0, SEEK_END)
        if size == 0:
            return needs_lf
        src.seek(size - 1)
        ends_with_lf = src.read(1) == b"\n"
        src.seek(0)
        if needs_lf:
            dst.write(b"\n")
        copyfileobj(src, dst)
    return not ends_with_lf


def merge_jsonl(sources: list[Path], output: Path) -> None:
    """Byte-merge validated JSONL inputs into one file.

    A valid JSONL file may omit LF after its final record. Insert LF at
    each non-empty file boundary where it is missing so that two records
    never share a physical line.
    """

    if any(_same_file(source, output) for source in sources):
        raise ValueError(
            f"merged JSONL path must differ from all input files: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary_dir:
        staged_output = Path(temporary_dir) / output.name
        with staged_output.open("wb") as dst:
            needs_lf = False
            for source in sources:
                needs_lf = _append_jsonl(source, dst, needs_lf)
        staged_output.replace(output)


def load_dataset_features(manifest_path: str):
    """Read the schema required for a merged JSONL file."""

    from datasets import Features

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return Features.from_dict(manifest["features"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SFT JSONL files")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--merge-jsonl",
        type=Path,
        default=None,
        help="also byte-merge the validated inputs into this JSONL path, "
        "inserting LF at non-empty file boundaries",
    )
    args = parser.parse_args()

    if any(_same_file(source, args.output) for source in args.data):
        parser.error("--output must differ from all input JSONL files")
    if args.merge_jsonl is not None and _same_file(args.output, args.merge_jsonl):
        parser.error("--output and --merge-jsonl must be different files")

    manifest = build_manifest(args.data)
    if args.merge_jsonl is not None:
        merge_jsonl(args.data, args.merge_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
