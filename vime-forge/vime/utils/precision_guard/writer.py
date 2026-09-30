"""Fingerprint artifacts on disk: coordinates -> path, records -> JSON.

JSON rather than a compact binary format: artifacts are committed, read by humans and
diffed. ``json.dumps`` uses ``repr`` for floats, so values round-trip exactly and
"identical" comparisons stay meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path


def records_path(root, *, backend: str, role: str, rollout_id: int, step_id: int, site: str) -> Path:
    """Artifact path, built purely from the collection coordinates."""
    return (
        Path(root)
        / str(backend)
        / str(role)
        / f"rollout_{int(rollout_id):06d}"
        / f"step_{int(step_id):04d}"
        / f"{site}.json"
    )


def write_records(path, records: dict) -> None:
    """Write JSON, one quantity per line, sorted by name.

    Sorting makes the same data produce a byte-identical file anywhere. One line per quantity
    is the granularity worth diffing, since a fingerprint changes as a whole, and it keeps
    each block sum off a line of its own, which is most of the file size.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join(
        f" {json.dumps(name, ensure_ascii=False)}: "
        f"{json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"
        for name, record in sorted(records.items())
    )
    path.write_text(("{\n" + body + "\n}" if body else "{}") + "\n", encoding="utf-8")


def read_records(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
