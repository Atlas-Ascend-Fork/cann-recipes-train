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

"""Versioned verifier specifications for Terminal-Bench training trials.

Training rows provide the instruction shown to the agent. A verifier sidecar
holds the hidden grading contract -- the oracle command, its timeout and the
expected /tests content hash -- keyed by ``task_id``. Keeping the two records
separate prevents verifier-only data from entering the harness prompt while
making reward computation deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
PROTOCOL = "terminalbench-v1"


class VerifierSpecError(ValueError):
    """A verifier manifest or its dataset/environment binding is invalid."""


def _required_string(record: dict[str, Any], key: str, *, location: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VerifierSpecError(f"{location}: {key} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class VerifierSpec:
    schema_version: int
    spec_id: str
    task_id: str
    protocol: str
    template_id: str | None
    image_digest: str
    workdir: str
    run_tests_cmd: str
    tests_sha256: str | None
    timeout_sec: int
    # Optional oracle payload (filename -> content). tmax images never bake
    # /tests (the upstream build scripts convert container_def only); the
    # orchestrator injects these into the sandbox after the agent runs,
    # matching upstream Terminal-Bench semantics.
    tests: dict[str, str] | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any], *, location: str) -> VerifierSpec:
        if record.get("schema_version") != 1:
            raise VerifierSpecError(f"{location}: schema_version must be 1")

        protocol = _required_string(record, "protocol", location=location)
        if protocol != PROTOCOL:
            raise VerifierSpecError(f"{location}: unsupported protocol {protocol!r}")

        template_id = record.get("template_id") or None
        if template_id is not None:
            try:
                UUID(template_id)
            except (ValueError, TypeError, AttributeError) as exc:
                raise VerifierSpecError(f"{location}: invalid template_id {template_id!r}") from exc

        image_digest = _required_string(record, "image_digest", location=location)
        if not _DIGEST_RE.fullmatch(image_digest):
            raise VerifierSpecError(f"{location}: invalid image_digest {image_digest!r}")

        workdir = _required_string(record, "workdir", location=location)
        if not workdir.startswith("/"):
            raise VerifierSpecError(f"{location}: workdir must be an absolute path")

        run_tests_cmd = _required_string(record, "run_tests_cmd", location=location)

        # Optional: null disables the pre-grading /tests tamper check (the
        # grader already skips it when absent). Prefer a recorded hash.
        tests_sha256 = record.get("tests_sha256")
        if tests_sha256 is not None:
            tests_sha256 = str(tests_sha256).strip()
            if not tests_sha256:
                tests_sha256 = None
            elif not _SHA256_RE.fullmatch(tests_sha256):
                raise VerifierSpecError(f"{location}: tests_sha256 must be a 64-character hex digest or null")

        timeout_sec = record.get("timeout_sec")
        if not isinstance(timeout_sec, int) or timeout_sec <= 0:
            raise VerifierSpecError(f"{location}: timeout_sec must be a positive integer")

        tests = record.get("tests")
        if tests is not None:
            if (
                not isinstance(tests, dict)
                or not tests
                or not all(
                    isinstance(name, str) and name and not PurePosixPath(name).is_absolute()
                    and ".." not in PurePosixPath(name).parts and isinstance(content, str)
                    for name, content in tests.items()
                )
            ):
                raise VerifierSpecError(
                    f"{location}: tests must be a non-empty mapping of filenames to content strings"
                )

        return cls(
            schema_version=1,
            spec_id=_required_string(record, "spec_id", location=location),
            task_id=_required_string(record, "task_id", location=location),
            protocol=protocol,
            template_id=template_id,
            image_digest=image_digest,
            workdir=workdir,
            run_tests_cmd=run_tests_cmd,
            tests_sha256=tests_sha256,
            timeout_sec=timeout_sec,
            tests=tests,
        )


class VerifierSpecRegistry:
    def __init__(self, specs: list[VerifierSpec]) -> None:
        self._by_task: dict[str, VerifierSpec] = {}
        self._by_spec_id: dict[str, VerifierSpec] = {}
        for spec in specs:
            if spec.task_id in self._by_task:
                raise VerifierSpecError(f"duplicate task_id in verifier manifest: {spec.task_id}")
            if spec.spec_id in self._by_spec_id:
                raise VerifierSpecError(f"duplicate spec_id in verifier manifest: {spec.spec_id}")
            self._by_task[spec.task_id] = spec
            self._by_spec_id[spec.spec_id] = spec

    def __len__(self) -> int:
        return len(self._by_task)

    def resolve_spec_id(self, spec_id: str) -> VerifierSpec:
        try:
            return self._by_spec_id[spec_id]
        except KeyError as exc:
            raise VerifierSpecError(f"no VerifierSpec for spec_id={spec_id!r}") from exc

    def resolve(
        self,
        task_id: str,
        *,
        spec_id: str | None = None,
        template_id: str | None = None,
        image_digest: str | None = None,
        workdir: str | None = None,
    ) -> VerifierSpec:
        try:
            spec = self._by_task[task_id]
        except KeyError as exc:
            raise VerifierSpecError(f"no VerifierSpec for task_id={task_id!r}") from exc

        expected = {
            "spec_id": spec.spec_id,
            "template_id": spec.template_id,
            "image_digest": spec.image_digest,
            "workdir": spec.workdir,
        }
        provided = {
            "spec_id": spec_id,
            "template_id": template_id,
            "image_digest": image_digest,
            "workdir": workdir,
        }
        for field, value in provided.items():
            if value is not None and value != expected[field]:
                raise VerifierSpecError(
                    f"task_id={task_id!r}: {field} does not match VerifierSpec {spec.spec_id!r}"
                )
        return spec


@lru_cache(maxsize=None)
def load_verifier_specs(path: str) -> VerifierSpecRegistry:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise VerifierSpecError(f"VerifierSpec manifest does not exist: {manifest_path}")

    specs: list[VerifierSpec] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        location = f"{manifest_path}:{line_number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerifierSpecError(f"{location}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise VerifierSpecError(f"{location}: each JSONL row must be an object")
        specs.append(VerifierSpec.from_record(record, location=location))

    if not specs:
        raise VerifierSpecError(f"VerifierSpec manifest is empty: {manifest_path}")
    return VerifierSpecRegistry(specs)


def apply_verifier_spec(md: dict[str, Any], registry: VerifierSpecRegistry) -> dict[str, Any]:
    """Bind the hidden grading contract to ephemeral rollout metadata.

    The spec is authoritative: run_tests_cmd / timeout_sec / tests_sha256 from
    the sidecar replace any inline defaults. Harness inputs continue to come
    exclusively from the row's instruction.
    """
    spec = registry.resolve(
        str(md.get("task_id") or ""),
        spec_id=str(md.get("verifier_spec_id") or "").strip() or None,
        template_id=str(md.get("template_id") or "").strip() or None,
        image_digest=str(md.get("image_digest") or "").strip() or None,
        workdir=str(md.get("workdir") or "").strip() or None,
    )
    grading = dict(md.get("grading") or {})
    grading["verifier_spec"] = spec
    grading["run_tests_cmd"] = spec.run_tests_cmd
    grading["timeout_sec"] = spec.timeout_sec
    grading["tests_sha256"] = spec.tests_sha256
    if spec.tests:
        grading["tests"] = dict(spec.tests)
    resolved = dict(md)
    resolved["grading"] = grading
    resolved["verifier_spec_id"] = spec.spec_id
    return resolved
