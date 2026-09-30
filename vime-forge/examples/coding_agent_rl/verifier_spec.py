"""Versioned, private verifier specifications for SWE training trials.

Training rows provide the problem shown to the agent.  A verifier sidecar holds
the hidden test patch and executable grading contract, keyed by ``instance_id``.
Keeping the two records separate prevents verifier-only data from being added to
the harness prompt while still making reward computation deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_EVAL_TYPES = frozenset({"fail_only", "pass_and_fail"})
_EVALUATORS = frozenset({"swebench", "swegym"})
_TEST_PATCH_OWNERS = frozenset({"verifier", "eval_script"})
_RUNTIME_SETUP_POLICIES = frozenset({"official-eval-script", "prebuilt-instance-image"})
_START_TEST_OUTPUT = ">>>>> Start Test Output"
_END_TEST_OUTPUT = ">>>>> End Test Output"
SUPPORTED_SWEGYM_REVISION = "16dd480cce9b27bf111a362d280881c6def5d2a7"


class VerifierSpecError(ValueError):
    """A verifier manifest or its dataset/environment binding is invalid."""


def _required_string(record: dict[str, Any], key: str, *, location: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VerifierSpecError(f"{location}: {key} must be a non-empty string")
    return value.strip()


def _required_text(record: dict[str, Any], key: str, *, location: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VerifierSpecError(f"{location}: {key} must be non-empty text")
    return value


def _required_string_list(record: dict[str, Any], key: str, *, location: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise VerifierSpecError(f"{location}: {key} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


@dataclass(frozen=True)
class VerifierSpec:
    schema_version: int
    spec_id: str
    instance_id: str
    protocol: str
    template_id: str
    image_digest: str
    base_commit: str
    repo: str
    version: str
    workdir: str
    eval_script: str
    log_parser: str
    eval_type: str
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    timeout_sec: int
    evaluator: str
    evaluator_revision: str | None
    test_patch_owner: str
    runtime_setup_policy: str

    @classmethod
    def from_record(cls, record: dict[str, Any], *, location: str) -> VerifierSpec:
        if record.get("schema_version") != 1:
            raise VerifierSpecError(f"{location}: schema_version must be 1")

        protocol = _required_string(record, "protocol", location=location)
        if protocol != "swebench-v5":
            raise VerifierSpecError(f"{location}: unsupported protocol {protocol!r}")

        template_id = _required_string(record, "template_id", location=location)
        try:
            UUID(template_id)
        except ValueError as exc:
            raise VerifierSpecError(f"{location}: invalid template_id {template_id!r}") from exc

        image_digest = _required_string(record, "image_digest", location=location)
        if not _DIGEST_RE.fullmatch(image_digest):
            raise VerifierSpecError(f"{location}: invalid image_digest {image_digest!r}")

        base_commit = _required_string(record, "base_commit", location=location).lower()
        if not _COMMIT_RE.fullmatch(base_commit):
            raise VerifierSpecError(f"{location}: invalid base_commit {base_commit!r}")

        evaluator = record.get("evaluator", "swebench")
        if not isinstance(evaluator, str) or evaluator not in _EVALUATORS:
            raise VerifierSpecError(f"{location}: unsupported evaluator {evaluator!r}")

        evaluator_revision = record.get("evaluator_revision")
        if evaluator_revision is not None:
            if not isinstance(evaluator_revision, str) or not _COMMIT_RE.fullmatch(
                evaluator_revision.lower()
            ):
                raise VerifierSpecError(
                    f"{location}: evaluator_revision must be a 40-character git commit"
                )
            evaluator_revision = evaluator_revision.lower()
        if evaluator == "swegym" and evaluator_revision is None:
            raise VerifierSpecError(f"{location}: swegym evaluator requires evaluator_revision")
        if evaluator == "swegym" and evaluator_revision != SUPPORTED_SWEGYM_REVISION:
            raise VerifierSpecError(
                f"{location}: unsupported SWE-Gym revision {evaluator_revision!r}; "
                f"expected {SUPPORTED_SWEGYM_REVISION}"
            )

        test_patch_owner = record.get("test_patch_owner", "verifier")
        if test_patch_owner not in _TEST_PATCH_OWNERS:
            raise VerifierSpecError(f"{location}: invalid test_patch_owner {test_patch_owner!r}")
        if evaluator == "swegym" and test_patch_owner != "eval_script":
            raise VerifierSpecError(
                f"{location}: swegym evaluator requires test_patch_owner='eval_script'"
            )

        runtime_setup_policy = record.get("runtime_setup_policy", "official-eval-script")
        if runtime_setup_policy not in _RUNTIME_SETUP_POLICIES:
            raise VerifierSpecError(
                f"{location}: invalid runtime_setup_policy {runtime_setup_policy!r}"
            )

        eval_script = _required_text(record, "eval_script", location=location)
        if evaluator == "swebench":
            start = eval_script.count(_START_TEST_OUTPUT)
            end = eval_script.count(_END_TEST_OUTPUT)
            if (
                start != 1
                or end != 1
                or eval_script.index(_START_TEST_OUTPUT) >= eval_script.index(_END_TEST_OUTPUT)
            ):
                raise VerifierSpecError(
                    f"{location}: eval_script must contain one ordered SWE-bench start/end marker pair"
                )

        eval_type = _required_string(record, "eval_type", location=location)
        if eval_type not in _EVAL_TYPES:
            raise VerifierSpecError(f"{location}: invalid eval_type {eval_type!r}")

        timeout_sec = record.get("timeout_sec", 600)
        if not isinstance(timeout_sec, int) or timeout_sec <= 0:
            raise VerifierSpecError(f"{location}: timeout_sec must be a positive integer")

        fail_to_pass = _required_string_list(record, "FAIL_TO_PASS", location=location)
        pass_to_pass = _required_string_list(record, "PASS_TO_PASS", location=location)
        if not fail_to_pass:
            raise VerifierSpecError(f"{location}: FAIL_TO_PASS must contain at least one test")

        return cls(
            schema_version=1,
            spec_id=_required_string(record, "spec_id", location=location),
            instance_id=_required_string(record, "instance_id", location=location),
            protocol=protocol,
            template_id=template_id,
            image_digest=image_digest,
            base_commit=base_commit,
            repo=_required_string(record, "repo", location=location),
            version=_required_string(record, "version", location=location),
            workdir=_required_string(record, "workdir", location=location),
            eval_script=eval_script,
            log_parser=_required_string(record, "log_parser", location=location),
            eval_type=eval_type,
            test_patch=_required_text(record, "test_patch", location=location),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            timeout_sec=timeout_sec,
            evaluator=evaluator,
            evaluator_revision=evaluator_revision,
            test_patch_owner=test_patch_owner,
            runtime_setup_policy=runtime_setup_policy,
        )

    def to_swebench_record(self) -> dict[str, Any]:
        """Return the normalized row consumed by ``swebench==5.0.2``."""
        return {
            "instance_id": self.instance_id,
            "image": self.template_id,
            "repo": self.repo,
            "version": self.version,
            "FAIL_TO_PASS": list(self.fail_to_pass),
            "PASS_TO_PASS": list(self.pass_to_pass),
            "log_parser": self.log_parser,
            "eval_type": self.eval_type,
            "eval_script": self.eval_script,
        }


class VerifierSpecRegistry:
    def __init__(self, specs: list[VerifierSpec]) -> None:
        self._by_instance: dict[str, VerifierSpec] = {}
        self._by_spec_id: dict[str, VerifierSpec] = {}
        for spec in specs:
            if spec.instance_id in self._by_instance:
                raise VerifierSpecError(f"duplicate instance_id in verifier manifest: {spec.instance_id}")
            if spec.spec_id in self._by_spec_id:
                raise VerifierSpecError(f"duplicate spec_id in verifier manifest: {spec.spec_id}")
            self._by_instance[spec.instance_id] = spec
            self._by_spec_id[spec.spec_id] = spec

    def __len__(self) -> int:
        return len(self._by_instance)

    def resolve_spec_id(self, spec_id: str) -> VerifierSpec:
        try:
            return self._by_spec_id[spec_id]
        except KeyError as exc:
            raise VerifierSpecError(f"no VerifierSpec for spec_id={spec_id!r}") from exc

    def resolve(
        self,
        instance_id: str,
        *,
        spec_id: str | None = None,
        template_id: str | None = None,
        image_digest: str | None = None,
        base_commit: str | None = None,
        repo: str | None = None,
        workdir: str | None = None,
        test_patch: str | None = None,
        fail_to_pass: list[str] | tuple[str, ...] | None = None,
        pass_to_pass: list[str] | tuple[str, ...] | None = None,
    ) -> VerifierSpec:
        try:
            spec = self._by_instance[instance_id]
        except KeyError as exc:
            raise VerifierSpecError(f"no VerifierSpec for instance_id={instance_id!r}") from exc

        expected = {
            "spec_id": spec.spec_id,
            "template_id": spec.template_id,
            "image_digest": spec.image_digest,
            "base_commit": spec.base_commit,
            "repo": spec.repo,
            "workdir": spec.workdir,
            "test_patch": spec.test_patch,
        }
        provided = {
            "spec_id": spec_id,
            "template_id": template_id,
            "image_digest": image_digest,
            "base_commit": base_commit.lower() if base_commit else None,
            "repo": repo,
            "workdir": workdir,
            "test_patch": test_patch,
        }
        for field, value in provided.items():
            if value is not None and value != expected[field]:
                raise VerifierSpecError(
                    f"instance_id={instance_id!r}: {field} does not match VerifierSpec {spec.spec_id!r}"
                )

        if fail_to_pass is not None and tuple(fail_to_pass) != spec.fail_to_pass:
            raise VerifierSpecError(f"instance_id={instance_id!r}: FAIL_TO_PASS does not match VerifierSpec")
        if pass_to_pass is not None and tuple(pass_to_pass) != spec.pass_to_pass:
            raise VerifierSpecError(f"instance_id={instance_id!r}: PASS_TO_PASS does not match VerifierSpec")
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
    """Bind hidden verifier data to ephemeral rollout metadata.

    The returned object is passed only to the verifier.  Harness inputs continue
    to come exclusively from ``problem_statement``/``SWE_PROMPT``.
    """
    grading = dict(md.get("grading") or {})
    instance = grading.get("sweb_instance") or {}
    spec = registry.resolve(
        str(md.get("instance_id") or ""),
        template_id=str(md.get("template_id") or "").strip() or None,
        image_digest=str(md.get("image_digest") or "").strip() or None,
        base_commit=str(instance.get("base_commit") or md.get("base_commit") or "").strip() or None,
        repo=str(instance.get("repo") or md.get("repo") or "").strip() or None,
        workdir=str(md.get("workdir") or "").strip() or None,
        test_patch=instance.get("test_patch"),
        fail_to_pass=instance.get("FAIL_TO_PASS"),
        pass_to_pass=instance.get("PASS_TO_PASS"),
    )
    grading["verifier_spec"] = spec
    resolved = dict(md)
    resolved["grading"] = grading
    resolved["verifier_spec_id"] = spec.spec_id
    return resolved
