#!/usr/bin/env python3
"""Prepare a reproducible SWE-Gym training bundle for AgentENV rollouts.

The command is deliberately split into resumable phases:

* ``inspect`` validates and normalizes the ProRL 293-row source JSONL.
* ``provision`` resolves immutable image digests and creates/reuses templates.
* ``qualify`` builds VerifierSpecs and runs empty/gold patch checks.
* ``finalize`` freezes only qualified rows into the three training inputs.
* ``prepare-all`` runs all phases in order.

Online dependencies (the E2B SDK and swebench) are imported lazily so the
offline inspect/finalize phases remain usable in a plain Python environment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "NovaSky-AI/SkyRL-v0-293-data"
DATASET_REVISION = "25f0e2f3b982d8dd0c1e59979341e5297e5aee31"
DATASET_SPLIT = "train"
EXPECTED_SOURCE_SHA256 = "8af4d6aa43b0cd4325e77ae712301443141f311d324397e53cfd6cd899f17cb1"
EXPECTED_ROWS = 293
WORKDIR = "/testbed"
SCHEMA_VERSION = 1
DEFAULT_REGISTRY_BASE = "https://docker.1ms.run"
REGISTRY_USER_AGENT = "vime-forge-swegym-preparer/1.0"
DEFAULT_TEMPLATE_CPU = 4
DEFAULT_TEMPLATE_MEMORY_MIB = 8192
DEFAULT_TEMPLATE_ALIAS_REVISION = "v1"
DEFAULT_PROBE_REQUEST_TIMEOUT = 600
DEFAULT_PROBE_SANDBOX_TIMEOUT = 300
DEFAULT_PROBE_COMMAND_TIMEOUT = 90
SWEGYM_PACKAGE_REVISION = "16dd480cce9b27bf111a362d280881c6def5d2a7"
DEFAULT_VERIFIER_TIMEOUT_SEC = 900
MONAI_VERIFIER_TIMEOUT_SEC = 1800

LEGACY_SWEBENCH_IMAGE_REPOS = {
    "marshmallow-code/marshmallow",
    "pydicom/pydicom",
    "pylint-dev/astroid",
    "pvlib/pvlib-python",
    "pyvista/pyvista",
    "sqlfluff/sqlfluff",
}

_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class PreparationError(RuntimeError):
    """A deterministic input, environment, or bundle invariant failed."""


class TemplateProbeError(PreparationError):
    """A staged Template probe operation failed."""

    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause_type = type(cause).__name__
        super().__init__(f"probe stage {stage!r} failed with {self.cause_type}: {cause}")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dumps(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, _json_dumps(value, pretty=True))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = "".join(_json_dumps(row) + "\n" for row in rows)
    _atomic_write_text(path, content)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PreparationError(f"JSONL file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreparationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise PreparationError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(row)
    return rows


def _required_text(record: dict[str, Any], key: str, *, location: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"{location}: {key} must be non-empty text")
    return value


def _required_string_list(record: dict[str, Any], key: str, *, location: str) -> list[str]:
    value = record.get(key)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PreparationError(f"{location}: {key} is not a JSON list") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise PreparationError(f"{location}: {key} must be a list of non-empty strings")
    return [item.strip() for item in value]


def registry_image_for_instance_id(instance_id: str) -> str:
    """Match the image naming policy used by ProRL's SWE-Gym example."""
    try:
        owner, repo_with_issue = instance_id.split("__", 1)
        repo, issue_id = repo_with_issue.rsplit("-", 1)
    except ValueError as exc:
        raise PreparationError(f"invalid SWE instance_id: {instance_id!r}") from exc
    if not owner or not repo or not issue_id:
        raise PreparationError(f"invalid SWE instance_id: {instance_id!r}")
    if f"{owner}/{repo}" in LEGACY_SWEBENCH_IMAGE_REPOS:
        return f"swebench/sweb.eval.x86_64.{owner}_1776_{repo}-{issue_id}:latest"
    suffix = instance_id.replace("__", "_s_").lower()
    return f"xingyaoww/sweb.eval.x86_64.{suffix}:latest"


def _validate_source_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    location = f"source row {row_index}"
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise PreparationError(f"{location}: metadata must be an object")
    instance = metadata.get("instance")
    if not isinstance(instance, dict):
        raise PreparationError(f"{location}: metadata.instance must be an object")

    instance_id = _required_text(instance, "instance_id", location=location).strip()
    metadata_id = _required_text(metadata, "instance_id", location=location).strip()
    if instance_id != metadata_id:
        raise PreparationError(f"{location}: metadata.instance_id does not match instance.instance_id")
    if metadata.get("split") != DATASET_SPLIT:
        raise PreparationError(f"{location}: expected split={DATASET_SPLIT!r}, got {metadata.get('split')!r}")

    repo = _required_text(instance, "repo", location=location).strip()
    if "/" not in repo:
        raise PreparationError(f"{location}: invalid repo {repo!r}")
    base_commit = _required_text(instance, "base_commit", location=location).strip().lower()
    if not _COMMIT_RE.fullmatch(base_commit):
        raise PreparationError(f"{location}: invalid base_commit {base_commit!r}")
    problem_statement = _required_text(instance, "problem_statement", location=location)
    _required_text(instance, "patch", location=location)
    test_patch = _required_text(instance, "test_patch", location=location)
    fail_to_pass = _required_string_list(instance, "FAIL_TO_PASS", location=location)
    pass_to_pass = _required_string_list(instance, "PASS_TO_PASS", location=location)
    if not fail_to_pass:
        raise PreparationError(f"{location}: FAIL_TO_PASS must not be empty")

    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], dict):
        raise PreparationError(f"{location}: prompt must contain exactly one chat message")
    if prompt[0].get("role") != "user" or not isinstance(prompt[0].get("content"), str):
        raise PreparationError(f"{location}: prompt must be a user text message")
    if prompt[0]["content"].strip() != problem_statement.strip():
        raise PreparationError(f"{location}: prompt content does not match problem_statement")

    normalized_instance = dict(instance)
    normalized_instance["instance_id"] = instance_id
    normalized_instance["repo"] = repo
    normalized_instance["base_commit"] = base_commit
    normalized_instance["problem_statement"] = problem_statement
    normalized_instance["test_patch"] = test_patch
    normalized_instance["FAIL_TO_PASS"] = fail_to_pass
    normalized_instance["PASS_TO_PASS"] = pass_to_pass
    return normalized_instance


def _candidate_row(instance: dict[str, Any], row_index: int) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    image = registry_image_for_instance_id(instance_id)
    remote = {key: value for key, value in instance.items() if key != "patch"}
    remote["image"] = image
    remote["workdir"] = WORKDIR
    problem_statement = str(instance["problem_statement"])
    return {
        "prompt": [{"role": "user", "content": problem_statement}],
        "label": instance_id,
        "metadata": {
            "instance_id": instance_id,
            "image": image,
            "workdir": WORKDIR,
            "problem_statement": problem_statement,
            "remote_env_info": remote,
            "dataset": {
                "id": DATASET_ID,
                "revision": DATASET_REVISION,
                "split": DATASET_SPLIT,
                "row_index": row_index,
            },
            "gold_solution_removed": True,
        },
    }


def _build_plan_row(instance: dict[str, Any], row_index: int, registry_base: str) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    image = registry_image_for_instance_id(instance_id)
    registry_host = urllib.parse.urlsplit(registry_base).netloc
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "row_index": row_index,
        },
        "instance_id": instance_id,
        "repo": str(instance["repo"]),
        "version": str(instance.get("version") or ""),
        "source_image_tag": image,
        "template_source_image_tag": f"{registry_host}/{image}",
        "workdir": WORKDIR,
        "base_commit": str(instance["base_commit"]),
    }


def _bundle_paths(bundle: Path) -> dict[str, Path]:
    return {
        "source_manifest": bundle / "source/source_manifest.json",
        "candidates": bundle / "state/candidate_rows.jsonl",
        "build_plan": bundle / "environment/template_build_plan.jsonl",
        "provision_results": bundle / "state/provision_results.jsonl",
        "provisioned_templates": bundle / "state/provisioned_templates.jsonl",
        "all_specs": bundle / "state/verifier_specs.all.jsonl",
        "qualification": bundle / "qualification/qualification_results.jsonl",
        "quarantine": bundle / "qualification/quarantined_instances.jsonl",
        "rejected": bundle / "qualification/rejected_instances.jsonl",
        "train": bundle / "rollout/train.no_gold.jsonl",
        "templates": bundle / "environment/agentenv_templates.jsonl",
        "specs": bundle / "verifier/verifier_specs.jsonl",
        "manifest": bundle / "preparation_manifest.json",
    }


def _load_quarantined_instances(
    bundle: Path,
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load the explicit do-not-block-training set for a prepared bundle.

    Quarantine is a derived-data gate only: the authoritative source and
    candidate rows remain unchanged.  Every active entry is excluded from
    qualification, final artifacts, rollout coverage, and training.
    """
    path = _bundle_paths(bundle)["quarantine"]
    if not path.is_file():
        return {}
    known_ids = {str(row["metadata"]["instance_id"]) for row in candidates}
    quarantined: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        instance_id = str(row.get("instance_id") or "").strip()
        if not instance_id:
            raise PreparationError(f"{path}:{line_number}: missing instance_id")
        if instance_id not in known_ids:
            raise PreparationError(f"{path}:{line_number}: unknown instance_id {instance_id!r}")
        if instance_id in quarantined:
            raise PreparationError(f"{path}:{line_number}: duplicate instance_id {instance_id!r}")
        if row.get("decision") != "do_not_block_training":
            raise PreparationError(
                f"{path}:{line_number}: quarantine decision must be 'do_not_block_training'"
            )
        excluded_from = row.get("excluded_from")
        required_gates = {"qualification", "finalization", "pure_rollout", "training"}
        if not isinstance(excluded_from, list) or not required_gates.issubset(set(excluded_from)):
            raise PreparationError(
                f"{path}:{line_number}: excluded_from must include {sorted(required_gates)}"
            )
        quarantined[instance_id] = row
    return quarantined


def inspect_source(
    source: Path,
    bundle: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
    registry_base: str = DEFAULT_REGISTRY_BASE,
) -> dict[str, Any]:
    source = source.resolve()
    actual_sha256 = _sha256(source)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise PreparationError(
            f"source SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256} ({source})"
        )
    source_rows = _read_jsonl(source)
    if len(source_rows) != expected_rows:
        raise PreparationError(f"expected {expected_rows} source rows, got {len(source_rows)}")

    candidates: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    instance_ids: set[str] = set()
    repositories: set[str] = set()
    for row_index, row in enumerate(source_rows):
        instance = _validate_source_row(row, row_index)
        instance_id = str(instance["instance_id"])
        if instance_id in instance_ids:
            raise PreparationError(f"duplicate instance_id at row {row_index}: {instance_id}")
        instance_ids.add(instance_id)
        repositories.add(str(instance["repo"]))
        candidates.append(_candidate_row(instance, row_index))
        plans.append(_build_plan_row(instance, row_index, registry_base))

    paths = _bundle_paths(bundle)
    _write_jsonl(paths["candidates"], candidates)
    _write_jsonl(paths["build_plan"], plans)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": DATASET_SPLIT},
        "source_path": str(source),
        "source_sha256": actual_sha256,
        "source_rows": len(source_rows),
        "unique_instance_ids": len(instance_ids),
        "unique_repositories": len(repositories),
        "registry_base": registry_base.rstrip("/"),
        "generated_at": _utc_now(),
        "outputs": {
            "candidate_rows": {"path": str(paths["candidates"]), "sha256": _sha256(paths["candidates"])},
            "template_build_plan": {
                "path": str(paths["build_plan"]),
                "sha256": _sha256(paths["build_plan"]),
            },
        },
    }
    _write_json(paths["source_manifest"], manifest)
    return manifest


def _parse_image_ref(image: str) -> tuple[str, str]:
    if "@" in image:
        repository, digest = image.rsplit("@", 1)
        if not _DIGEST_RE.fullmatch(digest):
            raise PreparationError(f"invalid image digest in {image!r}")
        return repository, digest
    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    if last_colon > last_slash:
        return image[:last_colon], image[last_colon + 1 :]
    return image, "latest"


def _bearer_parameters(header: str) -> dict[str, str]:
    if not header.lower().startswith("bearer "):
        raise PreparationError(f"unsupported registry authentication challenge: {header!r}")
    params: dict[str, str] = {}
    for match in re.finditer(r'(\w+)="([^"]*)"', header[7:]):
        params[match.group(1)] = match.group(2)
    if "realm" not in params:
        raise PreparationError(f"registry authentication challenge has no realm: {header!r}")
    return params


def _urlopen(request: urllib.request.Request, timeout: int):
    return urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context())


def resolve_registry_digest(image: str, registry_base: str, *, timeout: int = 60) -> str:
    """Resolve an OCI tag through a registry v2 HEAD request."""
    repository, reference = _parse_image_ref(image)
    if _DIGEST_RE.fullmatch(reference):
        return reference
    repository = repository.removeprefix("docker.io/").removeprefix("index.docker.io/")
    base = registry_base.rstrip("/")
    manifest_url = f"{base}/v2/{repository}/manifests/{urllib.parse.quote(reference, safe='')}"
    accept = ", ".join(
        (
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        )
    )
    # Some public Docker Hub mirrors reject Python's default urllib User-Agent
    # even though the same registry endpoint is available to OCI clients.
    headers = {"Accept": accept, "User-Agent": REGISTRY_USER_AGENT}

    def request_manifest(extra_headers: dict[str, str] | None = None):
        combined = dict(headers)
        combined.update(extra_headers or {})
        return _urlopen(urllib.request.Request(manifest_url, method="HEAD", headers=combined), timeout)

    try:
        response = request_manifest()
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise PreparationError(f"registry HEAD failed for {image}: HTTP {exc.code}") from exc
        challenge = exc.headers.get("WWW-Authenticate", "")
        params = _bearer_parameters(challenge)
        query = {
            key: value
            for key, value in (("service", params.get("service")), ("scope", params.get("scope")))
            if value
        }
        separator = "&" if "?" in params["realm"] else "?"
        token_url = params["realm"] + (separator + urllib.parse.urlencode(query) if query else "")
        try:
            with _urlopen(urllib.request.Request(token_url), timeout) as token_response:
                token_payload = json.load(token_response)
        except (OSError, ValueError, urllib.error.HTTPError) as token_exc:
            raise PreparationError(f"registry token request failed for {image}: {token_exc}") from token_exc
        token = token_payload.get("token") or token_payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise PreparationError(f"registry token response has no token for {image}")
        try:
            response = request_manifest({"Authorization": f"Bearer {token}"})
        except urllib.error.HTTPError as retry_exc:
            raise PreparationError(f"authenticated registry HEAD failed for {image}: HTTP {retry_exc.code}") from retry_exc
    except OSError as exc:
        raise PreparationError(f"registry HEAD failed for {image}: {exc}") from exc

    with response:
        digest = response.headers.get("Docker-Content-Digest", "").strip().lower()
    if not _DIGEST_RE.fullmatch(digest):
        raise PreparationError(f"registry returned no valid Docker-Content-Digest for {image}: {digest!r}")
    return digest


def _append_sdk_path(path: str | None) -> None:
    if not path:
        return
    sdk_path = Path(path).expanduser().resolve()
    if not sdk_path.is_dir():
        raise PreparationError(f"E2B SDK path is not a directory: {sdk_path}")
    if str(sdk_path) not in sys.path:
        sys.path.append(str(sdk_path))


def _api_json(
    url: str,
    api_key: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> Any:
    data = _json_dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise PreparationError(f"AgentENV API {method} {url} failed: HTTP {exc.code}: {detail}") from exc
    except OSError as exc:
        raise PreparationError(f"AgentENV API {method} {url} failed: {exc}") from exc
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise PreparationError(f"AgentENV API returned invalid JSON for {url}") from exc


def _template_alias(
    instance_id: str,
    digest: str,
    revision: str = DEFAULT_TEMPLATE_ALIAS_REVISION,
) -> str:
    if not re.fullmatch(r"v[1-9][0-9]*", revision):
        raise PreparationError(
            f"template alias revision must match 'v<positive integer>', got {revision!r}"
        )
    slug = re.sub(r"[^a-z0-9-]+", "-", instance_id.lower().replace("__", "-"))
    slug = re.sub(r"-+", "-", slug).strip("-")
    return f"swegym-{slug[:72]}-{digest.removeprefix('sha256:')[:8]}-{revision}"


def _resolve_template_alias(api_url: str, api_key: str, alias: str) -> str | None:
    encoded = urllib.parse.quote(alias, safe="")
    payload = _api_json(f"{api_url.rstrip('/')}/templates/aliases/{encoded}", api_key)
    if payload is None:
        return None
    template_id = payload.get("templateID") if isinstance(payload, dict) else None
    return template_id if isinstance(template_id, str) and template_id else None


def _run_template_probe(
    template_id: str,
    *,
    base_commit: str,
    api_url: str,
    sandbox_url: str,
    api_key: str,
    request_timeout: int = DEFAULT_PROBE_REQUEST_TIMEOUT,
    sandbox_timeout: int = DEFAULT_PROBE_SANDBOX_TIMEOUT,
    command_timeout: int = DEFAULT_PROBE_COMMAND_TIMEOUT,
) -> dict[str, Any]:
    from e2b import Sandbox  # type: ignore

    probe_started = time.monotonic()

    def event(stage: str, status: str, *, started: float | None = None, **details: Any) -> None:
        payload = {
            "event": "template_probe",
            "timestamp": _utc_now(),
            "template_id": template_id,
            "stage": stage,
            "status": status,
            **details,
        }
        if started is not None:
            payload["duration_sec"] = round(time.monotonic() - started, 3)
        print(_json_dumps(payload), flush=True)

    sandbox = None
    try:
        create_started = time.monotonic()
        event(
            "sandbox_create",
            "started",
            request_timeout=request_timeout,
            sandbox_timeout=sandbox_timeout,
        )
        try:
            sandbox = Sandbox.create(
                template=template_id,
                timeout=sandbox_timeout,
                api_url=api_url,
                api_key=api_key,
                sandbox_url=sandbox_url,
                request_timeout=request_timeout,
            )
        except Exception as exc:
            event(
                "sandbox_create",
                "failed",
                started=create_started,
                error_type=type(exc).__name__,
                error=str(exc)[:1000],
            )
            raise TemplateProbeError("sandbox_create", exc) from exc
        event(
            "sandbox_create",
            "completed",
            started=create_started,
            sandbox_id=str(getattr(sandbox, "sandbox_id", "")),
        )

        commands = {
            "architecture": "uname -m",
            "git_version": "git --version",
            "python_version": "python --version 2>&1 || python3 --version 2>&1",
            "pytest_version": (
                "cd /tmp && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --version 2>&1 || "
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest --version 2>&1"
            ),
            "head": f"git -C {WORKDIR} rev-parse HEAD",
            "status": f"git -C {WORKDIR} status --porcelain",
        }
        values: dict[str, str] = {}
        for name, command in commands.items():
            stage = f"command:{name}"
            command_started = time.monotonic()
            event(stage, "started", command_timeout=command_timeout)
            try:
                result = sandbox.commands.run(command, timeout=command_timeout)
            except Exception as exc:
                event(
                    stage,
                    "failed",
                    started=command_started,
                    error_type=type(exc).__name__,
                    error=str(exc)[:1000],
                )
                raise TemplateProbeError(stage, exc) from exc
            values[name] = (result.stdout or "").strip()
            event(
                stage,
                "completed",
                started=command_started,
                return_code=getattr(result, "exit_code", None),
            )

        validation_started = time.monotonic()
        event("validation", "started")
        try:
            if values["head"].lower() != base_commit.lower():
                raise PreparationError(
                    f"template {template_id} HEAD mismatch: expected {base_commit}, got {values['head']}"
                )
            if values["status"]:
                raise PreparationError(
                    f"template {template_id} has a dirty {WORKDIR}: {values['status'][:200]}"
                )
        except Exception as exc:
            event(
                "validation",
                "failed",
                started=validation_started,
                error_type=type(exc).__name__,
                error=str(exc)[:1000],
            )
            raise TemplateProbeError("validation", exc) from exc
        event("validation", "completed", started=validation_started)
        event("probe", "completed", started=probe_started)
        return {
            "validated_at": _utc_now(),
            "template_status": "ready",
            "architecture": values["architecture"],
            "git_version": values["git_version"],
            "python_version": values["python_version"],
            "pytest_version": values["pytest_version"],
            "worktree_clean": True,
            "head_matches_base_commit": True,
        }
    finally:
        if sandbox is not None:
            kill_started = time.monotonic()
            event("sandbox_kill", "started")
            try:
                sandbox.kill()
            except Exception as exc:
                # Cleanup failure is observable, but must not hide the stage
                # that caused the probe itself to fail.
                event(
                    "sandbox_kill",
                    "failed",
                    started=kill_started,
                    error_type=type(exc).__name__,
                    error=str(exc)[:1000],
                )
            else:
                event("sandbox_kill", "completed", started=kill_started)


def _provision_one(
    plan: dict[str, Any],
    *,
    existing: dict[str, dict[str, Any]],
    registry_base: str,
    api_url: str,
    sandbox_url: str,
    api_key: str,
    cpu_count: int,
    memory_mib: int,
    retries: int,
    template_alias_revision: str,
    probe_request_timeout: int,
    probe_sandbox_timeout: int,
    probe_command_timeout: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    instance_id = str(plan["instance_id"])
    started_at = _utc_now()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            existing_record = existing.get(instance_id)
            if existing_record is not None:
                if (
                    existing_record.get("base_commit") == plan["base_commit"]
                    and existing_record.get("repo") == plan["repo"]
                    and existing_record.get("workdir") == plan["workdir"]
                    and existing_record.get("source_image_tag") == plan["source_image_tag"]
                ):
                    validation = _run_template_probe(
                        str(existing_record["template_id"]),
                        base_commit=str(plan["base_commit"]),
                        api_url=api_url,
                        sandbox_url=sandbox_url,
                        api_key=api_key,
                        request_timeout=probe_request_timeout,
                        sandbox_timeout=probe_sandbox_timeout,
                        command_timeout=probe_command_timeout,
                    )
                    record = dict(existing_record)
                    record["validation"] = validation
                    return (
                        {
                            "instance_id": instance_id,
                            "status": "ready",
                            "action": "reused_existing_mapping",
                            "attempts": attempt,
                            "started_at": started_at,
                            "finished_at": _utc_now(),
                        },
                        record,
                    )

            digest = resolve_registry_digest(str(plan["source_image_tag"]), registry_base)
            registry_host = urllib.parse.urlsplit(registry_base).netloc
            source_image = f"{_parse_image_ref(str(plan['source_image_tag']))[0]}@{digest}"
            template_source_image = f"{registry_host}/{source_image}"
            alias = _template_alias(instance_id, digest, template_alias_revision)
            template_id = _resolve_template_alias(api_url, api_key, alias)
            action = "reused_alias"
            if template_id is None:
                from e2b import Template  # type: ignore

                template = (
                    Template()
                    .from_image(template_source_image)
                    .run_cmd("git config --system --add safe.directory '*'")
                    .run_cmd(
                        f"git -C {WORKDIR} reset --hard {plan['base_commit']} && "
                        f"git -C {WORKDIR} clean -fd"
                    )
                    .set_workdir(WORKDIR)
                )
                build = Template.build(
                    template,
                    alias,
                    cpu_count=cpu_count,
                    memory_mb=memory_mib,
                    skip_cache=False,
                    api_url=api_url,
                    api_key=api_key,
                    sandbox_url=sandbox_url,
                    request_timeout=120,
                )
                template_id = str(build.template_id)
                action = "created"

            validation = _run_template_probe(
                template_id,
                base_commit=str(plan["base_commit"]),
                api_url=api_url,
                sandbox_url=sandbox_url,
                api_key=api_key,
                request_timeout=probe_request_timeout,
                sandbox_timeout=probe_sandbox_timeout,
                command_timeout=probe_command_timeout,
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "dataset": plan["dataset"],
                "instance_id": instance_id,
                "repo": plan["repo"],
                "source_image_tag": plan["source_image_tag"],
                "source_image": source_image,
                "template_source_image": template_source_image,
                "image_digest": digest,
                "template_id": template_id,
                "template_alias": alias,
                "workdir": plan["workdir"],
                "base_commit": plan["base_commit"],
                "validation": validation,
            }
            return (
                {
                    "instance_id": instance_id,
                    "status": "ready",
                    "action": action,
                    "attempts": attempt,
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                },
                record,
            )
        except Exception as exc:  # keep processing the remaining 293 rows
            last_error = exc
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
    assert last_error is not None
    failure = {
        "instance_id": instance_id,
        "status": "retryable",
        "action": "failed",
        "attempts": retries,
        "error_type": type(last_error).__name__,
        "error": str(last_error)[:1000],
        "started_at": started_at,
        "finished_at": _utc_now(),
    }
    if isinstance(last_error, TemplateProbeError):
        failure["probe_stage"] = last_error.stage
        failure["cause_type"] = last_error.cause_type
    return failure, None


def provision_templates(
    bundle: Path,
    *,
    existing_map: Path | None,
    registry_base: str,
    api_url: str,
    sandbox_url: str,
    api_key: str,
    sdk_path: str | None,
    concurrency: int,
    retries: int,
    cpu_count: int,
    memory_mib: int,
    template_alias_revision: str = DEFAULT_TEMPLATE_ALIAS_REVISION,
    probe_request_timeout: int = DEFAULT_PROBE_REQUEST_TIMEOUT,
    probe_sandbox_timeout: int = DEFAULT_PROBE_SANDBOX_TIMEOUT,
    probe_command_timeout: int = DEFAULT_PROBE_COMMAND_TIMEOUT,
    instance_ids: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    _append_sdk_path(sdk_path)
    all_plans = _read_jsonl(_bundle_paths(bundle)["build_plan"])
    plans = all_plans
    if instance_ids:
        known = {str(plan["instance_id"]) for plan in plans}
        missing = sorted(instance_ids - known)
        if missing:
            raise PreparationError(f"unknown instance_id values: {missing}")
        plans = [plan for plan in plans if str(plan["instance_id"]) in instance_ids]
    if limit is not None:
        plans = plans[:limit]
    existing_rows = _read_jsonl(existing_map) if existing_map and existing_map.is_file() else []
    existing = {str(row["instance_id"]): row for row in existing_rows}
    paths = _bundle_paths(bundle)
    prior_results = {
        str(row["instance_id"]): row
        for row in (_read_jsonl(paths["provision_results"]) if paths["provision_results"].is_file() else [])
    }
    provisioned = {
        str(row["instance_id"]): row
        for row in (
            _read_jsonl(paths["provisioned_templates"]) if paths["provisioned_templates"].is_file() else []
        )
    }
    pending = [plan for plan in plans if str(plan["instance_id"]) not in provisioned]

    def checkpoint() -> None:
        # Selection flags are intended for canaries and retries.  Never let a
        # partial invocation erase checkpoints produced by an earlier run.
        ordered_ids = [str(plan["instance_id"]) for plan in all_plans]
        _write_jsonl(paths["provision_results"], [prior_results[item] for item in ordered_ids if item in prior_results])
        _write_jsonl(paths["provisioned_templates"], [provisioned[item] for item in ordered_ids if item in provisioned])

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _provision_one,
                plan,
                existing=existing,
                registry_base=registry_base,
                api_url=api_url,
                sandbox_url=sandbox_url,
                api_key=api_key,
                cpu_count=cpu_count,
                memory_mib=memory_mib,
                retries=retries,
                template_alias_revision=template_alias_revision,
                probe_request_timeout=probe_request_timeout,
                probe_sandbox_timeout=probe_sandbox_timeout,
                probe_command_timeout=probe_command_timeout,
            ): plan
            for plan in pending
        }
        for future in as_completed(futures):
            plan = futures[future]
            result, mapping = future.result()
            instance_id = str(plan["instance_id"])
            prior_results[instance_id] = result
            if mapping is not None:
                provisioned[instance_id] = mapping
            checkpoint()
            print(
                f"provision {len(prior_results)}/{len(plans)} {instance_id}: "
                f"{result['status']} ({result['action']})",
                flush=True,
            )

    checkpoint()
    selected_ids = {str(plan["instance_id"]) for plan in plans}
    selected_results = [row for key, row in prior_results.items() if key in selected_ids]
    counts = {
        "ready": sum(row.get("status") == "ready" for row in selected_results),
        "retryable": sum(row.get("status") == "retryable" for row in selected_results),
        "total": len(plans),
    }
    return counts


def _load_source_instances(source: Path) -> dict[str, dict[str, Any]]:
    instances: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(_read_jsonl(source)):
        instance = _validate_source_row(row, row_index)
        instances[str(instance["instance_id"])] = instance
    return instances


def _hydra_pytest_policy(eval_script: str, repo: str) -> str:
    if repo != "facebookresearch/hydra" or "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in eval_script:
        return eval_script
    lines = eval_script.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("pytest "):
            indent = line[: len(line) - len(stripped)]
            lines[index] = indent + "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 " + stripped
            return "".join(lines)
    raise PreparationError("Hydra eval script does not contain a direct pytest command")


def _activate_pinned_swegym_source(source: Path | None) -> Path:
    configured = source or (
        Path(os.environ["VIME_SWEGYM_SOURCE_PATH"])
        if os.environ.get("VIME_SWEGYM_SOURCE_PATH")
        else None
    )
    if configured is None:
        raise PreparationError(
            "official SWE-Gym source is required; pass --swegym-source or set "
            "VIME_SWEGYM_SOURCE_PATH"
        )
    root = configured.expanduser().resolve()
    required = root / "swegym/harness/test_spec.py"
    if not required.is_file():
        raise PreparationError(f"invalid SWE-Gym source tree: missing {required}")
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreparationError(f"cannot determine SWE-Gym source revision at {root}: {exc}") from exc
    revision = result.stdout.strip().lower()
    if revision != SWEGYM_PACKAGE_REVISION:
        raise PreparationError(
            f"SWE-Gym source revision mismatch: expected {SWEGYM_PACKAGE_REVISION}, got {revision}"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ["VIME_SWEGYM_SOURCE_PATH"] = str(root)
    return root


def _load_swegym_eval_spec_builder(source: Path | None):
    _activate_pinned_swegym_source(source)
    try:
        from swegym.harness.constants import MAP_REPO_VERSION_TO_SPECS  # type: ignore
        from swegym.harness.test_spec import (  # type: ignore
            TestSpec,
            make_eval_script_list,
        )
    except Exception as exc:
        raise PreparationError(f"official SWE-Gym eval builder is unavailable: {exc}") from exc

    def build(instance: dict[str, Any]):
        instance_id = str(instance["instance_id"]).lower()
        repo = str(instance["repo"]).lower()
        version = str(instance["version"])
        base_commit = str(instance["base_commit"])
        test_patch = str(instance["test_patch"])
        env_name = WORKDIR.strip("/")
        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
        eval_script_list = make_eval_script_list(
            instance,
            specs,
            env_name,
            WORKDIR,
            base_commit,
            test_patch,
        )
        return TestSpec(
            instance_id=instance_id,
            repo=repo,
            version=version,
            repo_script_list=[],
            eval_script_list=eval_script_list,
            env_script_list=[],
            arch="x86_64",
            FAIL_TO_PASS=list(instance["FAIL_TO_PASS"]),
            PASS_TO_PASS=list(instance["PASS_TO_PASS"]),
        )

    return build


def _prebuilt_image_eval_policy(eval_script: str, repo: str) -> tuple[str, str]:
    """Keep evaluation hermetic when the immutable instance image is complete.

    SWE-Gym's DVC spec repeats the full image-build install command during each
    evaluation.  It includes an unpinned GitHub dependency and several network
    package installs.  AgentENV starts from the already-built instance image, so
    repeating that command is both unnecessary and non-hermetic.
    """
    if repo != "iterative/dvc":
        return eval_script, "official-eval-script"
    lines = eval_script.splitlines(keepends=True)
    filtered = [line for line in lines if "mock-ssh-server" not in line]
    removed = len(lines) - len(filtered)
    if removed != 1:
        raise PreparationError(
            f"DVC official eval script expected one runtime install line, removed {removed}"
        )
    return "".join(filtered), "prebuilt-instance-image"


def _swegym_eval_script(instance: dict[str, Any], make_test_spec: Any) -> tuple[str, str]:
    test_spec = make_test_spec(instance)
    eval_script = getattr(test_spec, "eval_script", None)
    if not isinstance(eval_script, str) or not eval_script.strip():
        raise PreparationError(f"{instance['instance_id']}: SWE-Gym produced no eval script")
    eval_script, runtime_setup_policy = _prebuilt_image_eval_policy(
        eval_script, str(instance["repo"])
    )
    return _hydra_pytest_policy(eval_script, str(instance["repo"])), runtime_setup_policy


def _verifier_timeout(repo: str) -> int:
    if repo == "Project-MONAI/MONAI":
        return MONAI_VERIFIER_TIMEOUT_SEC
    return DEFAULT_VERIFIER_TIMEOUT_SEC


def build_verifier_specs(
    source: Path,
    bundle: Path,
    *,
    swegym_source: Path | None = None,
) -> list[dict[str, Any]]:
    make_eval_spec = _load_swegym_eval_spec_builder(swegym_source)

    instances = _load_source_instances(source)
    mappings = _read_jsonl(_bundle_paths(bundle)["provisioned_templates"])
    specs: list[dict[str, Any]] = []
    errors: list[str] = []
    for mapping in mappings:
        instance_id = str(mapping["instance_id"])
        instance = instances.get(instance_id)
        if instance is None:
            errors.append(f"{instance_id}: missing source instance")
            continue
        try:
            eval_script, runtime_setup_policy = _swegym_eval_script(instance, make_eval_spec)
            specs.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "spec_id": f"{instance_id}@swegym-{SWEGYM_PACKAGE_REVISION[:8]}-v2",
                    "instance_id": instance_id,
                    "protocol": "swebench-v5",
                    "evaluator": "swegym",
                    "evaluator_revision": SWEGYM_PACKAGE_REVISION,
                    "test_patch_owner": "eval_script",
                    "runtime_setup_policy": runtime_setup_policy,
                    "template_id": mapping["template_id"],
                    "image_digest": mapping["image_digest"],
                    "base_commit": mapping["base_commit"],
                    "repo": instance["repo"],
                    "version": str(instance.get("version") or "unknown"),
                    "workdir": WORKDIR,
                    "eval_script": eval_script,
                    "log_parser": "swegym-repo-specific",
                    "eval_type": "pass_and_fail",
                    "test_patch": instance["test_patch"],
                    "FAIL_TO_PASS": instance["FAIL_TO_PASS"],
                    "PASS_TO_PASS": instance["PASS_TO_PASS"],
                    "timeout_sec": _verifier_timeout(str(instance["repo"])),
                }
            )
        except Exception as exc:
            errors.append(f"{instance_id}: {type(exc).__name__}: {exc}")
    if errors:
        preview = "\n".join(errors[:20])
        raise PreparationError(f"failed to build {len(errors)} VerifierSpecs:\n{preview}")
    _write_jsonl(_bundle_paths(bundle)["all_specs"], specs)
    return specs


def _eval_result_record(outcome: Any) -> dict[str, Any]:
    return {
        "reward": float(outcome.reward),
        "status": str(outcome.status),
        "infra_failure": bool(outcome.infra_failure),
        "reason": outcome.reason,
        "applied_cleanly": bool(outcome.applied_cleanly),
    }


def _reusable_qualification_results(
    rows: list[dict[str, Any]],
    registry: Any,
) -> dict[str, dict[str, Any]]:
    reusable: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("qualification_status") not in {"qualified", "rejected"}:
            continue
        instance_id = str(row.get("instance_id") or "")
        try:
            spec = registry.resolve(instance_id)
        except Exception:
            continue
        if row.get("verifier_spec_id") == spec.spec_id:
            reusable[instance_id] = row
    return reusable


async def _qualify_instances(
    source: Path,
    bundle: Path,
    *,
    concurrency: int,
    timeout_sec: int,
    instance_ids: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    from examples.coding_agent_rl import swe
    from examples.coding_agent_rl.template_mapping import apply_template_mapping, load_template_mapping
    from examples.coding_agent_rl.verifier_spec import apply_verifier_spec, load_verifier_specs
    from vime.utils.types import Sample

    paths = _bundle_paths(bundle)
    candidates = _read_jsonl(paths["candidates"])
    quarantined = _load_quarantined_instances(bundle, candidates)
    instances = _load_source_instances(source)
    mappings = _read_jsonl(paths["provisioned_templates"])
    mapping_ids = {str(row["instance_id"]) for row in mappings}
    specs = _read_jsonl(paths["all_specs"])
    spec_ids = {str(row["instance_id"]) for row in specs}
    all_eligible = [
        row
        for row in candidates
        if str(row["metadata"]["instance_id"]) not in quarantined
        and str(row["metadata"]["instance_id"]) in mapping_ids
        and str(row["metadata"]["instance_id"]) in spec_ids
    ]
    active_ids = {
        str(row["metadata"]["instance_id"])
        for row in candidates
        if str(row["metadata"]["instance_id"]) not in quarantined
    }
    missing_runtime = sorted(active_ids - mapping_ids)
    missing_specs = sorted(active_ids - spec_ids)
    if missing_runtime or missing_specs:
        raise PreparationError(
            "active qualification scope is incomplete: "
            f"missing_templates={missing_runtime[:10]} missing_specs={missing_specs[:10]}"
        )
    eligible = all_eligible
    if instance_ids:
        known = {str(row["metadata"]["instance_id"]) for row in candidates}
        missing = sorted(instance_ids - known)
        if missing:
            raise PreparationError(f"unknown instance_id values: {missing}")
        eligible = [row for row in eligible if str(row["metadata"]["instance_id"]) in instance_ids]
    if limit is not None:
        eligible = eligible[:limit]
    mapping = load_template_mapping(str(paths["provisioned_templates"]))
    registry = load_verifier_specs(str(paths["all_specs"]))
    previous_rows = _read_jsonl(paths["qualification"]) if paths["qualification"].is_file() else []
    previous = _reusable_qualification_results(previous_rows, registry)
    semaphore = asyncio.Semaphore(concurrency)

    def checkpoint() -> None:
        # Preserve results outside a --limit/--instance-id canary selection.
        ordered_ids = [str(row["metadata"]["instance_id"]) for row in all_eligible]
        _write_jsonl(paths["qualification"], [previous[item] for item in ordered_ids if item in previous])

    async def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(row["metadata"]["instance_id"])
        source_instance = instances[instance_id]
        started_at = _utc_now()
        try:
            sample = Sample(prompt=row["prompt"], label=row["label"], metadata=row["metadata"])
            md = swe.get_metadata(sample, swe.PROTOCOL_SWEBENCH)
            md = apply_template_mapping(md, mapping)
            md = apply_verifier_spec(md, registry)
            spec = md["grading"]["verifier_spec"]
            async with semaphore:
                empty = await swe.run_evaluation(md, diff_text="", timeout_sec=timeout_sec)
                gold = await swe.run_evaluation(
                    md,
                    diff_text=str(source_instance["patch"]),
                    timeout_sec=timeout_sec,
                )
            empty_record = _eval_result_record(empty)
            gold_record = _eval_result_record(gold)
            if empty.infra_failure or gold.infra_failure:
                status = "retryable"
                reason = "infrastructure_failure"
            elif not empty.applied_cleanly or not gold.applied_cleanly:
                status = "rejected"
                reason = "patch_application_failure"
            elif float(empty.reward) != 0.0:
                status = "rejected"
                reason = "empty_patch_unexpectedly_resolved"
            elif float(gold.reward) != 1.0:
                status = "rejected"
                reason = "gold_patch_not_resolved"
            else:
                status = "qualified"
                reason = None
            return {
                "schema_version": SCHEMA_VERSION,
                "instance_id": instance_id,
                "verifier_spec_id": spec.spec_id,
                "evaluator": spec.evaluator,
                "evaluator_revision": spec.evaluator_revision,
                "qualification_status": status,
                "reason": reason,
                "empty": empty_record,
                "gold": gold_record,
                "started_at": started_at,
                "finished_at": _utc_now(),
            }
        except Exception as exc:
            return {
                "schema_version": SCHEMA_VERSION,
                "instance_id": instance_id,
                "qualification_status": "retryable",
                "reason": "qualification_exception",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "started_at": started_at,
                "finished_at": _utc_now(),
            }

    pending = [row for row in eligible if str(row["metadata"]["instance_id"]) not in previous]
    tasks = [asyncio.create_task(evaluate(row)) for row in pending]
    selected_ids = {str(row["metadata"]["instance_id"]) for row in eligible}
    completed = sum(instance_id in previous for instance_id in selected_ids)
    for task in asyncio.as_completed(tasks):
        result = await task
        previous[str(result["instance_id"])] = result
        completed += 1
        checkpoint()
        print(
            f"qualify {completed}/{len(eligible)} {result['instance_id']}: "
            f"{result['qualification_status']}",
            flush=True,
        )

    checkpoint()
    selected_results = [row for key, row in previous.items() if key in selected_ids]
    counts = {
        "qualified": sum(row.get("qualification_status") == "qualified" for row in selected_results),
        "rejected": sum(row.get("qualification_status") == "rejected" for row in selected_results),
        "retryable": sum(row.get("qualification_status") == "retryable" for row in selected_results),
        "eligible": len(eligible),
    }
    return counts


def qualify(
    source: Path,
    bundle: Path,
    *,
    api_url: str,
    sandbox_url: str,
    api_key: str,
    sdk_path: str | None,
    concurrency: int,
    timeout_sec: int,
    swegym_source: Path | None = None,
    instance_ids: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    # Qualification is a CPU-side orchestration process.  Configure the
    # E2B-compatible AgentENV client in-process and avoid importing torch_npu
    # when the caller intentionally has not sourced CANN.
    os.environ["E2B_API_URL"] = api_url
    os.environ["E2B_SANDBOX_URL"] = sandbox_url
    os.environ["E2B_API_KEY"] = api_key
    os.environ["VIME_AGENT_E2B_SELECTOR_MODE"] = "template"
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    if sdk_path:
        os.environ["VIME_AGENT_E2B_SDK_PATH"] = sdk_path
    _append_sdk_path(sdk_path)
    build_verifier_specs(source, bundle, swegym_source=swegym_source)
    return asyncio.run(
        _qualify_instances(
            source,
            bundle,
            concurrency=concurrency,
            timeout_sec=timeout_sec,
            instance_ids=instance_ids,
            limit=limit,
        )
    )


def _assert_no_gold_leak(row: dict[str, Any], gold_patch: str) -> None:
    serialized_prompt = _json_dumps(row.get("prompt"))
    if gold_patch and gold_patch in _json_dumps(row):
        raise PreparationError(f"{row.get('label')}: gold patch leaked into candidate row")
    private_keys = {"test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "eval_script", "verifier_spec"}
    prompt_value = row.get("prompt")
    if isinstance(prompt_value, list):
        for message in prompt_value:
            if isinstance(message, dict) and private_keys.intersection(message):
                raise PreparationError(f"{row.get('label')}: verifier key leaked into prompt")
    if any(marker in serialized_prompt for marker in (">>>>> Start Test Output", ">>>>> End Test Output")):
        raise PreparationError(f"{row.get('label')}: eval script marker leaked into prompt")


def finalize(source: Path, bundle: Path) -> dict[str, Any]:
    paths = _bundle_paths(bundle)
    candidates = _read_jsonl(paths["candidates"])
    mappings = _read_jsonl(paths["provisioned_templates"])
    specs = _read_jsonl(paths["all_specs"])
    results = _read_jsonl(paths["qualification"])
    source_instances = _load_source_instances(source)

    quarantined = _load_quarantined_instances(bundle, candidates)
    candidate_ids = [str(row["metadata"]["instance_id"]) for row in candidates]
    active_candidate_ids = [instance_id for instance_id in candidate_ids if instance_id not in quarantined]
    result_by_id = {str(row["instance_id"]): row for row in results}
    missing = [instance_id for instance_id in active_candidate_ids if instance_id not in result_by_id]
    if missing:
        raise PreparationError(f"qualification has no result for {len(missing)} instances: {missing[:10]}")
    spec_id_by_instance = {str(row["instance_id"]): str(row["spec_id"]) for row in specs}
    stale = [
        instance_id
        for instance_id in active_candidate_ids
        if result_by_id[instance_id].get("verifier_spec_id") != spec_id_by_instance.get(instance_id)
    ]
    if stale:
        raise PreparationError(
            f"qualification uses stale or missing VerifierSpec IDs for {len(stale)} instances: "
            f"{stale[:10]}"
        )
    retryable = [
        result_by_id[instance_id]
        for instance_id in active_candidate_ids
        if result_by_id[instance_id].get("qualification_status") == "retryable"
    ]
    if retryable:
        raise PreparationError(f"qualification still has {len(retryable)} retryable instances")

    qualified_ids = {
        instance_id
        for instance_id in active_candidate_ids
        if result_by_id[instance_id].get("qualification_status") == "qualified"
    }
    train_rows = [row for row in candidates if str(row["metadata"]["instance_id"]) in qualified_ids]
    template_rows = [row for row in mappings if str(row["instance_id"]) in qualified_ids]
    spec_rows = [row for row in specs if str(row["instance_id"]) in qualified_ids]
    if not (len(train_rows) == len(template_rows) == len(spec_rows) == len(qualified_ids)):
        raise PreparationError(
            "qualified artifact cardinality mismatch: "
            f"train={len(train_rows)} templates={len(template_rows)} specs={len(spec_rows)} "
            f"qualified={len(qualified_ids)}"
        )

    for row in train_rows:
        instance_id = str(row["metadata"]["instance_id"])
        _assert_no_gold_leak(row, str(source_instances[instance_id]["patch"]))

    rejected = [
        result_by_id[instance_id]
        for instance_id in active_candidate_ids
        if result_by_id[instance_id].get("qualification_status") != "qualified"
    ]
    _write_jsonl(paths["train"], train_rows)
    _write_jsonl(paths["templates"], template_rows)
    _write_jsonl(paths["specs"], spec_rows)
    _write_jsonl(paths["rejected"], rejected)

    source_manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": source_manifest["dataset"],
        "source": {
            "path": source_manifest["source_path"],
            "sha256": source_manifest["source_sha256"],
            "rows": source_manifest["source_rows"],
        },
        "counts": {
            "input": len(candidates),
            "eligible": len(active_candidate_ids),
            "qualified": len(qualified_ids),
            "rejected": len(rejected),
            "retryable": 0,
            "quarantined": len(quarantined),
        },
        "qualified_instance_ids": [item for item in candidate_ids if item in qualified_ids],
        "rejected_instance_ids": [str(row["instance_id"]) for row in rejected],
        "quarantined_instance_ids": [item for item in candidate_ids if item in quarantined],
        "outputs": {},
        "generated_at": _utc_now(),
    }
    for key in ("train", "templates", "specs", "qualification", "rejected"):
        manifest["outputs"][key] = {"path": str(paths[key]), "sha256": _sha256(paths[key])}
    if paths["quarantine"].is_file():
        manifest["outputs"]["quarantine"] = {
            "path": str(paths["quarantine"]),
            "sha256": _sha256(paths["quarantine"]),
        }
    _write_json(paths["manifest"], manifest)
    return manifest


def _api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    path = Path(args.api_key_file).expanduser()
    if not path.is_file():
        raise PreparationError(f"AgentENV API key file does not exist: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise PreparationError(f"AgentENV API key file is empty: {path}")
    return value


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _add_common_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)


def _add_online(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-url", default=os.environ.get("E2B_API_URL", "http://127.0.0.1:18000"))
    parser.add_argument(
        "--sandbox-url", default=os.environ.get("E2B_SANDBOX_URL", "http://127.0.0.1:18000")
    )
    parser.add_argument("--api-key", default=os.environ.get("E2B_API_KEY", ""))
    parser.add_argument(
        "--api-key-file",
        default=os.environ.get("VIME_AGENTENV_API_KEY_FILE", "/home/developer/.config/agentenv/api-key"),
    )
    parser.add_argument("--sdk-path", default=os.environ.get("VIME_AGENT_E2B_SDK_PATH", ""))


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--limit", type=_positive)


def _add_swegym_source(parser: argparse.ArgumentParser) -> None:
    configured = os.environ.get("VIME_SWEGYM_SOURCE_PATH")
    parser.add_argument(
        "--swegym-source",
        type=Path,
        default=Path(configured) if configured else None,
        help=f"official SWE-Gym source at revision {SWEGYM_PACKAGE_REVISION}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate and normalize the source JSONL")
    _add_common_source(inspect_parser)
    inspect_parser.add_argument("--expected-rows", type=_positive, default=EXPECTED_ROWS)
    inspect_parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    inspect_parser.add_argument("--registry-base", default=DEFAULT_REGISTRY_BASE)

    provision_parser = subparsers.add_parser("provision", help="create/reuse AgentENV templates")
    provision_parser.add_argument("--bundle", type=Path, required=True)
    provision_parser.add_argument("--existing-template-map", type=Path)
    provision_parser.add_argument("--registry-base", default=DEFAULT_REGISTRY_BASE)
    provision_parser.add_argument("--concurrency", type=_positive, default=2)
    provision_parser.add_argument("--retries", type=_positive, default=3)
    provision_parser.add_argument("--cpu-count", type=_positive, default=DEFAULT_TEMPLATE_CPU)
    provision_parser.add_argument("--memory-mib", type=_positive, default=DEFAULT_TEMPLATE_MEMORY_MIB)
    provision_parser.add_argument(
        "--template-alias-revision",
        default=os.environ.get("SWEGYM_TEMPLATE_ALIAS_REVISION", DEFAULT_TEMPLATE_ALIAS_REVISION),
    )
    provision_parser.add_argument(
        "--probe-request-timeout",
        type=_positive,
        default=int(os.environ.get("SWEGYM_PROBE_REQUEST_TIMEOUT", DEFAULT_PROBE_REQUEST_TIMEOUT)),
    )
    provision_parser.add_argument(
        "--probe-sandbox-timeout",
        type=_positive,
        default=int(os.environ.get("SWEGYM_PROBE_SANDBOX_TIMEOUT", DEFAULT_PROBE_SANDBOX_TIMEOUT)),
    )
    provision_parser.add_argument(
        "--probe-command-timeout",
        type=_positive,
        default=int(os.environ.get("SWEGYM_PROBE_COMMAND_TIMEOUT", DEFAULT_PROBE_COMMAND_TIMEOUT)),
    )
    _add_online(provision_parser)
    _add_selection(provision_parser)

    qualify_parser = subparsers.add_parser("qualify", help="build specs and run empty/gold verification")
    _add_common_source(qualify_parser)
    qualify_parser.add_argument("--concurrency", type=_positive, default=2)
    qualify_parser.add_argument("--timeout-sec", type=_positive, default=900)
    _add_online(qualify_parser)
    _add_swegym_source(qualify_parser)
    _add_selection(qualify_parser)

    finalize_parser = subparsers.add_parser("finalize", help="freeze qualified training artifacts")
    _add_common_source(finalize_parser)

    all_parser = subparsers.add_parser("prepare-all", help="run every preparation phase")
    _add_common_source(all_parser)
    all_parser.add_argument("--expected-rows", type=_positive, default=EXPECTED_ROWS)
    all_parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    all_parser.add_argument("--existing-template-map", type=Path)
    all_parser.add_argument("--registry-base", default=DEFAULT_REGISTRY_BASE)
    all_parser.add_argument("--provision-concurrency", type=_positive, default=2)
    all_parser.add_argument("--qualify-concurrency", type=_positive, default=2)
    all_parser.add_argument("--retries", type=_positive, default=3)
    all_parser.add_argument("--cpu-count", type=_positive, default=DEFAULT_TEMPLATE_CPU)
    all_parser.add_argument("--memory-mib", type=_positive, default=DEFAULT_TEMPLATE_MEMORY_MIB)
    all_parser.add_argument(
        "--template-alias-revision",
        default=os.environ.get("SWEGYM_TEMPLATE_ALIAS_REVISION", DEFAULT_TEMPLATE_ALIAS_REVISION),
    )
    all_parser.add_argument(
        "--probe-request-timeout",
        type=_positive,
        default=int(os.environ.get("SWEGYM_PROBE_REQUEST_TIMEOUT", DEFAULT_PROBE_REQUEST_TIMEOUT)),
    )
    all_parser.add_argument(
        "--probe-sandbox-timeout",
        type=_positive,
        default=int(os.environ.get("SWEGYM_PROBE_SANDBOX_TIMEOUT", DEFAULT_PROBE_SANDBOX_TIMEOUT)),
    )
    all_parser.add_argument(
        "--probe-command-timeout",
        type=_positive,
        default=int(os.environ.get("SWEGYM_PROBE_COMMAND_TIMEOUT", DEFAULT_PROBE_COMMAND_TIMEOUT)),
    )
    all_parser.add_argument("--timeout-sec", type=_positive, default=900)
    _add_online(all_parser)
    _add_swegym_source(all_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_source(
                args.source,
                args.bundle,
                expected_rows=args.expected_rows,
                expected_sha256=args.expected_source_sha256,
                registry_base=args.registry_base,
            )
            print(_json_dumps(result, pretty=True), end="")
            return 0
        if args.command == "provision":
            result = provision_templates(
                args.bundle,
                existing_map=args.existing_template_map,
                registry_base=args.registry_base,
                api_url=args.api_url,
                sandbox_url=args.sandbox_url,
                api_key=_api_key(args),
                sdk_path=args.sdk_path,
                concurrency=args.concurrency,
                retries=args.retries,
                cpu_count=args.cpu_count,
                memory_mib=args.memory_mib,
                template_alias_revision=args.template_alias_revision,
                probe_request_timeout=args.probe_request_timeout,
                probe_sandbox_timeout=args.probe_sandbox_timeout,
                probe_command_timeout=args.probe_command_timeout,
                instance_ids=set(args.instance_id) or None,
                limit=args.limit,
            )
            print(_json_dumps(result, pretty=True), end="")
            return 2 if result["retryable"] else 0
        if args.command == "qualify":
            result = qualify(
                args.source,
                args.bundle,
                api_url=args.api_url,
                sandbox_url=args.sandbox_url,
                api_key=_api_key(args),
                sdk_path=args.sdk_path,
                concurrency=args.concurrency,
                timeout_sec=args.timeout_sec,
                swegym_source=args.swegym_source,
                instance_ids=set(args.instance_id) or None,
                limit=args.limit,
            )
            print(_json_dumps(result, pretty=True), end="")
            return 2 if result["retryable"] else 0
        if args.command == "finalize":
            result = finalize(args.source, args.bundle)
            print(_json_dumps(result, pretty=True), end="")
            return 0
        if args.command == "prepare-all":
            inspect_source(
                args.source,
                args.bundle,
                expected_rows=args.expected_rows,
                expected_sha256=args.expected_source_sha256,
                registry_base=args.registry_base,
            )
            provision_result = provision_templates(
                args.bundle,
                existing_map=args.existing_template_map,
                registry_base=args.registry_base,
                api_url=args.api_url,
                sandbox_url=args.sandbox_url,
                api_key=_api_key(args),
                sdk_path=args.sdk_path,
                concurrency=args.provision_concurrency,
                retries=args.retries,
                cpu_count=args.cpu_count,
                memory_mib=args.memory_mib,
                template_alias_revision=args.template_alias_revision,
                probe_request_timeout=args.probe_request_timeout,
                probe_sandbox_timeout=args.probe_sandbox_timeout,
                probe_command_timeout=args.probe_command_timeout,
            )
            if provision_result["retryable"]:
                raise PreparationError(
                    f"provisioning left {provision_result['retryable']} retryable instances; resume after diagnosis"
                )
            qualification_result = qualify(
                args.source,
                args.bundle,
                api_url=args.api_url,
                sandbox_url=args.sandbox_url,
                api_key=_api_key(args),
                sdk_path=args.sdk_path,
                concurrency=args.qualify_concurrency,
                timeout_sec=args.timeout_sec,
                swegym_source=args.swegym_source,
            )
            if qualification_result["retryable"]:
                raise PreparationError(
                    f"qualification left {qualification_result['retryable']} retryable instances; resume after diagnosis"
                )
            result = finalize(args.source, args.bundle)
            print(_json_dumps(result, pretty=True), end="")
            return 0
        raise AssertionError(f"unknown command: {args.command}")
    except PreparationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
