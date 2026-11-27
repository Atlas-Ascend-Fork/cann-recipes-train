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

"""Terminal-Bench dataset preparer: source JSONL -> prepared training bundle.

Stages (mirroring the SWE-Gym preparer's contract, with Terminal-Bench
semantics). Task images are NOT built here: every task template must already
exist in AgentENV (e.g. created via ``aenv pull``), or be available as OCI
images for Harbor. This module converts the corpus and assembles the bundle.
VIME_AGENT_SANDBOX_BACKEND selects the provider; Harbor does not need E2B credentials.

  from-tasks   Convert a Terminal-Bench task tree (task.yaml + tests/) into a
               source JSONL. No sandbox or registry access.
  from-tmax    Convert the allenai/TMax-15K corpus into a source JSONL,
               dropping tasks whose released data cannot yield a usable
               environment. No sandbox or registry access.
  filter-tasks /
  sample-tasks Select the training subset; optionally emit a task-id list
               (labels.txt) that feeds ``gen-bundle --instances``.
  gen-bundle   Join an instance list against the corpus, discover the ready
               AgentENV template for each task (alias -> catalog scan ->
               aenv CLI; read-only by default), probe a sandbox (/tests
               presence + content hash, optional pristine-oracle qualification
               that must FAIL on the untouched environment) and export the
               bundle: rollout/train.no_gold.jsonl (prompt rows, no grading
               data), environment/agentenv_templates.jsonl,
               verifier/verifier_specs.jsonl and preparation_manifest.json.

Source JSONL row schema::

    {"task_id": "hello-world", "instruction": "...",
     "image": "ghcr.io/laude-institute/tb-hello-world:latest",
     "workdir": "/root",                       # optional, default /root
     "run_tests_cmd": "bash /tests/run-tests.sh",  # optional
     "timeout_sec": 900}                       # optional

Only stdlib at module level; e2b and yaml are imported lazily so from-tasks and
from-tmax run on a bare preparation host. DEFAULT_* constants mirror
terminalbench.py -- keep them in sync.
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
import concurrent.futures
import json
import logging
import os
import random
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROTOCOL = "terminalbench-v1"

# Mirror of terminalbench.py defaults (kept local so this module never imports
# the torch-backed task layer on a bare prep host).
DEFAULT_WORKDIR = "/root"
DEFAULT_TESTS_PATH = "/tests/run-tests.sh"
DEFAULT_INITIAL_STATE_PATH = "/tests/test_initial_state.py"
DEFAULT_RUN_TESTS_CMD = "bash /tests/run-tests.sh"
DEFAULT_TIMEOUT_SEC = 900
TESTS_HASH_CMD = "cd /tests && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1"

# TMax-15K per-complexity grading budgets (seconds).
TMAX_COMPLEXITY_TIMEOUTS = (("intricate", 2400), ("complex", 1800), ("moderate", 1200), ("short", 900))


def _oracle_targets(run_tests_cmd: str) -> list[str]:
    """Absolute-path tokens of the grading command (mirrors terminalbench.py)."""
    return [token for token in shlex.split(run_tests_cmd) if token.startswith("/")]

REGISTRY_USER_AGENT = "vime-tbench-preparer/1.0"
DEFAULT_TEMPLATE_ALIAS_REVISION = "v1"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TASK_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class PreparationError(RuntimeError):
    """A preparation stage could not complete."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(_json_dumps(row) + "\n" for row in rows)
    _atomic_write_text(path, payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreparationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise PreparationError(f"{path}:{line_number}: each JSONL row must be an object")
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# from-tasks: Terminal-Bench task tree -> source JSONL
# ---------------------------------------------------------------------------

def _parse_task_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise PreparationError("PyYAML is required for from-tasks (pip install pyyaml)") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PreparationError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise PreparationError(f"{path}: task.yaml must be a mapping")
    return data


def from_tasks(tasks_root: Path, output: Path, *, image_template: str) -> int:
    """Walk <tasks_root>/<task-dir>/{task.yaml, tests/run-tests.sh} trees."""
    if not tasks_root.is_dir():
        raise PreparationError(f"tasks root is not a directory: {tasks_root}")
    rows: list[dict[str, Any]] = []
    for task_dir in sorted(p for p in tasks_root.iterdir() if p.is_dir()):
        task_yaml = task_dir / "task.yaml"
        if not task_yaml.is_file():
            continue  # not a task directory
        tests_script = task_dir / "tests" / "run-tests.sh"
        if not tests_script.is_file():
            raise PreparationError(f"{task_dir}: tests/run-tests.sh is missing")
        data = _parse_task_yaml(task_yaml)
        instruction = data.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise PreparationError(f"{task_yaml}: instruction must be non-empty text")
        task_id = task_dir.name
        image = data.get("image")
        if not isinstance(image, str) or not image.strip():
            image = image_template.format(task_id=task_id)
        timeout_sec = data.get("timeout_sec") or data.get("timeout") or DEFAULT_TIMEOUT_SEC
        if not isinstance(timeout_sec, int) or timeout_sec <= 0:
            raise PreparationError(f"{task_yaml}: timeout_sec must be a positive integer")
        rows.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "tests": {
                    str(p.relative_to(task_dir / "tests")): p.read_text()
                    for p in sorted((task_dir / "tests").rglob("*"))
                    if p.is_file()
                },
                "image": image.strip(),
                "workdir": data.get("workdir") or DEFAULT_WORKDIR,
                "run_tests_cmd": data.get("run_tests_cmd") or DEFAULT_RUN_TESTS_CMD,
                "timeout_sec": timeout_sec,
            }
        )
    if not rows:
        raise PreparationError(f"no task directories found under {tasks_root}")
    _write_jsonl(output, rows)
    logger.info("wrote %s source rows to %s", len(rows), output)
    return 0


# ---------------------------------------------------------------------------
# Registry digest resolution (adapted from the SWE-Gym preparer)
# ---------------------------------------------------------------------------

def _parse_image_ref(image: str) -> tuple[str, str]:
    if "@" in image:
        repository, digest = image.rsplit("@", 1)
        if not _DIGEST_RE.fullmatch(digest):
            raise PreparationError(f"invalid image digest in {image!r}")
        return repository, digest
    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    if last_colon > last_slash:
        return image[:last_colon], image[last_colon + 1:]
    return image, "latest"


def _split_registry(image: str, registry_base: str | None) -> tuple[str, str]:
    """Return (registry API base, repository-without-host).

    With ``registry_base`` unset, the host is taken from the image reference
    itself (ghcr.io/..., docker.io/...). With it set, that mirror serves both
    digest resolution and template pulls, like the SWE-Gym flow.
    """
    repository, _ = _parse_image_ref(image)
    if registry_base:
        host = urllib.parse.urlsplit(registry_base).netloc
        for prefix in (f"{host}/", "docker.io/", "index.docker.io/"):
            repository = repository.removeprefix(prefix)
        return registry_base.rstrip("/"), repository
    first, _, rest = repository.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        return f"https://{first}", rest
    return "https://registry-1.docker.io", repository.removeprefix("docker.io/").removeprefix("index.docker.io/")


def _bearer_parameters(header: str) -> dict[str, str]:
    if not header.lower().startswith("bearer "):
        raise PreparationError(f"unsupported registry authentication challenge: {header!r}")
    params: dict[str, str] = {}
    for match in re.finditer(r'(\w+)="([^"]*)"', header[7:]):
        params[match.group(1)] = match.group(2)
    if "realm" not in params:
        raise PreparationError(f"registry authentication challenge has no realm: {header!r}")
    return params


def resolve_registry_digest(image: str, registry_base: str | None, *, timeout: int = 60) -> str:
    """Resolve an OCI tag through a registry v2 HEAD request."""
    repository, reference = _parse_image_ref(image)
    if _DIGEST_RE.fullmatch(reference):
        return reference
    base, repository = _split_registry(image, registry_base)
    manifest_url = f"{base}/v2/{repository}/manifests/{urllib.parse.quote(reference, safe='')}"
    accept = ", ".join(
        (
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        )
    )
    headers = {"Accept": accept, "User-Agent": REGISTRY_USER_AGENT}

    def request_manifest(extra_headers: dict[str, str] | None = None):
        combined = dict(headers)
        combined.update(extra_headers or {})
        request = urllib.request.Request(manifest_url, method="HEAD", headers=combined)
        return urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context())

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
            token_request = urllib.request.Request(token_url, headers={"User-Agent": REGISTRY_USER_AGENT})
            with urllib.request.urlopen(
                token_request, timeout=timeout, context=ssl.create_default_context()
            ) as token_response:
                token_payload = json.load(token_response)
        except (OSError, ValueError) as token_exc:
            raise PreparationError(f"registry token request failed for {image}: {token_exc}") from token_exc
        token = token_payload.get("token") or token_payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise PreparationError(f"registry token response has no token for {image}")
        try:
            response = request_manifest({"Authorization": f"Bearer {token}"})
        except urllib.error.HTTPError as retry_exc:
            raise PreparationError(
                f"authenticated registry HEAD failed for {image}: HTTP {retry_exc.code}"
            ) from retry_exc
    except OSError as exc:
        raise PreparationError(f"registry HEAD failed for {image}: {exc}") from exc

    with response:
        digest = response.headers.get("Docker-Content-Digest", "").strip().lower()
    if not _DIGEST_RE.fullmatch(digest):
        raise PreparationError(f"registry returned no valid Docker-Content-Digest for {image}: {digest!r}")
    return digest


# ---------------------------------------------------------------------------
# AgentENV API helpers
# ---------------------------------------------------------------------------

def _append_sdk_path(path: str | None) -> None:
    if not path:
        return
    sdk_path = Path(path).expanduser().resolve()
    if not sdk_path.is_dir():
        raise PreparationError(f"E2B SDK path is not a directory: {sdk_path}")
    if str(sdk_path) not in sys.path:
        sys.path.append(str(sdk_path))


def _template_alias(task_id: str, digest: str, revision: str = DEFAULT_TEMPLATE_ALIAS_REVISION) -> str:
    if not re.fullmatch(r"v[1-9][0-9]*", revision):
        raise PreparationError(f"template alias revision must match 'v<positive integer>', got {revision!r}")
    slug = re.sub(r"[^a-z0-9-]+", "-", task_id.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return f"tbench-{slug[:72]}-{digest.removeprefix('sha256:')[:8]}-{revision}"


@dataclass
class _SandboxSession:
    """One synchronous E2B sandbox lifetime against AgentENV."""

    template_id: str
    api_url: str
    sandbox_url: str
    api_key: str
    timeout: int

    def __enter__(self):
        from e2b import Sandbox  # type: ignore

        self._sandbox = Sandbox.create(
            template=self.template_id,
            api_key=self.api_key,
            api_url=self.api_url,
            sandbox_url=self.sandbox_url,
            timeout=self.timeout,
        )
        return self._sandbox

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._sandbox.kill()
        except Exception:
            pass


def _probe_template(plan: dict[str, Any], template_id: str, *, api_url: str, sandbox_url: str, api_key: str,
                    sandbox_timeout: int, command_timeout: int) -> dict[str, Any]:
    """Boot a sandbox from the template and capture the oracle's ground truth.

    Records: workdir presence, the grading command's oracle targets, the
    canonical /tests content hash later enforced at every grading, and (when
    the image ships one) a /tests/test_initial_state.py run that filters out
    environments whose services did not come up.
    """
    with _SandboxSession(template_id, api_url, sandbox_url, api_key, sandbox_timeout) as sandbox:
        _inject_probe_tests(sandbox, plan)
        result = sandbox.commands.run(f"test -d {plan['workdir']}", timeout=command_timeout)
        if result.exit_code != 0:
            raise PreparationError(
                f"template probe failed for {plan['task_id']}: workdir check failed "
                f"(exit={result.exit_code}): {(result.stderr or '')[:200]}"
            )
        for target in _oracle_targets(str(plan["run_tests_cmd"])) or [DEFAULT_TESTS_PATH]:
            result = sandbox.commands.run(f"test -e {target}", timeout=command_timeout)
            if result.exit_code != 0:
                raise PreparationError(
                    f"template probe failed for {plan['task_id']}: oracle target {target} missing "
                    f"(exit={result.exit_code})"
                )
        result = sandbox.commands.run(TESTS_HASH_CMD, timeout=command_timeout)
        if result.exit_code != 0 or not result.stdout.strip():
            raise PreparationError(
                f"template probe failed for {plan['task_id']}: /tests hash command failed "
                f"(exit={result.exit_code})"
            )
        tests_sha256 = result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{64}", tests_sha256):
            raise PreparationError(f"template probe failed for {plan['task_id']}: invalid /tests hash {tests_sha256!r}")
        validation: dict[str, Any] = {"template_status": "ready", "tests_sha256": tests_sha256}
        initial = sandbox.commands.run(f"test -f {DEFAULT_INITIAL_STATE_PATH}", timeout=command_timeout)
        if initial.exit_code == 0:
            run = sandbox.commands.run(
                f"python3 -m pytest {DEFAULT_INITIAL_STATE_PATH} -q", timeout=command_timeout * 5
            )
            if run.exit_code != 0:
                raise PreparationError(
                    f"template probe failed for {plan['task_id']}: initial-state test failed "
                    f"(exit={run.exit_code}): {(run.stdout or '')[-300:]}"
                )
            validation["initial_state"] = "passed"
        else:
            validation["initial_state"] = "absent"
        return validation


def _bundle_paths(bundle: Path) -> dict[str, Path]:
    return {
        "templates": bundle / "environment/agentenv_templates.jsonl",
        "qualification": bundle / "qualification/qualification_results.jsonl",
        "instances_tsv": bundle / "state/instances.tsv",
        "train_rows": bundle / "rollout/train.no_gold.jsonl",
        "verifier_specs": bundle / "verifier/verifier_specs.jsonl",
        "manifest": bundle / "preparation_manifest.json",
    }


def _qualify_one(plan: dict[str, Any], record: dict[str, Any], *, api_url: str, sandbox_url: str,
                 api_key: str, sandbox_timeout: int) -> dict[str, Any]:
    task_id = str(plan["task_id"])
    budget = int(plan["timeout_sec"])
    with _SandboxSession(str(record["template_id"]), api_url, sandbox_url, api_key,
                         sandbox_timeout + budget) as sandbox:
        _inject_probe_tests(sandbox, plan)
        try:
            result = sandbox.commands.run(str(plan["run_tests_cmd"]), timeout=budget)
        except Exception as exc:
            return {"task_id": task_id, "qualification_status": "retryable",
                    "reason": f"oracle_rpc_error:{type(exc).__name__}",
                    "checked_at": _utc_now()}
    if result.exit_code == 0:
        status, reason = "rejected", "oracle_passes_on_pristine_environment"
    else:
        status, reason = "qualified", f"oracle_exit_{result.exit_code}"
    return {"task_id": task_id, "qualification_status": status, "reason": reason,
            "pristine_exit_code": result.exit_code, "checked_at": _utc_now()}


# ---------------------------------------------------------------------------
# from-tmax: allenai/TMax-15K -> source JSONL
# ---------------------------------------------------------------------------

def _tmax_timeout_for(complexity: str) -> int:
    for prefix, timeout in TMAX_COMPLEXITY_TIMEOUTS:
        if complexity.lower().startswith(prefix):
            return timeout
    return DEFAULT_TIMEOUT_SEC


def from_tmax(source: Path, output: Path, *, image_template: str, keep_files_tasks: bool,
              keep_arch_pinned: bool, limit: int | None) -> int:
    """Convert the TMax-15K corpus into this example's source JSONL.

    No image building happens here: task templates must already exist in
    AgentENV (``--image-template`` only records the OCI reference each task is
    expected to have a ready template for). Tasks whose container_def pulls
    fixtures from the authors' cluster (%files) or pins x86 binaries are
    dropped by default -- their environments cannot be reproduced from the
    released data, so no ready template can exist for them either.
    """
    rows = _load_tmax_rows(source)
    if not rows:
        raise PreparationError(f"no rows found in {source}")

    kept: list[dict[str, Any]] = []
    dropped: dict[str, int] = {"malformed": 0, "external_fixtures": 0, "arch_pinned": 0, "duplicate_task_id": 0}
    seen: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        container_def = str(row.get("container_def") or "")
        if not task_id or not container_def or "description" not in row or "test_final_state" not in row:
            dropped["malformed"] += 1
            continue
        if task_id in seen:
            dropped["duplicate_task_id"] += 1
            continue
        if not keep_files_tasks and "%files" in container_def:
            dropped["external_fixtures"] += 1
            continue
        if not keep_arch_pinned and re.search(r"x86_64|amd64|i[36]86", container_def, flags=re.IGNORECASE):
            dropped["arch_pinned"] += 1
            continue
        seen.add(task_id)

        kept.append(
            {
                "task_id": task_id,
                "instruction": str(row["description"]),
                "tests": {"test_final_state.py": str(row["test_final_state"])},
                "image": image_template.format(task_id=task_id),
                "workdir": str(row.get("workdir") or "/home/user"),
                "run_tests_cmd": "python3 -m pytest /tests/test_final_state.py -q",
                "timeout_sec": _tmax_timeout_for(str(row.get("task_complexity") or "")),
                "metadata": {
                    "domain": str(row.get("domain") or ""),
                    "task_complexity": str(row.get("task_complexity") or ""),
                    "skill_type": str(row.get("skill_type") or ""),
                    "language": str(row.get("language") or ""),
                    "scenario": str(row.get("scenario") or ""),
                },
            }
        )
        if limit is not None and len(kept) >= limit:
            break

    if not kept:
        raise PreparationError(f"no TMax tasks survived filtering (dropped: {dropped})")
    _write_jsonl(output, kept)
    logger.info(
        "from-tmax: kept %s / %s tasks -> %s; dropped %s", len(kept), len(rows), output, dropped
    )
    return 0


# ---------------------------------------------------------------------------
# filter-tasks: subset selection over any bundle/source JSONL
# ---------------------------------------------------------------------------

def _load_id_list(value: str | None) -> set[str] | None:
    """Accept 'a,b,c' or a path to a newline-separated id file.

    Blank lines and '#' comment lines are ignored in file mode.
    """
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        ids = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    else:
        ids = {item.strip() for item in value.split(",") if item.strip()}
    return ids or None


def _dotted_lookup(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def filter_tasks(source: Path, output: Path, *, include_ids: str | None, exclude_ids: str | None,
                 where: list[str] | None, limit: int | None, labels: Path | None) -> int:
    """Select a training subset by task id lists and metadata field equality.

    ``--where`` compares dotted paths against string values, e.g.
    ``--where metadata.domain=data_science``. Conditions are ANDed. ``--labels``
    optionally writes the kept task ids as a one-per-line list for
    ``gen-bundle --instances``.
    """
    rows = _read_jsonl(source)
    include = _load_id_list(include_ids)
    exclude = _load_id_list(exclude_ids)
    conditions: list[tuple[str, str]] = []
    for clause in where or []:
        if "=" not in clause:
            raise PreparationError(f"--where expects key=value, got {clause!r}")
        conditions.append(tuple(clause.split("=", 1)))  # type: ignore[arg-type]

    kept = []
    for row in rows:
        task_id = str(row.get("label") or _dotted_lookup(row, "metadata.task_id") or row.get("task_id") or "")
        if include is not None and task_id not in include:
            continue
        if exclude is not None and task_id in exclude:
            continue
        if any(str(_dotted_lookup(row, key)) != value for key, value in conditions):
            continue
        kept.append(row)
    if limit is not None:
        kept = kept[:limit]
    if not kept:
        raise PreparationError(f"no rows matched the filter out of {len(rows)}")
    _write_jsonl(output, kept)
    if labels is not None:
        labels.write_text(
            "".join(
                f"{str(row.get('label') or _dotted_lookup(row, 'metadata.task_id') or row.get('task_id') or '')}\n"
                for row in kept
            ),
            encoding="utf-8",
        )
    logger.info(
        "filter-tasks: kept %s / %s rows -> %s%s",
        len(kept), len(rows), output, "; labels -> " + str(labels) if labels is not None else "",
    )
    return 0


# ---------------------------------------------------------------------------
# sample-tasks: seeded random subset (optionally proportional per domain),
# adapted from the upstream tmax repo's sample_tmax_instances.py
# ---------------------------------------------------------------------------

def _proportional_allocation(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation of ``total`` picks, proportional to counts."""
    available = sum(counts.values())
    total = min(total, available)
    exact = {name: count * total / available for name, count in counts.items()}
    alloc = {name: int(quota) for name, quota in exact.items()}
    remaining = total - sum(alloc.values())
    by_remainder = sorted(exact, key=lambda name: (-(exact[name] - alloc[name]), name))
    for name in by_remainder[:remaining]:
        alloc[name] += 1
    return {name: n for name, n in alloc.items() if n > 0}


def sample_tasks(source: Path, output: Path, labels: Path | None, *, total: int, seed: int,
                 strategy: str, where: list[str] | None) -> int:
    """Draw a reproducible random subset and write it as JSONL + a task-id list.

    ``strategy=proportional`` (default) allocates the total across
    ``metadata.domain`` values in proportion to their counts, mirroring the
    upstream per-dataset sampler; it falls back to uniform sampling when rows
    carry no domain metadata. The labels file feeds ``gen-bundle --instances``.
    """
    rows = _read_jsonl(source)
    conditions: list[tuple[str, str]] = []
    for clause in where or []:
        if "=" not in clause:
            raise PreparationError(f"--where expects key=value, got {clause!r}")
        conditions.append(tuple(clause.split("=", 1)))  # type: ignore[arg-type]
    if conditions:
        rows = [
            row for row in rows
            if all(str(_dotted_lookup(row, key)) == value for key, value in conditions)
        ]
    if not rows:
        raise PreparationError(f"no rows to sample from in {source}")
    if total <= 0:
        raise PreparationError("--total must be a positive integer")

    rng = random.Random(seed)
    domains = [str(_dotted_lookup(row, "metadata.domain") or "") for row in rows]
    if strategy == "proportional" and all(domains):
        groups: dict[str, list[int]] = {}
        for index, domain in enumerate(domains):
            groups.setdefault(domain, []).append(index)
        alloc = _proportional_allocation({name: len(ix) for name, ix in groups.items()}, total)
        chosen: list[int] = []
        for name, count in sorted(alloc.items()):
            chosen.extend(rng.sample(groups[name], count))
        sampled = [rows[index] for index in sorted(chosen)]
        logger.info("sample-tasks: proportional allocation across domains: %s", alloc)
    else:
        if strategy == "proportional":
            logger.warning("sample-tasks: no metadata.domain on rows; falling back to uniform sampling")
        sampled = rng.sample(rows, min(total, len(rows)))

    _write_jsonl(output, sampled)
    if labels is not None:
        labels.write_text(
            "".join(f"{str(row.get('task_id'))}\n" for row in sampled), encoding="utf-8"
        )
    logger.info(
        "sample-tasks: drew %s / %s rows (seed=%s) -> %s%s",
        len(sampled), len(rows), seed, output, "; labels -> " + str(labels) if labels is not None else "",
    )
    return 0


# ---------------------------------------------------------------------------
# gen-bundle: instance list + ready AgentENV templates -> training bundle.
# Mirrors tmp/scripts/gen_swegym_pipeline.sh in the vime-forge worktree:
# labels/instance.txt -> (digest + template discovery) -> TSV cache ->
# dataset join -> config triple. Templates must already exist in AgentENV
# (e.g. created via `aenv pull`); discovery is read-only unless --auto-pull.
# ---------------------------------------------------------------------------

def _api_get_json_soft(url: str, api_key: str, *, timeout: int = 30) -> Any:
    """GET returning None on ANY 4xx (some AgentENV builds answer unknown
    aliases with 400 instead of 404); raise on transport/5xx."""
    request = urllib.request.Request(url, headers={"X-API-Key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500:
            return None
        raise PreparationError(f"AgentENV API GET {url} failed: HTTP {exc.code}") from exc
    except OSError as exc:
        raise PreparationError(f"AgentENV API GET {url} failed: {exc}") from exc
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise PreparationError(f"AgentENV API returned invalid JSON for {url}") from exc


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    return re.sub(r"-+", "-", slug).strip("-")


def _load_instance_entries(path: Path, id_extract: str) -> list[str]:
    """One task id per line; '#' comments, blank lines and quotes tolerated.

    ``id_extract=last`` takes the last '/' segment of each entry so upstream
    tmax-style ``<dataset>/<hash>`` lists can be fed in when the final segment
    is the TMax task id.
    """
    ids: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().strip('"').strip("'").strip()
        if not line or line.startswith("#"):
            continue
        if id_extract == "last" and "/" in line:
            line = line.rsplit("/", 1)[1]
        ids.append(line)
    seen: set[str] = set()
    ordered = [i for i in ids if not (i in seen or seen.add(i))]
    if not ordered:
        raise PreparationError(f"no instance entries found in {path}")
    return ordered


def _parse_instance_map(tsv_path: Path, image_template: str | None) -> list[dict[str, Any]]:
    """Discovery-bypass TSV: ``task_id template_id image_digest [image_tag]``."""
    instances: list[dict[str, Any]] = []
    for line_number, raw in enumerate(tsv_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise PreparationError(
                f"{tsv_path}:{line_number}: need >=3 columns (task_id template_id image_digest); got {len(parts)}"
            )
        image_tag = parts[3].strip() if len(parts) > 3 and parts[3].strip() else ""
        if not image_tag:
            if not image_template:
                raise PreparationError(
                    f"{tsv_path}:{line_number}: no image_tag column and no --image-template to derive it"
                )
            image_tag = image_template.format(task_id=parts[0])
        instances.append(
            {
                "task_id": parts[0].strip(),
                "template_id": parts[1].strip(),
                "image_digest": parts[2].strip(),
                "image_tag": image_tag,
            }
        )
    if not instances:
        raise PreparationError(f"no instances found in {tsv_path}")
    return instances


_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_DIGEST_HEADER_RE = re.compile(r"(?im)^docker-content-digest:\s*(\S+)")


def _resolve_digest_via_ssh(repo: str, tag: str, ssh_target: str, remote_base: str) -> tuple[str | None, str | None]:
    """One-shot remote manifest HEAD when the registry is not directly reachable."""
    url = f"{remote_base.rstrip('/')}/v2/{repo}/manifests/{urllib.parse.quote(tag, safe='')}"
    command = ["ssh", "-4", ssh_target, "curl", "-sSI", "-H", f"Accept: {_MANIFEST_ACCEPT}", url]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"ssh digest lookup failed to run: {exc}"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        return None, f"ssh digest lookup exited {result.returncode}: {tail[-1] if tail else '(no output)'}"
    match = _DIGEST_HEADER_RE.search(result.stdout)
    if match is None:
        return None, "ssh digest lookup returned no Docker-Content-Digest header"
    return match.group(1), None


_TEMPLATE_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)


def _catalog_find(api_url: str, api_key: str, digest: str, task_id: str, slug: str) -> str | None:
    """Scan the /v2/templates catalog for an entry matching digest/id/slug."""
    for path in ("/v2/templates", "/templates"):
        payload = _api_get_json_soft(f"{api_url.rstrip('/')}{path}", api_key, timeout=60)
        items = payload if isinstance(payload, list) else None
        if isinstance(payload, dict):
            items = payload.get("templates") or payload.get("items") or payload.get("data")
        if not isinstance(items, list) or not items:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            blob = json.dumps(item)
            if digest in blob or task_id in blob or slug in blob.lower():
                for key in ("templateID", "template_id", "id", "uuid"):
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        return value
    return None


def _aenv_cli_find(aenv_bin: str, task_id: str, slug: str) -> str | None:
    """Parse `aenv templates list` for a UUID on lines mentioning the task."""
    try:
        result = subprocess.run(
            [aenv_bin, "templates", "list"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if task_id in line or slug in line.lower():
            for token in line.split():
                if _TEMPLATE_UUID_RE.fullmatch(token):
                    return token
    return None


def _aenv_pull_cli(aenv_bin: str, image_tag: str, name: str, *, cpu: int, memory_mib: int,
                   timeout: float) -> tuple[str | None, str | None]:
    """Build a template from the registry image via the aenv CLI."""
    try:
        result = subprocess.run(
            [aenv_bin, "pull", image_tag, "--name", name, "--cpu", str(cpu), "--memory", str(memory_mib)],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"aenv pull failed to run: {exc}"
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        tail = output.strip().splitlines()
        return None, f"aenv pull exited {result.returncode}: {tail[-1] if tail else '(no output)'}"
    match = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", output)
    return (match.group(0) if match else None), None


def _load_tmax_rows(source: Path) -> list[dict[str, Any]]:
    """Load the TMax corpus (parquet or JSONL) as raw rows."""
    if source.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise PreparationError("pyarrow is required for parquet input (pip install pyarrow)") from exc
        return pq.read_table(source).to_pylist()
    return _read_jsonl(source)


def _tests_payload(row: dict) -> dict[str, str] | None:
    from pathlib import PurePosixPath
    tests = row.get("tests")
    if tests is None and row.get("test_final_state"):
        tests = {"test_final_state.py": str(row["test_final_state"])}
    if tests is not None:
        if not isinstance(tests, dict) or not tests or any(
            not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts or not isinstance(content, str)
            for name, content in tests.items()
        ):
            raise PreparationError("Oracle payload must contain relative paths and text content")
    return tests


def _inject_probe_tests(sandbox, plan):
    if plan.get("tests"):
        sandbox.commands.run("rm -rf /tests && mkdir -p /tests", timeout=30)
        for name, content in plan["tests"].items():
            sandbox.commands.run("mkdir -p " + shlex.quote(str(Path("/tests") / name).rsplit("/", 1)[0]), timeout=30)
            sandbox.files.write("/tests/" + name, content)


async def _gen_harbor_bundle(args) -> int:
    """Prepare the same bundle layout using immutable OCI images, without E2B."""
    from .sandbox import create_sandbox
    from .template_mapping import TemplateMappingEntry
    from .verifier_spec import VerifierSpec

    rows = _load_tmax_rows(args.source)
    selected = set(_load_instance_entries(args.instances, args.id_extract)) if args.instances else None
    mapped = (
        {r["task_id"]: r for r in _parse_instance_map(args.instance_map, args.image_template)}
        if args.instance_map else {}
    )
    if mapped:
        selected = set(mapped)
    train, environments, specs, qualifications = [], [], [], []
    seen = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if selected is not None and task_id not in selected:
            continue
        if not task_id or task_id in seen:
            raise PreparationError(f"Missing or duplicate task id: {task_id!r}")
        seen.add(task_id)
        entry = mapped.get(task_id, {})
        image = entry.get("image_tag") or row.get("image") or (args.image_template or "").format(task_id=task_id)
        if not image:
            raise PreparationError(f"{task_id}: image or --image-template required")
        repo, ref = _parse_image_ref(image)
        digest = entry.get("image_digest") or (
            ref if ref.startswith("sha256:") else resolve_registry_digest(image, args.registry_base)
        )
        source_image = f"{repo}@{digest}"
        workdir = args.workdir or row.get("workdir") or "/home/user"
        instruction = row.get("instruction") or row.get("description")
        if not isinstance(instruction, str) or not instruction.strip():
            raise PreparationError(f"{task_id}: instruction required")
        budget = args.timeout_sec or row.get("timeout_sec") or _tmax_timeout_for(str(row.get("task_complexity", "")))
        oracle = args.run_tests_cmd or row.get("run_tests_cmd") or "python3 -m pytest /tests/test_final_state.py -q"
        tests = _tests_payload(row)
        template_id = entry.get("template_id")
        if template_id == "-":
            template_id = None
        env = dict(schema_version=1, task_id=task_id, source_image=source_image,
                   source_image_tag=image, image_digest=digest, template_id=template_id,
                   template_alias=None, workdir=workdir)
        spec = dict(schema_version=1, spec_id=f"tbench-spec-{task_id}", task_id=task_id,
                    protocol=PROTOCOL, template_id=template_id, image_digest=digest,
                    workdir=workdir, run_tests_cmd=oracle, timeout_sec=budget,
                    tests_sha256=None, tests=tests)
        TemplateMappingEntry.from_record(env, location=task_id)
        VerifierSpec.from_record(spec, location=task_id)
        if not args.no_probe or args.qualify:
            async with create_sandbox(source_image) as sb:
                await sb.exec("test -d " + shlex.quote(workdir), check=True, timeout=args.probe_command_timeout)
                if tests:
                    await sb.exec("rm -rf /tests && mkdir -p /tests", check=True, timeout=30, idempotent=False)
                    for name, content in tests.items():
                        await sb.write_file("/tests/" + name, content, user="root")
                for path in _oracle_targets(oracle):
                    await sb.exec("test -e " + shlex.quote(path), check=True, timeout=args.probe_command_timeout)
                _, value, _ = await sb.exec(TESTS_HASH_CMD, check=True, timeout=args.probe_command_timeout)
                if not re.fullmatch(r"[0-9a-f]{64}", value.strip()):
                    raise PreparationError(f"{task_id}: invalid tests hash")
                spec["tests_sha256"] = value.strip()
                if args.qualify:
                    code, _, _ = await sb.exec(oracle, timeout=budget, idempotent=False)
                    qualifications.append(
                        dict(
                            task_id=task_id,
                            qualification_status="qualified" if code else "rejected",
                            reason=(
                                "pristine_oracle_failed" if code else "oracle_passes_on_pristine_environment"
                            ),
                        )
                    )
                    if code == 0:
                        continue
        train.append(dict(
            prompt=instruction,
            label=task_id,
            metadata=dict(task_id=task_id, image=source_image, workdir=workdir),
        ))
        environments.append(env)
        specs.append(spec)
    if selected is not None and selected - seen:
        raise PreparationError(f"Selected tasks absent from source: {sorted(selected - seen)}")
    if not train:
        raise PreparationError("No tasks left for bundle")
    paths = _bundle_paths(args.bundle)
    for key, values in (
        ("train_rows", train),
        ("templates", environments),
        ("verifier_specs", specs),
        ("qualification", qualifications),
    ):
        _write_jsonl(paths[key], values)
    _write_json(paths["manifest"], dict(schema_version=1, protocol=PROTOCOL, sandbox_backend="harbor",
                                      source=str(args.source), source_sha256=_sha256_file(args.source),
                                      task_count=len(train), probe_disabled=args.no_probe, generated_at=_utc_now()))
    logger.info("Prepared %s Harbor tasks in %s", len(train), args.bundle)
    return 0


def gen_bundle(args: argparse.Namespace) -> int:
    """instance.txt + ready templates -> rollout/templates/verifier/manifest."""
    if os.environ.get("VIME_AGENT_SANDBOX_BACKEND", "agentenv") == "harbor":
        return asyncio.run(_gen_harbor_bundle(args))
    source = args.source
    api_url = (
        args.api_url
        or os.environ.get("AENV_API_URL", "").strip()
        or os.environ.get("E2B_API_URL", "").strip()
        or "http://127.0.0.1:8000"
    ).rstrip("/")
    api_key_file = Path(
        args.api_key_file
        or os.environ.get("AENV_API_KEY_FILE", "").strip()
        or os.environ.get("VIME_AGENTENV_API_KEY_FILE", "").strip()
        or "/var/lib/aenv/secrets/api-key"
    )
    aenv_bin = args.aenv_bin or shutil.which("aenv") or ""
    registry_ssh = (args.registry_ssh or "").strip()
    registry_remote_base = (args.registry_remote_base or "").strip() or "http://127.0.0.1:5000"
    bundle = args.bundle
    paths = _bundle_paths(bundle)

    # ── instances: TSV passthrough (no discovery) or list -> discovery ──────
    if args.instance_map:
        instances = _parse_instance_map(Path(args.instance_map), args.image_template)
        logger.info(
            "gen-bundle: %s instances from TSV %s (discovery skipped)", len(instances), args.instance_map
        )
    else:
        if not args.instances:
            raise PreparationError("gen-bundle needs --instances <list file> or --instance-map <tsv>")
        if not args.image_template:
            raise PreparationError("gen-bundle needs --image-template (e.g. myreg/tmax/{task_id}:arm64) in list mode")
        task_ids = _load_instance_entries(Path(args.instances), args.id_extract)

    # ── dataset join: every instance must exist in the source ───────────────
    # Accepts both the raw TMax corpus (task_id/description/test_final_state)
    # and any source JSONL produced by from-tmax / from-tasks (task_id/instruction).
    rows = _load_tmax_rows(source)
    by_task = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        instruction = row.get("description") or row.get("instruction")
        if task_id and isinstance(instruction, str) and instruction.strip():
            by_task.setdefault(task_id, row)
    if args.instance_map:
        wanted = [inst["task_id"] for inst in instances]
    else:
        wanted = task_ids
    missing_dataset = [task_id for task_id in wanted if task_id not in by_task]
    if missing_dataset:
        raise PreparationError(
            f"{len(missing_dataset)} instance ids not found in {source}: {missing_dataset[:5]}..."
        )
    logger.info("gen-bundle: matched %s/%s instances in %s", len(wanted), len(wanted), source)

    # ── discovery: digest + template id (skipped in TSV mode) ───────────────
    if not args.instance_map:
        try:
            api_key = api_key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise PreparationError(f"cannot read API key file {api_key_file}: {exc}") from exc
        if not api_key:
            raise PreparationError(f"API key file is empty: {api_key_file}")
        try:
            with urllib.request.urlopen(f"{api_url}/health", timeout=10) as response:
                logger.info("[ok] AgentENV reachable at %s (health HTTP %s)", api_url, response.status)
        except OSError as exc:
            raise PreparationError(f"AgentENV unreachable at {api_url}: {exc}") from exc

        instances, missing = [], []
        for task_id in task_ids:
            image_tag = args.image_template.format(task_id=task_id)
            repo, tag = _parse_image_ref(image_tag)
            try:
                digest = resolve_registry_digest(image_tag, args.registry_base)
            except Exception as exc:
                digest, digest_err = None, str(exc)
                if registry_ssh:
                    digest, ssh_err = _resolve_digest_via_ssh(repo, tag, registry_ssh, registry_remote_base)
                    if digest is None:
                        digest_err = f"{digest_err}; ssh fallback: {ssh_err}"
                if digest is None:
                    missing.append((task_id, f"digest lookup failed: {digest_err}"))
                    logger.warning("[miss] %s: %s", task_id, digest_err)
                    continue

            slug = _slugify(task_id)
            candidates = [task_id]
            if args.alias_format:
                candidates.append(args.alias_format.format(task_id=task_id, slug=slug, arch=args.arch))
            template_id, matched = None, None
            for alias in candidates:
                encoded = urllib.parse.quote(alias, safe="")
                payload = _api_get_json_soft(f"{api_url}/templates/aliases/{encoded}", api_key)
                value = payload.get("templateID") if isinstance(payload, dict) else None
                if isinstance(value, str) and value:
                    template_id, matched = value, f"alias:{alias}"
                    break
            if not template_id:
                template_id = _catalog_find(api_url, api_key, digest, task_id, slug)
                if template_id:
                    matched = "catalog-scan"
            if not template_id and aenv_bin:
                template_id = _aenv_cli_find(aenv_bin, task_id, slug)
                if template_id:
                    matched = "aenv-cli"
            if not template_id and args.auto_pull and aenv_bin:
                logger.info("[pull] %s: aenv pull %s --name %s", task_id, image_tag, task_id)
                pulled_id, pull_err = _aenv_pull_cli(
                    aenv_bin, image_tag, task_id,
                    cpu=args.pull_cpu, memory_mib=args.pull_memory_mib, timeout=args.pull_timeout,
                )
                if pulled_id:
                    template_id, matched = pulled_id, "aenv-pull"
                elif pull_err:
                    missing.append((task_id, pull_err))
                    logger.warning("[miss] %s: %s", task_id, pull_err)
                    continue
            if not template_id:
                missing.append((task_id, "template not found on server"))
                logger.warning(
                    "[miss] %s: no template (tried alias/catalog/aenv-cli%s); pull it first: aenv pull %s --name %s",
                    task_id, "/aenv-pull" if args.auto_pull else "", image_tag, task_id,
                )
                continue
            logger.info("[ok] %s  via=%s  template=%s  digest=%s…", task_id, matched, template_id, digest[:19])
            instances.append({"task_id": task_id, "template_id": template_id,
                              "image_digest": digest, "image_tag": image_tag})
        if not instances:
            raise PreparationError(f"no instances resolved ({len(missing)} missing)")
        output_tsv = args.output_tsv or paths["instances_tsv"]
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        with output_tsv.open("w", encoding="utf-8") as fh:
            fh.write("# task_id\ttemplate_id\timage_digest\timage_tag\n")
            for inst in instances:
                fh.write("\t".join([inst["task_id"], inst["template_id"],
                                    inst["image_digest"], inst["image_tag"]]) + "\n")
        logger.info(
            "gen-bundle: wrote %s row(s) to %s (rerun via --instance-map to skip discovery)",
            len(instances), output_tsv,
        )
        if missing:
            logger.warning("%s instance(s) missing:", len(missing))
            for task_id, reason in missing:
                logger.warning("  %s: %s", task_id, reason)

    # ── probe: /tests hash + oracle presence (+ optional pristine qualify) ──
    api_key = None
    if not args.no_probe or args.qualify:
        try:
            api_key = api_key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise PreparationError(f"cannot read API key file {api_key_file}: {exc}") from exc
        _append_sdk_path(args.sdk_path)

    plans = {}
    for inst in instances:
        row = by_task[inst["task_id"]]
        row_meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        complexity = str(row.get("task_complexity") or row_meta.get("task_complexity") or "")
        timeout = args.timeout_sec or row.get("timeout_sec") or _tmax_timeout_for(complexity)
        if not isinstance(timeout, int) or timeout <= 0:
            raise PreparationError(f"{inst['task_id']}: timeout_sec must be a positive integer, got {timeout!r}")
        plans[inst["task_id"]] = {
            "task_id": inst["task_id"],
            "tests": _tests_payload(row),
            "instruction": str(row.get("description") or row.get("instruction") or ""),
            "workdir": args.workdir or str(row.get("workdir") or "/home/user"),
            "run_tests_cmd": str(
                args.run_tests_cmd or row.get("run_tests_cmd")
                or "python3 -m pytest /tests/test_final_state.py -q"
            ),
            "timeout_sec": timeout,
            "metadata": {
                "domain": str(row.get("domain") or row_meta.get("domain") or ""),
                "task_complexity": complexity,
                "skill_type": str(row.get("skill_type") or row_meta.get("skill_type") or ""),
                "language": str(row.get("language") or row_meta.get("language") or ""),
                "scenario": str(row.get("scenario") or row_meta.get("scenario") or ""),
            },
        }

    validations: dict[str, dict[str, Any]] = {}
    if not args.no_probe:
        def probe_one(inst: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
            try:
                validation = _probe_template(
                    plans[inst["task_id"]], inst["template_id"],
                    api_url=api_url, sandbox_url=args.sandbox_url or api_url, api_key=api_key,
                    sandbox_timeout=args.probe_sandbox_timeout, command_timeout=args.probe_command_timeout,
                )
                return inst["task_id"], validation, None
            except Exception as exc:
                return inst["task_id"], None, f"{type(exc).__name__}: {str(exc)[:200]}"

        failures = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.probe_concurrency)) as pool:
            for task_id, validation, error in pool.map(probe_one, instances):
                if error is None:
                    validations[task_id] = validation
                    logger.info("[probe] %s: ready tests_sha256=%s…", task_id, validation["tests_sha256"][:12])
                else:
                    failures += 1
                    logger.warning("[probe-miss] %s: %s", task_id, error)
        if failures:
            instances = [inst for inst in instances if inst["task_id"] in validations]
            if not instances:
                raise PreparationError("all probes failed")

    qualifications: dict[str, dict[str, Any]] = {}
    if args.qualify:
        def qualify_one(inst: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            return inst["task_id"], _qualify_one(
                plans[inst["task_id"]], {"template_id": inst["template_id"]},
                api_url=api_url, sandbox_url=args.sandbox_url or api_url, api_key=api_key,
                sandbox_timeout=args.probe_sandbox_timeout,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.qualify_concurrency)) as pool:
            for task_id, outcome in pool.map(qualify_one, instances):
                qualifications[task_id] = outcome
                logger.info(
                    "[qualify] %s: %s (%s)", task_id, outcome["qualification_status"], outcome.get("reason")
                )
        rejected = [t for t, o in qualifications.items() if o["qualification_status"] != "qualified"]
        if rejected:
            logger.warning("gen-bundle: dropping %s non-qualified tasks: %s...", len(rejected), rejected[:5])
            instances = [inst for inst in instances if inst["task_id"] not in set(rejected)]
            if not instances:
                raise PreparationError("no tasks left after qualification")
        _write_jsonl(paths["qualification"], qualifications.values())

    # ── bundle export ───────────────────────────────────────────────────────
    train_rows, template_rows, spec_rows = [], [], []
    for inst in instances:
        task_id = inst["task_id"]
        plan = plans[task_id]
        digest = inst["image_digest"]
        repo, _ = _parse_image_ref(inst["image_tag"])
        source_image = f"{repo}@{digest}"
        validation = validations.get(task_id, {})
        train_rows.append(
            {"prompt": plan["instruction"], "label": task_id,
             "metadata": {"task_id": task_id, "image": source_image, "workdir": plan["workdir"],
                          **plan["metadata"]}}
        )
        template_rows.append(
            {"schema_version": SCHEMA_VERSION, "task_id": task_id,
             "source_image": source_image, "source_image_tag": inst["image_tag"],
             "image_digest": digest, "template_id": inst["template_id"],
             "template_alias": _template_alias(task_id, digest), "workdir": plan["workdir"]}
        )
        spec_rows.append(
            {"schema_version": SCHEMA_VERSION, "spec_id": f"tbench-spec-{task_id}", "task_id": task_id,
             "protocol": PROTOCOL, "template_id": inst["template_id"],
             "image_digest": digest, "workdir": plan["workdir"],
             "run_tests_cmd": plan["run_tests_cmd"],
             "tests_sha256": validation.get("tests_sha256"), "timeout_sec": plan["timeout_sec"], "tests": plan["tests"]}
        )

    _write_jsonl(paths["train_rows"], train_rows)
    _write_jsonl(paths["templates"], template_rows)
    _write_jsonl(paths["verifier_specs"], spec_rows)
    _write_json(
        paths["manifest"],
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "generator": "gen-bundle",
            "source": str(source),
            "source_sha256": _sha256_file(source),
            "instances": str(args.instance_map or args.instances),
            "generated_at": _utc_now(),
            "task_count": {"total": len(train_rows), "probed": len(validations),
                           "qualified": len(qualifications)},
            "probe_disabled": bool(args.no_probe),
            "artifacts": {
                "train_rows": str(paths["train_rows"]),
                "templates": str(paths["templates"]),
                "verifier_specs": str(paths["verifier_specs"]),
            },
        },
    )
    hash_note = "with /tests hash" if validations else "WITHOUT /tests hash (tamper check disabled)"
    logger.info(
        "gen-bundle: %s tasks exported to %s (%s);\n"
        "  point VIME_TBENCH_PREPARED_BUNDLE=%s on the training machine",
        len(train_rows), bundle, hash_note, bundle,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    from_tasks_cmd = commands.add_parser("from-tasks", help="Terminal-Bench task tree -> source JSONL")
    from_tasks_cmd.add_argument("--tasks-root", type=Path, required=True)
    from_tasks_cmd.add_argument("--output", type=Path, required=True)
    from_tasks_cmd.add_argument(
        "--image-template", default="ghcr.io/laude-institute/tb-{task_id}:latest",
        help="Image reference template used when task.yaml has no image field",
    )

    from_tmax_cmd = commands.add_parser(
        "from-tmax", help="allenai/TMax-15K -> source JSONL (no image building)"
    )
    from_tmax_cmd.add_argument("--source", type=Path, required=True, help="TMax-15K parquet or JSONL")
    from_tmax_cmd.add_argument("--output", type=Path, required=True)
    from_tmax_cmd.add_argument(
        "--image-template", required=True,
        help="OCI reference template of the ready task images, e.g. registry/tmax/{task_id}:arm64",
    )
    from_tmax_cmd.add_argument(
        "--keep-files-tasks", action="store_true",
        help="Keep tasks whose %files reference unavailable cluster fixtures (no ready template can exist)",
    )
    from_tmax_cmd.add_argument(
        "--keep-arch-pinned", action="store_true",
        help="Keep tasks pinning x86 binaries in their container def",
    )
    from_tmax_cmd.add_argument("--limit", type=int)

    gen_cmd = commands.add_parser(
        "gen-bundle",
        help="instance list + ready AgentENV templates -> training bundle "
             "(mirrors tmp/scripts/gen_swegym_pipeline.sh; templates must already exist)",
    )
    gen_cmd.add_argument(
        "--source", type=Path, required=True,
        help="TMax-15K parquet/JSONL or a source JSONL from from-tmax/from-tasks",
    )
    gen_cmd.add_argument("--bundle", type=Path, required=True)
    gen_input = gen_cmd.add_mutually_exclusive_group()
    gen_input.add_argument(
        "--instances", type=Path,
        help="instance list: one task id per line ('#' comments/blank lines tolerated; tmax instance.txt style)",
    )
    gen_input.add_argument(
        "--instance-map", type=Path,
        help="Discovery-bypass TSV: 'task_id template_id image_digest [image_tag]' per line",
    )
    gen_cmd.add_argument(
        "--id-extract", choices=("whole", "last"), default="whole",
        help="last: take the final '/' segment of each entry (for <dataset>/<task_id> lists)",
    )
    gen_cmd.add_argument("--image-template", help="e.g. myreg/tmax/{task_id}:arm64 (required in list mode)")
    gen_cmd.add_argument("--arch", default="arm64", help="Placeholder for --alias-format")
    gen_cmd.add_argument("--alias-format", help="Extra alias probe pattern ({task_id}/{slug}/{arch})")
    gen_cmd.add_argument("--registry-base", help="Registry v2 base URL; defaults to the image ref's host")
    gen_cmd.add_argument("--registry-ssh", help="user@host for one-shot remote digest lookups")
    gen_cmd.add_argument("--registry-remote-base", default="http://127.0.0.1:5000")
    gen_cmd.add_argument("--api-url", help="Defaults to AENV_API_URL / E2B_API_URL / http://127.0.0.1:8000")
    gen_cmd.add_argument("--sandbox-url", help="Defaults to the API URL")
    gen_cmd.add_argument(
        "--api-key-file", type=Path,
        help="Defaults to AENV_API_KEY_FILE / VIME_AGENTENV_API_KEY_FILE / /var/lib/aenv/secrets/api-key",
    )
    gen_cmd.add_argument("--sdk-path", help="Optional path to a compatible e2b SDK site-packages")
    gen_cmd.add_argument("--aenv-bin", help="aenv CLI path; defaults to PATH lookup")
    gen_cmd.add_argument(
        "--auto-pull", action="store_true",
        help="aenv pull missing templates (default off: read-only discovery). "
             "Sandboxes inherit the template's cpu/memory, so set --pull-cpu/--pull-memory-mib deliberately",
    )
    gen_cmd.add_argument("--pull-cpu", type=int, default=16)
    gen_cmd.add_argument("--pull-memory-mib", type=int, default=65536)
    gen_cmd.add_argument("--pull-timeout", type=float, default=1800)
    gen_cmd.add_argument(
        "--no-probe", action="store_true",
        help="Skip sandbox probing; verifier specs get tests_sha256=null (tamper check disabled)",
    )
    gen_cmd.add_argument("--probe-concurrency", type=int, default=2)
    gen_cmd.add_argument("--probe-sandbox-timeout", type=int, default=600)
    gen_cmd.add_argument("--probe-command-timeout", type=int, default=120)
    gen_cmd.add_argument(
        "--qualify", action="store_true",
        help="Also run the pristine-oracle qualification (must FAIL on the untouched environment)",
    )
    gen_cmd.add_argument("--qualify-concurrency", type=int, default=2)
    gen_cmd.add_argument("--workdir")
    gen_cmd.add_argument("--run-tests-cmd")
    gen_cmd.add_argument("--timeout-sec", type=int, help="Override per-task timeout (default: by complexity)")
    gen_cmd.add_argument("--output-tsv", type=Path, help="Discovery cache (default <bundle>/state/instances.tsv)")

    filter_cmd = commands.add_parser(
        "filter-tasks", help="Subset a source/bundle JSONL by task ids or metadata equality"
    )
    filter_cmd.add_argument("--source", type=Path, required=True)
    filter_cmd.add_argument("--output", type=Path, required=True)
    filter_cmd.add_argument("--include-task-ids", help="'a,b,c' or a newline-separated id file")
    filter_cmd.add_argument("--exclude-task-ids", help="'a,b,c' or a newline-separated id file")
    filter_cmd.add_argument(
        "--where", action="append",
        help="Dotted-path equality, e.g. --where metadata.domain=data_science; repeatable (ANDed)",
    )
    filter_cmd.add_argument("--limit", type=int)
    filter_cmd.add_argument(
        "--labels", type=Path, default=None,
        help="Optional task-id list file (one per line) for gen-bundle --instances",
    )

    sample_cmd = commands.add_parser(
        "sample-tasks", help="Seeded random subset (proportional per domain) as JSONL + labels file"
    )
    sample_cmd.add_argument("--source", type=Path, required=True)
    sample_cmd.add_argument("--output", type=Path, required=True, help="Sampled subset JSONL")
    sample_cmd.add_argument(
        "--labels", type=Path, default=None,
        help="Optional task-id list file (one per line) for gen-bundle --instances",
    )
    sample_cmd.add_argument("--total", type=int, required=True, help="Number of tasks to draw")
    sample_cmd.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    sample_cmd.add_argument(
        "--strategy", choices=("proportional", "uniform"), default="proportional",
        help="proportional: allocate across metadata.domain by count (upstream tmax behaviour); "
             "falls back to uniform when no domain metadata exists",
    )
    sample_cmd.add_argument(
        "--where", action="append",
        help="Optional pre-filter before sampling, e.g. --where metadata.task_complexity=short task (...)",
    )

    args = parser.parse_args()

    if args.command == "from-tasks":
        return from_tasks(args.tasks_root, args.output, image_template=args.image_template)
    if args.command == "from-tmax":
        return from_tmax(
            args.source, args.output, image_template=args.image_template,
            keep_files_tasks=args.keep_files_tasks, keep_arch_pinned=args.keep_arch_pinned,
            limit=args.limit,
        )
    if args.command == "gen-bundle":
        return gen_bundle(args)
    if args.command == "filter-tasks":
        return filter_tasks(
            args.source, args.output, include_ids=args.include_task_ids,
            exclude_ids=args.exclude_task_ids, where=args.where, limit=args.limit,
            labels=args.labels,
        )
    if args.command == "sample-tasks":
        return sample_tasks(
            args.source, args.output, args.labels, total=args.total, seed=args.seed,
            strategy=args.strategy, where=args.where,
        )
    raise AssertionError("unreachable")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
