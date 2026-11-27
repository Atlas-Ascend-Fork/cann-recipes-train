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

"""Shared AgenticRL configuration; standard library only, with no runtime activation.

The recipe launcher resolves aliases and provider settings once. Workers receive
canonical task variables and a selected-provider snapshot, never API key contents.
Standalone generators and sandbox tools may continue using the legacy variables.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shlex
from collections.abc import Mapping

logger = logging.getLogger(__name__)


TASK_PREFIXES = {"swegym": "SWE", "terminalbench": "TB"}
SNAPSHOT_KEY = "VIME_INTERNAL_SANDBOX_CONFIG"
RL_DEFAULTS = {
    "AGENT": "codex",
    "ROUTER_POLICY": "cache_aware", "ENABLE_PREFIX_CACHING": "1",
    "TRIAL_CONCURRENCY": "8", "BOOT_CONCURRENCY": "8", "BOOT_RETRIES": "3",
    "AGENT_TIME_BUDGET_SEC": "3600", "EVAL_TIMEOUT_SEC": "3600",
    "ROLLOUT_GUARD_SEC": "8400", "VLLM_MAX_NUM_SEQS": "64",
    "SAVE_INTERVAL": "6", "DUMP_MODE": "rollout",
    "KL_COV": "off", "KL_COV_RATIO": "2e-3", "KL_COV_COEF": "1.0",
}
# Parameters shared with single-turn RL belong to the model script and the launch
# command, so the task resolver no longer carries them. Each retired field names
# its replacement: an existing local file fails loudly instead of being ignored.
RETIRED_FIELDS = {
    "TRAIN_TP": "set TRAIN_TP in the local file",
    "TRAIN_DP": "set TRAIN_DP in the local file",
    "ROLLOUT_DEVICES": "set ROLLOUT_GPUS in the local file",
    "ROLLOUT_TP": "set VLLM_RANKS_PER_ENGINE in the local file",
    "TITAN_CONFIG": "set TITAN_CONFIG in the local file",
    "MAX_TOKENS_PER_GPU": "set MAX_TOKENS_PER_GPU in the local file",
    "ROLLOUT_BATCH_SIZE": "pass --rollout-batch-size after --",
    "N_SAMPLES_PER_PROMPT": "pass --n-samples-per-prompt after --",
    "NUM_EPOCHS": "pass --num-rollout after --",
    "ROLLOUT_TEMPERATURE": "pass --rollout-temperature after --",
    "ROLLOUT_TOP_K": "pass --rollout-top-k after --",
    "LEARNING_RATE": "pass the model recipe's learning-rate argument after --",
}
BUNDLE_FIELDS = {
    "prompt": ("PROMPT_DATA", "rollout/train.no_gold.jsonl"),
    "environment": ("ENVIRONMENT_MAP", "environment/agentenv_templates.jsonl"),
    "verifier": ("VERIFIER_SPEC_MAP", "verifier/verifier_specs.jsonl"),
    "manifest": ("PREPARATION_MANIFEST", "preparation_manifest.json"),
}
COMMON_FIELDS = {
    "lifetime_sec": "VIME_AGENT_SANDBOX_LIFETIME_SEC",
    "boot_concurrency": "VIME_SANDBOX_BOOT_CONCURRENCY",
    "boot_retries": "VIME_SANDBOX_BOOT_RETRIES",
}
PROVIDER_FIELDS = {
    "agentenv": {
        "api_url": "E2B_API_URL", "sandbox_url": "E2B_SANDBOX_URL",
        "api_key_file": "VIME_AGENTENV_API_KEY_FILE", "sdk_path": "VIME_AGENT_E2B_SDK_PATH",
        "selector_mode": "VIME_AGENT_E2B_SELECTOR_MODE",
        "image_metadata_key": "VIME_AGENT_SANDBOX_IMAGE_METADATA_KEY",
        "size": "VIME_AGENT_E2B_SANDBOX_SIZE", "rpc_retries": "VIME_AGENT_SANDBOX_RPC_RETRIES",
    },
    "harbor": {
        "environment": "VIME_HARBOR_ENVIRONMENT", "hosts": "VIME_HARBOR_HOSTS",
        "cpus": "VIME_HARBOR_CPUS", "memory_mb": "VIME_HARBOR_MEMORY_MB",
        "registry": "VIME_HARBOR_IMAGE_REGISTRY", "startup_timeout": "VIME_HARBOR_STARTUP_TIMEOUT",
        "acquire_timeout": "VIME_HARBOR_ACQUIRE_TIMEOUT",
        "marker_cancel_grace": "VIME_HARBOR_MARKER_CANCEL_GRACE_SEC",
        "prepare_concurrency": "VIME_HARBOR_PREPARE_CONCURRENCY",
        "prepare_acquire_timeout": "VIME_HARBOR_PREPARE_ACQUIRE_TIMEOUT_SEC",
        "prepare_queue_limit": "VIME_HARBOR_PREPARE_QUEUE_LIMIT",
    },
}
SANDBOX_ALIASES = {
    "VIME_AGENT_SANDBOX_LIFETIME_SEC": "SWE_SANDBOX_LIFETIME_SEC",
    "VIME_AGENT_SANDBOX_IMAGE_METADATA_KEY": "SWE_SANDBOX_IMAGE_METADATA_KEY",
    "VIME_AGENT_SANDBOX_RPC_RETRIES": "SWE_RPC_RETRIES",
    "VIME_AGENT_E2B_SANDBOX_SIZE": "SWE_E2B_SANDBOX_SIZE",
}
AGENT_FIELDS = (
    "VIME_AGENT_MAX_CONTEXT_TOKENS", "VIME_AGENT_MAX_TURNS_PER_SESSION",
    "VIME_AGENT_TASK_CONDA_SH", "VIME_AGENT_TASK_CONDA_PREFIX",
    "VIME_AGENT_NODE_TARBALL", "VIME_AGENT_CODEX_TARBALL", "VIME_AGENT_CODEX_VERSION",
    "VIME_AGENT_CODEX_MODEL_PROFILE", "VIME_AGENT_CODEX_EXTRA_ARGS", "VIME_AGENT_CODEX_EXTRA_ENVS",
    "VIME_AGENT_CODEX_CONTEXT_RESERVE_TOKENS", "VIME_AGENT_CODEX_AUTO_COMPACT_TOKENS",
    "VIME_AGENT_CC_TARBALL", "VIME_AGENT_CC_EXTRA_ARGS", "VIME_AGENT_CC_EXTRA_ENVS",
    "VIME_AGENT_CC_TOOLS", "VIME_AGENT_CC_MAX_OUTPUT_TOKENS",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
    "VIME_AGENT_ARTIFACT_BASE_URL", "VIME_AGENT_NPM_ONLINE", "VIME_AGENT_NPM_INSTALL_TIMEOUT_SEC",
    "VIME_AGENT_WORKSPACE_PREP_TIMEOUT_SEC", "VIME_AGENT_MARKER_TIMEOUT_SEC",
    "VIME_AGENT_MARKER_TIMEOUT_RETRIES", "VIME_FORK_MERGE_MAX_RESPONSE_TOKENS",
    "VIME_AGENT_DEBUG_TRACE_DIR", "VIME_AGENT_TRACE_WRITER",
)



def _environment(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _alias(task: str, name: str) -> str:
    prefix = TASK_PREFIXES[task]
    if name == "PREPARED_BUNDLE":
        return "VIME_SWEGYM_PREPARED_BUNDLE" if task == "swegym" else "VIME_TBENCH_PREPARED_BUNDLE"
    if name == "PROMPT":
        return "SWE_CC_PROMPT" if task == "swegym" else "TB_PROMPT"
    suffix = {"ROLLOUT_DEVICES": "ROLLOUT_GPUS", "ENVIRONMENT_MAP": "AGENTENV_TEMPLATE_MAP"}.get(name, name)
    return f"{prefix}_{suffix}"


def _legacy_aliases(task: str) -> dict[str, str]:
    fields = (
        *RL_DEFAULTS, *RETIRED_FIELDS, "PREPARED_BUNDLE", "PROMPT", "TRAIN_PROTOCOL", "EVAL_PROTOCOL",
        *(item[0] for item in BUNDLE_FIELDS.values()),
    )
    return {_alias(task, name): f"VIME_RL_{name}" for name in fields}


def _read_alias(env: Mapping[str, str], key: str, alias: str, default: str | None = None) -> str | None:
    if key in env and alias in env and env[key] != env[alias]:
        raise ValueError(f"Conflicting configuration: {key} and {alias}; keep one value")
    return env[key] if key in env else env.get(alias, default)


def rl_value(task: str, name: str, default: str | None = None, env: Mapping[str, str] | None = None) -> str | None:
    """Read a canonical field or its task-specific alias; preserve explicit empty values."""
    source = _environment(env)
    if name in ("BOOT_CONCURRENCY", "BOOT_RETRIES"):
        value = effective_sandbox_env(source).get(f"VIME_SANDBOX_{name}")
        if value is not None:
            return value
    return _read_alias(source, f"VIME_RL_{name}", _alias(task, name), default)


def bundle_path(task: str, kind: str, env: Mapping[str, str] | None = None) -> str | None:
    """Resolve a bundle sidecar, including legacy standalone per-file configuration."""
    source = _environment(env)
    suffix, relative = BUNDLE_FIELDS[kind]
    explicit = rl_value(task, suffix, env=source)
    if explicit is not None:
        return explicit
    bundle = rl_value(task, "PREPARED_BUNDLE", env=source)
    return str(Path(bundle) / relative) if bundle else None


def _backend(env: Mapping[str, str]) -> str:
    backend = env.get("VIME_AGENT_SANDBOX_BACKEND", "agentenv").strip().lower()
    if backend not in PROVIDER_FIELDS:
        raise ValueError("VIME_AGENT_SANDBOX_BACKEND must be agentenv or harbor")
    return backend


def _snapshot_fields(backend: str) -> set[str]:
    return {"VIME_AGENT_SANDBOX_BACKEND", *COMMON_FIELDS.values(), *PROVIDER_FIELDS[backend].values()}


def resolve_sandbox_env(
    env: Mapping[str, str] | None = None, defaults: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Select one provider without loading a credential or a provider SDK.

    Explicit deployment variables override JSON values. A launcher-created
    snapshot is authoritative in workers, including fields absent at resolution.
    """
    source = _environment(env)
    backend = _backend(source)
    if source.get(SNAPSHOT_KEY):
        try:
            snapshot = json.loads(source[SNAPSHOT_KEY])
            values = snapshot["environment"]
            if snapshot["backend"] != backend or not isinstance(values, dict):
                raise ValueError
            if not set(values) <= _snapshot_fields(backend) or not all(isinstance(v, str) for v in values.values()):
                raise ValueError
            if values.get("VIME_AGENT_SANDBOX_BACKEND") != backend:
                raise ValueError
        except (ValueError, TypeError, KeyError):
            raise ValueError("Invalid selected sandbox configuration snapshot") from None
        return {**values, SNAPSHOT_KEY: source[SNAPSHOT_KEY]}

    result = {k: v for k, v in (defaults or {}).items() if k in _snapshot_fields(backend)}
    configured = source.get("VIME_AGENT_SANDBOX_CONFIG", "")
    if configured:
        try:
            data = json.loads(Path(configured).expanduser().read_text())
        except (OSError, ValueError):
            raise ValueError("VIME_AGENT_SANDBOX_CONFIG must name a readable JSON configuration file") from None
        if not isinstance(data, dict) or not isinstance(data.get(backend), dict):
            raise ValueError(f"VIME_AGENT_SANDBOX_CONFIG requires a {backend} object")
        common = data.get("common", {})
        if not isinstance(common, dict):
            raise ValueError("VIME_AGENT_SANDBOX_CONFIG common must be an object")
        sections = ((common, COMMON_FIELDS, "common"), (data[backend], PROVIDER_FIELDS[backend], backend))
        for section, fields, label in sections:
            unknown = set(section) - set(fields)
            if unknown:
                # Do not echo values, including a mistakenly supplied inline key.
                raise ValueError(f"Unsupported fields in VIME_AGENT_SANDBOX_CONFIG {label}")
            for field, value in section.items():
                if field == "hosts":
                    if not isinstance(value, list):
                        raise ValueError("VIME_AGENT_SANDBOX_CONFIG harbor.hosts must be a list")
                    result[fields[field]] = json.dumps(value, separators=(",", ":"))
                elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    result[fields[field]] = str(value)
                else:
                    raise ValueError(f"Invalid value type in VIME_AGENT_SANDBOX_CONFIG {label}.{field}")
    for key in _snapshot_fields(backend) - {"VIME_AGENT_SANDBOX_BACKEND"}:
        alias = SANDBOX_ALIASES.get(key)
        value = _read_alias(source, key, alias) if alias else source.get(key)
        if value is not None:
            result[key] = value
    result["VIME_AGENT_SANDBOX_BACKEND"] = backend
    result[SNAPSHOT_KEY] = json.dumps(
        {"backend": backend, "environment": result}, separators=(",", ":"), sort_keys=True,
    )
    return result


def effective_sandbox_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Expose the existing adapter keys while preventing live overrides of a snapshot."""
    source = _environment(env)
    resolved = resolve_sandbox_env(source)
    hidden = {SNAPSHOT_KEY, *COMMON_FIELDS.values(), *SANDBOX_ALIASES.values()}
    for fields in PROVIDER_FIELDS.values():
        hidden.update(fields.values())
    return {**{k: v for k, v in source.items() if k not in hidden}, **resolved}


def apply_agentenv_config() -> None:
    """Translate the selected profile for the SDK, without opening api_key_file."""
    if _backend(os.environ) != "agentenv":
        return
    effective = effective_sandbox_env()
    for key in PROVIDER_FIELDS["agentenv"].values():
        if key in effective:
            os.environ[key] = effective[key]
        else:
            os.environ.pop(key, None)


def sandbox_backend(env: Mapping[str, str] | None = None) -> str:
    """The selected provider name, read from the effective sandbox configuration."""
    return _backend(effective_sandbox_env(env))


def validate_sandbox_config() -> None:
    """Fail at import time rather than on the first sandbox of a run.

    The provider module is imported lazily: it imports this one, and a task only
    needs the provider it selected.
    """
    if sandbox_backend() == "harbor":
        from .harbor_sandbox import HarborConfig

        HarborConfig.from_env()


def resolve_launch_config(
    task: str, repo_root: str | Path, env: Mapping[str, str] | None = None, *, warn: bool = False,
) -> dict[str, str]:
    """Resolve the two Qwen3.5 recipes; activation and run creation stay in Shell."""
    source = dict(_environment(env))
    # Canonical and task-prefixed names for a retired field are both rejected;
    # the bare shell variable named in the message is the supported way.
    for field, replacement in RETIRED_FIELDS.items():
        for name in (f"VIME_RL_{field}", _alias(task, field)):
            if source.get(name):
                raise ValueError(f"{name} is no longer supported; {replacement}")
    root = Path(repo_root)
    # Resolve the external sandbox file only once, below; BOOT values here
    # are legacy fallbacks until the selected provider snapshot is available.
    result = {f"VIME_RL_{key}": _read_alias(source, f"VIME_RL_{key}", _alias(task, key), value)
              for key, value in RL_DEFAULTS.items()}
    # Historically :- treated empty values as unset.
    for key, value in RL_DEFAULTS.items():
        if result[f"VIME_RL_{key}"] == "":
            result[f"VIME_RL_{key}"] = value
    result["VIME_RL_TASK"] = task
    result["VIME_RL_TRAIN_PROTOCOL"] = rl_value(
        task, "TRAIN_PROTOCOL", "swebench" if task == "swegym" else "terminalbench", source,
    )
    for field in ("PROMPT", "EVAL_PROTOCOL"):
        value = rl_value(task, field, env=source)
        if value is not None:
            result[f"VIME_RL_{field}"] = value
    bundle_default = root / "prepared" / ("swegym-293" if task == "swegym" else "terminalbench")
    bundle = rl_value(task, "PREPARED_BUNDLE", str(bundle_default), source) or str(bundle_default)
    result["VIME_RL_PREPARED_BUNDLE"] = bundle
    for suffix, relative in BUNDLE_FIELDS.values():
        path = str(Path(bundle) / relative)
        explicit = rl_value(task, suffix, env=source)
        if explicit is not None and explicit != path:
            raise ValueError(
                f"VIME_RL_{suffix} is derived from VIME_RL_PREPARED_BUNDLE; remove the conflicting file override"
            )
        result[f"VIME_RL_{suffix}"] = path
    result["PROMPT_DATA"] = result["VIME_RL_PROMPT_DATA"]

    defaults = {
        "VIME_AGENT_MAX_TURNS_PER_SESSION": "72", "VIME_AGENT_MAX_CONTEXT_TOKENS": "32768",
        "VIME_AGENT_NODE_TARBALL": "/mnt/workspace/agentenv-artifacts/node-v22.20.0-linux-x64.tar.xz",
        "VIME_AGENT_CODEX_TARBALL": "/mnt/workspace/agentenv-artifacts/vime-codex-offline-0.153.0.tgz",
        "VIME_AGENT_CODEX_MODEL_PROFILE": "Qwen3.5-4B",
        "ADAPTER_BIND_HOST": "0.0.0.0", "ADAPTER_PORT": "18001",
        "ADAPTER_PUBLIC_HOST": "192.0.2.1", "ADAPTER_PUBLIC_PORT": "18080",
    }
    for key, value in defaults.items():
        result[key] = source.get(key) or value
    result["VIME_AGENT_TASK_CONDA_SH"] = source.get(
        "VIME_AGENT_TASK_CONDA_SH", "/opt/miniconda3/etc/profile.d/conda.sh" if task == "swegym" else "",
    )
    result["VIME_AGENT_TASK_CONDA_PREFIX"] = source.get(
        "VIME_AGENT_TASK_CONDA_PREFIX", "/opt/miniconda3/envs/testbed" if task == "swegym" else "",
    )
    if task == "swegym":
        result["VIME_SWEGYM_SOURCE_PATH"] = (
            source.get("VIME_SWEGYM_SOURCE_PATH") or "/home/developer/dependencies/SWE-Bench-Package-16dd480c"
        )
    for key in AGENT_FIELDS:
        if key in source:
            result[key] = source[key]
    result.setdefault("VIME_AGENT_DEBUG_TRACE_DIR", "")
    result.setdefault("VIME_AGENT_TRACE_WRITER", "")
    lifetime = "4500" if task == "swegym" else str(int(result["VIME_RL_ROLLOUT_GUARD_SEC"]) + 600)
    sandbox_defaults = {
        "VIME_AGENT_SANDBOX_LIFETIME_SEC": lifetime,
        "VIME_SANDBOX_BOOT_CONCURRENCY": result["VIME_RL_BOOT_CONCURRENCY"],
        "VIME_SANDBOX_BOOT_RETRIES": result["VIME_RL_BOOT_RETRIES"],
        "VIME_AGENT_SANDBOX_RPC_RETRIES": "6",
        "VIME_AGENTENV_API_KEY_FILE": "/home/developer/.config/agentenv/api-key",
        "E2B_API_URL": "http://127.0.0.1:18000", "E2B_SANDBOX_URL": "http://127.0.0.1:18000",
        "VIME_AGENT_E2B_SDK_PATH": "/home/developer/.venvs/agentenv-harness/lib/python3.11/site-packages",
        "VIME_AGENT_E2B_SELECTOR_MODE": "template", "VIME_HARBOR_ENVIRONMENT": "remote_docker",
    }
    result.update(resolve_sandbox_env(source, sandbox_defaults))
    for key in ("BOOT_CONCURRENCY", "BOOT_RETRIES"):
        result[f"VIME_RL_{key}"] = result[f"VIME_SANDBOX_{key}"]
    if "VIME_AGENT_SANDBOX_CONFIG" in source:
        result["VIME_AGENT_SANDBOX_CONFIG"] = source["VIME_AGENT_SANDBOX_CONFIG"]
    if warn:
        aliases = {old: new for old, new in _legacy_aliases(task).items() if old in source}
        aliases.update({old: new for new, old in SANDBOX_ALIASES.items() if old in source})
        if aliases:
            migrations = "; ".join(f"{old} -> {new}" for old, new in sorted(aliases.items()))
            logger.warning("Legacy AgenticRL configuration: %s", migrations)
    return result


def worker_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The explicit worker allowlist; provider credentials are inherited by Ray."""
    source = _environment(env)
    keys = set(AGENT_FIELDS) | {
        "RUN_NAME", "OUTPUT_DIR", "VIME_SWEGYM_SOURCE_PATH",
        "ADAPTER_BIND_HOST", "ADAPTER_PORT", "ADAPTER_PUBLIC_HOST", "ADAPTER_PUBLIC_PORT",
    }
    fields = (
        *RL_DEFAULTS, "TASK", "PREPARED_BUNDLE", "PROMPT", "TRAIN_PROTOCOL", "EVAL_PROTOCOL",
        *(item[0] for item in BUNDLE_FIELDS.values()),
    )
    keys.update(f"VIME_RL_{key}" for key in fields)
    result = {key: source[key] for key in sorted(keys) if key in source}
    # A frozen snapshot is sufficient for adapters; legacy names are retained
    # only for their SDK boundary and existing diagnostics.
    result.update(resolve_sandbox_env(source))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    shell = sub.add_parser("shell")
    shell.add_argument("--task", choices=tuple(TASK_PREFIXES), required=True)
    shell.add_argument("--repo-root", required=True)
    shell.add_argument("--legacy-aliases", action="store_true", help=argparse.SUPPRESS)
    sub.add_parser("workers")
    args = parser.parse_args(argv)
    try:
        if args.operation == "shell":
            values = resolve_launch_config(args.task, args.repo_root, warn=True)
            if args.legacy_aliases:
                # Only the old sourced helper API needs its variable names back.
                # Use resolved values so inherited aliases agree in Ray workers.
                values.update({old: values[new] for old, new in _legacy_aliases(args.task).items() if new in values})
            # The fresh Ray head inherits the parent environment. Remove
            # consumed aliases so workers cannot re-read stale legacy values.
            # stdout is Shell source consumed by the launcher; keep diagnostics on stderr.
            print("unset " + " ".join(sorted(_legacy_aliases(args.task))))
            print("unset " + " ".join(sorted(SANDBOX_ALIASES.values())))
            for key, value in sorted(values.items()):
                print(f"export {key}={shlex.quote(value)}")
        elif args.operation == "workers":
            # stdout carries the worker environment array evaluated by the launcher.
            print("RT_ENV=(" + " ".join(shlex.quote(f"{key}={value}") for key, value in worker_env().items()) + ")")
    except (ValueError, KeyError, TypeError) as exc:
        logger.error("AgenticRL configuration error: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
