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

"""Terminal-Bench task layer: dataset parsing, workspace prep, same-sandbox oracle grading.

One protocol only (``terminalbench``): a task is a terminal instruction plus an
OCI image whose /tests holds the oracle. The agent works inside a sandbox booted
from the task template; grading runs ``run_tests_cmd`` in that SAME sandbox after
the agent finishes and maps exit code 0 to reward 1. No git, no diff extraction,
no external evaluator package.

Dataset row shape (bundle ``rollout/train.no_gold.jsonl``)::

    {"prompt": <instruction>, "label": <task_id>,
     "metadata": {"task_id": ..., "image": <source image ref>, "workdir": "/root"}}

The hidden grading contract (run_tests_cmd / timeout / tests hash) is bound at
rollout time from the VerifierSpec sidecar, never from the prompt row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from vime.agent import sandbox as agent_sandbox
from vime.agent.config import rl_value
from vime.agent.sandbox import Sandbox, SandboxCommandTimeout
if TYPE_CHECKING:
    from vime.utils.types import Sample

logger = logging.getLogger(__name__)

PROTOCOL_TERMINALBENCH = "terminalbench"

DEFAULT_WORKDIR = "/root"
DEFAULT_TESTS_PATH = "/tests/run-tests.sh"
DEFAULT_RUN_TESTS_CMD = "bash /tests/run-tests.sh"
DEFAULT_TIMEOUT_SEC = 900

# Canonical, order-independent hash over the whole /tests tree. The preparer
# records this value at probe time; run_evaluation recomputes it before
# grading so an agent that rewrote the oracle scores zero instead of one.
TESTS_HASH_CMD = "cd /tests && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1"

TB_PROMPT = rl_value(
    "terminalbench", "PROMPT",
    "Read TASK.md in the current directory and complete the task it describes. "
    "You may inspect and modify the environment freely, but do NOT read, modify "
    "or delete anything under /tests. Verify your work before finishing. When "
    "finished, print a one-line summary and exit.",
)


@dataclass(frozen=True)
class EvalResult:
    """Structured grading outcome with infrastructure failures kept separate."""

    reward: float
    status: str
    infra_failure: bool = False
    reason: str | None = None
    report: dict[str, Any] | None = None


def _oracle_targets(run_tests_cmd: str) -> list[str]:
    """Absolute-path tokens of the grading command, used for fail-fast checks."""
    import shlex

    return [token for token in shlex.split(run_tests_cmd) if token.startswith("/")]


def _coerce_prompt(prompt: Any) -> str:
    """Extract the user-message text from a prompt (str or chat-message list)."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        from vime.agent.adapters.common import flatten_content
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                return flatten_content(m.get("content"))
    return ""


def get_metadata(sample: Sample, protocol: str = PROTOCOL_TERMINALBENCH) -> dict[str, Any]:
    if protocol != PROTOCOL_TERMINALBENCH:
        raise ValueError(f"unsupported protocol {protocol!r}; expected {PROTOCOL_TERMINALBENCH!r}")
    m = sample.metadata or {}
    label = sample.label if (isinstance(sample.label, str) and len(sample.label) < 256) else None
    grading = m.get("grading") if isinstance(m.get("grading"), dict) else {}
    return {
        "protocol": PROTOCOL_TERMINALBENCH,
        "task_id": m.get("task_id") or m.get("instance_id") or label or "unknown",
        "image": m.get("image"),
        "workdir": m.get("workdir") or DEFAULT_WORKDIR,
        "instruction": m.get("instruction") or _coerce_prompt(sample.prompt),
        "grading": {
            "run_tests_cmd": grading.get("run_tests_cmd") or DEFAULT_RUN_TESTS_CMD,
            "timeout_sec": grading.get("timeout_sec") or DEFAULT_TIMEOUT_SEC,
            "tests_sha256": grading.get("tests_sha256"),
        },
    }


def evaluability_check(md: dict) -> str | None:
    """Return a reason string when the row cannot be graded, else None."""
    if md.get("protocol") != PROTOCOL_TERMINALBENCH:
        return f"protocol_row_mismatch:{md.get('protocol')!r}"
    if not md.get("task_id") or md["task_id"] == "unknown":
        return "missing_task_id"
    if not md.get("image"):
        return "missing_image"
    if not md.get("workdir"):
        return "missing_workdir"
    if not (md.get("instruction") or "").strip():
        return "missing_instruction"
    grading = md.get("grading") or {}
    if not (grading.get("run_tests_cmd") or "").strip():
        return "missing_run_tests_cmd"
    timeout = grading.get("timeout_sec")
    if not isinstance(timeout, int) or timeout <= 0:
        return "invalid_timeout_sec"
    return None


async def prepare_workspace(sb: Sandbox, workdir: str, md: dict) -> None:
    """Prep the agent sandbox: agent user and TASK.md. No oracle here.

    Idempotent and git-free. The oracle is deliberately NOT placed during the
    agent session: an oracle visible to the agent leaks the grading contract
    (it can read expected values and reverse-engineer them, or plant a fake
    always-passing test). run_evaluation injects the payload after the session
    ends. With no spec payload the image itself must bake the oracle, and that
    legacy path keeps the fail-fast existence check.
    """
    await agent_sandbox.ensure_agent_user(sb, workdir)
    await sb.write_file(f"{workdir}/TASK.md", md.get("instruction") or "", user="agent")
    grading = md.get("grading") or {}
    if grading.get("tests"):
        return  # injected at grading time; nothing to reveal or check now
    run_tests_cmd = (grading.get("run_tests_cmd") or DEFAULT_RUN_TESTS_CMD).strip()
    targets = _oracle_targets(run_tests_cmd)
    if targets:
        import shlex

        check = " && ".join(f"test -e {shlex.quote(target)}" for target in targets)
        code, _, _ = await sb.exec(check, user="root", timeout=60)
        if code != 0:
            raise RuntimeError(f"oracle missing for {md.get('task_id')}: {check} failed")


def _is_timeout_error(exc: BaseException) -> bool:
    # Python 3.11+ unifies asyncio.TimeoutError with TimeoutError; the e2b SDK
    # raises its own TimeoutException for command deadlines.
    return type(exc).__name__ == "TimeoutException" or isinstance(exc, (TimeoutError, SandboxCommandTimeout))


async def run_evaluation(sb: Sandbox, md: dict, *, timeout_sec: int | None = None) -> EvalResult:
    """Grade in the agent's own sandbox: inject the oracle, then run it.

    The effective budget is the smaller of the caller's budget and the task's
    VerifierSpec timeout. Exit 0 maps to reward 1.0; a rewritten /tests maps to
    reward 0.0 regardless of the oracle's exit code.
    """
    grading = md.get("grading") or {}
    run_tests_cmd = (grading.get("run_tests_cmd") or DEFAULT_RUN_TESTS_CMD).strip()
    spec_timeout = grading.get("timeout_sec")
    budget = int(timeout_sec or spec_timeout or DEFAULT_TIMEOUT_SEC)
    if isinstance(spec_timeout, int) and spec_timeout > 0:
        budget = min(budget, spec_timeout)
    task_id = md.get("task_id")
    expected_hash = grading.get("tests_sha256")

    # Deferred oracle injection: the agent session has ended, so the grading
    # contract can finally enter the sandbox. Wipe /tests first -- anything the
    # agent planted there (a fake always-passing test, or a regular file named
    # /tests) is replaced by the authoritative spec payload.
    tests = grading.get("tests") or {}
    if tests:
        code, _, err = await sb.exec("rm -rf /tests && mkdir -p /tests", user="root", timeout=30, idempotent=False)
        if code != 0:
            return EvalResult(
                0.0, "verifier_error", infra_failure=True,
                reason=f"tests_dir_reset_failed:{(err or '')[:120]}",
            )
        for name, content in tests.items():
            target = f"/tests/{name.lstrip('/')}"
            try:
                await sb.write_file(target, content, user="root")
            except Exception as exc:
                return EvalResult(
                    0.0, "verifier_error", infra_failure=True,
                    reason=f"oracle_injection_failed:{type(exc).__name__}",
                )

    if expected_hash:
        try:
            code, actual, _ = await sb.exec(TESTS_HASH_CMD, user="root", timeout=120)
        except Exception as exc:
            return EvalResult(0.0, "verifier_error", infra_failure=True, reason=f"hash_rpc_error:{type(exc).__name__}")
        if code != 0:
            return EvalResult(0.0, "tests_missing", infra_failure=True, reason="tests_hash_command_failed")
        actual = actual.strip()
        if actual != expected_hash:
            return EvalResult(
                0.0,
                "tests_tampered",
                reason=f"hash_mismatch:{actual[:12]}!={expected_hash[:12]}",
                report={"tests_sha256_expected": expected_hash, "tests_sha256_actual": actual},
            )

    try:
        exit_code, stdout, stderr = await sb.exec(run_tests_cmd, user="root", timeout=budget, idempotent=False)
    except Exception as exc:
        if _is_timeout_error(exc):
            return EvalResult(0.0, "tests_timeout", reason=f"oracle_exceeded_{budget}s")
        return EvalResult(0.0, "verifier_error", infra_failure=True, reason=f"oracle_rpc_error:{type(exc).__name__}")

    report = {
        "run_tests_cmd": run_tests_cmd,
        "timeout_sec": budget,
        "exit_code": exit_code,
        "stdout_tail": (stdout or "")[-2000:],
        "stderr_tail": (stderr or "")[-2000:],
    }
    if exit_code == 0:
        logger.info("[terminal_agent_rl] %s: oracle passed", task_id)
        return EvalResult(1.0, "solved", report=report)
    logger.info("[terminal_agent_rl] %s: oracle failed exit=%d", task_id, exit_code)
    return EvalResult(0.0, "tests_failed", reason=f"oracle_exit_{exit_code}", report=report)
