"""SWE task layer: dataset parsing, workspace prep, diff capture, fresh-sandbox eval.

One module, two grading protocols selected per-call (never an import-time side
effect):

  - "scaleswe" (default): scaleswe data shape (image_url + pre_commands +
    swepro/eval_cmd/f2p_script); custom "exit 0 == solved" grading.
  - "swebench": SWE-bench Verified (remote_env_info.{image,base_commit,
    test_patch,FAIL_TO_PASS,PASS_TO_PASS,version}); inline metadata uses the
    current SWE-bench evaluator. An explicitly configured VerifierSpec selects
    prepared SWE-bench grading or the pinned SWE-Gym evaluator.

The only thing that varies by protocol is the dataset schema and how a
diff is scored. Everything sandbox-side (prepare_workspace / git_diff /
apply_diff / pre_commands) is shared and lives here once.
``get_metadata(sample, protocol)`` produces the ``md`` dict; the
protocol-specific grading payload is carried under ``md["grading"]``
and is opaque to generate.py (which only reads instance_id / image / workdir).

Harness-agnostic on purpose -- nothing here is Claude-specific. ``SWE_PROMPT`` is
the task instruction (semantics, not CLI syntax). The only place a task meets a
harness is the prompt, which the orchestrator passes into ``harness.run()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vime.agent import sandbox as agent_sandbox
from vime.agent.adapters.common import flatten_content
from vime.agent.sandbox import E2BSandbox, Sandbox, exec_and_wait
from vime.utils.types import Sample

try:
    from swebench.harness.grading import get_eval_report  # type: ignore
    from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore

    _SWEBENCH_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - import-time diagnostic
    get_eval_report = None  # type: ignore
    make_test_spec = None  # type: ignore
    _SWEBENCH_IMPORT_ERROR = _exc

# Prepared SWE-bench sidecars use the v5 record API. Keep this optional and
# separate from the upstream inline-instance API above.
try:
    from swebench.harness.grading import get_eval_report as get_verifier_eval_report  # type: ignore
    from swebench.harness.utils import make_test_spec as make_verifier_test_spec  # type: ignore

    _SWEBENCH_VERIFIER_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - import-time diagnostic
    get_verifier_eval_report = None  # type: ignore
    make_verifier_test_spec = None  # type: ignore
    _SWEBENCH_VERIFIER_IMPORT_ERROR = _exc

from .verifier_spec import VerifierSpec

logger = logging.getLogger(__name__)

PROTOCOL_SCALESWE = "scaleswe"
PROTOCOL_SWEBENCH = "swebench"

# Paths inside the sandbox (avoid clashes with image-shipped paths).
_PATCH = "/workspace/__cagent_patch__.diff"
_TEST_PATCH = "/workspace/__cagent_test_patch__.diff"
_EVAL_SCRIPT = "/workspace/__cagent_eval.sh"
_PRE = "/workspace/__cagent_pre__.sh"
_F2P = "/workspace/__cagent_f2p__.py"
_SWEPRO_DIR = "/workspace/swepro_eval"

SWE_PROMPT = os.environ.get(
    "SWE_CC_PROMPT",
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit.",
)


def _activate_configured_swegym_source() -> None:
    configured = os.environ.get("VIME_SWEGYM_SOURCE_PATH")
    if not configured:
        return
    root = Path(configured).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_activate_configured_swegym_source()

try:
    from swegym.harness.grading import get_eval_report as get_swegym_eval_report  # type: ignore

    _SWEGYM_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - import-time diagnostic
    get_swegym_eval_report = None  # type: ignore
    _SWEGYM_IMPORT_ERROR = _exc


@dataclass(frozen=True)
class EvalResult:
    """Structured grading outcome with infrastructure failures kept separate."""

    reward: float
    applied_cleanly: bool
    status: str
    infra_failure: bool = False
    reason: str | None = None
    report: dict[str, Any] | None = None


def get_metadata(sample: Sample, protocol: str = PROTOCOL_SCALESWE) -> dict[str, Any]:
    if protocol == PROTOCOL_SWEBENCH:
        return _metadata_swebench(sample)
    return _metadata_scaleswe(sample)


def _metadata_scaleswe(sample: Sample) -> dict[str, Any]:
    """scaleswe shape: flat ``metadata.*`` (+ a few ``remote_env_info`` fallbacks).

    ``f2p_script`` (a self-contained pytest file ending in
    ``sys.exit(pytest.main(...))``) is carried verbatim; the grader materializes
    and runs it via ``write_file`` so no shell-quoting workaround is needed here.
    """
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    label = sample.label if (isinstance(sample.label, str) and len(sample.label) < 256) else None
    swepro = m.get("swepro")
    eval_cmd = m.get("eval_cmd")
    f2p_script = rem.get("f2p_script")
    looks_swebench = bool(rem.get("test_patch")) and not (swepro or eval_cmd or f2p_script)
    return {
        "protocol": PROTOCOL_SCALESWE,
        "instance_id": m.get("instance_id") or rem.get("instance_id") or label or "unknown",
        "image": m.get("image") or rem.get("image_url"),
        "workdir": m.get("workdir") or rem.get("workdir"),
        "problem_statement": m.get("problem_statement") or _coerce_prompt(sample.prompt),
        "looks_swebench": looks_swebench,
        "grading": {
            "swepro": swepro,
            "eval_cmd": eval_cmd,
            "f2p_script": f2p_script,
            "pre_commands": m.get("pre_commands") or rem.get("pre_commands"),
        },
    }


def _metadata_swebench(sample: Sample) -> dict[str, Any]:
    """SWE-bench Verified shape: carry the full instance dict through so
    make_test_spec gets every field it needs (version, hints_text, ...)."""
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    instance = {
        "instance_id": rem.get("instance_id") or "unknown",
        "repo": rem.get("repo") or "",
        "version": rem.get("version"),
        "base_commit": rem.get("base_commit") or "",
        "problem_statement": rem.get("problem_statement") or _coerce_prompt(sample.prompt),
        "hints_text": rem.get("hints_text") or "",
        "test_patch": rem.get("test_patch") or "",
        "FAIL_TO_PASS": rem.get("FAIL_TO_PASS"),
        "PASS_TO_PASS": rem.get("PASS_TO_PASS"),
        "environment_setup_commit": rem.get("environment_setup_commit"),
    }
    return {
        "protocol": PROTOCOL_SWEBENCH,
        "instance_id": instance["instance_id"],
        "image": rem.get("image"),
        "workdir": rem.get("workdir") or "/testbed",
        "problem_statement": instance["problem_statement"],
        "grading": {"sweb_instance": instance},
    }


def _coerce_prompt(prompt) -> str:
    """Extract the user-message text from a prompt (str or chat-message list)."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                return flatten_content(m.get("content"))
    return ""


def evaluability_check(md: dict) -> str | None:
    if md.get("protocol") == PROTOCOL_SWEBENCH:
        return _evaluability_check_swebench(md)
    return "protocol_row_mismatch:looks_swebench" if md.get("looks_swebench") else None


def _resolve_verifier_spec(md: dict) -> VerifierSpec | None:
    """Allow inline grading only when no prepared verifier contract is selected.

    A configured manifest or an attached verifier binding must never fall back
    to inline grading when its specification is missing or invalid.
    """
    grading = md.get("grading") or {}
    if "verifier_spec" in grading:
        spec = grading["verifier_spec"]
        if not isinstance(spec, VerifierSpec):
            raise ValueError("invalid_verifier_spec")
        return spec
    if md.get("verifier_spec_id") or os.environ.get("SWE_VERIFIER_SPEC_MAP", "").strip():
        raise ValueError("missing_verifier_spec")
    return None


def _evaluability_check_swebench(md: dict) -> str | None:
    grading = md.get("grading", {})
    inst = grading.get("sweb_instance") or {}
    try:
        spec = _resolve_verifier_spec(md)
    except ValueError as exc:
        return str(exc)
    if spec is not None and spec.evaluator == "swegym":
        if _SWEGYM_IMPORT_ERROR is not None:
            return f"swegym_import_failed:{type(_SWEGYM_IMPORT_ERROR).__name__}"
    elif spec is not None:
        if _SWEBENCH_VERIFIER_IMPORT_ERROR is not None:
            return f"swebench_verifier_import_failed:{type(_SWEBENCH_VERIFIER_IMPORT_ERROR).__name__}"
    elif _SWEBENCH_IMPORT_ERROR is not None:
        return f"swebench_import_failed:{type(_SWEBENCH_IMPORT_ERROR).__name__}"
    if not inst.get("repo"):
        return "missing_repo"
    if not inst.get("base_commit"):
        return "missing_base_commit"
    if not (inst.get("test_patch") or "").strip():
        return "missing_test_patch"
    try:
        _ = _build_test_spec(spec if spec is not None else inst).eval_script
    except Exception as e:  # KeyError on unknown repo/version, etc.
        return f"make_test_spec_failed:{type(e).__name__}"
    return None


# ---------------------------------------------------------------------------
# Workspace prep (agent sandbox, before harness.run)
# ---------------------------------------------------------------------------
async def prepare_workspace(sb: Sandbox, workdir: str, md: dict) -> None:
    """Prep the agent sandbox, then drop PROBLEM_STATEMENT.md.

    Assumes the agent user already owns ``workdir`` (the harness's ``run()`` calls
    ``ensure_agent_user``; the orchestrator runs this before ``run()`` and the
    agent user is created lazily there). To stay independent of call order we
    create the agent user here too -- it is idempotent.
    """
    await agent_sandbox.ensure_agent_user(sb, workdir)
    if md.get("protocol") == PROTOCOL_SCALESWE:
        grading = md.get("grading") or {}
        swepro = grading.get("swepro")
        if swepro:
            await apply_before_repo_set_cmd(sb, workdir, swepro)
        pre_commands = grading.get("pre_commands")
        if pre_commands:
            await apply_pre_commands(sb, workdir, pre_commands)
    await sb.write_file(
        f"{workdir}/PROBLEM_STATEMENT.md",
        md.get("problem_statement") or "",
        user="agent",
    )


async def apply_before_repo_set_cmd(sb: Sandbox, workdir: str, swepro: dict) -> None:
    """Run swepro['before_repo_set_cmd'] in the sandbox if present (no-op if not)."""
    before = swepro.get("before_repo_set_cmd")
    if not before:
        return
    payload = f"set -e\ncd {workdir}\n{before}\n"
    await sb.exec(
        "mkdir -p /workspace/swepro_setup && chown agent:agent /workspace/swepro_setup", user="root", check=True
    )
    await sb.write_file("/workspace/swepro_setup/before.sh", payload, user="agent")
    await sb.exec("bash /workspace/swepro_setup/before.sh", user="agent", check=False, timeout=600)


async def apply_pre_commands(sb: Sandbox, workdir: str, pre: list[str] | str) -> None:
    # Public: also called for the work sandbox to keep its baseline aligned with
    # eval (sweb-style pre_commands typically `git checkout <base_sha> -f`, so
    # skipping in the work sandbox makes the model's diff context mismatch the
    # eval base -> 100% apply failure).
    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in (pre or []) if c)
    await sb.write_file(_PRE, "set -e\n" + body, user="agent")
    await sb.exec(f"chmod 755 {_PRE} && cd {workdir} && bash {_PRE}", user="agent", check=False, timeout=600)


# ---------------------------------------------------------------------------
# Diff capture (agent sandbox, after harness.run)
# ---------------------------------------------------------------------------
async def git_diff(sb: Sandbox, workdir: str) -> str:
    cmd = f"cd {workdir} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/'"
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out


# ---------------------------------------------------------------------------
# Eval dispatch (fresh sandbox, apply diff, run dataset tests)
# ---------------------------------------------------------------------------
async def run_evaluation(md: dict, *, diff_text: str, timeout_sec: int) -> EvalResult:
    """Uniform entry point: dispatch to the protocol's grader.

    No-test-cheating guarantee (both grading protocols): the eval sandbox is built from
    the same image but starts CLEAN, so only the model-produced diff affects
    reward."""
    if md.get("protocol") == PROTOCOL_SWEBENCH:
        return await _grade_swebench(md, diff_text, timeout_sec)
    return await _grade_scaleswe(md, diff_text, timeout_sec)


# ---------------------------------------------------------------------------
# scaleswe grader
# ---------------------------------------------------------------------------
async def _grade_scaleswe(md: dict, diff_text: str, timeout_sec: int) -> EvalResult:
    """Three mutually-exclusive grading paths, in priority order: swepro test
    harness, a shell ``eval_cmd``, or a self-contained ``f2p_script`` pytest
    file. All resolve to "exit 0 == solved", reward is 1.0 iff solved."""
    image = md["image"]
    workdir = md["workdir"]
    grading = md.get("grading") or {}
    swepro = grading.get("swepro")
    eval_cmd = grading.get("eval_cmd")
    f2p_script = grading.get("f2p_script")
    pre_commands = grading.get("pre_commands")

    if not (swepro or eval_cmd or f2p_script):
        logger.warning("[swe.scaleswe] no swepro/eval_cmd/f2p_script; reward=0")
        return EvalResult(0.0, True, "configuration_error", reason="missing_scaleswe_grader")

    async with E2BSandbox(image) as ev:
        await agent_sandbox.ensure_agent_user(ev, workdir)
        if swepro:
            await _setup_swepro_assets(ev, swepro)
            await apply_before_repo_set_cmd(ev, workdir, swepro)
        if pre_commands:
            await apply_pre_commands(ev, workdir, pre_commands)

        applied = await _apply_diff(ev, workdir, diff_text)
        if not applied:
            return EvalResult(0.0, False, "invalid_patch", reason="model_patch_apply_failed")

        if swepro:
            r = await _run_swepro(ev, workdir, swepro, timeout_sec)
        elif eval_cmd:
            r = await _run_eval_cmd(ev, workdir, eval_cmd, timeout_sec)
        else:
            r = await _run_f2p_script(ev, workdir, f2p_script, timeout_sec)
        return EvalResult(r, True, "resolved" if r == 1.0 else "unresolved")


async def _setup_swepro_assets(ev: Sandbox, swepro: dict) -> None:
    await ev.exec(f"mkdir -p {_SWEPRO_DIR} && chmod 777 {_SWEPRO_DIR}", user="root", check=True)
    for k, dst in [("run_script_path", "run_script.sh"), ("parser_script_path", "parser.py")]:
        host_p = swepro.get(k)
        if host_p:
            await ev.write_file(f"{_SWEPRO_DIR}/{dst}", Path(host_p), user="root")
    await ev.exec(f"chmod 755 {_SWEPRO_DIR}/* && chown -R agent:agent {_SWEPRO_DIR}", user="root", check=True)


async def _apply_diff(ev: Sandbox, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await ev.write_file(_PATCH, diff_text, user="agent")
    # First-success-wins ladder collapsed into one exec (one sandbox round-trip).
    ladder = " || ".join(
        f"({cmd})"
        for cmd in (
            f"git apply --3way --whitespace=nowarn {_PATCH}",
            f"git apply --whitespace=nowarn {_PATCH}",
            f"patch -p1 --no-backup-if-mismatch < {_PATCH}",
        )
    )
    ec, _, _ = await ev.exec(f"cd {workdir} && ({ladder})", user="agent", check=False, timeout=120)
    return ec == 0


async def _run_swepro(ev: Sandbox, workdir: str, swepro: dict, timeout: int) -> float:
    test_arg = ",".join(swepro.get("selected_test_files") or [])
    stdout_f = f"{_SWEPRO_DIR}/stdout.log"
    stderr_f = f"{_SWEPRO_DIR}/stderr.log"
    result_f = f"{_SWEPRO_DIR}/result.json"
    await ev.exec(
        f"cd {workdir} && bash {_SWEPRO_DIR}/run_script.sh {json.dumps(test_arg)} > {stdout_f} 2> {stderr_f} || true",
        user="agent",
        check=False,
        timeout=timeout,
    )
    await ev.exec(
        f"python3 {_SWEPRO_DIR}/parser.py {stdout_f} {stderr_f} {result_f}",
        user="agent",
        check=False,
        timeout=120,
    )
    raw = await ev.read_file(result_f, user="agent")
    parsed = json.loads(raw) if raw else {"tests": []}
    passed = {t["name"] for t in parsed.get("tests", []) if t.get("status") == "PASSED"}
    required = set(swepro.get("fail_to_pass") or []) | set(swepro.get("pass_to_pass") or [])
    solved = bool(required) and required.issubset(passed)
    return 1.0 if solved else 0.0


async def _run_eval_cmd(ev: Sandbox, workdir: str, cmd: str, timeout: int) -> float:
    ec, _, _ = await ev.exec(f"cd {workdir} && {cmd}", user="agent", check=False, timeout=timeout)
    return 1.0 if ec == 0 else 0.0


async def _run_f2p_script(ev: Sandbox, workdir: str, script: str, timeout: int) -> float:
    # sweb f2p_script is a self-contained pytest file ending in
    # `sys.exit(pytest.main([...]))`; write it verbatim (no shell quoting) and
    # let python's exit code carry the pass/fail signal.
    await ev.write_file(_F2P, script, user="agent")
    ec, _, _ = await ev.exec(f"cd {workdir} && python {_F2P}", user="agent", check=False, timeout=timeout)
    return 1.0 if ec == 0 else 0.0


# Mirror of swebench.harness.run_evaluation.GIT_APPLY_CMDS: try each in order,
# first success wins. The `patch --fuzz` tier rescues diffs `git apply` rejects.
_GIT_APPLY_CMDS = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)


async def _apply_model_patch(ev: Sandbox, workdir: str) -> bool:
    """Apply the uploaded model patch via GIT_APPLY_CMDS; True if applied
    (or empty). Empty patch is a no-op success -- eval then scores it 0 on its
    own (no source change -> tests still fail)."""
    ladder = " || ".join(f"{cmd} {_PATCH}" for cmd in _GIT_APPLY_CMDS)
    cmd = (
        f"cd {workdir} && git config --global --add safe.directory {workdir} "
        f"&& if [ -s {_PATCH} ]; then {ladder}; fi"
    )
    ec, _, _ = await ev.exec(cmd, user="root", check=False, timeout=120)
    return ec == 0


def _build_test_spec(spec: VerifierSpec | dict):
    """Build grader input from an explicit verifier or the existing inline row."""
    if isinstance(spec, dict):
        return make_test_spec(spec)  # type: ignore[misc]
    if not isinstance(spec, VerifierSpec):
        raise TypeError("expected VerifierSpec or inline instance dict")
    if spec.evaluator == "swegym":
        return SimpleNamespace(
            instance_id=spec.instance_id.lower(),
            repo=spec.repo,
            version=spec.version,
            FAIL_TO_PASS=list(spec.fail_to_pass),
            PASS_TO_PASS=list(spec.pass_to_pass),
            eval_script=spec.eval_script,
        )
    return make_verifier_test_spec(spec.to_swebench_record())  # type: ignore[misc]


async def _apply_test_patch(ev: Sandbox, workdir: str) -> bool:
    """Apply the authoritative hidden tests; a conflict is a model-patch failure."""
    ec, _, _ = await ev.exec(
        f"cd {workdir} && git apply --verbose {_TEST_PATCH}",
        user="root",
        check=False,
        timeout=120,
        idempotent=False,
    )
    return ec == 0


def _eval_report_from_log(
    ts,
    instance_id: str,
    diff_text: str,
    log: str,
    *,
    spec: VerifierSpec | None = None,
) -> dict:
    """Dispatch captured logs to the selected SWE-Gym or SWE-bench grader."""
    prediction = {
        "instance_id": instance_id,
        "model_patch": diff_text or "",
        "model_name_or_path": "swe",
    }
    if spec is not None and spec.evaluator == "swegym":
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / ts.instance_id
            log_dir.mkdir()
            log_path = log_dir / "test_output.txt"
            log_path.write_text(log, encoding="utf-8")
            return get_swegym_eval_report(  # type: ignore[misc]
                ts,
                prediction,
                str(log_path),
                True,
            )

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    try:
        tmp.write(log)
        tmp.flush()
        tmp.close()
        report_fn = get_verifier_eval_report if spec is not None else get_eval_report
        return report_fn(  # type: ignore[misc]
            test_spec=ts,
            prediction=prediction,
            test_log_path=tmp.name,
            include_tests_status=True,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _ratio(d: dict) -> tuple[int, int]:
    """(passed, total) from a {success: [...], failure: [...]} bucket."""
    passed, failed = d.get("success", []), d.get("failure", [])
    return len(passed), len(passed) + len(failed)


def _log_swebench_result(instance_id: str, exit_code, info: dict, log: str) -> None:
    """Emit the per-instance grading outcome with test-bucket ratios; on a
    non-resolved row that parsed NO test lines, surface the log tail so failures
    (missing pytest plugin, conda not activated, ...) can be diagnosed."""
    if info.get("resolved"):
        logger.info("[swe.swebench] %s: reward=1 exit_code=%s", instance_id, exit_code)
        return
    ts_status = info.get("tests_status") or {}
    f2p_pass, f2p_total = _ratio(ts_status.get("FAIL_TO_PASS", {}))
    p2p_pass, p2p_total = _ratio(ts_status.get("PASS_TO_PASS", {}))
    nothing_parsed = not (f2p_total or p2p_total)
    tail = log[-800:] if nothing_parsed else ""
    logger.info(
        "[swe.swebench] %s: reward=0 exit_code=%s patch_applied=%s F2P=(%d/%d) P2P=(%d/%d)%s",
        instance_id,
        exit_code,
        bool(info.get("patch_successfully_applied")),
        f2p_pass,
        f2p_total,
        p2p_pass,
        p2p_total,
        f" tail={tail!r}" if tail else "",
    )


async def _grade_swebench(md: dict, diff_text: str, timeout_sec: int) -> EvalResult:
    """reward=1.0 iff sweb's get_eval_report declares the instance ``resolved``."""
    instance_id = md["instance_id"]
    try:
        spec = _resolve_verifier_spec(md)
    except ValueError as exc:
        return EvalResult(
            0.0,
            True,
            "infra_error",
            infra_failure=True,
            reason=str(exc),
        )

    evaluator = spec.evaluator if spec is not None else "swebench"
    if evaluator == "swegym":
        import_error = _SWEGYM_IMPORT_ERROR
    elif spec is not None:
        import_error = _SWEBENCH_VERIFIER_IMPORT_ERROR
    else:
        import_error = _SWEBENCH_IMPORT_ERROR
    if import_error is not None:
        logger.error(
            "[swe.swebench] %s: %s evaluator import failed: %r; reward=0",
            instance_id,
            evaluator,
            import_error,
        )
        return EvalResult(
            0.0,
            True,
            "infra_error",
            infra_failure=True,
            reason=f"{evaluator}_import_failed:{type(import_error).__name__}",
        )

    try:
        ts = _build_test_spec(spec if spec is not None else md["grading"]["sweb_instance"])
        eval_sh = ts.eval_script
    except Exception as e:
        logger.warning("[swe.swebench] %s: VerifierSpec/TestSpec failed: %s", instance_id, e)
        return EvalResult(
            0.0,
            True,
            "infra_error",
            infra_failure=True,
            reason=f"make_test_spec_failed:{type(e).__name__}",
        )

    image = md["image"]
    if not image:
        logger.warning("[swe.swebench] %s: missing template selector", instance_id)
        return EvalResult(0.0, True, "infra_error", infra_failure=True, reason="missing_template")

    async with E2BSandbox(image) as ev:
        writes = [
            ev.write_file(_PATCH, diff_text or "", user="root"),
            ev.write_file(_EVAL_SCRIPT, eval_sh, user="root"),
        ]
        if spec is not None and spec.test_patch_owner == "verifier":
            writes.append(ev.write_file(_TEST_PATCH, spec.test_patch, user="root"))
        await asyncio.gather(*writes)
        # Apply the model patch first (eval_script assumes it is already applied);
        # if no apply strategy works, the instance is unsolvable -- skip the eval.
        if not await _apply_model_patch(ev, md["workdir"]):
            logger.warning("[swe.swebench] %s: model patch failed to apply; reward=0", instance_id)
            return EvalResult(0.0, False, "invalid_patch", reason="model_patch_apply_failed")
        if spec is not None and spec.test_patch_owner == "verifier":
            if not await _apply_test_patch(ev, md["workdir"]):
                logger.warning(
                    "[swe.swebench] %s: hidden test patch failed to apply; reward=0",
                    instance_id,
                )
                return EvalResult(0.0, False, "invalid_patch", reason="test_patch_conflict")
        exit_code, log = await exec_and_wait(
            ev,
            cmd=f"bash {_EVAL_SCRIPT}",
            user="root",
            time_budget_sec=min(timeout_sec, spec.timeout_sec) if spec is not None else timeout_sec,
            tag="eval",
            want_output=True,
        )

    if exit_code < 0:
        return EvalResult(0.0, True, "infra_error", infra_failure=True, reason="verifier_timeout")

    try:
        report = _eval_report_from_log(ts, instance_id, diff_text, log, spec=spec)
    except Exception as e:
        logger.warning(
            "[swe.swebench] %s: get_eval_report failed: %s; reward=0 (tail=%r)",
            instance_id,
            e,
            log[-600:],
        )
        return EvalResult(
            0.0,
            True,
            "infra_error",
            infra_failure=True,
            reason=f"grading_exception:{type(e).__name__}",
        )

    info = report.get(instance_id, {})
    _log_swebench_result(instance_id, exit_code, info, log)
    if not info.get("patch_successfully_applied"):
        return EvalResult(
            0.0,
            False,
            "infra_error",
            infra_failure=True,
            reason=info.get("infra_failure_reason") or "unparseable_test_output",
            report=info,
        )
    if info.get("infra_failure"):
        return EvalResult(
            0.0,
            bool(info.get("patch_successfully_applied")),
            "infra_error",
            infra_failure=True,
            reason=info.get("infra_failure_reason") or "unparseable_test_output",
            report=info,
        )
    resolved = bool(info.get("resolved"))
    return EvalResult(
        1.0 if resolved else 0.0,
        bool(info.get("patch_successfully_applied")),
        "resolved" if resolved else "unresolved",
        report=info,
    )
