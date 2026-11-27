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

"""Bundle binding and same-sandbox grading; no service access."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from examples.terminal_agent_rl import terminalbench as tb
from examples.terminal_agent_rl import template_mapping as tm, verifier_spec as vs
from examples.terminal_agent_rl import prepare_terminalbench as prep
from vime.agent.sandbox import SandboxCommandTimeout

DIGEST = "sha256:" + "a" * 64
IMAGE = "registry.example/task@" + DIGEST
TEMPLATE = "11111111-1111-4111-8111-111111111111"


def environment(template=TEMPLATE):
    return dict(schema_version=1, task_id="task", source_image=IMAGE, source_image_tag="registry.example/task:latest",
                image_digest=DIGEST, template_id=template, workdir="/task")


def verifier(template=TEMPLATE):
    return dict(schema_version=1, task_id="task", spec_id="spec", protocol="terminalbench-v1",
                template_id=template, image_digest=DIGEST, workdir="/task", run_tests_cmd="bash /tests/check.sh",
                timeout_sec=10, tests_sha256=None, tests={"check.sh": "exit 1"})


class BindingTests(unittest.TestCase):
    def test_same_bundle_selects_each_backend(self):
        mapping = tm.TemplateMapping([tm.TemplateMappingEntry.from_record(environment(), location="test")])
        registry = vs.VerifierSpecRegistry([vs.VerifierSpec.from_record(verifier(), location="test")])
        for backend, image in (("agentenv", TEMPLATE), ("harbor", IMAGE)):
            with patch.dict(os.environ, {"VIME_AGENT_SANDBOX_BACKEND": backend}):
                md = tm.apply_template_mapping(dict(task_id="task", image=IMAGE, workdir="/task"), mapping)
                md = vs.apply_verifier_spec(md, registry)
                self.assertEqual(md["image"], image)
                self.assertEqual(md["grading"]["tests"], {"check.sh": "exit 1"})

    def test_harbor_only_mapping_rejects_agentenv(self):
        mapping = tm.TemplateMapping([tm.TemplateMappingEntry.from_record(environment(None), location="test")])
        with (
            patch.dict(os.environ, {"VIME_AGENT_SANDBOX_BACKEND": "agentenv"}),
            self.assertRaises(tm.TemplateMappingError),
        ):
            tm.apply_template_mapping(dict(task_id="task", image=IMAGE), mapping)

    def test_verifier_mismatch_is_rejected(self):
        reg = vs.VerifierSpecRegistry([vs.VerifierSpec.from_record(verifier(), location="test")])
        with self.assertRaises(vs.VerifierSpecError):
            reg.resolve("task", image_digest="sha256:" + "b" * 64)

    def test_oracle_paths_cannot_escape_tests(self):
        for name in ("../outside", "/absolute", "a/../../outside"):
            with self.subTest(name=name), self.assertRaises(vs.VerifierSpecError):
                vs.VerifierSpec.from_record({**verifier(), "tests": {name: "bad"}}, location="test")


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_oracle_only_written_during_evaluation(self):
        sb = SimpleNamespace(write_file=AsyncMock(), exec=AsyncMock(return_value=(0, "", "")))
        md = dict(
            instruction="Do the task",
            workdir="/task",
            grading={"tests": {"check.sh": "exit 0"}, "run_tests_cmd": "bash /tests/check.sh", "timeout_sec": 10},
        )
        with patch.object(tb.agent_sandbox, "ensure_agent_user", AsyncMock()):
            await tb.prepare_workspace(sb, "/task", md)
        self.assertEqual([c.args[0] for c in sb.write_file.call_args_list], ["/task/TASK.md"])
        result = await tb.run_evaluation(sb, md, timeout_sec=20)
        self.assertEqual(result.reward, 1)
        self.assertEqual(sb.exec.call_args.kwargs["timeout"], 10)
        self.assertFalse(sb.exec.call_args.kwargs["idempotent"])
        self.assertEqual(sb.write_file.call_args_list[-1].args[0], "/tests/check.sh")

    async def test_failure_timeout_and_infrastructure_are_distinct(self):
        for outcome, status, infra in (
            ((1, "", ""), "tests_failed", False),
            (SandboxCommandTimeout("limit"), "tests_timeout", False),
            (ConnectionError("link"), "verifier_error", True),
        ):
            sb = SimpleNamespace(exec=AsyncMock())
            if isinstance(outcome, Exception):
                sb.exec.side_effect = outcome
            else:
                sb.exec.return_value = outcome
            result = await tb.run_evaluation(sb, {"grading": {"run_tests_cmd": "check", "timeout_sec": 10}})
            self.assertEqual((result.status, result.infra_failure, result.reward), (status, infra, 0))

    async def test_hash_mismatch_prevents_oracle_execution(self):
        sb = SimpleNamespace(exec=AsyncMock(return_value=(0, "wrong", "")))
        result = await tb.run_evaluation(sb, {"grading": {"tests_sha256": "a" * 64}})
        self.assertEqual(result.status, "tests_tampered")
        sb.exec.assert_awaited_once()


class PreparationTests(unittest.TestCase):
    def test_harbor_bundle_keeps_oracle_out_of_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "tasks.jsonl"
            source.write_text(
                json.dumps(dict(
                    task_id="task", instruction="visible", image=IMAGE, workdir="/task",
                    tests={"check.sh": "hidden"}, run_tests_cmd="bash /tests/check.sh", timeout_sec=10,
                )) + "\n"
            )
            args = SimpleNamespace(source=source, bundle=root / "bundle", instances=None, instance_map=None,
                                   image_template=None, id_extract="whole", workdir=None, run_tests_cmd=None,
                                   timeout_sec=None, registry_base=None, no_probe=True, qualify=False)
            with (
                patch.dict(os.environ, {"VIME_AGENT_SANDBOX_BACKEND": "harbor"}),
                patch(
                    "examples.terminal_agent_rl.sandbox.create_sandbox",
                    side_effect=AssertionError("no sandbox expected"),
                ),
            ):
                prep.gen_bundle(args)
            paths = prep._bundle_paths(args.bundle)
            self.assertNotIn("hidden", paths["train_rows"].read_text())
            self.assertIn("hidden", paths["verifier_specs"].read_text())
            entry = tm.load_template_mapping(str(paths["templates"]))
            reg = vs.load_verifier_specs(str(paths["verifier_specs"]))
            with patch.dict(os.environ, {"VIME_AGENT_SANDBOX_BACKEND": "harbor"}):
                md = tm.apply_template_mapping(dict(task_id="task", image=IMAGE, workdir="/task"), entry)
                self.assertEqual(vs.apply_verifier_spec(md, reg)["grading"]["tests"], {"check.sh": "hidden"})


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real orchestrator with heavyweight model/server boundaries
        # replaced. Sandbox contracts, bundle loaders and task grading stay real.
        import importlib
        import sys
        from types import ModuleType
        from contextlib import contextmanager

        def mod(name, **attrs):
            m = ModuleType(name)
            m.__dict__.update(attrs)
            return m

        class Sample:
            class Status:
                ABORTED = "aborted"
                COMPLETED = "completed"
        cls.Sample = Sample

        class SingletonMeta(type):
            pass

        replacements = {
            "vime.agent.adapters": mod("vime.agent.adapters", AnthropicAdapter=object, ResponsesAdapter=object),
            "vime.agent.aiohttp_threaded": mod(
                "vime.agent.aiohttp_threaded", FilteredAccessLogger=object, run_app_in_thread=lambda *a, **k: None
            ),
            "vime.agent.harness": mod("vime.agent.harness", ClaudeCodeHarness=object, CodexHarness=object),
            "vime.utils.misc": mod("vime.utils.misc", SingletonMeta=SingletonMeta),
            "vime.utils.processing_utils": mod("vime.utils.processing_utils", load_tokenizer=lambda *a, **k: None),
            "vime.utils.types": mod("vime.utils.types", Sample=Sample),
        }
        with patch.dict(sys.modules, replacements), patch.dict(os.environ, {}, clear=True):
            cls.gen = importlib.import_module("examples.terminal_agent_rl.generate")

    async def test_preparation_permit_covers_create_install_workspace_and_releases(self):
        from contextlib import asynccontextmanager
        order = []

        @asynccontextmanager
        async def permit(sb):
            order.append("permit")
            yield
            order.append("release")

        class SB:
            async def __aenter__(self):
                order.append("create")
                return self

            async def __aexit__(self, *args):
                order.append("close")

        async def install(sb):
            order.append("install")

        async def workspace(*args):
            order.append("workspace")

        gen = self.gen
        with (
            patch.object(gen, "sandbox_preparation", permit),
            patch.object(gen, "create_sandbox", return_value=SB()),
            patch.object(gen, "HARNESS_CLS", return_value=SimpleNamespace(install_cli=install)),
            patch.object(tb, "prepare_workspace", workspace),
        ):
            async with gen.boot_agent_sandbox(IMAGE, "task", {"workdir": "/task"}):
                order.append("agent")
        self.assertEqual(order, ["permit", "create", "install", "workspace", "release", "agent", "close"])

    async def test_admission_timeout_does_not_requeue(self):
        from vime.agent.sandbox import SandboxAdmissionTimeout

        class AdmissionTimeoutContext:
            async def __aenter__(self):
                raise SandboxAdmissionTimeout("full")

            async def __aexit__(self, exc_type, exc, exc_tb):
                return False

        sb = SimpleNamespace(__aexit__=AsyncMock())
        with (
            patch.object(self.gen, "sandbox_preparation", return_value=AdmissionTimeoutContext()),
            patch.object(self.gen, "create_sandbox", return_value=sb) as create,
        ):
            with self.assertRaises(SandboxAdmissionTimeout):
                async with self.gen.boot_agent_sandbox(IMAGE, "task", {}):
                    pass
        create.assert_called_once()
        sb.__aexit__.assert_awaited_once()

    async def test_rollout_grades_after_agent_and_always_drops_session(self):
        from contextlib import asynccontextmanager
        gen = self.gen
        order = []

        @asynccontextmanager
        async def boot(*args):
            order.append("boot")
            yield object()
            order.append("close")

        async def run(*args, **kwargs):
            order.append("agent")
            return 0

        async def grade(*args, **kwargs):
            order.append("grade")
            return tb.EvalResult(1.0, "solved")

        adapter = SimpleNamespace(open_session=lambda *a, **k: None,
                                  finish_session=AsyncMock(return_value=[SimpleNamespace()]), drop_session=AsyncMock())
        sample = SimpleNamespace(
            prompt="visible", label="task", metadata={"image": IMAGE, "workdir": "/task"},
            session_id=None, index=0, group_index=0,
        )
        mapping = tm.TemplateMapping([tm.TemplateMappingEntry.from_record(environment(), location="test")])
        registry = vs.VerifierSpecRegistry([vs.VerifierSpec.from_record(verifier(), location="test")])
        with (
            patch.dict(os.environ, {"VIME_AGENT_SANDBOX_BACKEND": "harbor"}),
            patch.object(gen, "TEMPLATE_MAPPING", mapping),
            patch.object(gen, "VERIFIER_SPECS", registry),
            patch.object(
                gen, "_AdapterService",
                return_value=SimpleNamespace(adapter=adapter, max_context_len=100, adapter_url="http://model"),
            ),
            patch.object(gen, "boot_agent_sandbox", boot),
            patch.object(gen, "HARNESS_CLS", return_value=SimpleNamespace(run=run)),
            patch.object(tb, "run_evaluation", grade),
        ):
            await gen.generate(SimpleNamespace(), sample, {})
        self.assertEqual(order, ["boot", "agent", "grade", "close"])
        self.assertEqual(adapter.finish_session.call_args.kwargs["reward"], 1.0)
        adapter.drop_session.assert_awaited_once()
