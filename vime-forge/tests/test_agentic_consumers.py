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

"""Shared launch settings reach real task and sandbox consumers without services."""

import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from vime.agent.harbor_sandbox import HarborConfig
from vime.agent.sandbox import E2BSandbox


def _module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


def _task_modules(env):
    """Only replace model/server imports; configuration and task code stay real."""
    class SingletonMeta(type):
        pass

    class ClaudeHarness:
        pass

    class CodexHarness:
        pass

    replacements = {
        "vime.agent.adapters": _module(
            "vime.agent.adapters", AnthropicAdapter=object, ResponsesAdapter=object,
        ),
        "vime.agent.adapters.common": _module(
            "vime.agent.adapters.common", flatten_content=lambda value: value,
        ),
        "vime.agent.aiohttp_threaded": _module(
            "vime.agent.aiohttp_threaded", FilteredAccessLogger=object,
            run_app_in_thread=lambda *args, **kwargs: None,
        ),
        "vime.agent.harness": _module(
            "vime.agent.harness", ClaudeCodeHarness=ClaudeHarness, CodexHarness=CodexHarness,
        ),
        "vime.utils.misc": _module("vime.utils.misc", SingletonMeta=SingletonMeta),
        "vime.utils.processing_utils": _module(
            "vime.utils.processing_utils", load_tokenizer=lambda *args, **kwargs: None,
        ),
        "vime.utils.types": _module("vime.utils.types", Sample=object),
    }
    names = (
        "examples.swe_agent_rl.swe",
        "examples.terminal_agent_rl.terminalbench",
        "examples.swe_agent_rl.generate",
        "examples.terminal_agent_rl.generate",
    )
    missing = object()
    saved_attributes = []
    for name in {*names, *replacements}:
        parent_name, _, attribute = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            saved_attributes.append((parent, attribute, getattr(parent, attribute, missing)))
    try:
        with patch.dict(sys.modules, replacements), patch.dict(os.environ, env, clear=True):
            # Execute module-level settings again even if another test imported them.
            for name in names:
                sys.modules.pop(name, None)
            return tuple(importlib.import_module(name) for name in names)
    finally:
        # Importlib also installs children on parent packages, outside sys.modules.
        for parent, attribute, previous in saved_attributes:
            if previous is missing:
                parent.__dict__.pop(attribute, None)
            else:
                setattr(parent, attribute, previous)


class TaskConsumerTests(unittest.TestCase):
    def test_both_tasks_use_canonical_agent_prompt_and_budgets(self):
        env = dict(
            VIME_RL_AGENT="codex", VIME_RL_PROMPT="shared instruction",
            VIME_RL_AGENT_TIME_BUDGET_SEC="2500", VIME_RL_EVAL_TIMEOUT_SEC="700",
            VIME_RL_ROLLOUT_GUARD_SEC="4100", VIME_RL_BOOT_CONCURRENCY="7",
            VIME_RL_BOOT_RETRIES="4",
        )
        swe, terminalbench, swe_gen, terminal_gen = _task_modules(env)
        self.assertEqual(swe.SWE_PROMPT, "shared instruction")
        self.assertEqual(terminalbench.TB_PROMPT, "shared instruction")
        for gen in (swe_gen, terminal_gen):
            with self.subTest(module=gen.__name__):
                self.assertEqual(gen.AGENT_NAME, "codex")
                self.assertIs(gen.HARNESS_CLS, gen.CodexHarness)
                self.assertEqual(gen.CONFIG.agent_time_budget_sec, 2500)
                self.assertEqual(gen.CONFIG.eval_timeout_sec, 700)
                self.assertEqual(gen.CONFIG.rollout_guard_sec, 4100)
                self.assertEqual(gen.CONFIG.boot_concurrency, 7)
                self.assertEqual(gen.CONFIG.boot_retries, 4)

    def test_common_boot_snapshot_reaches_both_generators(self):
        snapshot = json.dumps({"backend": "harbor", "environment": {
            "VIME_AGENT_SANDBOX_BACKEND": "harbor", "VIME_SANDBOX_BOOT_CONCURRENCY": "4",
            "VIME_SANDBOX_BOOT_RETRIES": "3",
            "VIME_HARBOR_HOSTS": json.dumps([dict(ssh="root@cpu", python="/opt/python",
                worker="/opt/worker.py", root="/opt/state", slots=8)]),
        }})
        _, _, swe_gen, terminal_gen = _task_modules({
            "VIME_AGENT_SANDBOX_BACKEND": "harbor", "VIME_INTERNAL_SANDBOX_CONFIG": snapshot,
            "VIME_AGENT_SANDBOX_CONFIG": "/unavailable/on-worker.json", "VIME_RL_BOOT_CONCURRENCY": "99",
        })
        for gen in (swe_gen, terminal_gen):
            self.assertEqual(gen.CONFIG.boot_concurrency, 4)
            self.assertEqual(gen.CONFIG.boot_retries, 3)

    def test_standalone_defaults_remain_task_specific(self):
        _, _, swe_gen, terminal_gen = _task_modules({})
        for gen in (swe_gen, terminal_gen):
            self.assertEqual(gen.AGENT_NAME, "claude_code")
            self.assertEqual(gen.CONFIG.agent_time_budget_sec, 1800)
        self.assertEqual(swe_gen.CONFIG.eval_timeout_sec, 600)
        self.assertEqual(terminal_gen.CONFIG.eval_timeout_sec, 900)
        self.assertEqual(swe_gen.CONFIG.rollout_guard_sec, 2580)
        self.assertEqual(terminal_gen.CONFIG.rollout_guard_sec, 2880)

    def test_legacy_prompt_and_agent_inputs_remain_supported(self):
        swe, terminalbench, swe_gen, terminal_gen = _task_modules(dict(
            SWE_AGENT="codex", TB_AGENT="claude_code",
            SWE_CC_PROMPT="swe instruction", TB_PROMPT="terminal instruction",
        ))
        self.assertEqual(swe.SWE_PROMPT, "swe instruction")
        self.assertEqual(terminalbench.TB_PROMPT, "terminal instruction")
        self.assertEqual(swe_gen.AGENT_NAME, "codex")
        self.assertEqual(terminal_gen.AGENT_NAME, "claude_code")

    def test_canonical_bundle_prevents_inline_verifier_fallback(self):
        swe, *_ = _task_modules({})
        with patch.dict(os.environ, {"VIME_RL_PREPARED_BUNDLE": "/bundle"}, clear=True):
            with self.assertRaisesRegex(ValueError, "missing_verifier_spec"):
                swe._resolve_verifier_spec({"grading": {}})
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(swe._resolve_verifier_spec({"grading": {}}))


class SandboxConsumerTests(unittest.TestCase):
    def test_selected_profile_reaches_both_adapters_without_credentials(self):
        host = dict(ssh="root@cpu", python="/opt/python", worker="/opt/worker.py",
                    root="/opt/state", slots=8)
        profile = {
            "common": {"lifetime_sec": 9000},
            "agentenv": {"api_url": "http://agentenv", "sandbox_url": "http://sandbox",
                         "api_key_file": "/does-not-exist/key", "selector_mode": "template",
                         "rpc_retries": 5, "size": "lg"},
            "harbor": {"hosts": [host], "cpus": 8, "memory_mb": 16384,
                       "prepare_concurrency": 2, "prepare_queue_limit": 20},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sandboxes.json"
            path.write_text(json.dumps(profile))
            env = {"VIME_AGENT_SANDBOX_CONFIG": str(path), "VIME_AGENT_SANDBOX_BACKEND": "agentenv"}
            with patch.dict(os.environ, env, clear=True):
                sandbox = E2BSandbox("template")
                self.assertEqual(sandbox.timeout, 9000)
                self.assertEqual(sandbox.selector_mode, "template")
                self.assertEqual(sandbox.rpc_retries, 5)
                self.assertEqual(sandbox.size, "lg")
                self.assertNotIn("E2B_API_KEY", os.environ)
            with patch.dict(os.environ, {**env, "VIME_AGENT_SANDBOX_BACKEND": "harbor"}, clear=True):
                config = HarborConfig.from_env()
                self.assertEqual(config.hosts, (host,))
                self.assertEqual(config.lifetime, 9000)
                self.assertEqual(config.cpus, 8)
                self.assertEqual(config.memory_mb, 16384)
                self.assertEqual(config.prepare_concurrency, 2)
                self.assertEqual(config.prepare_queue_limit, 20)
                self.assertNotIn("E2B_API_KEY", os.environ)


class AgentEnvSDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_gets_profile_urls_without_reading_credential_file(self):
        profile = {"agentenv": {
            "api_url": "http://agentenv", "sandbox_url": "http://sandbox",
            "selector_mode": "template", "api_key_file": "/does-not-exist/key",
        }}
        create = AsyncMock(return_value=SimpleNamespace(sandbox_id="fake-sandbox"))
        sdk = _module("e2b", AsyncSandbox=SimpleNamespace(create=create))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sandboxes.json"
            path.write_text(json.dumps(profile))
            with patch.dict(os.environ, {"VIME_AGENT_SANDBOX_CONFIG": str(path)}, clear=True), \
                    patch.dict(sys.modules, {"e2b": sdk}):
                sandbox = E2BSandbox("template-id")
                self.assertIs(await sandbox.__aenter__(), sandbox)
                create.assert_awaited_once_with(template="template-id", timeout=3600)
                self.assertEqual(os.environ["E2B_API_URL"], "http://agentenv")
                self.assertEqual(os.environ["E2B_SANDBOX_URL"], "http://sandbox")
                self.assertNotIn("E2B_API_KEY", os.environ)


if __name__ == "__main__":
    unittest.main()
