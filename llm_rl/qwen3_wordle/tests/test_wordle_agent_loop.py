# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wordle_reward import compute_score  # noqa: E402


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _identity_decorator(_name):
    def decorator(loop_type):
        return loop_type

    return decorator


def _identity(function):
    return function


def _stub_modules():
    agent_api = types.ModuleType("verl.experimental.agent_loop.agent_loop")

    class AgentLoopBase:
        pass

    class AgentLoopMetrics:
        pass

    class AgentLoopOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    agent_api.AgentLoopBase = AgentLoopBase
    agent_api.AgentLoopMetrics = AgentLoopMetrics
    agent_api.AgentLoopOutput = AgentLoopOutput
    agent_api.register = _identity_decorator

    rollout_trace = types.ModuleType("verl.utils.rollout_trace")
    rollout_trace.rollout_trace_op = _identity
    return {
        "verl": _package("verl"),
        "verl.experimental": _package("verl.experimental"),
        "verl.experimental.agent_loop": _package("verl.experimental.agent_loop"),
        agent_api.__name__: agent_api,
        "verl.utils": _package("verl.utils"),
        rollout_trace.__name__: rollout_trace,
    }


def _load_wordle_agent_loop():
    modules = _stub_modules()
    path = Path(__file__).resolve().parents[1] / "torchtitan-npu" / "wordle_agent_loop.py"
    spec = importlib.util.spec_from_file_location("_wordle_agent_loop_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load wordle_agent_loop.py")

    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def _load_base_wordle_agent_loop():
    patch_path = (
        Path(__file__).resolve().parents[1]
        / "patches"
        / "0001-wordle-agent-loop.patch"
    )
    patch_text = patch_path.read_text()
    section = patch_text.split(
        "diff --git a/verl/experimental/agent_loop/wordle_agent_loop.py "
        "b/verl/experimental/agent_loop/wordle_agent_loop.py\n",
        1,
    )[1]
    hunk_start = section.index("@@ -0,0 +1,")
    source_start = section.index("\n", hunk_start) + 1
    source_lines = []
    for line in section[source_start:].splitlines():
        if line == "-- ":
            break
        if not line.startswith("+"):
            raise AssertionError(f"unexpected line in added source: {line!r}")
        source_lines.append(line[1:])

    modules = _stub_modules()
    module = types.ModuleType("_base_wordle_agent_loop_test")
    module.__file__ = str(patch_path)
    with patch.dict(sys.modules, modules):
        exec(compile("\n".join(source_lines), str(patch_path), "exec"), module.__dict__)
    return module


class _Tokenizer:
    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        return token_ids[0]


class TestWordleAgentLoopParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_wordle_agent_loop()
        cls.agent_loop_type = cls.module.WordleAgentLoop

    def test_rejects_multiple_canonical_guess_tags(self):
        self.assertIsNone(self._parse("<guess>[apple]</guess><guess>[crane]</guess>"))

    def test_accepts_one_canonical_guess_tag(self):
        self.assertEqual(self._parse("<guess>[crane]</guess>"), "crane")

    def test_final_turn_guess_is_recorded_and_reaches_reward(self):
        game = self.module.WordleGameState(
            messages=[],
            answer="crane",
            max_turns=6,
            prompt_ids=[],
            request_id="test",
        )
        game.turn = 5
        game.guessed_words.add("vivid")

        agent_loop = self.agent_loop_type.__new__(self.agent_loop_type)
        agent_loop.tokenizer = _Tokenizer()
        asyncio.run(
            getattr(agent_loop, "_process_response")(
                game,
                SimpleNamespace(token_ids=["<guess>[crown]</guess>"]),
            )
        )

        self.assertTrue(game.game_over)
        self.assertEqual(game.turn, 6)
        self.assertEqual(game.guessed_words, {"vivid", "crown"})
        result = compute_score(
            solution_str="<guess>[vivid]</guess><guess>[crown]</guess>",
            ground_truth=game.answer,
            extra_info={
                "num_turns": game.turn,
                "formatted_turns": game.formatted_turns,
                "is_correct": game.is_correct,
                "guessed_words": sorted(game.guessed_words),
            },
        )
        self.assertAlmostEqual(result["partial"], 0.5)
        self.assertEqual(result["correct"], 0.0)

    def test_base_loop_respects_remaining_budget_before_awarding_win(self):
        module = _load_base_wordle_agent_loop()
        loop = module.WordleAgentLoop.__new__(module.WordleAgentLoop)
        loop.max_turns = 6
        loop.response_length = 1
        loop.prompt_length = 8
        loop.tokenizer = _BudgetTokenizer()
        loop.server_manager = _BudgetServerManager()

        async def apply_chat_template(_messages, remove_system_prompt=False):
            return [99]

        loop.apply_chat_template = apply_chat_template
        result = asyncio.run(
            loop.run(
                sampling_params={"temperature": 0.0},
                raw_prompt=[],
                answer="crane",
                request_id="budget-test",
            )
        )

        self.assertEqual(loop.server_manager.calls, 1)
        self.assertEqual(loop.server_manager.max_tokens, 1)
        self.assertEqual(result.response_ids, [1])
        self.assertFalse(result.extra_fields["is_correct"])
        reward = compute_score(
            solution_str="<guess>[",
            ground_truth="crane",
            extra_info=result.extra_fields,
        )
        self.assertEqual(reward["correct"], 0.0)
        self.assertEqual(reward["score"], 0.0)

    def _parse(self, response: str) -> str | None:
        agent_loop = self.agent_loop_type.__new__(self.agent_loop_type)
        agent_loop.tokenizer = _Tokenizer()
        return getattr(agent_loop, "_parse_guess")([response])


class _BudgetTokenizer:
    eos_token_id = 99
    responses = {1: "<guess>[", 2: "crane]</guess>"}

    @staticmethod
    def encode(_text, add_special_tokens=False):
        return [10, 11]

    @classmethod
    def decode(cls, token_ids, skip_special_tokens=True):
        return "".join(cls.responses[token] for token in token_ids)


class _BudgetServerManager:
    def __init__(self):
        self.calls = 0
        self.max_tokens = None

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.calls += 1
        self.max_tokens = sampling_params["max_tokens"]
        token_ids = [1, 2][: self.max_tokens]
        return SimpleNamespace(
            token_ids=token_ids,
            log_probs=[],
            routed_experts=None,
        )

if __name__ == "__main__":
    unittest.main()
