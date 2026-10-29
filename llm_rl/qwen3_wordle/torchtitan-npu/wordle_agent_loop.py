# Adapted from verl's ToolAgentLoop implementation:
# https://github.com/volcengine/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Wordle multi-turn environment for verl's unified engine workers."""

import logging
import os
import re
from typing import Any

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.utils.rollout_trace import rollout_trace_op


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

CANONICAL_GUESS_RE = re.compile(r"<guess>\s*\[([A-Za-z]{5})\]\s*</guess>")
OPEN_GUESS_RE = re.compile(r"<guess\b", re.IGNORECASE)

DEFAULT_SYSTEM_PROMPT = (
    "You are a competitive game player. Make sure you read the game "
    "instructions carefully, and always follow the required format.\n\n"
    "In each turn, think step-by-step, then output your guess exactly as "
    "<guess>[word]</guess>."
)


class WordleGameState:
    """Mutable token and game state for one Wordle rollout."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        answer: str,
        max_turns: int,
        prompt_ids: list[int],
        request_id: str,
    ) -> None:
        self.messages = messages
        self.answer = answer.lower()
        self.max_turns = max_turns
        self.request_id = request_id
        self.prompt_ids = list(prompt_ids)
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn = 0
        self.formatted_turns = 0
        self.guessed_words: set[str] = set()
        self.game_over = False
        self.is_correct = False
        self.routed_experts = None

    def to_output(self, response_length: int) -> AgentLoopOutput:
        mask_length = len(self.response_mask)
        if mask_length:
            response_ids = self.prompt_ids[-mask_length:]
            prompt_ids = self.prompt_ids[:-mask_length]
        else:
            response_ids = []
            prompt_ids = list(self.prompt_ids)

        retained_length = min(response_length, len(response_ids))
        response_ids = response_ids[:retained_length]
        response_mask = self.response_mask[:retained_length]
        response_logprobs = (
            self.response_logprobs[:retained_length]
            if self.response_logprobs
            else None
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            num_turns=self.turn,
            metrics=AgentLoopMetrics(),
            routed_experts=self.routed_experts,
            extra_fields={
                "answer": self.answer,
                "num_turns": self.turn,
                "formatted_turns": self.formatted_turns,
                "is_correct": self.is_correct,
                "guessed_words": sorted(self.guessed_words),
            },
        )


class WordleAgentLoop(AgentLoopBase):
    """Run a six-turn Wordle game against an asynchronous vLLM server."""

    def __init__(self, *args, tools=None, **kwargs) -> None:
        del tools
        super().__init__(*args, **kwargs)
        self.max_turns = (
            getattr(self.rollout_config.multi_turn, "max_user_turns", 6) or 6
        )
        self.response_length = self.rollout_config.response_length

    @staticmethod
    def _compute_feedback(guess: str, answer: str) -> str:
        result: list[str | None] = []
        answer_characters: list[str | None] = list(answer)

        for index in range(5):
            if guess[index] == answer_characters[index]:
                result.append("G")
                answer_characters[index] = None
            else:
                result.append(None)

        for index in range(5):
            if result[index] is not None:
                continue
            if guess[index] in answer_characters:
                result[index] = "Y"
                matched_index = answer_characters.index(guess[index])
                answer_characters[matched_index] = None
            else:
                result[index] = "X"

        return " ".join(value for value in result if value is not None)

    @staticmethod
    def _has_canonical_format(text: str) -> bool:
        """Check that a response contains exactly one canonical guess tag."""
        return (
            len(CANONICAL_GUESS_RE.findall(text)) == 1
            and len(OPEN_GUESS_RE.findall(text)) == 1
        )

    @rollout_trace_op
    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        game = await self._create_game(kwargs)

        while not game.game_over:
            output = await self._generate_response(game, sampling_params)
            if output is None:
                game.game_over = True
                break
            await self._process_response(game, output)

        return game.to_output(self.response_length)

    async def _create_game(self, kwargs: dict[str, Any]) -> WordleGameState:
        messages = list(kwargs["raw_prompt"])
        answer = kwargs.get("answer", "")
        if not answer:
            raise ValueError("Wordle environment requires 'answer' in dataset")

        if not any(message.get("role") == "system" for message in messages):
            messages.insert(
                0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
            )

        prompt_ids = await self.apply_chat_template(
            messages, remove_system_prompt=False
        )
        game = WordleGameState(
            messages=messages,
            answer=answer,
            max_turns=self.max_turns,
            prompt_ids=prompt_ids,
            request_id=kwargs.get("request_id", ""),
        )

        return game

    async def _process_response(
        self, game: WordleGameState, output: Any
    ) -> None:
        guess = self._parse_guess(output.token_ids)
        game.turn += 1
        response_text = self.tokenizer.decode(
            output.token_ids, skip_special_tokens=True
        )
        game.formatted_turns += int(self._has_canonical_format(response_text))
        if guess is None:
            if game.turn >= game.max_turns:
                game.game_over = True
                return
            added = await self._add_env_message(
                game,
                "Invalid Move: No valid 5-letter guess found. "
                "Resubmit exactly as <guess>[word]</guess>.",
            )
            if not added:
                game.game_over = True
            return

        is_repeat = guess in game.guessed_words
        if not is_repeat:
            game.guessed_words.add(guess)

        if guess == game.answer:
            game.is_correct = True
            game.game_over = True
            return

        if game.turn >= game.max_turns:
            game.game_over = True
            return

        environment_message = self._build_environment_message(
            game, guess, is_repeat
        )
        game.messages.append({"role": "assistant", "content": response_text})
        if not await self._add_env_message(game, environment_message):
            game.game_over = True

    def _build_environment_message(
        self, game: WordleGameState, guess: str, is_repeat: bool
    ) -> str:
        if is_repeat:
            return (
                "You attempted an invalid move. Reason: You have already "
                f"guessed '{guess}' before. Please try a different word. "
                "Please resubmit a valid move and remember to follow the "
                "game rules."
            )

        feedback = self._compute_feedback(guess, game.answer)
        remaining_turns = game.max_turns - game.turn
        return (
            f"\n{' '.join(guess.upper())}\n{feedback}\n"
            f"You have {remaining_turns} guesses left."
        )

    async def _generate_response(
        self,
        game: WordleGameState,
        sampling_params: dict[str, Any],
    ) -> Any:
        remaining_tokens = self.response_length - len(game.response_mask)
        if remaining_tokens <= 0:
            return None

        turn_params = dict(sampling_params)
        turn_params["max_tokens"] = min(512, remaining_tokens)
        try:
            output = await self.server_manager.generate(
                request_id=game.request_id,
                prompt_ids=game.prompt_ids,
                sampling_params=turn_params,
            )
        except Exception:
            logger.exception(
                "Wordle generation failed for %s", game.request_id
            )
            return None

        if output is None or not output.token_ids:
            return None

        game.prompt_ids += output.token_ids
        game.response_mask += [1] * len(output.token_ids)
        if output.log_probs:
            game.response_logprobs += output.log_probs
        if output.routed_experts is not None:
            game.routed_experts = output.routed_experts
        return output

    def _parse_guess(self, token_ids: list[int]) -> str | None:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        matches = CANONICAL_GUESS_RE.findall(text)
        if len(matches) != 1 or len(OPEN_GUESS_RE.findall(text)) != 1:
            return None

        return matches[0].lower()

    async def _add_env_message(
        self, game: WordleGameState, content: str
    ) -> bool:
        eos_id = self.tokenizer.eos_token_id
        transition_text = ""
        if not game.prompt_ids or game.prompt_ids[-1] != eos_id:
            transition_text += "<|im_end|>"
        transition_text += (
            "\n<|im_start|>user\n"
            + content
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        transition_ids = self.tokenizer.encode(
            transition_text, add_special_tokens=False
        )

        if len(game.response_mask) + len(transition_ids) >= self.response_length:
            return False

        game.prompt_ids += transition_ids
        game.response_mask += [0] * len(transition_ids)
        if game.response_logprobs:
            game.response_logprobs += [0.0] * len(transition_ids)
        game.messages.append({"role": "user", "content": content})
        return True
