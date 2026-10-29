# coding=utf-8
# Adapted from Prime Intellect's Verifiers Wordle environment at:
# https://github.com/PrimeIntellect-ai/verifiers/blob/main/environments/wordle/wordle.py
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

"""Wordle reward function for verl GRPO training.

This module provides ``compute_score``, compatible with verl's
``custom_reward_function`` interface. It is invoked per-rollout with the
generated solution_str and dataset fields.

Reward Components:
  - correct_answer (1.0): exact match with secret word
  - partial_answer (0.0-0.8): partial credit per green/yellow letter
  - length_bonus (0.0-1.0): bonus for shorter correct solutions
  - format_reward (0.0-1.0): fraction of turns using <guess>[word]</guess>

The final reward uses the same four-part shape as the Verifiers Wordle rubric.
Unlike the upstream environment, this recipe intentionally keeps the best
partial guess in the trajectory as a denser shaping signal.
"""

import re


CANONICAL_GUESS_RE = re.compile(r"<guess>\s*\[([A-Za-z]{5})\]\s*</guess>")
OPEN_GUESS_RE = re.compile(r"<guess\b", re.IGNORECASE)


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: str = "",
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, float | int]:
    """Compute Wordle reward for a single rollout (verl-compatible signature).

    Args:
        data_source: Dataset identifier.
        solution_str: The model's decoded completion text.
        ground_truth: The secret word.
        extra_info: Additional fields from agent loop.

    Returns:
        A dictionary containing the total reward and individual components.
    """
    answer = ground_truth.lower().strip()
    guesses = _extract_guesses(solution_str)
    loop_info = extra_info or {}

    loop_is_correct = loop_info.get("is_correct")
    if isinstance(loop_is_correct, bool):
        is_correct = loop_is_correct
    else:
        is_correct = bool(guesses and guesses[-1] == answer)

    # 1. Correct answer reward
    correct_reward = 1.0 if is_correct else 0.0

    # 2. Partial answer reward (best match among guesses)
    accepted_guesses = loop_info.get("guessed_words")
    if not is_correct and isinstance(accepted_guesses, (list, tuple, set)):
        partial_guesses = [
            guess.lower() for guess in accepted_guesses if isinstance(guess, str)
        ]
    else:
        partial_guesses = guesses
    partial = _partial_answer(partial_guesses, answer) if not is_correct else 0.0

    # The agent loop records every consumed turn, including malformed replies.
    # Falling back to observed opening tags keeps this function usable in
    # standalone tests and with older verl versions.
    num_turns = _get_num_turns(solution_str, extra_info, len(guesses))

    # 3. Length bonus (fewer consumed turns is better)
    length_bonus = 1.0 / max(num_turns, 1) if is_correct else 0.0

    # 4. Format reward: fraction of actual turns using the canonical format.
    # Weight 0.2 matches the Verifiers Wordle rubric.
    canonical_turns = _get_formatted_turns(solution_str, extra_info, num_turns)
    format_reward = min(canonical_turns / num_turns, 1.0) if num_turns else 0.0

    total = correct_reward + partial + length_bonus + 0.2 * format_reward

    return {
        "score": total,
        "correct": correct_reward,
        "partial": partial,
        "length_bonus": length_bonus,
        "format": format_reward,
        "num_guesses": len(guesses),
    }


def _extract_guesses(text: str) -> list[str]:
    """Extract canonical ``<guess>[word]</guess>`` guesses in order."""
    return [word.lower() for word in CANONICAL_GUESS_RE.findall(text)]


def _get_num_turns(text: str, extra_info: dict | None, num_guesses: int) -> int:
    """Return the number of turns consumed by the agent loop."""
    raw_num_turns = (extra_info or {}).get("num_turns")
    if raw_num_turns is not None and not isinstance(raw_num_turns, bool):
        try:
            num_turns = int(raw_num_turns)
        except (TypeError, ValueError, OverflowError):
            num_turns = 0
        if num_turns > 0:
            return num_turns

    return max(len(OPEN_GUESS_RE.findall(text)), num_guesses)


def _get_formatted_turns(
    text: str, extra_info: dict | None, num_turns: int
) -> int:
    """Return turns containing exactly one canonical guess response."""
    raw_formatted_turns = (extra_info or {}).get("formatted_turns")
    if raw_formatted_turns is not None and not isinstance(raw_formatted_turns, bool):
        try:
            formatted_turns = int(raw_formatted_turns)
        except (TypeError, ValueError, OverflowError):
            formatted_turns = -1
        if formatted_turns >= 0:
            return min(formatted_turns, num_turns)

    # Older verl versions do not propagate AgentLoopOutput.extra_fields to the
    # reward function. In that case, fall back to counting canonical tags.
    return min(len(CANONICAL_GUESS_RE.findall(text)), num_turns)


def _partial_answer(guesses: list[str], answer: str) -> float:
    """Best partial credit among all guesses."""
    if not guesses:
        return 0.0
    best = 0.0
    for guess in guesses:
        if len(guess) != len(answer):
            continue
        greens = sum(1 for i in range(len(answer)) if guess[i] == answer[i])
        yellows = 0
        ans_chars = list(answer)
        gue_chars = list(guess)
        # Mark greens first
        for i in range(len(answer)):
            if gue_chars[i] == ans_chars[i]:
                ans_chars[i] = None
                gue_chars[i] = None
        # Count yellows
        for i in range(len(answer)):
            if gue_chars[i] is not None and gue_chars[i] in ans_chars:
                yellows += 1
                ans_chars[ans_chars.index(gue_chars[i])] = None
        score = 0.2 * greens + 0.1 * yellows
        if score > best:
            best = score
    return best
