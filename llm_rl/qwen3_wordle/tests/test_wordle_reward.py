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

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wordle_reward import _partial_answer, compute_score  # noqa: E402


class TestWordleReward(unittest.TestCase):
    def test_correct_answer_uses_actual_turn_count(self):
        result = compute_score(
            solution_str="<guess>[apple]</guess><guess>[crane]</guess>",
            ground_truth="crane",
            extra_info={"num_turns": 2, "formatted_turns": 2},
        )

        self.assertAlmostEqual(result["score"], 1.7)
        self.assertEqual(result["correct"], 1.0)
        self.assertEqual(result["partial"], 0.0)
        self.assertEqual(result["length_bonus"], 0.5)
        self.assertEqual(result["format"], 1.0)

    def test_partial_reward_keeps_best_guess(self):
        result = compute_score(
            solution_str="<guess>[crown]</guess><guess>[vivid]</guess>",
            ground_truth="crane",
            extra_info={"num_turns": 2, "formatted_turns": 2},
        )

        self.assertAlmostEqual(result["partial"], 0.5)
        self.assertAlmostEqual(result["score"], 0.7)

    def test_partial_reward_handles_duplicate_letters(self):
        self.assertAlmostEqual(_partial_answer(["allee"], "apple"), 0.5)

    def test_missing_formatted_turn_reduces_format_reward(self):
        result = compute_score(
            solution_str="<guess>[vivid]</guess>no guess in the second turn",
            ground_truth="crane",
            extra_info={"num_turns": 2, "formatted_turns": 1},
        )

        self.assertEqual(result["format"], 0.5)
        self.assertAlmostEqual(result["score"], 0.1)

    def test_noncanonical_guess_is_rejected(self):
        result = compute_score(
            solution_str="<guess>crane</guess>",
            ground_truth="crane",
            extra_info={"num_turns": 1, "formatted_turns": 0},
        )

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["correct"], 0.0)
        self.assertEqual(result["format"], 0.0)
        self.assertEqual(result["num_guesses"], 0)

    def test_agent_format_count_prevents_multiple_tag_hack(self):
        result = compute_score(
            solution_str="<guess>[apple]</guess><guess>[crane]</guess>",
            ground_truth="vivid",
            extra_info={"num_turns": 1, "formatted_turns": 0},
        )

        self.assertEqual(result["format"], 0.0)

    def test_invalid_multi_guess_turn_does_not_receive_win_reward(self):
        result = compute_score(
            solution_str="<guess>[apple]</guess><guess>[crane]</guess>",
            ground_truth="crane",
            extra_info={
                "num_turns": 1,
                "formatted_turns": 0,
                "is_correct": False,
                "guessed_words": [],
            },
        )

        self.assertEqual(result["correct"], 0.0)
        self.assertEqual(result["partial"], 0.0)
        self.assertEqual(result["length_bonus"], 0.0)
        self.assertEqual(result["score"], 0.0)

    def test_invalid_multi_guess_turn_does_not_receive_partial_reward(self):
        result = compute_score(
            solution_str="<guess>[apple]</guess><guess>[crane]</guess>",
            ground_truth="apple",
            extra_info={
                "num_turns": 1,
                "formatted_turns": 0,
                "is_correct": False,
                "guessed_words": [],
            },
        )

        self.assertEqual(result["partial"], 0.0)
        self.assertEqual(result["score"], 0.0)

    def test_empty_rollout_has_zero_reward(self):
        result = compute_score(solution_str="", ground_truth="crane")

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["num_guesses"], 0)


if __name__ == "__main__":
    unittest.main()
