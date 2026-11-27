# Copyright (c) 2026 The CANN Contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_sft.qwen36_ascendc.torchtitan.data_process import merge_jsonl


class MergeJsonlTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_dir = TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.root = Path(temporary_dir.name)
        self.output = self.root / "train.jsonl"

    def test_inserts_only_missing_boundary_newlines(self) -> None:
        first = self.root / "first.jsonl"
        second = self.root / "second.jsonl"
        third = self.root / "third.jsonl"
        empty = self.root / "empty.jsonl"
        first_record = '{"input":"算子","output":"a"}'.encode("utf-8")
        second_record = b'{"prompt":"q","response":"b"}\r\n'
        third_record = b'{"input":"q","output":"c"}'
        first.write_bytes(first_record)
        second.write_bytes(second_record)
        third.write_bytes(third_record)
        empty.touch()

        merge_jsonl([empty, first, empty, second, third, empty], self.output)

        self.assertEqual(
            self.output.read_bytes(),
            first_record + b"\n" + second_record + third_record,
        )

    def test_rejects_paths_that_alias_an_input(self) -> None:
        source = self.root / "input.jsonl"
        record = b'{"input":"q","output":"a"}'
        source.write_bytes(record)
        symlink = self.root / "symlink.jsonl"
        symlink.symlink_to(source)
        hardlink = self.root / "hardlink.jsonl"
        hardlink.hardlink_to(source)

        for output in (source, symlink, hardlink):
            with self.subTest(output=output.name):
                with self.assertRaisesRegex(ValueError, "must differ"):
                    merge_jsonl([source], output)
        self.assertEqual(source.read_bytes(), record)

    def test_failed_merge_preserves_existing_output(self) -> None:
        source = self.root / "input.jsonl"
        source.write_bytes(b'{"input":"q","output":"a"}')
        previous = b'{"input":"q","output":"previous"}\n'
        self.output.write_bytes(previous)

        with self.assertRaises(FileNotFoundError):
            merge_jsonl([source, self.root / "missing.jsonl"], self.output)

        self.assertEqual(self.output.read_bytes(), previous)
