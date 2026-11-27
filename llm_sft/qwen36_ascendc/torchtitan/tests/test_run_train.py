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

import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


LAUNCHER = Path(__file__).resolve().parents[1] / "run_train.sh"


class RunTrainTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_dir = TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.root = Path(temporary_dir.name)
        tools = self.root / "tools"
        tools.mkdir()
        (self.root / "assets").mkdir()
        (self.root / "first source.jsonl").write_text("{}\n", encoding="utf-8")
        self.output = self.root / "run"
        self.env = {
            "PATH": os.pathsep.join((str(tools), os.defpath)),
            "TRACE_DIR": str(self.root),
            "HF_ASSETS_PATH": "assets",
            "DUMP_FOLDER": str(self.output),
            "DATA_FILES": "first source.jsonl",
            "CANNBOT_TRAIN_STEPS": "10",
            "CANNBOT_CHECKPOINT_INTERVAL": "5",
        }
        python = tools / "python3"
        python.write_text(
            r"""#!/bin/bash
if [[ $1 != -m ]]; then
  printf '%s\0' "$@" > "$TRACE_DIR/python.args"
fi
""",
            encoding="utf-8",
        )
        python.chmod(0o755)
        torchrun = tools / "torchrun"
        torchrun.write_text(
            r"""#!/bin/bash
printf '%s\0' "$DATA_FILES" "$DATA_MANIFEST" \
  "$CANNBOT_TRAIN_STEPS" "$CANNBOT_CHECKPOINT_INTERVAL" "$@" \
  > "$TRACE_DIR/torchrun.args"
""",
            encoding="utf-8",
        )
        torchrun.chmod(0o755)

    def run_launcher(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", str(LAUNCHER), *args],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def arguments(self, command: str) -> list[str]:
        trace = (self.root / f"{command}.args").read_text(encoding="utf-8")
        return trace.removesuffix("\0").split("\0")

    def test_invalid_inputs_fail_before_preparing_data(self) -> None:
        cases = (
            ("DATA_FILES", ",first source.jsonl"),
            ("DATA_FILES", "first source.jsonl,"),
            ("DATA_FILES", "first source.jsonl,,first source.jsonl"),
            ("DATA_FILES", "first source.jsonl\nfirst source.jsonl"),
            ("DATA_FILES", "missing.jsonl"),
            ("DATA_FILES", "assets"),
            ("HF_ASSETS_PATH", "missing-model"),
            ("DUMP_FOLDER", str(self.root / "run,invalid")),
            ("CANNBOT_TRAIN_STEPS", "0"),
            ("CANNBOT_CHECKPOINT_INTERVAL", "1.5"),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                original = self.env[key]
                self.env[key] = value
                result = self.run_launcher()
                self.env[key] = original
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(key, result.stderr)
                self.assertFalse(self.output.exists())
                self.assertFalse((self.root / "run,invalid").exists())
                self.assertFalse((self.root / "python.args").exists())
                self.assertFalse((self.root / "torchrun.args").exists())

    def test_preserves_input_order_and_handles_option_prefixed_paths(self) -> None:
        (self.root / "-second.jsonl").write_text("{}\n", encoding="utf-8")
        (self.root / "-assets").mkdir()
        self.env.update(
            DATA_FILES="first source.jsonl,-second.jsonl",
            HF_ASSETS_PATH="-assets",
            DUMP_FOLDER="-run",
        )

        result = self.run_launcher()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.arguments("python"),
            [
                str(LAUNCHER.with_name("data_process.py")),
                "--data",
                "first source.jsonl",
                "./-second.jsonl",
                "--output",
                "./-run/data_manifest.json",
                "--merge-jsonl",
                "./-run/train.jsonl",
            ],
        )
        args = self.arguments("torchrun")
        self.assertEqual(
            args[:4], ["./-run/train.jsonl", "./-run/data_manifest.json", "10", "5"]
        )
        self.assertEqual(args[args.index("--hf-assets-path") + 1], "./-assets")
        self.assertEqual(args[args.index("--dump-folder") + 1], "./-run")

    def test_smoke_provides_step_counts(self) -> None:
        self.env["CANNBOT_SMOKE"] = "1"
        del self.env["CANNBOT_TRAIN_STEPS"]
        del self.env["CANNBOT_CHECKPOINT_INTERVAL"]

        result = self.run_launcher()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.arguments("torchrun")[2:4], ["3", "3"])

    def test_export_accepts_paths_without_training_environment(self) -> None:
        for name in ("HF_ASSETS_PATH", "DUMP_FOLDER", "DATA_FILES"):
            del self.env[name]

        result = self.run_launcher("export-hf", "a checkpoint", "-assets", "export")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.arguments("python"),
            [
                str(LAUNCHER.with_name("cannbot_recipe.py")),
                "export-hf",
                "--",
                "a checkpoint",
                "-assets",
                "export",
            ],
        )
        self.assertFalse((self.root / "torchrun.args").exists())
