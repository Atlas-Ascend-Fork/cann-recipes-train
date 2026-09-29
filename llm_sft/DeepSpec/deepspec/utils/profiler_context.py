# coding=utf-8
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

"""NPU profiler managers for inference and training.

Disabled profilers are no-ops and do not require torch_npu. Training schedules
advance once per completed optimizer step.
"""

import importlib
import json
import os
import subprocess
from pathlib import Path


class FakeContextManager:
    """Provide the profiler interface without collecting a trace."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    @staticmethod
    def start():
        return

    @staticmethod
    def step():
        return

    @staticmethod
    def stop():
        return


def _load_torch_npu_profiler():
    try:
        torch_npu = importlib.import_module("torch_npu")
        return torch_npu, torch_npu.profiler
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "NPU profiling is enabled, but torch_npu.profiler is unavailable."
        ) from exc


def create_profiler(
    enable_profiler=False,
    profile_save_path="prof",
    active=3,
    repeat=1,
    skip_first=3,
):
    if not enable_profiler:
        return FakeContextManager()

    _, profiler_api = _load_torch_npu_profiler()
    os.makedirs(profile_save_path, exist_ok=True)
    experimental_config = profiler_api._ExperimentalConfig(
        profiler_level=profiler_api.ProfilerLevel.Level1,
        aic_metrics=profiler_api.AiCMetrics.PipeUtilization,
    )
    return profiler_api.profile(
        activities=[
            profiler_api.ProfilerActivity.NPU,
            profiler_api.ProfilerActivity.CPU,
        ],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        experimental_config=experimental_config,
        schedule=profiler_api.schedule(
            wait=0,
            warmup=0,
            active=active,
            repeat=repeat,
            skip_first=skip_first,
        ),
        on_trace_ready=profiler_api.tensorboard_trace_handler(profile_save_path),
    )


class ProfilerManager:
    """Manage separate prefill and decode traces during inference."""

    def __init__(self, enable_profiler, profile_save_path):
        self.enable_profiler = enable_profiler
        self.profile_save_path = profile_save_path
        self.status = None
        self.current_profiler = FakeContextManager()

    def set_status(self, is_prefill):
        if not self.enable_profiler:
            return

        if self.status is None and is_prefill:
            self.status = "prefill"
            self.current_profiler = create_profiler(
                enable_profiler=self.enable_profiler,
                profile_save_path=os.path.join(self.profile_save_path, "prof", "prefill"),
                active=1,
                repeat=1,
                skip_first=0,
            )
            self.current_profiler.start()
        elif self.status == "prefill" and not is_prefill:
            self.status = "decode"
            self.current_profiler.stop()
            self.current_profiler = create_profiler(
                enable_profiler=self.enable_profiler,
                profile_save_path=os.path.join(self.profile_save_path, "prof", "decode"),
            )
            self.current_profiler.start()

    def step(self):
        self.current_profiler.step()

    def close(self):
        if self.enable_profiler and self.status is not None:
            self.current_profiler.stop()
            self.status = None

    def __del__(self):
        # Destructors may run during shutdown or partial initialization.
        try:
            self.close()
        except Exception:
            pass


def _int_field(config, name, minimum):
    value = getattr(config, name, None)
    if isinstance(value, bool):
        raise ValueError(f"train.npu_profiling.{name} must be an integer")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"train.npu_profiling.{name} must be an integer"
        ) from exc
    if value < minimum:
        raise ValueError(
            f"train.npu_profiling.{name} must be >= {minimum}, got {value}"
        )
    return value


def validate_training_profiler_config(config):
    """Validate the complete training profiling configuration."""
    required = (
        "enabled",
        "wait",
        "warmup",
        "active",
        "repeat",
        "skip_first",
        "output_dir",
        "with_stack",
        "record_shapes",
        "profile_memory",
    )
    missing = [name for name in required if not hasattr(config, name)]
    if missing:
        raise ValueError(
            "train.npu_profiling is missing fields: " + ", ".join(missing)
        )
    if not isinstance(config.enabled, bool):
        raise ValueError("train.npu_profiling.enabled must be a boolean")
    if config.output_dir is not None and not isinstance(
        config.output_dir, (str, os.PathLike)
    ):
        raise ValueError("train.npu_profiling.output_dir must be a path or null")

    result = {
        "enabled": config.enabled,
        "wait": _int_field(config, "wait", 0),
        "warmup": _int_field(config, "warmup", 0),
        "active": _int_field(config, "active", 1),
        "repeat": _int_field(config, "repeat", 1),
        "skip_first": _int_field(config, "skip_first", 0),
        "output_dir": config.output_dir,
    }
    for name in ("with_stack", "record_shapes", "profile_memory"):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"train.npu_profiling.{name} must be a boolean")
        result[name] = getattr(config, name)
    return result


def _git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


class TrainingProfilerManager:
    """Collect a per-rank training trace with optimizer-step scheduling."""

    def __init__(
        self,
        config,
        *,
        checkpoint_dir,
        rank,
        world_size,
        experiment_name,
        device_type,
    ):
        self.config = validate_training_profiler_config(config)
        self.enabled = self.config["enabled"]
        self.profiler = FakeContextManager()
        self.started = False
        self.output_dir = None
        if not self.enabled:
            return
        if device_type != "npu":
            raise RuntimeError(
                "train.npu_profiling.enabled=true requires an NPU runtime; "
                f"current device type is {device_type!r}."
            )

        torch_npu, profiler_api = _load_torch_npu_profiler()
        root = self.config["output_dir"] or os.path.join(checkpoint_dir, "profiling")
        self.output_dir = os.path.abspath(os.path.join(root, f"rank_{rank}"))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._write_metadata(
            rank=rank,
            world_size=world_size,
            experiment_name=experiment_name,
            torch_npu_version=getattr(torch_npu, "__version__", None),
        )
        self.profiler = self._create_profiler(profiler_api)

    def _write_metadata(self, *, rank, world_size, experiment_name, torch_npu_version):
        metadata = {
            "world_size": int(world_size),
            "rank": int(rank),
            "experiment_name": str(experiment_name),
            "schedule_unit": "optimizer_step",
            "schedule": {
                name: self.config[name]
                for name in ("wait", "warmup", "active", "repeat", "skip_first")
            },
            "options": {
                name: self.config[name]
                for name in ("with_stack", "record_shapes", "profile_memory")
            },
            "git_revision": _git_revision(),
            "torch_version": importlib.import_module("torch").__version__,
            "torch_npu_version": torch_npu_version,
        }
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _create_profiler(self, profiler_api):
        return profiler_api.profile(
            activities=[
                profiler_api.ProfilerActivity.NPU,
                profiler_api.ProfilerActivity.CPU,
            ],
            with_stack=self.config["with_stack"],
            record_shapes=self.config["record_shapes"],
            profile_memory=self.config["profile_memory"],
            schedule=profiler_api.schedule(
                wait=self.config["wait"],
                warmup=self.config["warmup"],
                active=self.config["active"],
                repeat=self.config["repeat"],
                skip_first=self.config["skip_first"],
            ),
            on_trace_ready=profiler_api.tensorboard_trace_handler(self.output_dir),
        )

    def start(self):
        if self.enabled and not self.started:
            self.profiler.start()
            self.started = True

    def step(self):
        if self.started:
            self.profiler.step()

    def close(self):
        if self.started:
            self.started = False
            self.profiler.stop()

    def close_best_effort(self):
        """Release profiling resources without masking a training exception."""
        try:
            self.close()
        except Exception as exc:
            print(f"Training NPU profiler cleanup failed: {exc!r}", flush=True)
