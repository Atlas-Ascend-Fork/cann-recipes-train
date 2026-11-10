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

"""Resolve example configuration, then invoke an unchanged native train loop."""

from __future__ import annotations

import importlib
import os

from examples.cann_bench_single_turn.config import add_example_arguments, resolve_run_settings


WORKER_ENVIRONMENT_NAMES = (
    'PATH',
    'LD_LIBRARY_PATH',
    'LD_PRELOAD',
    'PYTHONPATH',
    'VIME_LAUNCHER_RUN_ID',
    'TRAIN_ENTRYPOINT',
    'CANN_EVAL_URL',
    'CANN_EVAL_EVAL_URL',
    'CANN_EVAL_ALIAS',
    'CANN_ARCHIVE_DIR',
    'CANN_ASYNC_EVALUATION',
    'VIME_CANN_THINKING_CLOSE_REWARD',
    'VIME_VLLM_EXECUTABLE',
    'VIME_VLLM_PYTHON',
    'ROLLOUT_ENV',
    'VIME_CANN_HOME',
    'ASCEND_HOME_PATH',
    'ASCEND_OPP_PATH',
    'ASCEND_AICPU_PATH',
    'ASCEND_TOOLKIT_HOME',
    'ASCEND_CUSTOM_OPP_PATH',
    'NNAL_HOME',
    'ATB_HOME_PATH',
    'ATB_OPSRUNNER_PATH',
    'CC',
    'CXX',
    'TRITON_CACHE_DIR',
    'TMPDIR',
    'TENSORBOARD_DIR',
    'VLLM_ASCEND_ENABLE_NZ',
    'VLLM_USE_AOT_COMPILE',
    'VLLM_ASCEND_NPUGRAPH_CAPTURE_LIMIT',
    'PYTHONUNBUFFERED',
    'PYTHONDONTWRITEBYTECODE',
    'RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES',
    'HCCL_WHITELIST_DISABLE',
    'HCCL_CONNECT_TIMEOUT',
    'HCCL_DETERMINISTIC',
    'HCCL_HOST_SOCKET_PORT_RANGE',
    'HCCL_NPU_SOCKET_PORT_RANGE',
    'PYTORCH_NPU_ALLOC_CONF',
    'CUDA_DEVICE_MAX_CONNECTIONS',
    'TASK_QUEUE_ENABLE',
    'STREAMS_PER_DEVICE',
    'MULTI_STREAM_MEMORY_RESERVE',
    'CPU_AFFINITY_CONF',
    'OMP_NUM_THREADS',
    'no_proxy',
    'NO_PROXY',
)


def worker_environment():
    """Forward the prepared NPU runtime, without copying credentials or device IDs.

    Matches the environment channel used by main's TorchTitan Qwen3.5 launcher.
    Training-only overrides still belong to --train-env-vars.
    """
    return {name: os.environ[name] for name in WORKER_ENVIRONMENT_NAMES if name in os.environ}


def main():
    # Runtime imports (for example cv2) may mutate library paths. Capture the
    # prepared worker environment for direct invocations before those imports.
    prepared_environment = worker_environment()
    # Register task-specific options through the native parser extension.
    from vime.utils.arguments import parse_args

    args = resolve_run_settings(parse_args(add_custom_arguments=add_example_arguments))
    entrypoint = os.environ.get("TRAIN_ENTRYPOINT", "train_async.py")
    if args.debug_rollout_only:
        entrypoint = "train.py"
    if entrypoint not in {"train.py", "train_async.py"}:
        raise ValueError("TRAIN_ENTRYPOINT must be train.py or train_async.py")

    import ray

    # Ray Jobs already supplied this environment. Ray 2.48 rejects duplicate
    # env keys even when their values agree; direct invocations still need it.
    ray_options = {"address": os.environ.get("RAY_ADDRESS", "auto")}
    if not os.environ.get("RAY_JOB_CONFIG_JSON_ENV_VAR"):
        ray_options["runtime_env"] = {"env_vars": prepared_environment}
    ray.init(**ray_options)
    evaluator = None
    if args.cann_single_turn["async_evaluation"]:
        from examples.cann_bench_single_turn.deferred_evaluation import start

        evaluator = start(args, entrypoint)
    try:
        importlib.import_module(entrypoint.removesuffix(".py")).train(args)
    finally:
        if evaluator is not None:
            from examples.cann_bench_single_turn.deferred_evaluation import stop

            stop(evaluator)


if __name__ == "__main__":
    main()
