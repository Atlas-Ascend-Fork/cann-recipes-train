"""Device-specific launch policy for VIME tests and examples."""

import json
import os
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field


def _never() -> bool:
    return False


@dataclass(frozen=True)
class Platform:
    name: str
    ray_args: str
    env: dict = field(default_factory=dict)
    unsupported_features: frozenset = frozenset()
    torch_dist_convert: bool = True
    detect: Callable[[], bool] = _never

    def ray_start_args(self, num_devices: int) -> str:
        return self.ray_args.format(n=num_devices)


PLATFORMS: dict[str, Platform] = {}


def register(platform: Platform) -> None:
    PLATFORMS[platform.name] = platform


def registered_platforms() -> list[Platform]:
    return list(PLATFORMS.values())


def _detect_npu() -> bool:
    try:
        from vime.utils import accelerator

        return accelerator.is_available() and accelerator.device_type() == "npu"
    except (ImportError, RuntimeError):
        return False


register(Platform(name="cuda", ray_args="--num-gpus {n}"))

register(
    Platform(
        name="npu",
        ray_args="--num-gpus 0 --resources '{{\"NPU\": {n}}}'",
        detect=_detect_npu,
        torch_dist_convert=False,
        unsupported_features=frozenset({"deepep", "fp8_rollout"}),
        env={
            "PYTHONPATH": (
                "/root/Megatron-LM:/root/vime:"
                "/root/Megatron-Bridge/src:/root/mbridge:"
                "/root/MegatronAdaptor:/root/TransformerEngineNPU:"
                "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:"
                "/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge"
            ),
            "LD_LIBRARY_PATH": (
                "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:"
                "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:"
                "/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:"
                "/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:"
                "/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/:"
                "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:"
                "/usr/local/Ascend/cann/lib64:/usr/local/Ascend/cann/aarch64-linux/devlib"
            ),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
            "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
            "HCCL_CONNECT_TIMEOUT": "7200",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "VLLM_ASCEND_ENABLE_NZ": "0",
            "HYDRA_FULL_ERROR": "1",
            "TRANSFORMERS_VERBOSITY": "error",
        },
    )
)


def current_platform() -> Platform:
    """Resolve an explicit test platform, detected accelerator, or CUDA default."""
    override = os.environ.get("VIME_TEST_DEVICE")
    if override:
        return PLATFORMS[override.lower()]
    for platform in registered_platforms():
        if platform.detect():
            return platform
    return PLATFORMS["cuda"]


def launch_commands(
    platform: Platform,
    train_args: str,
    num_devices: int,
    megatron_model_type: str | None,
    repo_base_dir: str,
    train_script: str = "train.py",
    extra_env: dict | None = None,
    external_ray: bool = False,
    master_addr: str = "127.0.0.1",
) -> list[str]:
    """Build the ordered shell commands used to launch a training job."""
    all_env = {**platform.env, **(extra_env or {})}
    commands = [
        "pkill -9 -f '[v]llm serve|VLL[M]::'; sleep 3; "
        + ("" if external_ray else "ray stop --force; pkill -9 ray; ")
        + "pkill -9 vime; sleep 3; true; "
    ]
    if not external_ray:
        exports = "".join(f"export {key}={shlex.quote(str(value))} && " for key, value in all_env.items())
        commands.append(
            f"{exports}export PYTHONUNBUFFERED=1 && ray start --head --node-ip-address {master_addr} "
            f"{platform.ray_start_args(num_devices)} --disable-usage-stats"
        )

    runtime_env = {
        "env_vars": {
            "no_proxy": f"127.0.0.1,{master_addr}",
            "MASTER_ADDR": master_addr,
            **all_env,
        }
    }
    source = f'source "{repo_base_dir}/scripts/models/{megatron_model_type}.sh" && ' if megatron_model_type else ""
    model_args = "${MODEL_ARGS[@]}" if megatron_model_type else ""
    commands.append(
        f"export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && {source}"
        f'ray job submit --address="http://127.0.0.1:8265" '
        f"--runtime-env-json={shlex.quote(json.dumps(runtime_env))} "
        f"-- python3 {train_script} {model_args} {train_args}"
    )
    return commands
