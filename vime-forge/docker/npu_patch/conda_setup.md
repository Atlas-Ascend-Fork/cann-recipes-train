# Vime NPU Conda 环境搭建流程

本文档用于在宿主机上搭建 Vime NPU conda 环境，并用 0.5B 最小训练样例验证环境可用性。流程参考 `docker/Dockerfile.npu` 的安装顺序：先固定 vLLM 相关依赖，再用 constraints 保护关键版本，最后安装训练依赖和 Vime。

## 版本和源码

| 组件 | 推荐版本或 commit |
| --- | --- |
| Python | 3.11 |
| CANN Toolkit / A3 Ops | 9.0.0 |
| NNAL / ATB | 9.0.0 |
| torch / torch-npu | 2.10.0+cpu / 2.10.0 |
| numpy | 1.26.4 |
| triton-ascend | 3.2.1 |
更多python包版本见 ./requirements.txt

## 创建 conda 环境

如果当前 shell 还没有 `conda` 命令，先 source 自己 conda 安装目录下的初始化脚本，例如 `source ~/miniconda3/etc/profile.d/conda.sh`。后续命令默认继续在这个已激活环境中执行；如果换了新终端，只需要重新 `conda activate py311`。

```bash
conda create -y -n py311 python=3.11
conda activate py311

conda config --add channels https://repo.huaweicloud.com/ascend/repos/conda/
export PYTHONNOUSERSITE=1
conda install -y \
  ascend::cann-toolkit==9.0.0 \
  ascend::cann-a3-ops==9.0.0 \
  ascend::cann-nnal==9.0.0
```

## 安装路径变量

进入已经下载好的 Vime 仓库根目录。`WORKSPACE` 只用于放外部依赖源码，不再重新 clone Vime 本体；建议放到仓库外的大容量磁盘目录，不要放在 Vime 仓库目录内。

```bash
cd /path/to/vime

export VIME_ROOT=${VIME_ROOT:-$(pwd)}
export WORKSPACE=/path/to/vime-3rdparty
export PATCH_DIR="${VIME_ROOT}/docker/npu_patch"
export ASCEND_HOME=/path/to/Ascend

mkdir -p "${WORKSPACE}"
```

## Ascend 环境

```bash
source "${ASCEND_HOME}/cann-9.0.0/set_env.sh"
source "${ASCEND_HOME}/nnal/atb/set_env.sh"
```

下文按组件组织为 `clone -> checkout -> patch -> install`。示例命令优先使用镜像源；镜像不可用时，可改用下一行注释里的官方源，commit 保持不变。

## 配置 pip 源

普通 PyPI 源只负责常规 Python 包；vLLM-Ascend 依赖的 Ascend wheel 来自 Ascend PyPI 源，需要单独加到 `extra-index-url`：

```bash
# 常规 Python 包镜像源
python -m pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple

# 添加torch-cpu / triton-ascend 等源
python -m pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu https://mirrors.huaweicloud.com/ascend/repos/pypi"
python -m pip config set global.trusted-host "repo.huaweicloud.com mirrors.huaweicloud.com"
```

## 安装PyTorch & torch_npu
先固定 PyTorch / Ascend 基础包，避免后续依赖安装时改动 NPU 运行栈：

```bash
 # update gcc >= 10.0
sudo apt update
sudo apt install gcc-10 g++-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
sudo update-alternatives --config gcc
sudo update-alternatives --config g++


# 训练侧：torch==2.12.0 & torch_npu==2.12.0
python -m pip install \
  torch==2.12.0+cpu \
  torchvision==0.27.0+cpu \
  torchaudio==2.11.0+cpu
# 注意：更多包版本见 ./requirements.txt，最好提前全部装上(pip3 install --no-deps -r requirements.txt)。


# (a) 从源码编译torch_npu 或者 (b)下载编译好的whl包直接安装(推荐)。下面为源码编译(a)命令：
git clone -b v2.12.0-26.1.0 https://gitcode.com/Ascend/pytorch.git
cd pytorch && git checkout 6beb15962672c61e3aa78157e72cddd943e317f9 && git submodule update --init --recursive
git apply --whitespace=nowarn "${PATCH_DIR}/torch_npu.patch"  # 编译前不修改.h文件
bash ci/build.sh --python=3.11
pip install dist/torch_npu*.whl

# 注意： 安装好torch_npu2.12.0后，参照torch_npu.patch需要对安装目录里的torch_npu/csrc/core/npu/interface/AclInterface.h 和 torch_npu/csrc/core/npu/NPUException.h进行修改。否则后面编译算子的时候会与cann-9.0.0的头文件有冲突。

# 推理侧：torch==2.10.0 & torch_npu==2.10.0
python -m pip install \
  torch==2.10.0+cpu \
  torchvision==0.25.0+cpu \
  torchaudio==2.10.0+cpu
```
注意：如果选择走uv双虚拟环境路线，可先安装torch2.12.0，后面再进行环境克隆（得到.venv-train/和.venv-vllm/， 见 [UV样例验证](#UV样例验证)）。如果选择走统一环境路线，安装torch2.12.0，后面不需要更新或克隆环境。


## 安装 vLLM 相关依赖
vLLM：

```bash
export VLLM_VERSION=0.23.0
export VLLM_COMMIT=v0.23.0
git clone https://gitcode.com/vllm-project/vllm.git "${WORKSPACE}/vllm"
# 官方源：git clone https://github.com/vllm-project/vllm.git "${WORKSPACE}/vllm"
git -C "${WORKSPACE}/vllm" checkout "${VLLM_COMMIT}"
git -C "${WORKSPACE}/vllm" apply --whitespace=nowarn "${PATCH_DIR}/vllm.patch"
VLLM_TARGET_DEVICE=empty python -m pip install -v -e "${WORKSPACE}/vllm" --no-build-isolation --no-deps
```

vLLM-Ascend：

```bash
export VLLM_ASCEND_COMMIT=v0.23.0
git clone https://gitcode.com/gh_mirrors/vl/vllm-ascend.git "${WORKSPACE}/vllm-ascend"
# 官方源：git clone https://github.com/vllm-project/vllm-ascend.git "${WORKSPACE}/vllm-ascend"
git -C "${WORKSPACE}/vllm-ascend" fetch origin main
git -C "${WORKSPACE}/vllm-ascend" checkout "${VLLM_ASCEND_COMMIT}"
git -C "${WORKSPACE}/vllm-ascend" submodule update --init --recursive
git -C "${WORKSPACE}/vllm-ascend" apply --whitespace=nowarn "${PATCH_DIR}/vllm-ascend.patch"
python -m pip install triton-ascend==3.2.1
python -m pip install -v --no-build-isolation -e "${WORKSPACE}/vllm-ascend" --no-deps  # 添加no-deps避免变更包版本
```
Mooncake:

```bash
python3 -m pip install mooncake-transfer-engine-npu==0.3.11.post1
```
在Atlas 950系列服务器上部署推理服务时，请验证每台服务器上的`/lib/route.conf`、`/etc/hccl_rootinfo.json`和`/etc/hixlep`目录（描述UB链路拓扑）是否存在并配置正确。如果其中任何一项缺失或错误，请按照[HiXLEP配置文件生成指南](https://gitcode.com/cann/hixl/wiki/A5%20LocalCommRes%E9%85%8D%E7%BD%AE%E6%8C%87%E5%8D%97.md)生成所需内容。生成`/etc/hixlep`时，请使用\"D2D场景\"。

配置好后，使用时需在pd分离的配置项"kv_connector_extra_config"中添加配置项(可参考[pd_qwen30b_sp_ep_a5.yaml](../../scripts/configs/pd_qwen30b_sp_ep_a5.yaml))：
```bash
  "ascend_local_comm_res_path": "/etc/hixlep"
```

安装完成后生成 constraints，后续安装 Vime 依赖时用它防止 `torch`、`torch-npu`、`numpy`、`vllm`、`vllm-ascend` 等关键包被改动：

```bash
python - <<'PY'
import importlib.metadata as m

names = [
    "numpy",
    "ray",
    "torch",
    "torch-npu",
    "torchvision",
    "transformers",
    "triton-ascend",
    "vllm",
    "vllm-ascend",
]

with open("/tmp/py311-constraints.txt", "w") as f:
    for name in names:
        f.write(f"{name}=={m.version(name)}\n")
PY
```

如果 constraints 生成时提示某个包不存在，先补齐 vLLM 相关依赖安装，不要跳过 constraints。

## 安装训练依赖（推理侧可选择不安装）

### torch_memory_saver（必装）

Vime 训练 actor 当前会 import `torch_memory_saver`。Python 环境没有稳定公开 wheel 时，可以按 `Dockerfile.npu` 从 `sgl-kernel-npu` 构建，或者复用同架构同 Python 版本已经构建好的 wheel。

```bash
git clone --depth 1 --branch 2026.6.0 https://github.com/sgl-project/sgl-kernel-npu.git "${WORKSPACE}/sgl-kernel-npu"
cd "${WORKSPACE}/sgl-kernel-npu"
bash build.sh -a kernels
bash build.sh -a memory-saver
python -m pip install --no-deps output/torch_memory_saver-*-linux_aarch64.whl
```

### TorchTitan (和Megatron二选一安装)
```bash
# 与 torchtitan-npu 的 requirements.txt 保持一致，升级 torchtitan-npu 时按那里的值同步。
TORCHTITAN_COMMIT=086bf6c166ec85c1298eb5596fa9bf95f6a2d840
TORCHTITAN_NPU_COMMIT=3500b075cf0a2476cf6641ee4868ec72a0073221

git clone https://gitcode.com/GitHub_Trending/to/torchtitan.git "${WORKSPACE}/torchtitan"
git -C "${WORKSPACE}/torchtitan" checkout "${TORCHTITAN_COMMIT}"
git -C "${WORKSPACE}/torchtitan" apply --whitespace=nowarn "${PATCH_DIR}/torchtitan.patch"
python -m pip install --no-deps --no-build-isolation -e "${WORKSPACE}/torchtitan"

git clone https://gitcode.com/cann/torchtitan-npu.git "${WORKSPACE}/torchtitan-npu"
git -C ${WORKSPACE}/torchtitan-npu checkout ${TORCHTITAN_NPU_COMMIT}
git -C "${WORKSPACE}/torchtitan-npu" apply --whitespace=nowarn "${PATCH_DIR}/torchtitan-npu.patch"
python -m pip install --no-deps --no-build-isolation -e "${WORKSPACE}/torchtitan-npu"
```

`torchtitan.patch` 中的 chunked cross-entropy 接线只作用于直接使用 TorchTitan
`Trainer` 的训练入口。Vime 的 TorchTitan backend 自行构造训练循环，目前不会启用这条
接线。升级 `TORCHTITAN_COMMIT` 时必须在新基线上重新应用并验证该补丁，不能沿用旧补丁而
忽略冲突。

`torchtitan-npu.patch` 用于降低 TorchTitan checkpoint save 阶段的 host memory 峰值。该 patch 会在 PyTorch DCP `SerializationFormat.TORCH_SAVE` 路径中释放已经写出的 CPU staging tensor 引用，避免大模型保存 checkpoint 时持续保留已写 tensor 导致 `rtMallocHost failed: driver error: out of memory`。默认开启，可通过以下环境变量临时回退原始行为：

```bash
export VIME_DCP_RELEASE_WRITTEN_TENSORS=0
```


### Megatron-LM
  Megatron-LM
  ```bash
  export MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862

  git clone https://gitcode.com/NVIDIA/Megatron-LM.git "${WORKSPACE}/Megatron-LM"
  # 官方源：git clone https://github.com/NVIDIA/Megatron-LM.git "${WORKSPACE}/Megatron-LM"
  git -C "${WORKSPACE}/Megatron-LM" checkout "${MEGATRON_COMMIT}"
  git -C "${WORKSPACE}/Megatron-LM" apply --whitespace=nowarn "${PATCH_DIR}/meagtron_comm.patch"
  git -C "${WORKSPACE}/Megatron-LM" apply --whitespace=nowarn "${PATCH_DIR}/megatron.patch"
  python -m pip install --no-deps --no-build-isolation -e "${WORKSPACE}/Megatron-LM"
  ```

  Megatron-Bridge：

  `Megatron-Bridge` 按 `Dockerfile.npu` 通过 `PYTHONPATH` 使用，不安装 package metadata，避免拉入 CUDA-only 依赖。
  ```bash
  export MEGATRON_BRIDGE_COMMIT=3fd3768045422d0aa5c97e90a4e6c659aea9acb9

  git clone --branch bridge https://github.com/radixark/Megatron-Bridge.git "${WORKSPACE}/Megatron-Bridge"
  git -C "${WORKSPACE}/Megatron-Bridge" checkout "${MEGATRON_BRIDGE_COMMIT}"
  git -C "${WORKSPACE}/Megatron-Bridge" apply --whitespace=nowarn "${PATCH_DIR}/megatron-bridge.patch"
  python -m pip install --no-build-isolation "nvidia-modelopt[torch]>=0.37.0"
  ```

  mbridge：

  ```bash
  export MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c

  git clone https://github.com/ISEEKYAN/mbridge.git "${WORKSPACE}/mbridge"
  git -C "${WORKSPACE}/mbridge" checkout "${MBRIDGE_COMMIT}"
  python -m pip install --no-deps --no-build-isolation -e "${WORKSPACE}/mbridge"
  ```

  MindSpeed：

  ```bash
  export MINDSPEED_COMMIT=fc63de5c48426dd019c3b3f39e65f5bdf56e4086

  git clone https://gitcode.com/Ascend/MindSpeed.git "${WORKSPACE}/MindSpeed"

  git -C "${WORKSPACE}/MindSpeed" checkout "${MINDSPEED_COMMIT}"
  git -C "${WORKSPACE}/MindSpeed" apply --whitespace=nowarn "${PATCH_DIR}/mindspeed.patch"
  python -m pip install --no-deps --no-build-isolation -e "${WORKSPACE}/MindSpeed"
  ```

### Vime（必装）：

```bash
python -m pip install --constraint /tmp/py311-constraints.txt --requirement "${VIME_ROOT}/requirements.txt"
python -m pip install --no-deps --no-build-isolation -e "${VIME_ROOT}"
```


## 环境检查

先设置运行前环境变量：

```bash
export VLLM_VERSION=0.23.0
export VLLM_USE_AOT_COMPILE=0 # for torch>2.10
export PYTHONPATH="${WORKSPACE}/Megatron-Bridge/src:${WORKSPACE}/Megatron-LM:${VIME_ROOT}:$PYTHONPATH"
```

```bash
python - <<'PY'
import importlib.util
import torch
import torch_npu
import vllm
import vllm_ascend
import megatron
import mindspeed
import torch_memory_saver
import vime
from megatron.bridge import AutoBridge

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", vllm.__version__)
print("vllm_ascend_C", importlib.util.find_spec("vllm_ascend.vllm_ascend_C"))
PY
```

`vllm_ascend.vllm_ascend_C` 必须能找到；否则说明 vLLM-Ascend 自定义算子没有构建好，后续可能在采样算子处失败。

## 0.5B 验证样例

0.5B 只是用来验证 Vime NPU conda 环境能完成最小训练链路。样例默认 1 卡 rollout、1 卡 actor，不开启 NPU graph。

运行前先编辑 `scripts/run-qwen2.5-0.5B-npu-smoke.sh` 顶部路径：

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3-4B --local-dir /tmp/models/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k  --local-dir /tmp/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024  --local-dir /tmp/datasets/aime-2024

WORKSPACE=/path/to/vime-3rdparty
MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
PROMPT_DATA=/path/to/dapo-math-17k.jsonl
```

然后启动样例：

```bash
bash "${VIME_ROOT}/scripts/run-qwen2.5-0.5B-npu-smoke.sh"
```

脚本会重启本机 Ray，运行前不要让本地 Ray 通信走代理。


## UV样例验证
如果走uv模式，需要设置`UV_MODE`、`VIME_UV_SETUP`环境变量，VIME_UV_SETUP会克隆并拆分环境，在.venv-vllm/环境中对torch/torch_npu进行降级(自动调用脚本 `scripts/setup-uv-split-envs.sh`)。第一次克隆后，之后运行脚本就不需要开启`VIME_UV_SETUP=1`。
```bash
UV_MODE=1 VIME_UV_SETUP=1 bash "scripts/torchtitan/run-qwen3-30B-A3B.sh"
```
