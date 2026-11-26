# DeepSeek-V4 RL 训练部署

## 概述

DeepSeek团队发布了模型DeepSeek-V4技术报告，本实践基于DeepSeek技术报告在CANN平台上进行rl后训练适配，支持在昇腾`Ascend 950`平台部署。

本文介绍在 **Ascend 950DT 系列、Linux ARM** 环境下，使用 Vime 作为rl流程控制框架后端对接torchtitan和vllm分别作为rl训练及推理侧框架。

- 本实践的相关特性可参见 [基于鲲鹏CPU、昇腾950和灵衢协同的Agentic RL训练实践](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/llm_rl/kunpeng_ascend_agentic_rl/zh/kunpeng_ascend_agentic_rl_infrastructure.md)



## 1. 硬件与系统要求

产品型号：**Ascend 950DT 系列**

操作系统：**Linux ARM**（`aarch64`）

| 项目 | 要求 |
|---|---|
| 计算资源 | 用户根据模型显存、吞吐目标和实际设备自行确定训练与推理节点数量、每节点 NPU 数及并行配置 |
| 驱动与固件 | 与本次 RL 镜像内的 CANN 和 Ascend 950 硬件匹配；多节点使用一致版本 |
| 多机网络 | 节点间 Ray、Gloo、HCCL 通信互通；PD 分离还需满足跨节点 KV 传输要求 |
| 存储 | 足够容纳模型、数据和 checkpoint；多机训练节点共同访问同一份 checkpoint 存储 |

在每台宿主机检查系统架构和 NPU 状态：

```bash
uname -s       # 预期为 Linux
uname -m       # 预期为 aarch64
npu-smi info   # 检查设备、驱动及固件状态
```

镜像和依赖包应使用 Linux ARM 对应的构建。驱动、固件及 CANN 的具体版本以待提供的 RL 镜像配套要求为准。显存占用取决于镜像、模型精度、序列长度及并发，本文未提供实测性能数据。

## 2. 环境搭建

### 2.1 获取镜像

使用预置 ARM 镜像，镜像中已包含本实践所需的 PyTorch、torch_npu、torchtitan、torchtitan-npu、vllm、vllm-ascend及所有python依赖 和 CANN 运行环境。从[950DT ARM 镜像地址](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/DeepSeek/ascend_rl_cann9.3.0_dsv4_aarch_950_image_v0.5.tar)
下载 docker 镜像，上传到 950 服务器的每个节点，并在每个节点执行：

```bash
docker load -i ascend_rl_cann9.3.0_dsv4_aarch_950_image_v0.5.tar
```

### 2.2 启动容器

在各个节点上通过如下脚本拉起容器，默认容器名为`ascend_rl`。请将权重路径和源码路径挂载到容器中。

```bash
docker run -u root -itd --ulimit nproc=65535:65535 --ipc=host \
    --device=/dev/davinci0     --device=/dev/davinci1 \
    --device=/dev/davinci2     --device=/dev/davinci3 \
    --device=/dev/davinci4     --device=/dev/davinci5 \
    --device=/dev/davinci6     --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    --device=/dev/ummu \
    --device=/dev/uburma \
    -v /usr/local/sbin/urma_perftest:/usr/local/sbin/urma_perftest -v /usr/bin/urma_admin:/usr/bin/urma_admin \
    -v /usr/lib64:/usr/lib64 -v /usr/lib:/usr/lib \
    -v /home/:/home \
    -v /data/:/data \
    -v /etc/localtime:/etc/localtime \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog \
    -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip:/root/.pip -v /etc/hosts:/etc/hosts \
    -v /usr/bin/hostname:/usr/bin/hostname \
    -v /etc/hixlep:/etc/hixlep \
    -v /lib/route.conf:/lib/route.conf \
    --net=host \
    --shm-size=128g \
    --privileged \
    --name ascend_rl ascend_rl_cann9.3.0_dsv4_aarch_950_image:v0.4 /bin/bash
```

容器示例保留 Ascend 950 的 `/dev/ummu`、`/dev/uburma`、URMA 工具及通信配置挂载；按实际宿主机安装路径调整。多机场景的存储与 PD 通信资源要求见第 4.2、4.5 节。

进入容器：

```bash
docker exec -it ascend_rl bash
```

### 2.3 源码下载
HOME_PATH 由用户指定，下面以 /data 为例：

```bash
export HOME_PATH=/data
cd ${HOME_PATH}
git clone https://gitcode.com/cann/cann-recipes-train.git
cd ${HOME_PATH}/cann-recipes-train/vime-forge
```

### 2.4 训推环境版本介绍

后续命令均在容器内执行。镜像应预装与本仓库匹配的依赖和 DSV4 算子。训练与推理环境由启动器分别调用。环境构建脚本会让两个 venv 继承基础 Python 的公共包，并在训练 venv 中覆盖训练侧 Torch；rollout 的 `vllm` 可以是指向基础环境入口的符号链接。因此应检查实际 `vllm` 可执行文件，不能仅凭 venv 名称判断使用的 Torch 版本。

训推侧使用两个独立的运行环境，版本必须来自同一套镜像和兼容的 Ascend 950 软件栈：

| 环境 | 运行时路径 | 组件与版本要求 |
|---|---|---|
| 训练环境 | `/app/uv-train` | Python、PyTorch、torch_npu、TorchTitan、TorchTitan-NPU、TorchAO-NPU、Ray、Vime；具体版本以镜像提供的版本清单为准 |
| 推理环境 | `/app/uv-vllm` | Python、vLLM、vLLM-Ascend 及推理侧 PyTorch/torch_npu；当前运行时通过 `VLLM_VERSION` 默认使用 `0.30.0` |
| CANN | `/usr/local/Ascend/cann` | 训练和推理共同使用，路径需与 `local.sh` 中的 `VIME_CANN_HOME` 一致 |
| NNAL / ATB | `/usr/local/Ascend/nnal/atb` | 推理和通信相关运行库，路径需与 `VIME_NNAL_HOME` 一致 |

训练和推理环境中的 PyTorch、torch_npu、vLLM-Ascend 与 CANN 必须相互兼容，不能分别安装未经验证的版本。镜像提供者应补充实际版本号；在版本确定前，README 不假定具体的 PyTorch、torch_npu、TorchTitan 或 vLLM-Ascend 小版本。

已有预装镜像时无需重新运行 `setup-dsv4-uv-env.sh`。该脚本用于构建环境，会替换目标 `WORKSPACE` 下的四个第三方源码目录，并需要调用者提供匹配的训练/推理 Torch 与 torch-npu wheel。

## 3. 准备模型与数据

可以从 Hugging Face、ModelScope 等平台下载所需的模型和数据集。启动脚本默认从 `${VIME_MODEL_ROOT}` 读取模型、从 `${VIME_DATA_ROOT}` 读取数据集

### 3.1 权重准备

  下载[DeepSeek-V4-Flash原始Hybrid FP8-MXFP4权重](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)，并上传到各节点的某个固定的路径下，比如`/data/models/deepseek_v4_hybrid_fp8_mxfp4`。


  由于训练过程中我们希望使用Hybrid MXFP8-MXFP4或整网MXFP8权重进行训练正向的计算及推理，因此我们在原始权重的基础上需要做一步权重转换。
  需要使用[convert_model.py](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/deepseek_v4/utils/convert_model.py) 脚本完成 Hybrid FP8-MXFP4 到 Hybrid MXFP8-MXFP4 / MXFP8 权重转换。

    >入参介绍：`input_fp8_hf_path`：原始权重路径；`output_hf_path`：转换后输出的权重路径；`quant_type`：量化模式；

    如果权重转换的运行环境为NPU，需要先执行：

    ```shell
    cann_path=/usr/local/Ascend/cann  # cann包安装路径
    source ${cann_path}/bin/setenv.bash
    ```
    权重转换拉起示例：

    ```shell
    # 转换为Hybrid MXFP8-MXFP4权重，适用于Ascend 950系列
    python convert_model.py --input_fp8_hf_path /data/models/deepseek_v4  --output_hf_path /data/models/deepseek_v4_hybrid_mxfp8_mxfp4 --quant_type w4a8-mx

    # 转换为MXFP8权重，适用于Ascend 950系列
    python convert_model.py --input_fp8_hf_path /data/models/deepseek_v4  --output_hf_path /data/models/deepseek_v4_mxfp8 --quant_type w8a8-mx
    ```

### 3.2 数据集准备

默认使用 DAPO Math 17k 作为训练数据，AIME-2024 或 math-500 作为评测数据：

```bash
mkdir -p ${VIME_DATA_ROOT}

# 下载训练数据集 (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir ${VIME_DATA_ROOT}/dapo-math-17k

# 下载评估数据集 (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir ${VIME_DATA_ROOT}/aime-2024

# 下载评估数据集 (math-500)，原始文件为 test.jsonl
hf download --repo-type dataset HuggingFaceH4/MATH-500 \
  --local-dir ${VIME_DATA_ROOT}/math-500

# 转换为 vime 评测格式，生成 ${VIME_DATA_ROOT}/math-500/math-500.jsonl
python vime-forge/tools/build_math500_eval_data.py \
  --input ${VIME_DATA_ROOT}/math-500/test.jsonl \
  --output ${VIME_DATA_ROOT}/math-500/math-500.jsonl
```

`build_math500_eval_data.py` 会把 MATH-500 的 `{"problem", "answer", "subject", "level", "unique_id"}`
逐行改写为 vime 评测所需的 `{"prompt": [...], "label": ..., "metadata": {...}}`：

- `prompt`：`[{"role": "user", "content": problem}]`，与 aime-2024 评测集的格式一致；
- `label`：原始 `answer`，供 `rm_type: math` 的 `\boxed{}` 判分使用；
- `metadata`：保留 `subject`、`level`、`unique_id`，便于按主题/难度细分。

生成的 `math-500.jsonl` 必须放在 `${VIME_DATA_ROOT}/math-500/math-500.jsonl`，该路径由
`vime-forge/scripts/configs/eval.yaml` 中的 math500 数据集项引用
（`n_samples_per_eval_prompt: 1`、`max_response_len: 8192`）。

> 如果下载节点访问 Hugging Face 受限，可先设置 `HF_ENDPOINT`（脚本通过 `hf_hub_url` 下载，会遵循该变量）；
> 也可以省略 `--input`，由脚本自行下载 `test.jsonl`。`--output` 省略时默认同样是
> `$VIME_DATA_ROOT/math-500/math-500.jsonl`。

也可以把已有数据放到该目录。启动脚本使用 `prompt` 和 `label` 字段，并启用 chat template；自定义 JSONL 每行可以采用以下形式：

```json
{"prompt":[{"role":"user","content":"What is 1 + 1? Put your final answer in \\boxed{}."}],"label":"2"}
```

`label` 是 math reward 使用的参考答案；应准备足够的题目供 rollout 采样。

## 4. 完整 DeepSeek-V4 Flash 多机部署

### 4.1 选择推理模式

使用完整 Flash 的 **43 个主干层、256 个路由专家**，关闭 MTP，配方为 `vime_dsv4_flash_43layers_256experts`。训练和推理节点数量由用户根据资源与吞吐需求自行决定，PD 模式不绑定固定的机器数量。

| 模式 | 配置方式 | 行为 |
|---|---|---|
| PD 混部（关闭 PD 分离） | `VLLM_CONFIG` 置空 | 常规 vLLM 实例同时执行 Prefill 和 Decode |
| PD 分离 | `VLLM_CONFIG` 指向包含 `prefill`、`decode` 组的 YAML | P、D 使用各自分配的资源，通过 KV connector 传输缓存 |

本流程保持训练与 rollout 分开分配资源，使用 `COLOCATE=off`。该变量控制训练与推理是否共卡；PD 开关由 `VLLM_CONFIG` 及其 YAML 内容决定。


### 4.2 节点、镜像与存储准备

在各节点按第 2 节启动同一镜像的容器，使用 host 网络。容器暴露的 NPU 数应与 `NPUS_PER_NODE` 一致；代码、Python 环境、模型和数据路径在各节点保持一致。建议采用以下目录：

```text
${HOME_PATH}/cann-recipes-train/vime-forge/    # 各节点使用相同代码版本
/models/DeepSeek-V4-Flash-MXFP8/               # 模型权重，各节点均可读取
${HOME_PATH}/dsv4-rl/data/                     # 训练数据
${HOME_PATH}/dsv4-rl/ckpt/                     # 训练节点共享的 checkpoint 存储
${HOME_PATH}/dsv4-rl/tb/                       # 日志与 TensorBoard 输出
```

模型可以在各节点本地复制，也可以共享挂载。**Checkpoint 目录必须让所有训练 rank 访问同一份存储**；相同路径名但各自独立的本地目录不能代替共享存储。

Ray、Gloo、HCCL 和推理通信所用网络必须互通。在各容器启动 Ray 前配置正确的 `GLOO_SOCKET_IFNAME`；平台所需的 HCCL 网卡或本机 IP 参数也应分别配置。Ray 默认端口包括 `6379`、dashboard `8265`、dashboard agent `52365` 和 worker `20000–20999`；还需允许 HCCL、vLLM、router 及 KV 传输使用的通信端口。

**节点角色由实际 IP 排序和资源分配共同决定。** 当前 placement group 按 IPv4 数值和设备 ID 排序，先分配训练所需资源，再分配 rollout 资源。PD 分离时按 YAML 的 server group 顺序分配推理资源。仅调整 `NODE_IP_LIST` 的列表顺序不能任意绑定训练/P/D 角色，应检查实际 placement 日志。

### 4.3 修改 local.sh 和启动参数

单机、多机统一使用 `scripts/torchtitan/run-dsv4.sh`。机器路径、资源及实验覆盖项集中配置在 `scripts/torchtitan/common/local.sh`。该文件不提交到仓库，需要在各节点分别准备，写法参照`local.sh.example`；首次使用时执行：

```bash
cd "${HOME_PATH}/cann-recipes-train/vime-forge"
cp scripts/torchtitan/common/local.sh.example scripts/torchtitan/common/local.sh
```

按实际安装路径、模型及数据目录修改，例如：

```bash
# scripts/torchtitan/common/local.sh
export VIME_CANN_HOME=/usr/local/Ascend/cann
export VIME_NNAL_HOME=/usr/local/Ascend/nnal/atb
export VIME_TRAIN_ENV=/app/uv-train
export VIME_VLLM_ENV=/app/uv-vllm
export VIME_WORKSPACE="${HOME_PATH}/dsv4-rl"
export VIME_MODEL_ROOT=/models
export VIME_DATA_ROOT="${VIME_WORKSPACE}/data"
export VIME_CKPT_ROOT="${VIME_WORKSPACE}/ckpt"
export VIME_TB_ROOT="${VIME_WORKSPACE}/tb"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUN_NAME=dsv4-flash
export MODE=sync
export COLOCATE=off
export SAVE_ON=off
export DSV4_RECIPE=vime_dsv4_flash_43layers_256experts
export NODE_IP_LIST="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4 10.0.0.5"
export NPUS_PER_NODE=8
export ROLLOUT_NODE_NUM=1
export EP_DEGREE=32
export GLOO_SOCKET_IFNAME=enp34s0f1  # 替换为本机实际通信网卡
export VLLM_CONFIG=""              # PD 分离配置见第 4.5 节
```

上例默认模型路径为 `${VIME_MODEL_ROOT}/DeepSeek-V4-Flash-MXFP8/`。目录名不同时，在 `local.sh` 中覆盖脚本显式传入的路径：

```bash
OVERRIDE_ARGS+=(
    --hf-checkpoint /models/your-converted-model
    --ref-load /models/your-converted-model
)
```

启动器先读取 `local.sh`，再由 `common/default.sh` 补齐未设置项。`OVERRIDE_ARGS` 追加在模型脚本参数之后；命令行 `--` 后的参数最后追加。也可用 `--local local.<experiment>.sh` 选择其他本地配置文件。

上例按 5 个节点、每节点 8 卡配置，其中训练 4 个节点（32 卡）、推理 1 个节点（8 卡）。将示例 IP 替换为实际地址，资源参数统一写在 `local.sh` 中。各节点使用相同的任务配置；`LOCAL_NODE_IP` 和网卡名按本机实际情况填写。

| 参数 | 配置说明 |
|---|---|
| `NODE_IP_LIST` | 全部节点 IP，以空格分隔；第一个为 Ray head；单机时不设置 |
| `NPUS_PER_NODE` | 每节点提供的 NPU 数；当前入口按各节点同卡数计算 |
| `ROLLOUT_NODE_NUM` | 推理节点数量；其余节点用于训练，训推分离时需为两侧都保留资源 |
| `ACTOR_GPUS` | 每个训练节点使用的卡数，多机默认取 `NPUS_PER_NODE`，不是训练总卡数 |
| `ROLLOUT_GPUS` | 推理总卡数，多机默认取 `ROLLOUT_NODE_NUM × NPUS_PER_NODE` |
| `EP_DEGREE` | 与训练总卡数、专家数及其他并行维度兼容的 expert parallel degree |
| `VLLM_RANKS_PER_ENGINE` | PD 混部时每个 engine 的卡数，默认取 `ROLLOUT_GPUS` |

公共 launcher 根据节点总数减去 `ROLLOUT_NODE_NUM` 得到 `ACTOR_NODE_NUM`；训练总卡数为 `ACTOR_NODE_NUM × ACTOR_GPUS`。例如 5 个节点、每节点 8 卡、1 个推理节点，对应训练 32 卡、推理 8 卡。完整模型的显存和并行约束仍需按实际部署验证。

GRPO、采样等参数在 `run-dsv4.sh` 中声明，也可通过 `local.sh` 的 `OVERRIDE_ARGS` 覆盖；优化器、默认学习率、训练精度和梯度裁剪等配方参数位于 `vime/backends/torchtitan_utils/models/deepseek_v4/config_registry.py`，详见第 5 节。

### 4.4 关闭 PD 分离：PD 混部

在 `scripts/torchtitan/common/local.sh` 中将配置置空：

```bash
export VLLM_CONFIG=""
```

`VLLM_RANKS_PER_ENGINE` 设为每个常规推理 engine 使用的卡数，并保证能整除 `ROLLOUT_GPUS`。当前入口同时用该值设置 `--vllm-data-parallel-size` 和 `--rollout-num-gpus-per-engine`，由同一个 engine 执行 Prefill 和 Decode。

从 PD 分离切回混部时，重新核对该并行参数，并清除命令行中遗留的 `--vllm-config` 覆盖项。训练与推理节点数量由用户独立设置。

### 4.5 开启 PD 分离

在 `scripts/torchtitan/common/local.sh` 中指定 PD YAML：

```bash
export VLLM_CONFIG="${HOME_PATH}/cann-recipes-train/vime-forge/scripts/configs/pd_dsv4_a5.yaml"
```

以 [pd_dsv4_a5.yaml](../../vime-forge/scripts/configs/pd_dsv4_a5.yaml) 为模板，根据用户选定的资源修改配置；模板内的数值是示例，不是固定的节点数量要求。

| 配置项 | 调整要求 |
|---|---|
| `worker_type` | 分别定义 `prefill` 与 `decode` 组 |
| 各组 `num_gpus` | 分别设置 P、D 总卡数，所有组的卡数之和须等于 `ROLLOUT_GPUS` |
| 各组 `num_gpus_per_engine` | 按 P、D engine 规模设置，并与组内卡数及物理节点布局匹配 |
| `kv_transfer_config` | P 使用 `kv_producer`，D 使用 `kv_consumer`，配置匹配的 `MooncakeHybridConnector` 通信参数 |
| `kv_connector_extra_config` 中的 P/D `dp_size`、`tp_size` | 与实际 engine 并行布局一致；不能只改资源数而保留旧的 connector 配置 |
| `update_weights` | 保持 `true`，让 P、D 都接收训练侧的权重更新 |

`VLLM_CONFIG` 非空时，当前 `run-dsv4.sh` 仅追加 `--vllm-config`，不再通过 `VLLM_RANKS_PER_ENGINE` 推导全局 DP。各组的 `num_gpus_per_engine`、`overrides.tensor_parallel_size` 和 `overrides.data_parallel_size` 应在 YAML 中设置，并与连接器并行配置一致。模板中 P、D 各 8 卡，合计 16 卡；第 4.3 节示例仅分配 8 张推理卡，不能直接使用该模板。保持 1 个推理节点时，需将 YAML 的 P、D 资源及其 TP/DP、连接器配置调整为总计 8 卡，并满足模型的显存要求。

P/D 分离需要镜像内可用的 Mooncake/HIXL 通信栈，以及目标机器对应的 `/etc/hixlep`、URMA 设备和通信配置；挂载见第 2 节。资源文件须有效，P/D 之间的 KV 传输网络需可达。

### 4.6 量化参数配置

量化参数需要与模型权重和推理侧配置保持一致。在 `scripts/torchtitan/common/local.sh` 中设置下面的变量，由统一入口 `run-dsv4.sh` 读取，并将修改同步到所有相关节点。

> DSV4 的 LI 统一采用 C4 量化，推理侧 KV cache 统一使用 FP8（`--vllm-kv-cache-dtype fp8`）；`ENABLE_MXFP4` 只控制 MoE 权重是否使用 MXFP4。下面两个场景的差异仅在 MoE 权重的量化格式。模型权重请使用第 3 节转换得到，并按第 4.3 节通过 `VIME_MODEL_ROOT` 或 `OVERRIDE_ARGS` 指向该目录。

#### 场景 1：MoE MXFP8 + LI C4

该场景的 MoE 权重使用 MXFP8；LI 使用 C4 量化。使用 DSV4 拉起脚本默认配置即可，训练侧量化参数对应：

```bash
--extension.quantization.recipe all_block_fp8
--extension.quantization.enable-quantized-training
```

#### 场景 2：MoE MXFP4 + LI C4

该场景对 MoE 权重启用 MXFP4，在 `local.sh` 中设置：

```bash
export ENABLE_MXFP4=1
```

### 4.7 训练和推理侧 CP 配置

训练侧和推理侧的 Context Parallel（CP）需要分别配置。启用前请确认 CP 度数与训练 world size、rollout 卡数及模型并行布局相容，并将配置同步到所有相关节点。

#### 训练侧 CP

在 `local.sh` 中设置训练侧 CP 度数，`run-dsv4.sh` 会生成对应的 `--parallelism.context_parallel_degree` 和 `--parallelism.spmd_backend` 参数：

```bash
export CP_DEGREE=2
export SPMD_BACKEND=spmd_types
```

`CP_DEGREE` 应能整除训练侧可用的并行资源；如果同时调整 EP、TP 或训练节点数，需要重新检查各并行维度的乘积和显存占用。

#### 推理侧 CP

在 `local.sh` 中为每个推理 engine 设置 TP/DP，例如：

```bash
export VLLM_RANKS_PER_ENGINE=8
OVERRIDE_ARGS+=(
    --vllm-additional-config '{"enable_dsa_cp":true,"enable_flashcomm1":true}'
    --vllm-tensor-parallel-size 8
    --vllm-data-parallel-size 1
)
```

上例为每个 8 卡 engine 使用 TP=8、DP=1。PD 分离时，TP/DP 以各组 YAML 的 `overrides` 为准，并同步检查连接器的并行信息。

### 4.8 启动与运行检查

完成资源和 PD 模式配置后，在每个节点的容器中执行同一条启动命令：

```bash
cd "${HOME_PATH}/cann-recipes-train/vime-forge"
bash scripts/torchtitan/run-dsv4.sh
```

先启动 head，再启动其余节点，或由调度器并行启动。Head 提交任务后等待资源就绪；worker 加入 Ray 后保持运行，需要终端或调度器持续托管。多网卡环境可在本节点启动前设置 `LOCAL_NODE_IP=<本节点在列表中的IP>`。

检查 Ray 的存活节点和 NPU 总数是否与 `NODE_IP_LIST`、`NPUS_PER_NODE` 一致；检查实际训练 world size、EP 和 rollout 资源是否与配置一致。混部应启动 regular rollout；分离应启动 P/D 两类实例，router 获得两侧地址，并完成 KV 传输。

首次权重同步完成后应开始生成回复，训练更新后再次同步权重。首次验证可将 `--num-rollout` 设为 `2`，采样 batch 和训练 batch 则按实际训练并行度配置。日志与恢复方法见第 6 节。

## 5. 关键训练参数

默认执行同步 GRPO 训练，共 200 个 rollout；每次采样 128 个 prompt，每个生成 8 个回复，`global-batch-size=1024`。vLLM 使用 `dummy` 加载，首次 rollout 前由训练侧推送真实权重。

`--num-rollout` 是整个任务的目标 rollout 数，恢复训练时也按该总数计算。修改 GRPO、采样及脚本显式传入的参数时编辑 `scripts/torchtitan/run-dsv4.sh`；修改节点、资源或 PD 模式时编辑 `scripts/torchtitan/common/local.sh`。精度、梯度裁剪和优化器配置由 [config_registry.py](../../vime-forge/vime/backends/torchtitan_utils/models/deepseek_v4/config_registry.py) 中的 `_vime_rl_config()` 提供。开始新实验时修改 `local.sh` 中的 `RUN_NAME` 或指定新的 `--save` 目录，避免从已有 checkpoint 自动续训。

下表列出当前 shell 启动脚本显式传入的参数。

| 参数 | 默认值 / 行为 |
|---|---|
| `--advantage-estimator` | `grpo`；当前 TorchTitan 后端只接入 GRPO |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.28` |
| `--use-kl-loss` / `--kl-loss-coef` | 启用 KL 路径，系数为 `0.0`；仍会创建 reference 路径 |
| `--entropy-coef` | `0.0` |
| `--lr_scheduler.decay_type` / `--lr_scheduler.min_lr_factor` / `--lr_scheduler.warmup_steps` | `linear` / `1.0` / `0`；分别表示线性调度、最低学习率相对初始学习率的比例和预热步数 |
| `--rollout-temperature` | `1` |
| `--vllm-gpu-memory-utilization` | `0.7` |
| `--vllm-server-concurrency` | `1024` |
| `--vllm-kv-cache-dtype` | `fp8` |
| `--vllm-compilation-config` | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` |
| `--use-dynamic-batch-size` / `--max-tokens-per-gpu` | 开启；token 上限默认等于 `SEQ_LEN` |
| `--use-rollout-routing-replay` | 开启；支持 rollout 侧 PD 共卡和 PD 分离两种部署方式 |
| `--fine-grained-memory-cleanup` | 开启，在训练阶段间释放缓存；可用 `--no-fine-grained-memory-cleanup` 关闭 |

`SEQ_LEN` 同时用于训练序列长度、rollout 上下文上限和 vLLM `max-model-len`。修改长度时优先修改本地配置，并保证 `MAX_RESPONSE_LEN < SEQ_LEN`。改变 batch 时需保持采样量与训练 batch 配套，例如当前脚本默认 `128 × 8 = 1024`。

公共 launcher 通过 `enable_batch_invariant()` 统一启用训练确定性、vLLM 确定性、rollout seed 和 RL batch invariance，运行环境固定设置 `HCCL_DETERMINISTIC=strict`、`LCCL_DETERMINISTIC=1`。

## 6. 日志、保存与恢复

以 `RUN_NAME=dsv4-flash`、同步训练为例，默认产物为：

```text
${HOME_PATH}/dsv4-rl/
  ckpt/dsv4-flash-sync/step-N/       # 训练 checkpoint，N 为从 0 开始的 rollout ID
  tb/dsv4-flash/<时间戳>/
    train.log                        # Ray job 的训练输出
    rerun.sh                         # 本次实际提交命令
    ...                              # TensorBoard 事件及其他运行输出
```

在 `local.sh` 设置 `SAVE_ON=on` 后，当前脚本以 `--save-interval 20` 定期保存，并保留最近 2 个 checkpoint；默认 `SAVE_ON=off` 不开启该保存配置。同步训练在最后一个 rollout 也会保存，实际频率可通过 `OVERRIDE_ARGS` 覆盖。

启动 TensorBoard 查看训练指标：

```bash
tensorboard --logdir ${HOME_PATH}/dsv4-rl/tb --host 0.0.0.0 --port 6006
```

相同保存目录中已有 checkpoint 时，默认从最新 step 恢复。保持所需的节点资源与 PD 模式配置，在全部节点重新运行：

```bash
# 确认 local.sh 中的资源和 PD 配置后，在各节点执行
cd ${HOME_PATH}/cann-recipes-train/vime-forge
bash scripts/torchtitan/run-dsv4.sh
```


## 7. 多机拉起命令

本例使用 5 个节点、每节点 8 卡，其中训练 4 个节点（32 卡）、推理 1 个节点（8 卡），采用 PD 混部（`VLLM_CONFIG=""`）。先在每个节点的容器中按第 4.3 节配置 `scripts/torchtitan/common/local.sh`：节点列表、卡数、训练配方和环境路径都写在该文件中，启动命令无需重复传入。

确认各节点的任务配置一致，并将示例 IP 和网卡名替换为实际值；模型、数据和共享 checkpoint 路径须在各节点可访问。多网卡环境可在各自的 `local.sh` 中设置 `LOCAL_NODE_IP`，填写本机在节点列表中的 IP。

在 **5 个节点的容器中分别执行相同命令**：

```bash
export HOME_PATH=/data
cd "${HOME_PATH}/cann-recipes-train/vime-forge"
bash scripts/torchtitan/run-dsv4.sh
```

`local.sh` 中的第一个 IP 对应 Ray head，负责提交任务；其余节点加入 Ray 集群并保持运行。各节点必须都执行，不能只在 head 执行。实际训练和推理角色以第 4.2 节的资源分配规则及 placement 日志为准。
