# 基于TorchTitan的DeepSeek-V3.2 32K长序列预训练优化实践样例

## 概述

本样例针对DeepSeek-V3.2模型，基于[TorchTitan开源框架](https://github.com/pytorch/torchtitan)，在64卡Atlas A3集群上完成32K长序列预训练全流程的优化适配。优化点介绍可参见[CANN+TorchTitan: DeepSeek-V3.2 32k长序列预训练昇腾优化实践](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/llm_pretrain/deepseek_v32/zh/deepseek-v32_pre_train_optimization.md)。

训练框架使用昇腾官方维护的 [torchtitan-npu](https://gitcode.com/cann/torchtitan-npu) 插件（本样例对应版本 `v0.2.2`），需在本样例目录下拉取后使用，详见下文「源码准备」。

## 硬件要求
产品型号：Atlas A3 系列

最少卡数：64张A3

## 目录结构说明
```bash
├─utils                              # 数据/权重转换工具脚本
│  ├─parquet2json.py                 # enwiki parquet → jsonl 转换
│  ├─weight_convert.sh               # FP8 → Bfloat16 权重转换入口
│  └─convert_model.py                # 权重转换核心脚本
└─torchtitan-npu                     # 运行时拉取，不纳入 git，见「源码准备」
```

## 环境准备
> 本样例直接使用昇腾官方公开镜像，无需自行构建 Dockerfile。镜像中已包含本样例所需的 CANN 软件栈与 Python 环境。

1. 准备镜像并启动容器。

```bash
# 获取Docker镜像
docker pull quay.io/ascend/cann:9.0.0-beta.1-a3-ubuntu22.04-py3.11

# 构建Docker容器
docker run \
--device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v ${HOST_WORKSPACE}:${HOST_WORKSPACE} \
-w ${HOST_WORKSPACE} \
--shm-size=100g \
--privileged=true \
-itd \
--net=host \
--name ${YOUR_CONTAINER_NAME} \
quay.io/ascend/cann:9.0.0-beta.1-a3-ubuntu22.04-py3.11 \
/bin/bash

# 进入容器
docker exec -it ${YOUR_CONTAINER_NAME} bash
```

> `HOST_WORKSPACE` 用于挂载宿主机上存放本样例代码、数据集与权重的目录（建议直接挂 cann-recipes-train 所在的父目录），`YOUR_CONTAINER_NAME` 自定义容器名。

2. 进入容器后安装 `ifconfig`（多机训练拉起脚本依赖）。

```bash
apt-get update && apt-get install -y net-tools
```

3. 源码准备。

```bash
# 下载本样例所在代码仓
git clone https://gitcode.com/cann/cann-recipes-train.git
# 进入示例所在路径
cd cann-recipes-train/llm_pretrain/deepseekv32/

# 拉取训练框架 torchtitan-npu v0.2.2（只拉这一个分支）
git clone -b v0.2.2 --single-branch https://gitcode.com/cann/torchtitan-npu.git

# 安装运行时依赖，并以可编辑模式安装 torchtitan-npu
cd torchtitan-npu
pip install -r requirements.txt
pip install -e .
cd ..
```

> **多机共享盘场景**：若多台机器挂载同一份共享盘上的 `torchtitan-npu/`，请**跳过** `pip install -e .`，
> 仅在每台机器各自执行 `pip install -r requirements.txt` 即可。
> `pip install -e .` 会在源码目录写入 `*.egg-info/` 和 `__editable__*.pth`，多机并发执行会争抢写共享盘。
> 跳过后启动训练时必须 `cd torchtitan-npu/` 再调用 `python -m torchtitan_npu.entry`（`run_train.sh` 默认就是这样调用的），
> 这样 Python 才能从 cwd 找到 `torchtitan_npu` 包。

## 数据集准备

首先创建数据集路径（位于 torchtitan-npu 下，与训练配置中的 `dataset_path` 对齐）

```bash
mkdir -p ./torchtitan-npu/tests/assets/enwiki
```

下载[enwiki 的parquet数据](https://huggingface.co/datasets/lsb/enwiki20230101)到`./torchtitan-npu/tests/assets`路径下面

可以使用下面的命令下载数据集

```bash
cd ./torchtitan-npu/tests/assets
hf download lsb/enwiki20230101 --repo-type=dataset --local-dir .
cd ../../..
```

然后使用parquet转换脚本将其转换成json格式

```bash
# 全量转换（parquet 总大小约 12GB，实测转换耗时约 5 分钟，转换后 jsonl 约 20GB）
python ./utils/parquet2json.py --input ./torchtitan-npu/tests/assets/data --output ./torchtitan-npu/tests/assets/enwiki/enwiki.jsonl

# debug 用，只取 10000 行
python ./utils/parquet2json.py --input ./torchtitan-npu/tests/assets/data --output ./torchtitan-npu/tests/assets/enwiki/enwiki.jsonl --max-rows 10000
```

## 模型权重准备
本样例使用的DeepSeek-V3.2模型权重准备方法如下：
```bash
# 从魔塔社区下载模型的基础文件（不加载权重实验也需要执行这步操作）
mkdir -p /data/models/DeepSeek-V3.2
pip install modelscope

# 下载DeepSeek-V3.2完整FP8权重至指定目录（此步骤需要目录所在磁盘有650GB以上空间）
modelscope download --model deepseek-ai/DeepSeek-V3.2 --local_dir /data/models/DeepSeek-V3.2
```

在各个节点上使用`weight_convert.sh`脚本完成FP8到Bfloat16权重转换。

>入参介绍：`input_fp8_hf_path`：原始fp8权重路径；`output_hf_path`：转换后输出的权重路径；`quant_mode`：量化模式

如果权重转换的运行环境为NPU，需要先执行：

```shell
cann_path=/usr/local/Ascend/ascend-toolkit/latest  # cann包安装路径
source ${cann_path}/bin/setenv.bash
```

权重转换拉起示例（在 `deepseekv32/` 目录下执行）：

```shell
# 转换为Bfloat16权重
bash utils/weight_convert.sh --input_fp8_hf_path /data/models/DeepSeek-V3.2 --output_hf_path /data/models/DeepSeek-V3.2-bf16 --quant_mode bfloat16
```

## pretrain执行

训练相关的启动脚本和toml配置均位于 `torchtitan-npu` 仓内，进入其目录后执行：

```shell
cd torchtitan-npu

# 示例执行，2层（1moe，1dense）裁剪模型（请提前配置toml文件中相关地址）
NGPU=16 CONFIG_FILE="./torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_debug.toml" bash ./scripts/run_train.sh

# 开启inductor compile
NGPU=16 CONFIG_FILE="./torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_debug.toml" bash ./scripts/run_train.sh --compile.enable --model.converters "npu_bypass_triton_codegen"

# 64卡671B参数全量预训练拉起脚本（要在8机上同时拉起），运行前需修改 scripts/run_train_multinodes.sh 中的 IPs、Network_Interface 变量
CONFIG_FILE="./torchtitan_npu/models/deepseek_v32/train_configs/deepseek_v32_671b_61layers_32k_128die.toml" bash ./scripts/run_train_multinodes.sh
```
