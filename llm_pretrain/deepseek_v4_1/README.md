# 基于TorchTitan-NPU的 DeepSeek-V4.1-Flash 训练部署指导

## 概述

本文面向 `torchtitan-npu` 仓中的 `DeepSeek-V4.1-Flash` 模型训练场景，介绍源码获取、数据准备、模型权重准备与 BF16 转换、镜像准备、Docker 容器拉起、训练配置说明以及多机训练启动方式。


## 硬件与软件要求

### DeepSeek-V4.1-Flash 模型基于[master分支](https://gitcode.com/cann/torchtitan-npu)的硬件与软件要求
| 项目 | 要求 |
| --- | --- |
| 产品型号 | Atlas A5 系列 |
| 最小卡数要求 | Flash 8 机 64 卡 A3，16 机 128 卡 A5 |
| 操作系统 | Linux ARM |
| 驱动版本 | Ascend HDK 25.5.6 |
| CANN版本 | 9.3.0 |
| 镜像版本 | [dsv4.1_train_torchtitan_a5_cann_master_newest_0916](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/deepseek_train/dsv4.1_train_torchtitan_a5_cann_master_newest_0916.tar.xz) |

> [!NOTE]
> **安装校验与版本选择建议：**
>
> 1. 请使用 `npu-smi info` 检查 Ascend NPU **固件**与**驱动**是否正确安装且版本匹配。
> 2. 请将 NPU 固件/驱动更新至 `25.5.6`，并下载安装对应的[固件与驱动包（25.5.6）](https://support.huawei.com/enterprise/zh/atlas-computing/ascend-hdk-pid-252764743/software/269306426?idAbsPath=fixnode01|23710424|251366513|254884019|261408772|252764743)。
> 3. 安装步骤请参考：[社区安装指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)。



## 源码准备

执行如下命令拉取源码：

```shell
mkdir -p /home/code && cd /home/code/
git clone -b master https://gitcode.com/cann/torchtitan-npu.git
cd torchtitan-npu
```

 cann-recipes 仓不归档对应的训练拉起脚本, 直接使用 torchtitan-npu 仓 examples/deepseek_v4_1 目录下拉起脚本



## 获取Docker镜像

```shell
docker load -i dsv4.1_train_torchtitan_a5_cann_master_newest_0916.tar.xz
```

加载后镜像版本如下：

```shell
dsv4.1_train_torchtitan_a5:cann_master_newest_0916
```

## 启动Docker容器

dsv4.1_train_torchtitan_a5:cann_master_newest_0916 镜像支持 DeepSeek-V4.1-Flash 模型，以该镜像的 docker 启动举例，执行如下命令启动容器：

```shell
docker run -u root -itd --ulimit nproc=65535:65535 --ipc=host \
        --device=/dev/davinci0 \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -v /usr/local/dcmi/:/usr/local/dcmi/ \
        -v /usr/local/sbin/npu-smi/:/usr/local/sbin/npu-smi/ \
        -v /var/log/:/var/log/ \
        -v /data/:/data/ \
        -v /home/:/home/ \
        -v /mnt/:/mnt/ \
        -v /etc/:/etc/ \
        -v /tmp/:/tmp/ \
        -v /npu/:/npu/ \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/sbin/:/usr/local/sbin/ \
        -v /dev/shm/:/dev/shm/ \
        -v /root/.pip:/root/.pip \
        -v /usr/bin/:/usr/bin/ \
        -v /usr/lib64/:/usr/lib64/ \
        -w / \
        --net=host \
        --entrypoint /bin/bash \
        --name dsv4.1_train_torchtitan_a5 \
        --privileged dsv4.1_train_torchtitan_a5:cann_master_newest_0916 \
        #-c "sleep infinity"
```

进入容器：

```shell
docker exec -it dsv4_train_torchtitan_v3.0 /bin/bash
```

在容器内执行环境变量初始化：

```shell
source /usr/local/Ascend/cann/set_env.sh
```

## 数据集准备

该配置默认使用仓内样例数据集：

```shell
dataset = "cc12m-test"
```

若直接按默认配置拉起训练，默认的 `cc12m-test` 数据集；若使用自定义数据集，请提前准备好数据目录，并在后续训练配置中修改 `dataset` 和 `dataset_path`。

## 模型权重准备

### DeepSeek-V4.1-Flash 模型权重准备

若使用 `DeepSeek-V4.1-Flash` 权重进行CPT训练，建议先从 [huggingface](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 下载原始权重和配置文件到统一目录，例如：

```shell
mkdir -p /data/models/DeepSeek-V4.1-Flash
```

将原始权重转换为 BF16 权重，输出目录与训练配置文件保持一致，例如：

```shell
mkdir -p /data/models/DeepSeek-V4.1-Flash-bf16
```

权重转换使用 `cann-recipes-train` 仓中的 `convert_bf16.py` 脚本。示例如下：

```shell
cd /home/code/
git clone https://gitcode.com/cann/cann-recipes-train.git
cd cann-recipes-train/llm_pretrain/deepseek_v4_1/utils
python3 convert_bf16.py \
  --input_hf_path /data/models/DeepSeek-V4.1-Flash \
  --output_hf_path /data/models/DeepSeek-V4.1-Flash-bf16
```



## 训练配置

### DeepSeek-V4.1-Flash 训练配置

本次训练单机使用配置文件 CONFIG=deepseek_v4_1_flash_40layers_16experts_multimodal
本次训练多机使用配置文件 CONFIG=deepseek_v4_1_flash
拉起训练前请重点确认以下路径配置与实际环境一致：



## 启动训练

* 进入各节点上的 `torchtitan-npu` 源码目录后执行如下命令，即可启动 `DeepSeek-V4.1-Flash` 单机 BF16 CPT训练任务, 默认使能 compile AutoFuse：

```shell
HF_ASSETS_PATH=/data/models/DeepSeek-V4.1-Flash/deepseek_v41_tokenizer \
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh \
--training.steps 500 --extension.quantization.no-enable-quantized-training
```

* 进入各节点上的 `torchtitan-npu` 源码目录后执行如下命令，即可启动 `DeepSeek-V4.1-Flash` 单机 MXFP8 CPT训练任务，默认使能 MXFP8 低精度, 默认使能 compile AutoFuse：

```shell
HF_ASSETS_PATH=/data/models/DeepSeek-V4.1-Flash/deepseek_v41_tokenizer \
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh --training.steps 500
```

* 根据使用实际的网卡、节点 IP等，修改多机训练脚本配置，参考 `torchtitan-npu` [快速上手](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/user-guides/quickstart.md) 文档中的“[多机训练任务](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/user-guides/quickstart.md#多机训练任务)”一节。


* 进入各节点上的 `torchtitan-npu` 源码目录后，在所有参与训练的节点上同时执行如下命令，即可启动 `DeepSeek-V4.1-Flash` 多机CPT训练任务，默认使能 MXFP8 低精度, 默认使能 compile AutoFuse：

```shell
HF_ASSETS_PATH=/data/models/DeepSeek-V4.1-Flash/deepseek_v41_tokenizer \
bash examples/deepseek_v4_1/deepseek_v4_1_flash_cpt_4k_a5.sh --training.steps 500
```

