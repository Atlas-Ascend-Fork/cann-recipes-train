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

- 本次训练多机使用配置文件 CONFIG=deepseek_v4_1_flash
- 拉起训练前请重点确认以下路径配置与实际环境一致：

## 启动 CPT 训练

* 根据使用实际的网卡、节点 IP等，修改多机训练脚本配置，参考 `torchtitan-npu` [快速上手](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/user-guides/quickstart.md) 文档中的“[多机训练任务](https://gitcode.com/cann/torchtitan-npu/blob/master/docs/user-guides/quickstart.md#多机训练任务)”一节。


* 进入各节点上的 `torchtitan-npu` 源码目录后，在所有参与训练的节点上同时执行如下命令，即可启动 `DeepSeek-V4.1-Flash` 多机CPT训练任务，默认使能 MXFP8 低精度, 默认使能 compile AutoFuse：

```shell
HF_ASSETS_PATH=/data/models/DeepSeek-V4.1-Flash/deepseek_v41_tokenizer \
bash examples/deepseek_v4_1/deepseek_v4_1_flash_cpt_4k_a5.sh --training.steps 500
```

## 启动图文混合 SFT 训练

SFT 的数据适配与官方 encoding 解耦：Encoder 动态调用模型目录内 `encoding/encoding.py` 的 `encode_messages` 接口，保持原始编码规则，训练侧生成 assistant loss mask。

典型数据集可从 Hugging Face 获取：[LLaVA-Instruct-150K 开源数据集下载链接](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/conversation_58k.json)。

按前文 CPT 一致多机训练方式配置实际网卡、节点 IP 和权重路径，再在各节点容器内依次执行：

1. 进入源码目录。

   ```shell
   cd /home/code/torchtitan-npu
   ```

2. 设置模型资源与数据集路径。

   ```shell
   export HF_ASSETS_PATH=/data/models/DeepSeek-V4.1-Flash/deepseek_v41_tokenizer
   export DATASET_PATH=/path/to/train.jsonl
   ```

3. 在所有参与训练的节点上同时启动 SFT。

   ```shell
   bash examples/deepseek_v4_1/deepseek_v4_1_flash_sft_4k_a5.sh \
     --training.steps 128 --lr-scheduler.total-steps 128
   ```

配置参数可在命令末尾覆盖。

使用 LLaVA 子集训练 128 步后，重新加载权重得到的训练后评测结果如下，评测对象为电脑桌图文对话：

| 指标 | LLaVA 子集 SFT 前 | LLaVA 子集 SFT 后（128 步） |
| --- | ---: | ---: |
| 交叉熵（CE） | 3.894351 | 5.180367e-6 |
| 监督 token 准确率 | 27.12% | 100% |
| 监督序列完全匹配 | 否 | 是 |

拟合实验中的两个数据样例：

| 图片 | 问题 | 训练答案 |
| --- | --- | --- |
| ![船](images/llava_129141.jpg) | What color is the boat in the image? | The boat in the image is blue and white. |
| ![电脑桌](images/llava_390311.jpg) | What color is the table where the computer is? | The table where the computer is placed is white. |

对于上表第 2 行的问题 "What color is the table where the computer is?"，模型生成回答 "The table where the computer is placed is white."，可见图文混合 SFT 训练之后的结果符合预期。
