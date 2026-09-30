# DeepSeek-V4-Flash-0731 LoRA SFT 使用指南

本文档说明如何基于 MindSpeed-LLM 对 DeepSeek-V4-Flash-0731 进行 64K no-think LoRA SFT。具体修改请参考[补丁说明](./patch_description.md)。

示例环境：

- CANN：9.1.0
- 硬件：两台 Atlas A3 节点，每台 8 卡，共 16 卡

## 目录

- [环境安装](#环境安装)
- [代码获取](#代码获取)
- [初始模型转换](#初始模型转换)
- [数据准备](#数据准备)
- [启动训练](#启动训练)
- [LoRA 合并与导出](#lora-合并与导出)
- [附录：训练与续训曲线](#附录训练与续训曲线)

## 环境安装

在每台训练节点的宿主机拉取镜像：

```bash
docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12
```

启动容器：

```bash
docker run -itd --name dsv4_lora \
  --ulimit nproc=65535:65535 \
  --privileged --network=host --ipc=host \
  --device=/dev/davinci0:/dev/davinci0 \
  --device=/dev/davinci1:/dev/davinci1 \
  --device=/dev/davinci2:/dev/davinci2 \
  --device=/dev/davinci3:/dev/davinci3 \
  --device=/dev/davinci4:/dev/davinci4 \
  --device=/dev/davinci5:/dev/davinci5 \
  --device=/dev/davinci6:/dev/davinci6 \
  --device=/dev/davinci7:/dev/davinci7 \
  --device=/dev/davinci8:/dev/davinci8 \
  --device=/dev/davinci9:/dev/davinci9 \
  --device=/dev/davinci10:/dev/davinci10 \
  --device=/dev/davinci11:/dev/davinci11 \
  --device=/dev/davinci12:/dev/davinci12 \
  --device=/dev/davinci13:/dev/davinci13 \
  --device=/dev/davinci14:/dev/davinci14 \
  --device=/dev/davinci15:/dev/davinci15 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /data/:/data/ \
  -v /home/:/home/ \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /root/.pip:/root/.pip \
  -v /etc/hosts:/etc/hosts \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12 \
  /bin/bash
```

进入容器，后续命令在容器内执行：

```bash
docker exec -it dsv4_lora bash
```

## 代码获取

获取指定版本的 MindSpeed-LLM 并应用本项目补丁。将 `RECIPE_DIR` 改为本目录在容器内的实际绝对路径，下例将训练代码放在 `/data/dsv4_sft/MindSpeed-LLM`：

```bash
RECIPE_DIR=/data/cann-recipes-train/llm_sft/dsv4_flash_ascendc/mindspeed-llm
mkdir -p /data/dsv4_sft
cd /data/dsv4_sft

git clone https://gitcode.com/Ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout --detach 0ebf60e35590f981bf86de42c453e283366e68c2

git apply --check --index "${RECIPE_DIR}/mindspeed-llm.patch"
git apply --index "${RECIPE_DIR}/mindspeed-llm.patch"
git submodule update --init 3rdparty/Megatron-LM 3rdparty/MindSpeed
```

补丁包含 MindSpeed 子模块版本调整，需要使用 `git apply --index`，再执行上述子模块初始化命令，以获取指定版本。

两台节点均执行上述步骤。后续命令均在 `MindSpeed-LLM` 根目录执行。

## 初始模型转换

准备 `deepseek-ai/DeepSeek-V4-Flash-0731` 的原始 Hugging Face 格式权重，目录中需包含权重分片、索引、配置及 tokenizer 文件。以下路径均为示例，请按实际位置设置：

```bash
export RAW_HF_DIR=/data/models/DeepSeek-V4-Flash-0731
export BF16_HF_DIR=/data/models/DeepSeek-V4-Flash-0731-BF16
export MCORE_DIR=/data/models/DeepSeek-V4-Flash-0731-MCore-TP1-PP2-EP16

bash examples/mcore/deepseek4_flash/ckpt_dequant_deepseek4_flash_0731_to_bf16.sh \
  "${RAW_HF_DIR}" "${BF16_HF_DIR}"

TARGET_TP=1 TARGET_PP=2 TARGET_EP=16 \
bash examples/mcore/deepseek4_flash/ckpt_convert_deepseek4_flash_0731_hf2mcore.sh \
  "${BF16_HF_DIR}" "${MCORE_DIR}"
```

每一步使用独立的新目录或空目录作为输出。反量化将原始混合 FP8/FP4 权重转为 BF16，保留原有 FP32 状态及整数路由表。HF→MCore 只转换 43 层模型主体，不转换附带的 dSPARK 模块；MCore 使用 44 个层位置，其中第 43 号为 noop，MTP=0。

## 数据准备

JSONL 每行包含且只包含 `instruction`、`input`、`output` 三个字符串字段。`instruction` 可以为空，`input` 和 `output` 必须非空。例如：

```json
{"instruction":"根据要求实现 AscendC 算子。","input":"算子需求及接口说明。","output":"完整的训练答案。"}
```

使用训练底座对应的 tokenizer，生成 64K no-think 数据：

```bash
bash examples/mcore/deepseek4_flash/prepare_data.sh \
  --input /data/datasets/train.jsonl \
  --tokenizer-name-or-path /data/models/DeepSeek-V4-Flash-0731-BF16 \
  --output-dir /data/datasets/dsv4_flash_0731_nothink_64k \
  --seq-length 65536
```

多个 JSONL 文件分别使用一个 `--input`，按传参顺序合并。例如：

```bash
bash examples/mcore/deepseek4_flash/prepare_data.sh \
  --input /data/datasets/train_1.jsonl \
  --input /data/datasets/train_2.jsonl \
  --input /data/datasets/train_3.jsonl \
  --tokenizer-name-or-path /data/models/DeepSeek-V4-Flash-0731-BF16 \
  --output-dir /data/datasets/dsv4_flash_0731_nothink_64k \
  --seq-length 65536
```

每个 `--input` 必须指向具体的 JSONL 文件，不能直接传入目录。处理同一目录中的多个文件时，也按上述方式逐个列出文件路径。

脚本使用 `--prompt-type deepseek4 --enable-thinking false`，并核对转换后的 token、label 和 loss mask。加入模板及结束符后的总长度 **大于等于 65536** 的样本默认整条跳过；保留样本不会被截断。设置 `--overlength-policy error` 可改为遇到这类样本时报错。

输出目录中的 `dataset_manifest.json` 记录保留样本数，训练脚本据此计算迭代数。文件名中的 `alpaca_packed` 是当前预处理器的命名，不表示已经把多个样本拼接在一起。

## 启动训练

补丁提供两份 no-think 启动脚本，默认配置如下：

| 配置 | 固定长度脚本 | 动态 padding 脚本（推荐） |
| --- | --- | --- |
| 文件名 | `lora_finetune_deepseek4_flash_0731_nothink_64k_A3_ptd.sh` | `lora_finetune_deepseek4_flash_0731_nothink_dynamic_64k_A3_ptd.sh` |
| padding | 每个 micro-batch 均补齐到 65536 | 按 micro-batch 内最长样本，向上补齐到 4096 的倍数，上限 65536 |
| GBS | 8 | 8 |
| LoRA rank / alpha | 32 / 64 | 32 / 64 |

两份脚本均为 TP1/PP2/CP16/EP16、MBS=1、LR=`1e-5`、5 epochs。动态 padding 复用同一份数据，不做跨样本 packing。两份脚本的默认 GBS、rank 和 alpha 已对齐；自定义参数时，若要只对比 padding 的影响，也应保持这些参数一致。两份脚本均支持通过 `GBS` 环境变量设置全局 batch size，动态脚本还支持通过 `LORA_R`、`LORA_ALPHA` 设置 LoRA 参数。

动态 padding 的长度按 `ceil(L_max / 4096) × 4096` 计算，`L_max` 是当前 micro-batch 内最长样本经过模板编码后的总 token 数。例如，9000 token 补齐到 12288，65000 token 补齐到 65536。

**建议优先使用动态 padding 脚本。** 在当前启动脚本配置下，实测动态 padding 平均每个 iteration 约 **14s**，固定长度约 **65s**。固定长度 padding 可作为固定输入形状的对照基线，便于性能分析和排查变长执行相关问题。

loss、grad_norm 曲线及耗时对比见[附录：固定长度与动态 padding](#固定长度与动态-padding)。

在两台节点分别设置相同的数据及模型路径。模型和数据须在两台节点可访问；节点本地保存的 checkpoint 分片需一并保留。

```bash
export MASTER_ADDR="<节点0训练网卡IP>"
export TOKENIZER_PATH=/data/models/DeepSeek-V4-Flash-0731-BF16
export DATA_OUTPUT_DIR=/data/datasets/dsv4_flash_0731_nothink_64k
export CKPT_SAVE_DIR=/data/checkpoints/dsv4_flash_0731_nothink_64k
export MCORE_DIR=/data/models/DeepSeek-V4-Flash-0731-MCore-TP1-PP2-EP16
```

以推荐的动态 padding 脚本为例，节点 0 执行：

```bash
NODE_RANK=0 bash examples/mcore/deepseek4_flash/lora_finetune_deepseek4_flash_0731_nothink_dynamic_64k_A3_ptd.sh \
  "${MCORE_DIR}"
```

节点 1 执行：

```bash
NODE_RANK=1 bash examples/mcore/deepseek4_flash/lora_finetune_deepseek4_flash_0731_nothink_dynamic_64k_A3_ptd.sh \
  "${MCORE_DIR}"
```

如需固定长度训练，两台节点都替换为表格中的固定长度脚本。脚本内的默认路径为通用示例，请根据实际环境设置上述环境变量。

默认 `STEPS_PER_EPOCH=ceil(保留样本数/GBS)`，训练迭代数为其 5 倍，每隔 `STEPS_PER_EPOCH` 保存一次。可用 `TRAIN_ITERS`、`SAVE_INTERVAL` 覆盖。脚本保存 LoRA 权重、router bias、optimizer 和 RNG 状态；设置 `RESUME_DIR` 为包含 `latest_checkpointed_iteration.txt` 的 checkpoint 根目录可续训，底座参数仍传入原来的 `MCORE_DIR`。续训时保持数据、并行方式、batch size 和学习率计划一致；未保存的 optimizer/RNG 状态不能由补丁补回。

从 epoch 3 续训的曲线及训练进度对照见[附录：动态 padding 从 epoch 3 续训](#动态-padding-从-epoch-3-续训)。

## LoRA 合并与导出

先将所选 iteration 在所有节点保存的 rank 分片汇集到一个完整的 checkpoint 根目录，并确保其 `latest_checkpointed_iteration.txt` 指向该 iteration。合并需要同时提供训练时使用的 MCore 底座和 BF16 HF 配置/tokenizer 目录。

`CANN_ENV_SCRIPT` 应指向容器内实际的 `set_env.sh`，下例使用 `/usr/local/Ascend/cann/set_env.sh`。两份训练脚本和合并脚本均默认使用 rank=32、alpha=64；若训练时自定义了这些参数，合并时须使用相同值：

```bash
CANN_ENV_SCRIPT=/usr/local/Ascend/cann/set_env.sh \
BASE_MCORE=/data/models/DeepSeek-V4-Flash-0731-MCore-TP1-PP2-EP16 \
LORA_MCORE=/data/checkpoints/dsv4_flash_0731_nothink_64k_complete \
HF_META=/data/models/DeepSeek-V4-Flash-0731-BF16 \
OUT_HF=/data/models/DeepSeek-V4-Flash-0731-LoRA-BF16 \
LORA_R=32 LORA_ALPHA=64 CONVERT_DEVICE=0 \
bash examples/mcore/deepseek4_flash/ckpt_convert_deepseek4_flash_lora_mcore2hf.sh
```

`OUT_HF` 使用新目录或空目录。合并脚本的 target modules 必须与训练一致，默认已对齐这两份训练脚本。输出为合并后的 BF16 HF 模型主体，并包含 tokenizer 文件及整理后的模型配置。

## 附录：训练与续训曲线

两组完整训练均为 GBS=8、每个 epoch 206 步，共 5 个 epoch、1030 步；动态 padding 续训从 epoch 3 结束后的第 619 步运行至第 1030 步。

横坐标为 epoch，每 206 个 iteration 对应 1 个 epoch。浅色线为逐 iteration 原始值，深色线为 15 步移动平均。loss 对应日志中的 `lm loss`，grad_norm 对应 `grad norm`。

### 固定长度与动态 padding

<p align="center">
  <img src="../../../docs/llm_sft/dsv4_flash_ascendc/figure/padding_training_metrics.png" alt="固定长度与动态 padding 的训练 loss 和 grad_norm 曲线" width="640">
</p>

| 运行 | 平均每 iteration 耗时 | Epoch 1 平均 loss | Epoch 5 平均 loss | Epoch 5 平均 grad_norm |
| --- | --- | --- | --- | --- |
| 固定长度 padding | 65.24 s | 0.143511 | 0.060550 | 0.110816 |
| 动态 padding | 14.00 s | 0.142875 | 0.060401 | 0.110650 |

平均耗时取日志 `elapsed time per iteration` 的逐步均值，包含首步；loss 和 grad_norm 为对应 epoch 的逐步均值。两次运行的 loss 均随训练下降，走势接近。

### 动态 padding 从 epoch 3 续训

<p align="center">
  <img src="../../../docs/llm_sft/dsv4_flash_ascendc/figure/dynamic_padding_resume_metrics.png" alt="动态 padding 连续训练与 epoch 3 续训的 loss 和 grad_norm 对照" width="640">
</p>

竖虚线标出 epoch 3 结束的第 **618** 步，此时已消费 **4944** 个样本。蓝线为完整连续训练，橙色虚线为第 **619** 步开始的续训；分界线之后展示两条训练曲线。两条曲线的移动平均均在续训边界重新开始计算。

| 区间 | 运行 | 平均 loss | 平均 grad_norm |
| --- | --- | --- | --- |
| Epoch 4（619–824） | 连续训练 | 0.063784 | 0.110937 |
| Epoch 4（619–824） | 续训 | 0.063803 | 0.112117 |
| Epoch 5（825–1030） | 连续训练 | 0.060401 | 0.110650 |
| Epoch 5（825–1030） | 续训 | 0.060449 | 0.110078 |
