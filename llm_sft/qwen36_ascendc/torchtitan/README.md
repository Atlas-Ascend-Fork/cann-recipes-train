# TorchTitan Ascend C 算子生成全参数 SFT 样例

本样例基于 **TorchTitan-NPU**，对 Qwen3.6-27B 进行 Ascend C 算子生成全参数监督微调。训练配置面向 Ascend C 算子生成语料和 CP8 × FSDP2，支持完整 DCP 保存与 Hugging Face 权重导出。
训练效果及训练过程报告请参考：[实践报告](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/llm_sft/qwen36_ascendc/zh/qwen36_ascendc_sft_report.md)。

## 文件说明

| 文件 | 职责 |
| --- | --- |
| `cannbot_recipe.py` | 配置训练、并行、优化器和 `NoThinkTemplateTokenizer` |
| `data_process.py` | 校验原始 JSONL schema 并转换样本字段 |
| `run_train.sh` | 启动训练、加载数据、保存权重 |
| `Dockerfile` | 固定依赖版本，构建训练环境 |

`run_train.sh` 按 `DATA_FILES` 中的顺序读取 JSONL 训练数据。`data_process.py` 只校验 schema，并在 dataloader 读取样本时转换字段。数据切分、tokenization、greedy packing、padding 和 dataloader 状态恢复统一使用 TorchTitan 官方 `ChatDataLoader`/`ChatDataset`。

## 环境版本

| 组件 | 发布构建版本或 revision |
| --- | --- |
| 硬件 | 8 卡 910C（16 die） |
| CANN | 9.2.0 |
| TorchTitan | `c91448d20480c7b294314e68976823050002ebec` |
| TorchTitan-NPU | `2afd4c01aa5b1bc9a860a2bd67bdd212b1b1a8f7` |

依赖版本和源码 revision 均直接固定在 Dockerfile 中，使用 Dockerfile 进行构建即可获得以上版本配套。

## 快速开始

### 1. 构建镜像

```bash
cd llm_sft/qwen36_ascendc/torchtitan
docker build -t cannbot-qwen36-sft:cannbot-domain .
```

容器运行时需要挂载合法获取的 Hugging Face 基础模型、JSONL 数据和独立输出目录，并透传 8 卡 910C（16 die）及宿主机驱动。

### 2. 校验训练数据

输入为 UTF-8 JSONL，支持以下两种原始字段：

```json
{"input":"Write an operator.","output":"..."}
{"prompt":"Write an operator.","response":"..."}
```

训练读取时分别映射 `input → user`、`output → assistant` 或 `prompt → user`、`response → assistant`。`run_train.sh` 会生成数据 manifest，并为混合字段提供明确 schema，避免 JSON loader 在文件边界重新推断字段。

### 3. 启动训练

```bash
export HF_ASSETS_PATH=/workspace/models/Qwen3.6-27B
export DATA_FILES=/workspace/data/transfer.jsonl,/workspace/data/rft.jsonl,/workspace/data/sampled.jsonl
export DUMP_FOLDER=/workspace/runs/qwen36-sft
export CANNBOT_TRAIN_STEPS=500
export CANNBOT_CHECKPOINT_INTERVAL=100
bash run_train.sh
```

`CANNBOT_TRAIN_STEPS` 是 optimizer step 数，`CANNBOT_CHECKPOINT_INTERVAL` 是同步 DCP 的保存间隔。它们应根据实际数据规模和训练计划设置，不由样本文件名或固定数据集推断。


### 4. 导出 Hugging Face 权重

```bash
bash run_train.sh export-hf \
  /workspace/runs/qwen36-sft/checkpoint/step-N \
  /workspace/models/Qwen3.6-27B \
  /workspace/exports/step-N
```


## 训练配置

| 参数 | 值 |
| --- | --- |
| world size / CP / FSDP / TP | 16 / 8 / 2 / 1 |
| local batch / DP / gradient accumulation | 1 / 2 / 1 |
| global batch size | 2 |
| optimizer | fused AdamW |
| learning rate | `1e-5` |
| Adam betas / eps | `(0.9, 0.999)` / `1e-8` |
| warmup / decay | 10% / cosine decay to zero |
| weight decay / max norm | 0 / `inf`（不裁剪） |
| template | `qwen3_6_nothink` |
| seed | 42 |

以上为基于已有数据实践获得的较优超参数配置。

## 数据构成

训练数据应满足“校验训练数据”一节所述的 JSONL schema。上层目录 `../data_generator` 提供配套的数据生成脚本，可将算子规格与代码转换为固定 Markdown 单轮格式的 JSONL 数据，具体用法见该目录 README。
