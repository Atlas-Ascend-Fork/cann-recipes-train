# TorchTitan Ascend C 算子生成全参数 SFT 样例

本样例基于 **TorchTitan-NPU**，对 Qwen3.6-27B 进行 Ascend C 算子生成全参数监督微调。训练配置面向 Ascend C 算子生成语料和 CP8 × FSDP2，支持完整 DCP 保存与 Hugging Face 权重导出。
训练效果及训练过程报告请参考：[实践报告](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/llm_sft/qwen36_ascendc/zh/qwen36_ascendc_sft_report.md)。

## 文件说明

| 文件 | 职责 |
| --- | --- |
| `cannbot_recipe.py` | 配置训练、并行、优化器和 `NoThinkTemplateTokenizer` |
| `data_process.py` | 校验并合并 JSONL 文件，转换样本字段 |
| `run_train.sh` | 启动训练、加载数据、保存权重 |
| `Dockerfile` | 固定依赖版本，构建训练环境 |

`run_train.sh` 调用 `data_process.py` 校验 schema，并按 `DATA_FILES` 中的顺序合并 JSONL 文件。相邻非空文件之间缺少换行时补一个 LF。`data_process.py` 也在 dataloader 读取样本时转换字段。数据切分、tokenization、greedy packing、padding 和 dataloader 状态恢复统一使用 TorchTitan 官方 `ChatDataLoader`/`ChatDataset`。

## 环境版本

| 组件 | 发布构建版本或 revision |
| --- | --- |
| 硬件 | 8 卡 Atlas A3（16 die） |
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

#### 启动并进入容器

在已安装昇腾驱动的 Atlas A3 宿主机上执行。示例使用默认驱动路径，透传 8 卡 Atlas A3 对应的 16 个设备节点。可将 `QWEN36_WORKSPACE` 改为宿主机上的绝对路径。

```bash
export QWEN36_WORKSPACE="${QWEN36_WORKSPACE:-$HOME/qwen36-sft}"
mkdir -p "$QWEN36_WORKSPACE"/{models,data,runs,exports}

docker run -dit --name cannbot-qwen36-sft --network=host --ipc=host \
  --device=/dev/davinci{0..15} \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
  --mount type=bind,src=/usr/local/dcmi,dst=/usr/local/dcmi,readonly \
  --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \
  --mount "type=bind,src=$QWEN36_WORKSPACE/models,dst=/workspace/models,readonly" \
  --mount "type=bind,src=$QWEN36_WORKSPACE/data,dst=/workspace/data,readonly" \
  --mount "type=bind,src=$QWEN36_WORKSPACE/runs,dst=/workspace/runs" \
  --mount "type=bind,src=$QWEN36_WORKSPACE/exports,dst=/workspace/exports" \
  cannbot-qwen36-sft:cannbot-domain

docker exec -it cannbot-qwen36-sft /bin/bash
```

进入容器后，工作目录是 `/opt/cannbot_torchtitan`。输入 `exit` 回到宿主机终端，容器会继续运行。在该宿主机终端完成第 2、3 步，第 4 步再进入容器。

### 2. 校验训练数据

开放数据可从[项目数据发布帖](https://gitcode.com/cann/cann-recipes-train/discussions/2)下载，按[数据生成说明](../data_generator/README.md)生成训练用的 JSONL，放入宿主机的 `$QWEN36_WORKSPACE/data` 目录。

输入为 UTF-8 JSONL，支持以下两种原始字段：

```json
{"input":"Write an operator.","output":"..."}
{"prompt":"Write an operator.","response":"..."}
```

训练读取时分别映射 `input → user`、`output → assistant` 或 `prompt → user`、`response → assistant`。`run_train.sh` 会生成数据 manifest，并为混合字段提供明确 schema，避免 JSON loader 在文件边界重新推断字段。

### 3. 获取基础模型权重

在宿主机用 Hugging Face CLI 下载完整的 [Qwen3.6-27B 模型目录](https://huggingface.co/Qwen/Qwen3.6-27B)，其中包含权重分片、配置和 tokenizer 文件。

```bash
python3 -m pip install -U huggingface_hub
hf download Qwen/Qwen3.6-27B --local-dir "$QWEN36_WORKSPACE/models/Qwen3.6-27B"
```

第 1 步的挂载使模型在容器内可通过 `/workspace/models/Qwen3.6-27B` 访问。训练脚本从 `HF_ASSETS_PATH` 指定的目录加载基础模型。

### 4. 启动训练

在宿主机再次执行第 1 步的 `docker exec` 命令，然后在容器中运行以下命令。

```bash
export HF_ASSETS_PATH=/workspace/models/Qwen3.6-27B
export DATA_FILES=/workspace/data/transfer.jsonl,/workspace/data/rft.jsonl,/workspace/data/sampled.jsonl
export DUMP_FOLDER=/workspace/runs/qwen36-sft
export CANNBOT_TRAIN_STEPS=500
export CANNBOT_CHECKPOINT_INTERVAL=100
bash run_train.sh
```

将 `DATA_FILES` 替换为容器内可读取的 JSONL 路径，按读取顺序用逗号分隔。输入路径不能含逗号或换行，`DUMP_FOLDER` 不能含逗号。

`CANNBOT_TRAIN_STEPS` 是 optimizer step 数，`CANNBOT_CHECKPOINT_INTERVAL` 是同步 DCP 的保存间隔。根据实际数据规模和训练计划，将两项都设为正整数。


### 5. 导出 Hugging Face 权重

在容器内执行导出。导出的权重保存在宿主机的 `$QWEN36_WORKSPACE/exports` 目录。

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
