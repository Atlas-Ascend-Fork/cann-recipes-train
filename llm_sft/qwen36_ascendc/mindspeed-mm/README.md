# Qwen3.6-27B SFT 使用指南

本文档说明如何基于 MindSpeed-MM 对 Qwen3.6-27B 进行 SFT（监督微调）。
训练效果及训练过程报告请参考：[实践报告](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/llm_sft/qwen36_ascendc/zh/qwen36_ascendc_sft_report.md)。

示例环境与版本信息：

- CANN：9.0.0
- 硬件：Atlas A3 NPU 16 die

## 目录

- [代码获取](#代码获取)
- [环境安装](#环境安装)
- [初始模型下载与转换](#初始模型下载与转换)
- [数据准备与转换](#数据准备与转换)
- [启动训练](#启动训练)
- [Checkpoint 后处理](#checkpoint-后处理)

## 代码获取

在工作目录执行以下命令，获取指定版本的 MindSpeed-MM 并应用本项目补丁：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
git checkout aaed711fd4750857f104e8d832766ed915ff9ef0

git apply --check ../mindspeed-mm.patch
git apply ../mindspeed-mm.patch
```

后续命令均在 `MindSpeed-MM` 根目录执行。若补丁文件不在上级目录，请将 `../mindspeed-mm.patch` 替换为实际路径。

## 环境安装

### 1. CANN 环境配置

请先按照昇腾软件安装指南配置驱动、固件以及 CANN Toolkit。本示例使用 **CANN 9.0.0**。

确认 CANN 环境变量已生效，例如按实际安装路径执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

### 2. Python 环境配置

推荐使用 Conda 进行 Python 包管理：

```bash
conda create -n qw36-sft python=3.11 -y
conda activate qw36-sft

bash scripts/install.sh --msid eb10b92
pip install transformers==5.2.0 accelerate==1.2.0
pip install triton-ascend==3.2.1 \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

# 如果出现 ModuleNotFoundError: No module named 'pkg_resources'，请使用低版本的 setuptools
pip install setuptools==81.0.0
```

## 初始模型下载与转换

### 1. 下载模型

模型下载地址：

ModelScope：https://www.modelscope.cn/models/Qwen/Qwen3.6-27B

HuggingFace：https://huggingface.co/Qwen/Qwen3.6-27B

将 Qwen3.6-27B 的 Hugging Face 格式权重放置到以下目录，目录名需与配置文件保持一致：

```text
MindSpeed-MM/ckpt/hf_path/Qwen3.6-27B/
```

目录中应包含模型权重、配置文件以及 tokenizer 相关文件（例如 `config.json`、`*.safetensors`、`tokenizer.json`）。

### 2. 转换为 DCP 格式

示例配置启用了 `init_model_with_meta_device`，因此训练前需要先将 Hugging Face 权重转换为 DCP：

```bash
bash cvt_ckpt_hf2dcp.sh
```

脚本使用固定输入和输出路径（如需修改路径请修改脚本）：

```text
输入：ckpt/hf_path/Qwen3.6-27B
输出：ckpt/dcp_path/Qwen3.6-27B
```

转换完成后，输出目录应包含 `release/` 和 `latest_checkpointed_iteration.txt`。配置文件中的 `training.load` 应指向 `ckpt/dcp_path/Qwen3.6-27B`，不要直接指向 `release/` 子目录。

## 数据准备与转换

### 1. 下载数据

数据集下载地址：https://gitcode.com/cann/cann-recipes-train/discussions/2

### 2. 生成数据集

使用当前仓库提供的 data_generator 生成数据集，在生成过程中会调用当前仓库提供的 prompt_generator 生成 prompt：

```bash
python path-to-data_generator/generate_data.py \
  --source-root path-to-data \
  --output-dir path-to-generated-data \
  --filter-ops exp foreach_addcdiv_scalar foreach_norm gelu masked_scale mish sigmoid swi_glu \
               apply_adam_w apply_rotary_pos_emb arg_max cross_entropy_loss cummin dynamic_quant gather gcd \
               grid_sampler_3d group_norm maximum resize_bilinear rms_norm scatter softmax unsorted_segment_sum \
  --clean
```

本项目的数据处理脚本接收 JSONL 文件。每行必须是一个 JSON 对象，支持以下任一格式：

```json
{"instruction": "任务描述", "input": "任务内容", "output": "答案"}
```

或：

```json
{"prompt": "任务内容", "response": "答案"}
```

其中 `instruction`、`input`、`output` 格式要求包含 `input` 和 `output`；`instruction` 非空时会与 `input` 以换行拼接。`prompt`、`response` 格式要求两个字段同时存在。

使用当前仓库提供的 data_generator 生成的数据集是上述的第一种格式。

### 3. 放置 JSONL 文件

在 MindSpeed-MM 根目录创建数据目录，并放入训练数据：

```bash
mkdir -p data/jsonl data/json
```

默认示例需要以下两个文件：

```text
data/jsonl/ops-data.jsonl
data/jsonl/sampled-data.jsonl
```

文件名可以调整，但需要同步修改 YAML 配置中的 `data.dataset_param.basic_parameters.dataset`。

### 4. 转换为训练 JSON

执行批量转换：

```bash
bash cvt_data_format_jsonl2json.sh \
  'data/jsonl/*.jsonl' \
  'data/json/*.json'
```

转换后的文件会写入 `data/json/`，并被示例配置中的 `./data/json/*.json` 自动读取。脚本会校验每行 JSON 语法、对象类型以及必需字段；输入存在错误时会直接报错，请修正后重新执行。

## 启动训练

将本项目提供的配置文件放置到 `MindSpeed-MM/examples/qwen3_6` 目录下，示例配置文件为：

```text
packing 全参 SFT 配置
examples/qwen3_6/qwen3_6_27B_config_packing64k_gbs2.yaml

unpacking 全参 SFT 配置
examples/qwen3_6/qwen3_6_27B_config_gbs16.yaml

unpacking LoRA SFT 配置
examples/qwen3_6/qwen3_6_27B_config_LoRA_gbs8.yaml
```

训练前请确认配置中的以下路径与本地目录一致：

- `data.dataset_param.preprocess_parameters.model_name_or_path`：原始 Hugging Face 模型目录
- `data.dataset_param.basic_parameters.dataset`：转换后的 JSON 数据
- `model.model_name_or_path`：原始 Hugging Face 模型目录
- `training.load`：转换后的 DCP 目录
- `training.save`：训练输出目录

### 1. 启动全参 SFT

启动后台训练（默认使用 16 张 NPU）：

```bash
# 使用 packing 配置
bash run_train_nohup.sh \
  examples/qwen3_6/qwen3_6_27B_config_packing64k_gbs2.yaml

# 使用 unpacking 配置
bash run_train_nohup.sh \
  examples/qwen3_6/qwen3_6_27B_config_gbs16.yaml
```

脚本会在 NPU 空闲后启动训练，并在 `logs/` 下写入 PID 文件和 workflow 日志。训练成功结束后会自动执行 checkpoint 格式转换后处理；训练失败时不会执行后处理。

### 2. 启动 LoRA SFT

启动后台训练（默认使用 16 张 NPU）：

```bash
bash run_train_nohup.sh \
  examples/qwen3_6/qwen3_6_27B_config_LoRA_gbs8.yaml
```

脚本会在 NPU 空闲后启动训练，并在 `logs/` 下写入 PID 文件和 workflow 日志。训练成功结束后会自动执行 checkpoint merge 后处理；训练失败时不会执行后处理。

## Checkpoint 后处理

以 packing 全参 SFT 配置为例，训练输出默认位于 `./outputs/packing64k-gbs2`。训练流程正常结束时会自动完成 checkpoints 的格式转换，如需手动将所有 iteration 的 DCP checkpoint 转换为 Hugging Face 格式，可执行：

```bash
bash cvt_ckpt_dcp2hf_batch.sh \
  ./outputs/packing64k-gbs2 \
  ./outputs/packing64k-gbs2-hf \
  examples/qwen3_6/qwen3_6_27B_config_packing64k_gbs2.yaml
```

脚本会遍历输入目录下的 `iter_*` 目录，并将结果写入带有对应 iteration 后缀的输出目录。原始 Hugging Face 模型目录默认从 `ckpt/hf_path/Qwen3.6-27B` 读取 tokenizer 等文件。

如需只转换单个 DCP checkpoint：

```bash
bash cvt_ckpt_dcp2hf.sh \
  ./outputs/packing64k-gbs2-iter_000010 \
  ./outputs/packing64k-gbs2/iter_000010
```

如需预览批处理脚本将处理的 checkpoint 而不实际转换，可设置：

```bash
CHECKPOINT_POSTPROCESS_DRY_RUN=1 bash cvt_ckpt_dcp2hf_batch.sh \
  ./outputs/packing64k-gbs2 \
  ./outputs/packing64k-gbs2-hf \
  examples/qwen3_6/qwen3_6_27B_config_packing64k_gbs2.yaml
```
