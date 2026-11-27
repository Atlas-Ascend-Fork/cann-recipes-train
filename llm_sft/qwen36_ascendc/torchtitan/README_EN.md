# TorchTitan Full-Parameter SFT Example for Ascend C Operator Generation

This example uses **TorchTitan-NPU** to perform full-parameter supervised fine-tuning of Qwen3.6-27B for Ascend C operator generation. The training configuration targets Ascend C operator-generation data and CP8 × FSDP2, with support for complete DCP checkpoint saving and Hugging Face weight export.
For training results and the training process report, see the [practice report](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/llm_sft/qwen36_ascendc/zh/qwen36_ascendc_sft_report.md).

## Files

| File | Purpose |
| --- | --- |
| `cannbot_recipe.py` | Configures training, parallelism, the optimizer, and `NoThinkTemplateTokenizer` |
| `data_process.py` | Validates the source JSONL schema and maps sample fields |
| `run_train.sh` | Starts training, loads data, and saves weights |
| `Dockerfile` | Pins dependency versions and builds the training environment |

`run_train.sh` reads JSONL files in `DATA_FILES` order. `data_process.py` validates the schema and maps fields when the dataloader reads a sample. TorchTitan's official `ChatDataLoader`/`ChatDataset` handles dataset splitting, tokenization, greedy packing, padding, and dataloader-state recovery.

## Environment

| Component | Version or revision |
| --- | --- |
| Hardware | 8 Atlas A3 cards (16 dies) |
| CANN | 9.2.0 |
| TorchTitan | `c91448d20480c7b294314e68976823050002ebec` |
| TorchTitan-NPU | `2afd4c01aa5b1bc9a860a2bd67bdd212b1b1a8f7` |

Dependency versions and source revisions are pinned directly in the Dockerfile. Building with the Dockerfile provides the version combination shown above.

## Quick start

### 1. Build the image

```bash
cd llm_sft/qwen36_ascendc/torchtitan
docker build -t cannbot-qwen36-sft:cannbot-domain .
```

Mount a legally obtained Hugging Face base model, the JSONL data, and an independent output directory when starting the container. Pass through all 8 Atlas A3 cards (16 dies) and the host driver.

### 2. Validate the training data

Input files are UTF-8 JSONL and support the following two source schemas:

```json
{"input":"Write an operator.","output":"..."}
{"prompt":"Write an operator.","response":"..."}
```

During training, the fields are mapped as `input → user` and `output → assistant`, or as `prompt → user` and `response → assistant`. `run_train.sh` creates a data manifest and provides an explicit schema for mixed fields, preventing the JSON loader from re-inferring fields at file boundaries.

### 3. Start training

```bash
export HF_ASSETS_PATH=/workspace/models/Qwen3.6-27B
export DATA_FILES=/workspace/data/transfer.jsonl,/workspace/data/rft.jsonl,/workspace/data/sampled.jsonl
export DUMP_FOLDER=/workspace/runs/qwen36-sft
export CANNBOT_TRAIN_STEPS=500
export CANNBOT_CHECKPOINT_INTERVAL=100
bash run_train.sh
```

`CANNBOT_TRAIN_STEPS` is the optimizer-step count, and `CANNBOT_CHECKPOINT_INTERVAL` is the synchronous DCP save interval. Set both values for the actual data size and training plan rather than inferring them from file names or a fixed dataset.

### 4. Export Hugging Face weights

```bash
bash run_train.sh export-hf \
  /workspace/runs/qwen36-sft/checkpoint/step-N \
  /workspace/models/Qwen3.6-27B \
  /workspace/exports/step-N
```

## Training configuration

| Parameter | Value |
| --- | --- |
| world size / CP / FSDP / TP | 16 / 8 / 2 / 1 |
| local batch / DP / gradient accumulation | 1 / 2 / 1 |
| global batch size | 2 |
| optimizer | fused AdamW |
| learning rate | `1e-5` |
| Adam betas / eps | `(0.9, 0.999)` / `1e-8` |
| warmup / decay | 10% / cosine decay to zero |
| weight decay / max norm | 0 / `inf` (no clipping) |
| template | `qwen3_6_nothink` |
| seed | 42 |

These are preferred hyperparameter settings obtained from practice with the existing data.

## Data composition

Training data must conform to the JSONL schemas described in "Validate the training data." The parent directory `../data_generator` provides the corresponding data-generation scripts, which convert operator specifications and code into JSONL data using a fixed single-turn Markdown format. See that directory's README for usage.
