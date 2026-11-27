# TorchTitan Full-Parameter SFT Example for Ascend C Operator Generation

This example uses **TorchTitan-NPU** to perform full-parameter supervised fine-tuning of Qwen3.6-27B for Ascend C operator generation. The training configuration targets Ascend C operator-generation data and CP8 × FSDP2, with support for complete DCP checkpoint saving and Hugging Face weight export.
For training results and the training process report, see the [practice report](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/llm_sft/qwen36_ascendc/zh/qwen36_ascendc_sft_report.md).

## Files

| File | Purpose |
| --- | --- |
| `cannbot_recipe.py` | Configures training, parallelism, the optimizer, and `NoThinkTemplateTokenizer` |
| `data_process.py` | Validates and merges JSONL files and maps sample fields |
| `run_train.sh` | Starts training, loads data, and saves weights |
| `Dockerfile` | Pins dependency versions and builds the training environment |

`run_train.sh` calls `data_process.py` to validate the schema and merge JSONL files in `DATA_FILES` order, inserting a missing LF between non-empty files. `data_process.py` also maps fields when the dataloader reads a sample. TorchTitan's official `ChatDataLoader`/`ChatDataset` handles dataset splitting, tokenization, greedy packing, padding, and dataloader-state recovery.

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

#### Start and enter the container

Run these commands on an Atlas A3 host with the Ascend driver installed. The example uses the default driver paths and passes through all 16 device nodes on an 8-card Atlas A3 system. You can set `QWEN36_WORKSPACE` to a different absolute path on the host.

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

The container opens in `/opt/cannbot_torchtitan`. Run `exit` to return to the host terminal. The container stays running. Complete steps 2 and 3 in that host terminal and reenter the container for step 4.

### 2. Validate the training data

Download the open data from the [project's dataset announcement](https://gitcode.com/cann/cann-recipes-train/discussions/2). Follow the [data generation instructions](../data_generator/README.md) and place the generated JSONL files in `$QWEN36_WORKSPACE/data` on the host.

Input files are UTF-8 JSONL and support the following two source schemas:

```json
{"input":"Write an operator.","output":"..."}
{"prompt":"Write an operator.","response":"..."}
```

During training, the fields are mapped as `input → user` and `output → assistant`, or as `prompt → user` and `response → assistant`. `run_train.sh` creates a data manifest and provides an explicit schema for mixed fields, preventing the JSON loader from re-inferring fields at file boundaries.

### 3. Download the base model weights

Use the Hugging Face CLI on the host to download the complete [Qwen3.6-27B model directory](https://huggingface.co/Qwen/Qwen3.6-27B), including the weight shards, configuration, and tokenizer files.

```bash
python3 -m pip install -U huggingface_hub
hf download Qwen/Qwen3.6-27B --local-dir "$QWEN36_WORKSPACE/models/Qwen3.6-27B"
```

The mount in step 1 makes the model available at `/workspace/models/Qwen3.6-27B` inside the container. The training launcher loads the base model from the directory specified by `HF_ASSETS_PATH`.

### 4. Start training

Run the `docker exec` command from step 1 again on the host, then run the following commands inside the container.

```bash
export HF_ASSETS_PATH=/workspace/models/Qwen3.6-27B
export DATA_FILES=/workspace/data/transfer.jsonl,/workspace/data/rft.jsonl,/workspace/data/sampled.jsonl
export DUMP_FOLDER=/workspace/runs/qwen36-sft
export CANNBOT_TRAIN_STEPS=500
export CANNBOT_CHECKPOINT_INTERVAL=100
bash run_train.sh
```

Set `DATA_FILES` to readable JSONL paths inside the container. List them in reading order and separate them with commas. Input paths must not contain commas or line breaks. `DUMP_FOLDER` must not contain commas.

`CANNBOT_TRAIN_STEPS` is the optimizer-step count, and `CANNBOT_CHECKPOINT_INTERVAL` is the synchronous DCP save interval. Set both to positive integers for the data size and training plan.

### 5. Export Hugging Face weights

Run the export inside the container. The exported weights are saved in `$QWEN36_WORKSPACE/exports` on the host.

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
