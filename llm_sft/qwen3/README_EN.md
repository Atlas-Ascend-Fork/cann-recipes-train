# Qwen3-1.7B SFT Training Example

## Hardware Requirements

- Hardware: Atlas A3. The training script uses one logical NPU device by default (`NPUS_PER_NODE=1`).

## Build Environment with Dockerfile

1. Clone the repository on the host and enter the example directory.
   ```bash
   git clone https://gitcode.com/cann/cann-recipes-train.git
   cd cann-recipes-train/llm_sft/qwen3
   ```

2. Build the image and start the container on the host.
   ```bash
   # The current directory contains the Dockerfile and build_docker.sh
   docker build -t qwen3_sft -f Dockerfile.cann.mindspeed.qwen3 .

   # Create and enter the container (default name: qwen3_sft, image: qwen3_sft:latest)
   bash build_docker.sh
   ```

   By default, `build_docker.sh` maps all 16 NPU device nodes (`/dev/davinci0` through `/dev/davinci15`) on an A3 server. The number of devices mapped into the container does not determine how many devices training uses: this example sets `NPUS_PER_NODE=1` and launches only one training process.

   If the environment exposes only some device nodes, edit the `--device` arguments in `build_docker.sh` to match the available nodes. For example, if only `/dev/davinci0` is available, keep `--device=/dev/davinci0` and the shared device nodes `/dev/davinci_manager`, `/dev/devmm_svm`, and `/dev/hisi_hdc`, and remove the mappings for `davinci1` through `davinci15`. For training on multiple devices, also update the training process count and the parallelism settings in the training and model conversion scripts.

   The script mounts this example directory from the host at `/workspace/qwen3` in the container. To use different container or image names, edit `container_name` and `image_name` in `build_docker.sh` and keep the image name consistent with the build command.

3. Prepare dependencies and training scripts inside the container.
   ```bash
   cd /workspace/qwen3
   # Copy the required dependency files already prepared in the image
   cp -r /workspace/MindSpeed-LLM MindSpeed-LLM
   cp -r qwen3_dense MindSpeed-LLM/examples/mcore

   cd MindSpeed-LLM
   ```

## Quick Start SFT Training Dependency Setup on One‑stop Platform

### Environment Requirements

- One‑stop platform image: CANN‑8.5.0‑A3 or CANN‑8.5.0‑A2

### Project and Dependency Setup

In the current directory (`cann-recipes-train/llm_sft/qwen3`), run:
```bash
bash build_project_platform.sh

# Enter the corresponding directory
cd MindSpeed-LLM
pip install pyarrow==21.0.0
```

This section only covers the environment preparation and project/dependency setup steps that are additionally required on the one‑stop platform. The following dataset preparation, model weight preparation, data conversion, model conversion, and SFT training steps are identical to those below; continue with the sections below.

## Dataset Preparation

All commands in this section are executed in `cann-recipes-train/llm_sft/qwen3/MindSpeed-LLM` (the current directory after the `cd MindSpeed-LLM` step above).

The Alpaca dataset used in this example can be prepared as follows:
```bash
# Create a dataset directory under the current directory (MindSpeed-LLM)
mkdir -p ./dataset
cd dataset/

# Install modelscope and download the Alpaca parquet files from ModelScope
pip install modelscope

modelscope download --dataset OmniData/alpaca --local_dir ./alpaca
cd ../
```

## Model Weight Preparation

The Qwen3-1.7B model weights used in this example can be downloaded as follows:
```bash
# Download the base model files from ModelScope into ./Qwen3-1.7B
mkdir ./Qwen3-1.7B
modelscope download --model Qwen/Qwen3-1.7B --local_dir ./Qwen3-1.7B
```

## Qwen3-1.7B SFT Training Steps

1. **Data conversion** – Convert raw data into the training format:
   ```bash
   bash examples/mcore/qwen3_dense/data_convert_qwen3_instruction.sh
   ```

2. **Model conversion** – Convert the HuggingFace format model to the Mcore training format:
   ```bash
   bash examples/mcore/qwen3_dense/ckpt_convert_qwen3_hf2mcore.sh
   ```

3. **SFT training** – Launch the training script:
   ```bash
   bash examples/mcore/qwen3_dense/tune_qwen3_4K_full_A3_ptd.sh
   ```

4. **Convert back** – Convert the trained model back to HuggingFace format:
   ```bash
   bash examples/mcore/qwen3_dense/ckpt_convert_qwen3_mcore2hf.sh
   ```

> **Note:** Modify the `ascend-toolkit` path in all scripts according to your actual environment. The default configuration is:
> ```
> source /usr/local/Ascend/ascend-toolkit/set_env.sh
> ```
> For the one‑stop platform, change it to (using CANN 8.5.0 as an example):
> ```
> source /home/developer/Ascend/cann-8.5.0/set_env.sh
> ```

---

## Appendix – Parameter Descriptions

### 1. Data Preparation

| Parameter | Description |
|-----------|-------------|
| `--input` | Path to the input data file(s). Can be a directory (all files in it will be processed) or a specific file. Supports `.parquet`, `.csv`, `.json`, `.jsonl`, `.txt`, `.arrow`. All files in the same directory must have the same format. |
| `--tokenizer-name-or-path` | Path to the tokenizer files. |
| `--tokenizer-type` | Tokenizer type; set to `PretrainedFromHF` when loading from a HuggingFace-format tokenizer directory. |
| `--output-prefix` | Prefix for the output data files. |
| `--handler-name` | For fine‑tuning with Alpaca‑style datasets, specify `AlpacaStyleInstructionHandler`. The `--map-keys` parameter is used to extract the corresponding columns. |
| `--workers` | Number of worker processes used for data preprocessing. |
| `--log-interval` | Interval between progress log updates. |
| `--enable-thinking` | Whether to enable thinking mode and add the reasoning tags `<think>` and `</think>` to samples. |
| `--seq-length` | Maximum sequence length; no truncation is applied when unspecified (default `None`). This example script does not set it explicitly, so the default is used. |
| `--prompt-type` | Specifies the model prompt template, default is `qwen3`. |

**Example of Alpaca‑style data:**
```json
[
   {
      "instruction": "Human instruction (required)",
      "input": "Human input (optional)",
      "output": "Model response (required)",
      "system": "System prompt (optional)",
      "history": [
         ["First round instruction (optional)", "First round response (optional)"],
         ["Second round instruction (optional)", "Second round response (optional)"]
      ]
   }
]
```

---

### 2. Model Conversion (HF → Mcore)

| Parameter | Description |
|-----------|-------------|
| `--load-model-type` | Type of the input model; `hf` for HuggingFace format. |
| `--save-model-type` | Type of the output model; `mg` for Mcore format. |
| `--target-tensor-parallel-size` | Tensor parallelism size. |
| `--target-pipeline-parallel-size` | Pipeline parallelism size. |
| `--load-dir` | Path to the input model. |
| `--save-dir` | Path to the output model. |
| `--moe-grouped-gemm` | Enable MoE grouped GEMM optimization. MoE models only; Qwen3-1.7B is a dense model, so this example script does not set it. |
| `--model-type-hf` | HuggingFace model type. |
| `--expert-tensor-parallel-size` | Expert tensor parallelism size; must be explicitly set to 1. |

---

### 3. SFT Training Parameters

- **Path Configuration**

| Parameter | Description |
|-----------|-------------|
| `CKPT_LOAD_DIR` | Path to load pretrained weights. |
| `CKPT_SAVE_DIR` | Path to save fine‑tuned model. |
| `DATA_PATH` | Path to the training dataset. |
| `TOKENIZER_PATH` | Path to the tokenizer files. |

- **Parallelism Configuration** – must be **consistent** with the model conversion settings

| Parameter | Description |
|-----------|-------------|
| `TP` | Tensor parallelism size. |
| `PP` | Pipeline parallelism size. |
| `CP` | Context parallelism size, default is 1. This example script does not enable it; it can be enabled for long‑sequence training, and must stay consistent with the model conversion settings. |
| `CP_TYPE` | Context parallelism algorithm; effective only when `CP` is enabled. This example script does not enable it. |

- **Training Configuration**

| Parameter | Description |
|-----------|-------------|
| `SEQ_LENGTH` | Sequence length. |
| `TRAIN_ITERS` | Number of training iterations. |
| `--micro-batch-size` | Batch size per GPU. |
| `--global-batch-size` | Global batch size. |
| `--lr` | Learning rate. |
| `--lr-decay-style` | Learning rate decay style. |
| `--min-lr` | Minimum learning rate. |
| `--weight-decay` | Weight decay. |
| `--lr-warmup-fraction` | Warmup fraction. |
| `--clip-grad` | Gradient clipping. |

- **Fine‑tuning Parameters**

| Parameter | Description |
|-----------|-------------|
| `--finetune` | Enable fine‑tuning mode. |
| `--stage` | Training stage; `sft` denotes supervised fine‑tuning. |
| `--is-instruction-dataset` | Use instruction dataset. |
| `--prompt-type` | Use Qwen3‑style prompt format. |

---

### 4. Model Conversion (Mcore → HF)

| Parameter | Description |
|-----------|-------------|
| `--load-dir` | Path to the input model (the saved checkpoint from training). |
| `--save-dir` | Path to the output HuggingFace model. |
| `--model-type-hf` | HuggingFace model type. |
