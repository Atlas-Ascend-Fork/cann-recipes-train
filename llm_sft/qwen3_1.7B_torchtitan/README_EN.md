# Qwen3-1.7B SFT Training Example

## Hardware Requirements

Number of cards: 1 A2 or 1 A3

## Quick Start for SFT Training on the One-Stop Platform

### Environment Requirements

Select the following one-stop platform template: cann_9.0.0-py3.11-A2-arm-20260715/cann_9.0.0-py3.11-A3-arm-20260715

### Project and Dependency Setup

Run the following commands in the current directory (`cann-recipes-train/llm_sft/qwen3_1.7B_torchtitan/`):

```bash
# Download the repository containing this example and install its dependencies
bash build_project_platform.sh
# Enter the directory
cd torchtitan-npu
# Quickly verify that the environment is working correctly
NGPU=1 bash scripts/run_train.sh
```

Expected output:

Running with configs: model and recipe resolved from the current master defaults
visible dies: 2
step: 1  loss: <finite>  grad_norm: <finite>
Training completed

Note: Update the `ascend-toolkit` path in the scripts above according to your environment. On the one-stop platform, modify it as follows (using CANN 9.0.0 as an example):

```
/home/developer/Ascend/cann-9.0.0/set_env.sh
```

## Preparing Model Weights

Prepare the Qwen3-1.7B model weights used in this example as follows:

```bash
# Download the base model files from ModelScope and store them in
# ./assets/hf/Qwen3-1.7B under the current directory
MODEL_DIR="./assets/hf/Qwen3-1.7B"
python3 -m pip install -U modelscope
modelscope download \
  --model PrimeIntellect/Qwen3-1.7B \
  --local_dir "$MODEL_DIR"

pwd
ls ./assets/hf/Qwen3-1.7B
```

## Preparing the Dataset

This example uses the `willcb/V3-wordle` dataset:

```bash
# 1. Download the dataset (the source file is already in Parquet format)
HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com hf download willcb/V3-wordle data/train-00000-of-00001.parquet --repo-type=dataset --local-dir ./assets/data/wordle_raw

# 2. Copy it to the data path expected by torchtitan
mkdir -p ./assets/data/wordle
cp ./assets/data/wordle_raw/data/train-00000-of-00001.parquet ./assets/data/wordle
```

The Wordle evaluation environment depends on the NLTK corpora (`words` and `averaged_perceptron_tagger`). The following commands download them from the official NLTK source and automatically extract them to `$HOME/developer/nltk_data` (this takes approximately seven minutes):

```bash
pip install nltk
python3 -c "
import nltk
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
"
```

Inspect the dataset format by loading it locally:

```bash
python3 -c "
from datasets import load_dataset
import pprint
ds = load_dataset('./assets/data/wordle', split='train')
pprint.pprint(next(iter(ds)), width=100, depth=3)
"
```

The `process_wordle_sample` processor concatenates `prompt` and `completion` into a complete `[system, user, assistant, user, assistant, ...]` message list.

## Baseline SFT Training and Wordle Evaluation

Use TorchTitan-NPU to perform SFT on Qwen3-1.7B with the Wordle dataset, and then interactively evaluate the trained model on Wordle using `vf-eval`.

### 1. Start SFT Training

Use TorchTitan-NPU to perform SFT on Qwen3-1.7B with the Wordle dataset:

```bash
MODULE=torchtitan_npu.models.qwen3 \
CONFIG=sft_qwen3_1_7b_wordle \
NGPU=1 \
bash scripts/run_train.sh \
--hf_assets_path "assets/hf/Qwen3-1.7B" \
--checkpoint.folder checkpoint_wordle_sft \
--checkpoint.last_save_in_hf \
--checkpoint.enable \
--checkpoint.initial_load_in_hf \
dataloader:chat_data_loader_config \
--dataloader.dataset_path "assets/data/wordle"
```

Parameter descriptions:

| Parameter | Description |
|---|---|
| `MODULE` | Uses the Qwen3 TorchTitan-NPU model configuration |
| `CONFIG` | Wordle SFT configuration file |
| `NGPU` | Number of GPUs/NPUs to use |
| `--checkpoint.folder` | Directory in which to save checkpoints |
| `--checkpoint.last_save_in_hf` | Saves the final checkpoint in Hugging Face format when training ends |
| `--checkpoint.initial_load_in_hf` | Initializes from a model in Hugging Face format |
| `chat_data_loader_config` | Uses the conversational-format data loader |
| `--dataloader.dataset_path` | Path to the Wordle SFT dataset |

After training is complete, the checkpoint is saved by default to:

```text
outputs/checkpoint_wordle_sft/
```

### 2. Prepare the Hugging Face Inference Checkpoint

`infer_server.py` uses Hugging Face `AutoTokenizer` to load the tokenizer.

Some TorchTitan checkpoints primarily store model weights and training state and do not include the complete tokenizer files. Therefore, the tokenizer configuration from the original Qwen3 model must be added to the checkpoint directory.

Run:

```bash
src=assets/hf/Qwen3-1.7B
dst=outputs/checkpoint_wordle_sft/step-20

test -d "$dst"

for name in \
config.json \
tokenizer.json \
tokenizer_config.json \
special_tokens_map.json
do
    if test -f "$src/$name"; then
        cp "$src/$name" "$dst/$name"
    fi
done

for name in \
generation_config.json \
vocab.json \
merges.txt
do
    if test -f "$src/$name"; then
        cp "$src/$name" "$dst/$name"
    fi
done

echo "Loadable checkpoint:"
find "$dst" -maxdepth 1 -type f -printf '%f\n' | sort
```

### 3. Start the Inference Service

```bash
source /home/developer/Ascend/cann-9.0.0/set_env.sh
# Stop an existing inference service (optional)
pkill -f infer_server.py || true

# Install inference dependencies
pip install transformers
pip install torchvision==0.27.0+cpu \
--index-url https://download.pytorch.org/whl/cpu
# Start the inference server

python3 scripts/infer_server.py \
--model ./outputs/checkpoint_wordle_sft/step-20 \
--port 8000 \
> /tmp/infer_server.log 2>&1 &

sleep 10

curl http://localhost:8000/health
# {"status":"ok"}
```

### 4. Configure the vf-eval Evaluation Environment

`vf-eval` is an interactive evaluation tool based on Prime-RL. It is used to test the model's reasoning ability in the Wordle environment.

The evaluation workflow is as follows:

1. The environment selects a secret word from the reserved evaluation words.
2. The inference server receives the system prompt and the Wordle game rules.
3. The model generates a prediction:

```xml
<think>
...
</think>
<guess>[word]</guess>
```

4. The Wordle environment parses the content inside `<guess>`.
5. It returns G/Y/X letter feedback based on the prediction.
6. The model continues making predictions based on the feedback.
7. The interaction runs for at most six rounds.
8. The final reward is calculated when the game ends.

The reward consists of the following components:

| Reward | Meaning |
|---|---|
| `correct_answer` | Whether the secret word was guessed correctly |
| `partial_answer` | Provides a partial reward based on the number of green/yellow letters |
| `length_bonus` | Encourages completing the task in fewer rounds |
| `format_reward` | Checks whether the `<guess>` output format is correct |

Initialize the vf-eval environment:

```bash
bash ../setup_prime_rl.sh
```

### 5. Run the Wordle Evaluation

```bash
# Activate the vf-eval environment:
PRIMERL_DIR=./prime-rl
source "$PRIMERL_DIR/.venv-wordle-legacy/bin/activate"

# Start the evaluation:
vf-eval wordle \
--num-examples 4 \
--rollouts-per-example 2 \
--api-base-url http://127.0.0.1:8000/v1 \
--max-concurrent 1 \
--verbose \
--temperature 0.6 \
--save-results
```

Parameter descriptions:

| Parameter | Description |
|---|---|
| `--num-examples` | Number of evaluation samples |
| `--rollouts-per-example` | Number of samples generated for each example |
| `--api-base-url` | Inference server address |
| `--max-concurrent` | Sends requests serially to avoid overloading the single-threaded server |
| `--temperature` | Generation sampling temperature |
| `--save-results` | Saves the evaluation results |

### Results Analysis

Reference performance comparison between the baseline model and the SFT model:

| Metric | Base Qwen3-1.7B | Wordle SFT | Change |
|------|----------------|------------|------|
| Average format_reward | 0.60 | 1.0 | +0.4 |
| Average correct_answer | 0.00 | 0.00 | No change |
| Average partial_answer | 0.00 | 0.25 | +0.25 |
| Average length_bonus | 0.00 | 0.00 | No change |
| Average reward | 0.22 | 0.4 | +0.18 |

Format correctness improved significantly: `format_reward` increased from 0.6 to 1.0, indicating that the SFT stage successfully taught the model to follow the XML interaction format required by the Wordle environment. This is consistent with the goal of using SFT to learn the output format.

Partial correctness improved: `partial_answer` increased from 0 to 0.25, indicating that the model began generating guesses containing correct letters (G/Y feedback) and gradually learned the game logic. However, the correct-guess rate remained at 0 (`correct_answer` was unchanged).

Overall reward increased: The average total reward rose from 0.22 to 0.4. The gain came mainly from improvements in formatting and partial correctness, but the final win rate did not improve (`length_bonus` remained at 0), indicating that the model was still unable to solve the words completely.

Comparison with the official baseline: The Qwen3-1.7B baseline score of 0.22 is close to the approximately 0.2 reported officially, and the post-SFT score of 0.4 is also broadly in line with expectations. After the RL stage, the average reward is expected to reach approximately 1.5 with a win rate of around 60%. This indicates that the current SFT model serves as the foundation for RL and that RL is needed to improve its strategic capabilities.
