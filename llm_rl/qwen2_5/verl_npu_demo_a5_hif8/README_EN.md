# Qwen2.5 HiF8 Rollout RL Example (A5)

## Overview

An Ascend 950 implementation of [`../verl_npu_demo`](../verl_npu_demo). The model, data set, chat template and reward function match the original sample; the differences are the platform, <term>**Atlas A5** Series Products</term> + CANN 9.2, and the added **HiF8 (W8A8_HIF8_PTS)** rollout path that speeds up reinforcement learning.

Both runs share the launcher [run_qwen2_5_1_5b_A5.sh](run_qwen2_5_1_5b_A5.sh); its first argument selects `bf16` (the BF16 control) or `hif8` (W8A8_HIF8).

## Supported Hardware Models

<term>**Atlas A5** Series Products</term>. A2/A3 follow the original sample's flow and are unaffected.

## Environment Preparation

Unless noted otherwise, run every command below from this sample's directory, `llm_rl/qwen2_5/verl_npu_demo_a5_hif8`.

The software stack differs from the original sample in order to support A5, and the two cannot be mixed:

| Component | Original sample | This sample |
| --- | --- | --- |
| CANN | 8.5.0 | **9.2.0** |
| Python | 3.11 | 3.11 |
| torch / torch_npu | from the image | `2.10.0` / `2.10.0.post4` |
| vLLM | pip `0.11.0` | from source `0fc695fc`, installs as `0.23.0+empty` |
| vLLM-Ascend | pip `0.11.0` | from source `f042ad88` |
| verl | `release/v0.6.1` | from source `483b8a00` (v0.9.0) |

Full commits:

```text
vLLM         0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
vLLM-Ascend  f042ad88882e22a43af323b0df5691467bad8553
verl         483b8a009ba3a97563edee3a19887e4862b8094a
```

### Base Image

The base image needs CANN 9.2.0 and must not have vLLM preinstalled. A5 maps to `950` on quay.io:

```shell
docker pull quay.io/ascend/cann:9.2.0-beta.2-950-ubuntu22.04-py3.11-devel
```

For the `docker run` arguments, see [the original sample's environment preparation](../verl_npu_demo/README.md#环境准备) (in Chinese).

### 1. Clone the Source

```shell
export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"   # change this to put the sources elsewhere
bash download_frameworks_source_code.sh
```

Clones the three repositories at the commits listed above. A directory already on the pinned commit is skipped, so the script can be re-run. vLLM-Ascend runs `git submodule update --init --recursive` to fetch Catlass, which the custom kernels need.

A new terminal window needs this `export` again. To reuse sources you have already downloaded, set it to where those sources actually are.

### 2. Apply Patches and Install

```shell
bash build_project.sh
```

`build_project.sh` calls `apply_all_patches.sh` itself, so there is no need to run it separately. The patches are:

```text
patches/
├── verl/
│   └── 0001-verl-feature-hif8_rollout.patch          # HiF8 weight quantization and refit routing
└── vllm_ascend/
    ├── 0001-vllm_ascend-feature-register_hif8.patch  # registers W8A8_HIF8_PTS
    ├── vllm_ascend/device/hif8_compat.py             # dtype and operator checks
    └── vllm_ascend/quantization/methods/w8a8_hif8.py # HiF8 linear
```

vLLM itself needs no changes. Every patch is dry-run first and fails outright if the source tree is on the wrong commit.

Equivalent manual steps (each install runs inside parentheses, so you stay in this sample's directory):

```shell
bash apply_all_patches.sh
pip install -r requirements.txt

# vLLM is installed empty; the kernels come from vllm-ascend. pip show vllm should report 0.23.0+empty
( cd "${WORKSPACE_DIR}/vllm" \
  && VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation )

# vLLM-Ascend builds the A5 custom kernels
( cd "${WORKSPACE_DIR}/vllm-ascend" \
  && export SOC_VERSION=ascend950pr_9579 COMPILE_CUSTOM_KERNELS=1 \
  && pip install -e . --no-deps --no-build-isolation )

( cd "${WORKSPACE_DIR}/verl" && pip install -e . --no-deps )
```

## Model Weight Preparation

Download Qwen2.5-1.5B-Instruct as in the original sample, to `${WORKSPACE_DIR}/verl/model/Qwen2_5_1_5B_Instruct`, then overwrite its `tokenizer_config.json` with the original sample's. If you have run the original sample before and the source file is no longer there, restore it from the repository first:

```shell
git checkout -- :/llm_rl/qwen2_5/verl_npu_demo/tokenizer_config.json
cp ../verl_npu_demo/tokenizer_config.json "${WORKSPACE_DIR}/verl/model/Qwen2_5_1_5B_Instruct/"
```

vLLM-Ascend needs a second model directory for the HiF8 path: it symlinks the BF16 weights and the tokenizer and adds a `quant_model_description.json`.

```shell
python3 make_quant_model_dir.py --model "${WORKSPACE_DIR}/verl/model/Qwen2_5_1_5B_Instruct" --scheme hif8
# -> ${WORKSPACE_DIR}/verl/model/Qwen2_5_1_5B_Instruct_hif8
```

That file holds:

| Key | Value |
| --- | --- |
| `quant_method` | `ascend` |
| `<layer>.weight` | `W8A8_HIF8_PTS` for the linears to quantize, `FLOAT` for the rest |
| `hif8_act_mode` | `cast` |
| `hif8_dst_type_max` | `16.0` |

The code default for `hif8_act_mode` is `pts`; `cast` is set explicitly for the best performance. For Qwen2.5-1.5B the weights to quantize are `q_proj`/`k_proj`/`v_proj`/`o_proj` and `gate_proj`/`up_proj`/`down_proj` across 28 layers, 196 weights in total.

## Data Set Preparation

Process MATH-lighteval with the preprocessing script that ships with verl, which produces `train.parquet` and `test.parquet`:

```shell
# If Hugging Face is unreachable, run this first: export HF_ENDPOINT=https://hf-mirror.com
python3 "${WORKSPACE_DIR}/verl/examples/data_preprocess/math_dataset.py" \
    --local_save_dir "${WORKSPACE_DIR}/verl/data/math/data"
```

The comparison needs a fixed validation set. The rule was fixed before any run: from the 5000 test rows, keep the 200 whose `extra_info.index % 25 == 0`.

```shell
cp make_val_subset.py "${WORKSPACE_DIR}/verl/"
# make_val_subset.py reads and writes relative paths, so it has to run under verl
( cd "${WORKSPACE_DIR}/verl" && python3 make_val_subset.py )
# ./data/math/data/test.parquet -> ./data/math/data/val_subset.parquet
```

The script prints the row count before and after filtering. Of these 200 rows, 5 have prompts longer than `data.max_prompt_length=512` under the Qwen2.5-1.5B-Instruct tokenizer and are dropped by `filter_overlong_prompts` during training, so **both runs score 195 rows**.

## RL Training

From this sample's directory, copy three files into the verl source tree and start the runs. `new_math_reward.py` is the original sample's reward function and this sample keeps no copy of it; use `cp` rather than `mv` so the original sample is not left without the file (if you have run the original sample before and it has been moved away, restore it with `git checkout -- :/llm_rl/qwen2_5/verl_npu_demo/new_math_reward.py`):

```shell
cp ../verl_npu_demo/new_math_reward.py a5_math_reward_threadsafe.py run_qwen2_5_1_5b_A5.sh "${WORKSPACE_DIR}/verl/"
cd "${WORKSPACE_DIR}/verl"

# BF16 rollout (the control)
ASCEND_RT_VISIBLE_DEVICES=0,1 bash run_qwen2_5_1_5b_A5.sh bf16

# HiF8 rollout; the actor stays BF16
ASCEND_RT_VISIBLE_DEVICES=0,1 bash run_qwen2_5_1_5b_A5.sh hif8
```

The two runs differ in three places: `bf16` uses the original model directory while `hif8` uses the one built by `make_quant_model_dir.py` and additionally passes `actor_rollout_ref.rollout.quantization=ascend`; the experiment name and the log name carry the precision. Every other setting is identical.

### Configuration Differences from the Original Sample

The original sample's script was written for verl 0.6.1 and is not compatible with verl 0.9.0:

| Item | Original sample (0.6.1) | This sample (0.9.0) |
| --- | --- | --- |
| Custom reward function | `custom_reward_function.path` | `reward.custom_reward_function.path` / `.name` |
| Rollout mode | implicit sync (offline `LLM`) | only `mode=async` |
| Trainer | single implementation | `trainer.use_v1` defaults to `true`, goes through TransferQueue |
| Cards / TP | 4 cards / TP2 (or one card) | 2 cards / TP2 |

### Reward Function

verl 0.9 scores inside `AgentLoopWorkerTQ` worker threads, where the `math_verify` timeout built on `signal.alarm()` does not work. [a5_math_reward_threadsafe.py](a5_math_reward_threadsafe.py) loads the original sample's `new_math_reward.py` (checking its sha256) and replaces only its `parse` / `verify` with versions that have the timeout turned off. The scoring logic is unchanged.

## New Source Files: 2 Files, 216 Lines

Naming and code structure follow the MXFP8 implementation already in upstream vLLM-Ascend (`w8a8_mxfp8.py` / `AscendW8A8MXFP8DynamicLinearMethod` / `device/mxfp_compat.py`). `w8a8_hif8.py` (172 lines) holds `AscendW8A8HiF8PTSLinearMethod`: the HiF8 linear, `cast`/`pts` activation and the refit buffer. `hif8_compat.py` (44 lines) checks dtypes and operators, and raises when an operator is missing rather than falling back silently.

## Measured Results

BF16 and HiF8, 10 steps each, two cards in the same domain.

**HiF8 is in effect**, on three independent pieces of evidence:

| Level | Evidence |
| --- | --- |
| The configuration was read | the worker prints `[W8A8_HIF8_PTS] activation mode: cast` (the default is `pts`) |
| The numbers separate | the overall mean of `rollout_actor_probs_pearson_corr` is 0.99807 for BF16 against 0.99194 for HiF8, with no overlap between the ranges |
| Every layer is covered | counting modules after the model loads: all 112 Linears are `AscendW8A8HiF8PTSLinearMethod` |

The 112 agrees with the 196 in the description file: vLLM fuses q/k/v into `qkv_proj` and gate/up into `gate_up_proj`, so 28 layers × 4 = 112.

### Accuracy Comparison

The primary metric is the reward mean at step 10 on the same fixed validation set, and **the two show no significant difference**.

The main conclusion comes from a test paired by seed. Two runs on the same seed share the initial weights and the data order.

| Test | Size | Mean difference (HiF8 − BF16) |
| --- | --- | ---: |
| **Paired by seed** | **13 pairs** | **−0.00122** |
| Group means compared directly | 8 runs per group | +0.00336 |

The 13 pairs, where the difference in each column is HiF8 minus BF16:

```text
seed  11      22      33      44      55      66      77      88      99      123     200     1234    2025
BF16  0.36385 0.37061 0.38349 0.38810 0.37974 0.38769 0.36128 0.39487 0.37333 0.37026 0.38410 0.35451 0.36615
HiF8  0.36328 0.37503 0.38410 0.39846 0.37410 0.36936 0.39538 0.35543 0.38429 0.36718 0.38949 0.36379 0.34223
```

The standard deviation of the paired differences is **σ ≈ 0.018**, so the smallest difference 13 pairs can resolve is about **0.011** (95% confidence); a difference below that is invisible at this scale. σ comes from the randomness of the training trajectory itself and is unrelated to rollout precision.

### Graph Mode

The HiF8 run was verified with `enforce_eager=False` and quantization works (the log shows `Capturing` and `torch.compile`, and `activation mode: cast` is printed as usual). First three steps of the same run, eager → graph: generation 31.4 / 26.7 / 32.7 s → **19.0 / 16.8 / 18.8 s**, full step 99 / 76 / 86 s → **73 / 57 / 63 s**, that is generation about 40% faster and the full step about 26% faster.

The launcher defaults to `enforce_eager=True`, and the accuracy comparison was done in eager mode as well.
