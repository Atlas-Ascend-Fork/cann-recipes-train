# Qwen2.5 HiF8 rollout 强化学习样例（A5）

## 概述

在Ascend 950实现 [`../verl_npu_demo`](../verl_npu_demo)。模型、数据集、chat template、奖励函数与原样例保持一致，区别是平台为 <term>Atlas A5 系列产品</term> + CANN 9.2，并添加 **HiF8（W8A8_HIF8_PTS）** rollout 功能以加速强化学习。

两组同用启动脚本 [run_qwen2_5_1_5b_A5.sh](run_qwen2_5_1_5b_A5.sh)，第一个参数选 `bf16`（BF16 对照）或 `hif8`（W8A8_HIF8）。

## 支持的产品型号

<term>Atlas A5系列产品</term>。A2/A3 走原样例流程，不受影响。

## 环境准备

以下命令除注明外，都在本样例目录 `llm_rl/qwen2_5/verl_npu_demo_a5_hif8` 下执行。

为支持A5环境，软件栈与原样例不一致，不能混用：

| 组件 | 原样例 | 本样例 |
| --- | --- | --- |
| CANN | 8.5.0 | **9.2.0** |
| Python | 3.11 | 3.11 |
| torch / torch_npu | 随镜像 | `2.10.0` / `2.10.0.post4` |
| vLLM | pip `0.11.0` | 源码 `0fc695fc`，装出来是 `0.23.0+empty` |
| vLLM-Ascend | pip `0.11.0` | 源码 `f042ad88` |
| verl | `release/v0.6.1` | 源码 `483b8a00`（v0.9.0） |

完整 commit：

```text
vLLM         0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
vLLM-Ascend  f042ad88882e22a43af323b0df5691467bad8553
verl         483b8a009ba3a97563edee3a19887e4862b8094a
```

### 基础镜像

基础镜像需要CANN 9.2.0，不应预装 vLLM 的。A5 在 quay.io 上对应 `950`：

```shell
docker pull quay.io/ascend/cann:9.2.0-beta.2-950-ubuntu22.04-py3.11-devel
```

`docker run` 参数见[原样例的环境准备](../verl_npu_demo/README.md#环境准备)。

### 1. 拉取源码

```shell
export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"   # 要换源码落点就改这里
bash download_frameworks_source_code.sh
```

按上表 commit 克隆三个仓库；已经在对应 commit 上的目录会跳过，所以脚本可以重复执行。vLLM-Ascend 会跑 `git submodule update --init --recursive` 拉 Catlass，编自定义算子要用。

换新的终端窗口要重新执行这条 `export`。复用已经下载过的源码，就设成那份源码的实际路径。

### 2. patch 并安装

```shell
bash build_project.sh
```

`build_project.sh` 会自动调用 `apply_all_patches.sh` 完成 patch 应用，不必单独运行。patch 的构成：

```text
patches/
├── verl/
│   └── 0001-verl-feature-hif8_rollout.patch          # HiF8 权重量化与 refit 路由
└── vllm_ascend/
    ├── 0001-vllm_ascend-feature-register_hif8.patch  # 注册 W8A8_HIF8_PTS
    ├── vllm_ascend/device/hif8_compat.py             # dtype 与算子检查
    └── vllm_ascend/quantization/methods/w8a8_hif8.py # HiF8 linear
```

vLLM 本身无需修改。每个 patch 先 dry-run，源码树 commit 不对就直接失败。

等价的手工步骤（每条安装在括号内执行，跑完仍在本样例目录）：

```shell
bash apply_all_patches.sh
pip install -r requirements.txt

# vLLM 装空壳，算子由 vllm-ascend 提供；装完 pip show vllm 应该是 0.23.0+empty
( cd "${WORKSPACE_DIR}/vllm" \
  && VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation )

# vLLM-Ascend 编 A5 自定义算子
( cd "${WORKSPACE_DIR}/vllm-ascend" \
  && export SOC_VERSION=ascend950pr_9579 COMPILE_CUSTOM_KERNELS=1 \
  && pip install -e . --no-deps --no-build-isolation )

( cd "${WORKSPACE_DIR}/verl" && pip install -e . --no-deps )
```

## 权重准备

和原样例一样下载 Qwen2.5-1.5B-Instruct，落点是 `${WORKSPACE_DIR}/verl/model/Qwen2_5_1_5B_Instruct`，再用原样例的 `tokenizer_config.json` 覆盖。若此前跑过原样例、源文件已不在，先从仓库取回：

```shell
git checkout -- :/llm_rl/qwen2_5/verl_npu_demo/tokenizer_config.json
cp ../verl_npu_demo/tokenizer_config.json "${WORKSPACE_DIR}/verl/model/Qwen2_5_1_5B_Instruct/"
```

由于 vLLM-Ascend 的需要，HiF8 路径须多一个模型目录：软链 BF16 的权重和 tokenizer，再加一份 `quant_model_description.json`。

```shell
python3 make_quant_model_dir.py --model "${WORKSPACE_DIR}/verl/model/Qwen2_5_1_5B_Instruct" --scheme hif8
# -> ${WORKSPACE_DIR}/verl/model/Qwen2_5_1_5B_Instruct_hif8
```

该文件的内容：

| 键 | 值 |
| --- | --- |
| `quant_method` | `ascend` |
| `<layer>.weight` | 要量化的 linear 填 `W8A8_HIF8_PTS`，其余 `FLOAT` |
| `hif8_act_mode` | `cast` |
| `hif8_dst_type_max` | `16.0` |

`hif8_act_mode` 代码里默认是 `pts`，显式采用 `cast` 以获得最优性能。Qwen2.5-1.5B 要量化的是 28 层的 `q_proj`/`k_proj`/`v_proj`/`o_proj` 和 `gate_proj`/`up_proj`/`down_proj`，共 196 个权重。

## 数据集准备

用 verl 自带的预处理脚本处理 MATH-lighteval，产出 `train.parquet` 和 `test.parquet`：

```shell
# 若网络无法连通 huggingface，先执行：export HF_ENDPOINT=https://hf-mirror.com
python3 "${WORKSPACE_DIR}/verl/examples/data_preprocess/math_dataset.py" \
    --local_save_dir "${WORKSPACE_DIR}/verl/data/math/data"
```

对照要固定验证集。规则在跑之前就定好：从 5000 条 test 里按 `extra_info.index % 25 == 0` 挑 200 条。

```shell
cp make_val_subset.py "${WORKSPACE_DIR}/verl/"
# make_val_subset.py 按相对路径读写，须在 verl 下跑
( cd "${WORKSPACE_DIR}/verl" && python3 make_val_subset.py )
# ./data/math/data/test.parquet -> ./data/math/data/val_subset.parquet
```

脚本会打印筛选前后的条数。这 200 条里有 5 条 prompt 按 Qwen2.5-1.5B-Instruct 的 tokenizer 超过 `data.max_prompt_length=512`，训练时被 `filter_overlong_prompts` 滤掉，**两组实际都是 195 条**。

## 强化学习训练

在本样例目录下把三个文件拷进 verl 源码树再开跑。`new_math_reward.py` 是原样例的奖励函数，本样例不另存副本；用 `cp` 不用 `mv`，以免原样例缺文件（若此前跑过原样例、它已被搬走，用 `git checkout -- :/llm_rl/qwen2_5/verl_npu_demo/new_math_reward.py` 取回）：

```shell
cp ../verl_npu_demo/new_math_reward.py a5_math_reward_threadsafe.py run_qwen2_5_1_5b_A5.sh "${WORKSPACE_DIR}/verl/"
cd "${WORKSPACE_DIR}/verl"

# BF16 rollout（对照）
ASCEND_RT_VISIBLE_DEVICES=0,1 bash run_qwen2_5_1_5b_A5.sh bf16

# HiF8 rollout，actor 仍为 BF16
ASCEND_RT_VISIBLE_DEVICES=0,1 bash run_qwen2_5_1_5b_A5.sh hif8
```

两者只差三处：`bf16` 用原始模型目录、`hif8` 用 `make_quant_model_dir.py` 造的那个并额外传 `actor_rollout_ref.rollout.quantization=ascend`；experiment_name 与日志名按精度区分。其余全部配置逐字相同。

### 相对原样例的配置差异

原样例脚本按 verl 0.6.1 编写，与 verl 0.9.0 存在不兼容：

| 项 | 原样例（0.6.1） | 本样例（0.9.0） |
| --- | --- | --- |
| 自定义奖励函数 | `custom_reward_function.path` | `reward.custom_reward_function.path` / `.name` |
| rollout 模式 | 隐式 sync（离线 `LLM`） | 仅支持 `mode=async` |
| trainer | 单一实现 | `trainer.use_v1` 默认 `true`，走 TransferQueue |
| 卡数 / TP | 4 卡 / TP2（或单卡） | 2 卡 / TP2 |

### 奖励函数

verl 0.9 在 `AgentLoopWorkerTQ` 工作线程中评分，`math_verify` 基于 `signal.alarm()` 的超时在非主线程不可用。[a5_math_reward_threadsafe.py](a5_math_reward_threadsafe.py) 加载原样例的 `new_math_reward.py`（校验 sha256），只把其 `parse` / `verify` 换成关闭超时的版本，评分逻辑不变。

## 新增源码：2 个文件、216 行

命名与代码结构和上游 vLLM-Ascend 现有的 MXFP8 实现对齐（`w8a8_mxfp8.py` / `AscendW8A8MXFP8DynamicLinearMethod` / `device/mxfp_compat.py`）。`w8a8_hif8.py`（172 行）是 `AscendW8A8HiF8PTSLinearMethod`：HiF8 linear、`cast`/`pts` 激活、refit buffer；`hif8_compat.py`（44 行）做 dtype 和算子检查，缺算子直接报错，不静默回退。

## 实测结果

BF16 和 HiF8 各 10 步、两卡同域。

**HiF8 确实生效**，三条独立证据：

| 层次 | 证据 |
| --- | --- |
| 配置读到了 | worker 打印 `[W8A8_HIF8_PTS] activation mode: cast`（默认是 `pts`） |
| 数值分得开 | `rollout_actor_probs_pearson_corr` 全程均值 BF16 0.99807 > HiF8 0.99194，区间不重叠 |
| 每层都落实 | 载入模型后数模块：112 个 Linear 全是 `AscendW8A8HiF8PTSLinearMethod` |

112 和描述文件里的 196 对得上：vLLM 把 q/k/v 合成 `qkv_proj`、gate/up 合成 `gate_up_proj`，28 层 × 4 = 112。

### 精度对照

主指标是同一固定验证集在 step10 的 reward mean，**两者无显著差异**。

主结论用按 seed 配对的检验。同 seed 的两次运行共享初始权重和数据顺序。

| 检验 | 规模 | 平均差（HiF8 − BF16） |
| --- | --- | ---: |
| **按 seed 配对** | **13 对** | **−0.00122** |
| 直接比组均值 | 每组 8 次 | +0.00336 |

13 对的逐对数据，每对差值即同列 HiF8 减 BF16：

```text
seed  11      22      33      44      55      66      77      88      99      123     200     1234    2025
BF16  0.36385 0.37061 0.38349 0.38810 0.37974 0.38769 0.36128 0.39487 0.37333 0.37026 0.38410 0.35451 0.36615
HiF8  0.36328 0.37503 0.38410 0.39846 0.37410 0.36936 0.39538 0.35543 0.38429 0.36718 0.38949 0.36379 0.34223
```

逐对差值的标准差 **σ ≈ 0.018**，所以 13 对能分辨的最小差异约 **0.011**（95% 置信）；小于这个量级的差异，本规模的实验看不出来。σ 来自训练轨迹本身的随机性，与 rollout 精度无关。

### graph 模式

HiF8 组开 `enforce_eager=False` 验证过，量化正常（日志里有 `Capturing`、`torch.compile`，`activation mode: cast` 照常打印）。同组前三步耗时，eager → graph：生成 31.4 / 26.7 / 32.7 s → **19.0 / 16.8 / 18.8 s**，整步 99 / 76 / 86 s → **73 / 57 / 63 s**，即生成快约 40%、整步快约 26%。

启动脚本默认 `enforce_eager=True`，精度对照也在 eager 下完成。
