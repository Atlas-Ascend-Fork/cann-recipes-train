# AscendC 算子生成强化学习样例

本样例基于 Vime、TorchTitan-NPU 和 vLLM-Ascend，对 Qwen3.5-4B、Qwen3.5-9B 和 Qwen3.6-27B 进行单轮强化学习。模型根据算子题目生成四文件代码，CANN-Bench 服务评测编译、精度与性能，训练端使用评测 reward 进行 GRPO 更新。

以下命令均在本仓库的 `vime-forge/` 目录执行。

## 环境与资源

使用已配置好的 Vime NPU 训练环境。评测服务需单独部署，支持 `/api/v1` 接口和 `1.x` 结果格式。

| 组件 | 参考版本 |
| --- | --- |
| TorchTitan | `086bf6c` |
| TorchTitan-NPU | `584a627` |
| 训练 PyTorch / torch-npu | `2.14.0.dev20260719+cpu` / `2.14.0.dev20260808` |
| Triton-Ascend | `3.2.1` |
| 推理 | 已包含 thinking budget 与 processed logprobs 支持的 vLLM / vLLM-Ascend `0.23.0` 环境 |

三个模型入口默认使用以下资源配置：

| 模型 | 训练 NPU | 推理 NPU | 总 NPU 数 |
| --- | --- | --- | --- |
| Qwen3.5-4B | 4，DP2 × CP2 | 4，TP4 | 8 |
| Qwen3.5-9B | 4，DP1 × CP4 | 4，TP4 | 8 |
| Qwen3.6-27B | 12，DP3 × CP4 | 4，TP4 | 16 |

DP、CP、TP 分别表示数据并行、上下文并行和张量并行。

## 准备题集

训练集和评测集均使用 UTF-8 编码的 JSONL 文件，每行一个 JSON 对象。例如：

```json
{"input":"算子描述、proto、cases、CPU golden 及四文件输出要求。","label":{"op_name":"div_v2","format":"4string"},"metadata":{"op_name":"div_v2","submission_format":"4string"}}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `input` | 字符串 | 完整的题目 prompt；换行在 JSON 字符串中写为 `\n`，不包含目标 AscendC 实现或参考答案 |
| `label` | 对象 | `op_name` 为评测服务注册的算子名，`format` 固定为 `4string`；不需要提供标准答案 |
| `metadata` | 对象 | 可扩展题目来源、版本等信息；若包含 `op_name` 或 `submission_format`，需与 `label` 一致 |

示例中的 `input` 仅说明内容结构，实际数据应填入完整题目。算子名、题目规格和目标平台需与评测服务一致；训练集与评测集应分开准备，避免题目泄漏。

通过 `DATA_PATH` 指定训练集；周期评测使用相同格式，通过 `EVAL_DATA_PATH` 指定：

```bash
export DATA_PATH=/path/to/train.jsonl
export EVAL_DATA_PATH=/path/to/eval.jsonl
```

未设置 `EVAL_DATA_PATH` 时不启用周期评测；启用后默认每 16 个 rollout 评测一次，每题生成 5 个样本。

## 启动训练

以 Qwen3.5-4B 为例，设置模型目录、训练数据和评测服务地址后启动：

```bash
export MODEL_DIR=/models/Qwen3.5-4B
export DATA_PATH="$PWD/examples/cann_bench_single_turn/data/train.jsonl"
export CANN_EVAL_URL="http://<eval-host>:8763"

bash examples/cann_bench_single_turn/run_qwen35_4b_single_node_npu_rl.sh
```

`MODEL_DIR` 指向包含 Hugging Face 权重、tokenizer 和模型配置的目录。默认训练 10 个 epoch，输出保存到 `outputs/cann_single_turn/<时间戳>/`。评测实验名自动生成为 `vime-cann-single-turn-<时间戳>`，无需额外设置 alias。

使用 9B 或 27B 时，更换模型目录和入口：

```bash
MODEL_DIR=/models/Qwen3.5-9B \
  bash examples/cann_bench_single_turn/run_qwen35_9b_single_node_npu_rl.sh

MODEL_DIR=/models/Qwen3.6-27B \
  bash examples/cann_bench_single_turn/run_qwen36_27b_single_node_npu_rl.sh
```

## 查看训练结果

默认输出目录包含：

```text
outputs/cann_single_turn/<时间戳>/
├── train.log          # 训练与评测日志
├── rollout_jsonl/     # 生成代码、reward 和逐 case 评测结果
├── tensorboard/       # TensorBoard 事件
└── checkpoints/       # 导出的 Hugging Face 权重
```

默认每 16 个 rollout 导出一次 HF 权重。导出内容不包含优化器状态，不支持完整断点续训。

### TensorBoard

```bash
tensorboard --logdir outputs/cann_single_turn --port 6006
```

打开 `http://127.0.0.1:6006/` 查看曲线。训练指标位于 `cann/train/`，周期评测指标位于 `eval/<题集>/`：

| 指标 | 含义 |
| --- | --- |
| `reward_mean` | 已记录的最终原始 reward 均值 |
| `reward_pass5` | 每题五个 reward 取最大值，再对题目求均值 |
| `compile_rate` | 编译通过比例 |
| `accuracy_rate` | 所有 case 都通过的样本比例 |
| `case_accuracy_rate` | 通过的 case 数占总 case 数的比例 |
| `mean_speedup` | 有性能测量的有效样本的平均加速比 |

`reward_pass5` 只在每个题组都恰有 5 个样本时输出，不是精度通过率。编译、精度及加速比统计排除基础设施失败；原始 reward 均值仍可包含已归档的故障占位 0，但这些故障样本不参与学习。

生成耗时及评测的排队、编译、运行耗时也会记录。缺失的性能测量不按加速比 0 统计。

## 常用配置

在启动命令前设置环境变量即可覆盖默认值：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `NUM_EPOCH` | `10` | 训练 epoch 数 |
| `NUM_ROLLOUT` | 未设置 | 设置后按固定 rollout 数运行，优先于 epoch 配置 |
| `LR` | `1e-6` | 学习率 |
| `ROLLOUT_BATCH_SIZE` | 4B/9B 为 `4`，27B 为 `8` | 每轮题目数 |
| `SAMPLES_PER_PROMPT` | 4B/9B 为 `4`，27B 为 `6` | 每题生成样本数 |
| `CONTEXT_LEN` / `RESPONSE_LEN` | `36864` / `36864` | 总上下文与响应长度上限 |
| `VLLM_MODEL_LEN` | `65536` | 推理模型最大上下文 |
| `CONCURRENCY` | `32` | 生成并发 |
| `EVAL_INTERVAL` / `SAVE_INTERVAL` | `16` / `16` | 周期评测与权重导出间隔 |
| `OUTPUT_DIR` | `outputs/cann_single_turn/<时间戳>` | 输出目录 |

例如调整 epoch 数和学习率：

```bash
NUM_EPOCH=5 LR=5e-7 \
  bash examples/cann_bench_single_turn/run_qwen35_4b_single_node_npu_rl.sh
```

评测地址、超时及任务奖惩设置见 [config.yaml](config.yaml)。

### Thinking budget

训练 rollout 与周期评测默认使用 `TEMPERATURE=0.6`；脚本不设置 top-p/top-k 截断，沿用框架默认值。

`THINKING_TOKEN_BUDGET` 默认为 `12288`，限制 thinking 部分的 token 数；设为 `none` 可关闭预算。

预算强制生成的结束标记不计 loss，但仍保留为后续答案的上下文；提前自然生成的结束标记仍参与学习。

预算位置不匹配时保留原 mask 并记录告警，不因该位置校验中断训练。当前不支持投机解码。

### 让评测与下一轮生成重叠

评测等待较长时，可开启：

```bash
CANN_ASYNC_EVALUATION=1 \
  bash examples/cann_bench_single_turn/run_qwen35_4b_single_node_npu_rl.sh
```

该开关默认关闭。开启后，本轮评测可与下一轮生成重叠，训练仍等待本轮 reward；周期评测不异步化。此模式使用默认的 `train_async.py`、训练/推理分卡和每轮权重更新配置，不支持 partial rollout 或额外 oversampling。

开启时，基础设施故障所在题组跳过学习，不补采新答案；关闭时沿用默认过滤与补采策略。

## Reward 与失败样本

样例配置当前选用 `scoring.py`：以服务分数除以 100 为基础，全精度结果额外加 `0.2`；有可用性能数据时补充性能奖励，不重复计入服务已有的 HAP。Reward 可以为负或大于 1，不是精度百分比。

可恢复的输出格式扣 `0.05`，不可恢复格式为 `-0.1`，明显重复输出为 `-1.0`。编译、运行和精度失败属于模型评测结果；网络、服务和环境故障不作为模型错误的零分样本。默认不启用自然结束 thinking 的 `0.05` 奖励。

### 切换评分策略

提供 `scoring.py` 和 `scoring_legacy.py` 两套可选的 reward 定义，区别在于全精度奖励、性能变换及缺失服务分数时的处理。具体公式见[评分对照](../cann_bench_evaluator/README.md#两种评分方式)。

切换评分方式时，先复制任务配置：

```bash
cp examples/cann_bench_single_turn/config.yaml /path/to/my-cann.yaml
```

例如选用 `scoring_legacy.py`，将副本中的评分路径改为：

```yaml
cann_scoring:
  function_path: examples.cann_bench_evaluator.scoring_legacy.score_result
```

启动时选择该配置：

```bash
bash examples/cann_bench_single_turn/run_qwen35_4b_single_node_npu_rl.sh \
  -- --custom-config-path /path/to/my-cann.yaml
```

选用 `scoring.py` 时，将路径设为 `examples.cann_bench_evaluator.scoring.score_result`。策略在运行初始化时确定，修改配置不会热更新正在运行的训练。两套公式及自定义评分方式见[共享评测与评分](../cann_bench_evaluator/README.md#独立评分)。
