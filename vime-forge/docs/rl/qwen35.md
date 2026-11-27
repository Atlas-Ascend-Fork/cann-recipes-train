# Qwen3.5-4B 单轮 RL 训练

[Qwen3.5-4B 模型入口](../../scripts/torchtitan/run-qwen35-4B.sh)使用 TorchTitan-NPU 训练、vLLM 推理，运行单轮 GRPO。环境准备见[快速开始](../quickstart.md#11-准备-npu-运行环境)，启动参数见[TorchTitan 启动器](../torchtitan-launcher.md)。

## 训练数据

训练入口默认从 `$VIME_MODEL_ROOT/Qwen3.5-4B` 加载模型，从 `$VIME_DATA_ROOT/dapo-math-17k.jsonl` 读取 DAPO-Math-17K 数据集。可在外部 `local.sh` 中分别设置 `MODEL_PATH` 和 `PROMPT_DATA` 来指定其他路径。

JSONL 文件每行一道题，`prompt` 是对话消息列表，`label` 是标准答案。训练入口使用模型的聊天模板格式化 `prompt`，并使用 `deepscaler` 根据 `label` 为模型的数学答案评分。

```json
{"prompt": [{"role": "user", "content": "计算 1+1，并将最终答案写在 \\boxed{} 中。"}], "label": "2"}
```

<a id="24-上下文并行cp"></a>

## 上下文并行（CP）

默认配置为 `TRAIN_TP=1 TRAIN_DP=4 TRAIN_CP=1`，使用四张训练卡，训练配置为 `vime_qwen35_4b`。KL-Cov 目前仅支持 CP=1。

在同样的四张训练卡上开启 CP=4，在外部 `local.sh` 中设置：

```bash
export TRAIN_TP=1 TRAIN_DP=1 TRAIN_CP=4
```

当 `TRAIN_CP>1` 时，Qwen3.5 的 Ascend 实现要求 `TRAIN_TP=1`。入口默认选择 `vime_qwen35_4b_cp`，并添加 `--packed-seq-alignment 64`，满足 Gated DeltaNet 对序列分片的对齐要求。若已显式设置 `TITAN_CONFIG`，请将其设为 `vime_qwen35_4b_cp`。

### CP=4 验证记录

单轮 RL 在四张训练卡和两个 TP=1 推理引擎上完成了一次训练更新。该次运行设置 `--num-rollout 1`，每批 8 道题，每题采样 4 次，共 32 条回答。推理设备数和 token 预算如下：

```bash
export ROLLOUT_GPUS=2 VLLM_RANKS_PER_ENGINE=1
export SEQ_LEN=18432 MAX_PROMPT_LEN=2048 MAX_RESPONSE_LEN=16384
export MAX_TOKENS_PER_GPU=8192
```
