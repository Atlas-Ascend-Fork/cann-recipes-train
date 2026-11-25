# Geo3K 多模态强化学习

本样例沿用 Vime 的单轮与多轮数据集，使用 TorchTitan 训练后端和 vLLM 推理后端，默认模型为 Qwen3.5-4B，训练与推理使用独立 NPU。

## 数据

数据集字段为 `problem`（题目）、`answer`（答案）、`images`（图像列表）。下载到本地配置指定的 `VIME_DATA_ROOT`：

```bash
hf download --repo-type dataset chenhegu/geo3k_imgurl \
    --local-dir "${VIME_DATA_ROOT}/geo3k_imgurl"
hf download --repo-type dataset VeraIsHere/geo3k_imgurl_processed \
    --local-dir "${VIME_DATA_ROOT}/geo3k_imgurl_processed"
```

单轮直接读取 `geo3k_imgurl/train.parquet`，使用 `deepscaler` 奖励；多轮读取已含工具调用提示词的 `geo3k_imgurl_processed/train.parquet`，使用 `math` 奖励。

数据路径由脚本中的 `GEO3K_TRAIN_DATA`、`GEO3K_EVAL_DATA` 指定，可在本地配置中覆盖。

## 启动

按照 [本地配置模板](../../scripts/torchtitan/common/local.sh.example) 设置模型、数据与运行环境，从 `vime-forge` 目录执行：

```bash
bash examples/geo3k_vlm/run-qwen35-4B-geo3k-torchtitan.sh --local local.mine.sh
bash examples/geo3k_vlm_multi_turn/run-qwen35-4B-geo3k-mt-torchtitan.sh --local local.mine.sh
```

卡数、批量与上下文长度为样例配置，应按设备容量调整；当前视觉输入适配要求 CP=1。多轮轮数由配置文件中的 `max_turns` 指定。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `run-qwen35-4B-geo3k-torchtitan.sh` | 单轮样例执行脚本 |
| `run-qwen35-4B-geo3k-mt-torchtitan.sh` | 多轮样例执行脚本 |
| `geo3k_vlm_multi_turn_config.yaml` | 指定 `max_turns` 与交互环境入口 |
| `geo3k_vlm_multi_turn/rollout.py` | 多轮 rollout：调用 vLLM 生成、构造 loss mask 与 log_probs、执行环境交互、按轮数上限提前停止 |
| `geo3k_vlm_multi_turn/env_geo3k.py` | 工具调用环境：解析 `<tool_call>`、批改答案、返回每轮反馈 |
