# Vime

[English](./README.md) · [代码仓库](https://github.com/vllm-project/vime)

[![文档](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.vllm.ai/projects/vime/zh-cn/latest/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/vllm-project/vime)

**Vime** 是基于 [slime](https://github.com/THUDM/slime) 的 RL scaling 用 LLM post-training 框架，以 [**vLLM**](https://github.com/vllm-project/vllm)（配合 [vllm-router](https://github.com/vllm-project/router)）作为 rollout 后端。本仓库在Vime基础之上面向 Ascend NPU，以 [**torchtitan**](https://github.com/pytorch/torchtitan) 作为训练后端，提供两大核心能力：

1. **高性能训练**：通过连接 torchtitan 与 vLLM，支持共卡与分离部署、同步与异步等模式的训练；
2. **灵活的数据生成**：通过自定义数据生成接口以及 server based engine，实现任意的训练数据生成流程。

> **维护说明**：torchtitan 是本仓库的主力训练后端。megatron 后端当前支持 Qwen3 系列部分模型，后续不再维护。

## 定位

本仓库是 [Vime](https://github.com/vllm-project/vime) 面向 Ascend NPU 的版本：rollout 侧沿用 Vime 的 vLLM 集成，训练侧以 torchtitan 替代 megatron，模型定义、FSDP2 并行与 DCP 检查点均来自 torchtitan 上游，训练流程与接口和 Vime 保持一致。

## 目录

- [Vime](#vime)
  - [定位](#定位)
  - [目录](#目录)
  - [支持的模型与硬件](#支持的模型与硬件)
  - [架构总览](#架构总览)
  - [快速开始](#快速开始)
    - [Agentic RL 示例](#agentic-rl-示例)
  - [参数说明](#参数说明)
  - [代码阅读路径](#代码阅读路径)
  - [开发指南](#开发指南)
  - [slime doc](#slime-doc)
  - [FAQ](#faq)
  - [致谢](#致谢)
  - [引用](#引用)

## 支持的模型与硬件

| 模型 | torchtitan 预置配置函数 | 硬件 |
|---|---|---|
| Qwen3-4B | `vime_qwen3_4b` | Atlas 800T A3、A5 |
| Qwen3-30B-A3B | `vime_qwen3_30b_a3b` | Atlas 800T A3、A5 |
| DeepSeek V4 Flash | `vime_dsv4_flash` | A5 |

预置配置位于 `vime/backends/torchtitan_utils/models/<模型族>/config_registry.py`。

## 架构总览

![arch](./imgs/arch_torchtitan.svg)

**模块说明**：

- **training (torchtitan)**：负责主训练流程，从 Data Buffer 读取数据，训练完后将权重同步至 rollout 模块；
- **rollout (vLLM + router)**：启动 vLLM 推理引擎并路由生成请求；自定义生成函数可以在其上封装多轮循环、工具调用、环境/沙盒交互和基于 verifier 的奖励；
- **data buffer**：桥梁模块，管理 prompt 初始化、自定义数据与 rollout 生成方法，包括通过同一接口产出样本的 agent 工作流。

## 快速开始

有关训练启动、配置覆盖与日志排障，请参考 [torchtitan 启动器](scripts/torchtitan/README.md)。训练后端的运行模式、配置和数据处理流程请参考 [torchtitan 训练后端](docs/torchtitan-backend.md)。

### Agentic RL 示例

Agent 工作负载通过 Vime 的定制接口接入标准 rollout / Data Buffer 循环，并不是独立框架：

- [`examples/coding_agent_rl`](examples/coding_agent_rl/README.md)：使用 Claude Code 或 Codex、沙盒工具、测试奖励和 token 精确轨迹片段的端到端 coding-agent RL；


## 参数说明

Vime 的参数分为三类：

1. **torchtitan 配置**：`--module` / `--config` 指定 `config_registry` 中的预置配置函数，由它返回预置配置，给出模型规格、优化器、精度等基线；`--<字段>.<子字段>` 形式的点分参数覆盖预置配置中的字段，例如 `--parallelism.tensor_parallel_degree 4`。详见 [torchtitan 训练后端](docs/torchtitan-backend.md) 第 6 章；
2. **vLLM 参数**：vLLM server 与 engine 相关选项以 `--vllm-` 为前缀（例如 `--vllm-gpu-memory-utilization`）。vllm-router 自身的选项以 `--router-` 传入（例如 `--router-policy round_robin`）；Vime 侧的 router 编排参数使用 `--vllm-router-` 前缀，包括 `--vllm-router-ip`、`--vllm-router-port` 和实际生效的 `--vllm-router-request-timeout-secs`。完整参数见 [vime/backends/vllm_utils/arguments.py](vime/backends/vllm_utils/arguments.py)。
3. **框架参数**：与 Vime 编排相关的开关（rollout 卡数、数据路径、RL 算法等），见 [vime/utils/arguments.py](vime/utils/arguments.py)。

`--rollout-num-gpus-per-engine` 对应每个 vLLM engine 的 tensor parallel size。默认 rollout 入口为 `vime.rollout.vllm_rollout.generate_rollout`。

## 代码阅读路径

建议从训练主循环开始，只在需要时继续深入：

```text
train.py: train
├─ vime/ray/placement_group.py       Ray 资源与 worker 初始化
├─ vime/ray/rollout.py              RolloutManager.generate：rollout 编排
│  └─ vime/rollout/vllm_rollout.py  样本生成与奖励计算
└─ vime/ray/actor_group.py          RayTrainGroup.async_train：训练调度
   └─ vime/backends/torchtitan_utils/actor.py
      ├─ engine.py                   torchtitan 模型执行
      └─ loss.py                     RL loss 与 advantage
```

第一次阅读时，可以把 `vime/utils/arguments.py` 当作配置入口。只有需要修改相关区域时，再深入 `vime/backends/vllm_utils/` 的部署细节和 `vime/backends/torchtitan_utils/update_weight/` 下的权重同步实现。

## 开发指南

- **欢迎贡献！** 若有功能建议、性能调优或使用体验反馈，欢迎提交 Issue / PR。

- 使用 [pre-commit](https://pre-commit.com/) 保证提交代码风格：

  ```bash
  apt install pre-commit -y
  pre-commit install

  # 运行 pre-commit 保证代码风格
  pre-commit run --all-files --show-diff-on-failure --color=always
  ```


## slime doc

Vime 由 slime 衍生而来。以下上游资源与本仓库文档仍沿用 slime 命名，可作为共享概念（数据生成、定制化、高级主题）的参考；其中与 megatron 相关的内容仅适用于 megatron 后端：

[![Documentation](https://img.shields.io/badge/slime_文档-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

- 上游仓库：[THUDM/slime](https://github.com/THUDM/slime)
