# SWE-Gym 代码修复任务

SWE-Gym 的强化学习要求Agent根据问题描述，在任务沙箱中阅读代码、修改仓库并运行测试。任务结束后，框架提取代码补丁，在独立的评测沙箱中应用补丁并运行该题的测试。评测器将问题判定为已解决时，奖励为 1，否则为 0。

Qwen3.5-4B 训练入口使用 TorchTitan-NPU 训练、vLLM-Ascend 推理，支持 Harbor 和 AgentENV 两种沙箱。默认Agent Harness为 Codex，也可选择 Claude Code。

## 使用指南

已有任务数据包（bundle）时，可直接阅读训练指南。首次准备数据时，在 NPU 服务器上运行准备脚本，通过 AgentENV 创建或复用模板，完成任务检查并导出 bundle。Harbor 训练可复用这份 bundle，需提前在各台 CPU 沙箱服务器上备齐与包中镜像摘要一致的任务镜像。

| 阅读目的 | 文档 |
| --- | --- |
| 准备任务数据包，核对模板、镜像和测试规格 | [SWE-Gym 数据准备](../../docs/agentic-rl/swe-agentic-rl/data-preparation.md) |
| 在 NPU 服务器上启动训练，查看奖励、日志和轨迹 | [SWE-Gym 训练指南](../../docs/agentic-rl/swe-agentic-rl/training.md) |
| 准备Agent Harness安装包，初始化沙箱内的运行环境 | [Agent Harness安装与初始化](../../docs/agentic-rl/env-and-config/harness-preparation.md) |
| 编写本地配置文件，查询参数覆盖顺序 | [AgenticRL 配置说明](../../docs/agentic-rl/env-and-config/configuration.md) |
| 部署 Harbor 或 AgentENV，配置沙箱到模型的回调连接 | [沙箱部署与配置](../../docs/agentic-rl/env-and-config/sandboxes.md) |

## 代码入口

- [训练脚本](run-qwen35-4B-swegym-torchtitan.sh) 加载 SWE-Gym 任务配置，调用公共 Qwen3.5-4B 训练脚本。
- [数据准备脚本](prepare_swegym.py) 检查源数据、准备模板、验证任务并导出训练数据包。
- [轨迹生成器](generate.py) 创建任务沙箱、运行Agent Harness，将交互轨迹和奖励交给训练框架。
- [任务评测代码](swe.py) 提取模型补丁，在独立沙箱中执行测试并计算奖励。

两个任务共用的启动脚本和配置样例见[公共配置目录](../common/)。其他任务的介绍见[强化学习样例索引](../README.md)。
