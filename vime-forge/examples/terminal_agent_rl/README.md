# Terminal／TMax 任务

将 Terminal-Bench 或 TMax 的终端任务用于 AgenticRL 训练；两种数据共用同一入口，通过 bundle 选择。训练使用 TorchTitan NPU 后端和 vLLM，支持 Harbor、AgentENV，以及 Codex、Claude Code。

## 使用指南

| 阅读目的 | 文档 |
| --- | --- |
| 准备任务数据包 | [数据准备](../../docs/agentic-rl/terminal-agentic-rl/data-preparation.md) |
| 配置、检查和启动训练 | [训练指南](../../docs/agentic-rl/terminal-agentic-rl/training.md) |
| 准备 Agent 安装包与初始化环境 | [Agent 安装与初始化](../../docs/agentic-rl/env-and-config/harness-preparation.md) |
| 查询环境变量及参数覆盖规则 | [配置说明](../../docs/agentic-rl/env-and-config/configuration.md) |
| 配置 Harbor／AgentENV 与网络 | [沙箱部署与配置](../../docs/agentic-rl/env-and-config/sandboxes.md) |

数据处理、bundle 导出和训练启动在 **NPU 服务器**执行；镜像下载、安装包制作及沙箱运行在 **CPU 沙箱服务器**完成。具体命令及执行位置见上述指南。

## 代码入口

- [训练入口](run-qwen35-4B-tbench-torchtitan.sh)：选择任务并调用公共 Qwen3.5 启动脚本。
- [轨迹生成器](generate.py)：执行任务并将轨迹交给训练框架。
- [公共脚本与配置模板](../common/)：两个任务共用的配置解析及样例。

其他任务的介绍见[强化学习样例索引](../README.md)。

<!-- 保留原章节链接，操作内容统一见上方训练与数据准备指南。 -->
<a id="21-数据包结构和后端要求"></a>
<a id="22-转换原始任务"></a>
<a id="23-通过-harbor-生成-bundle"></a>
<a id="24-通过-agentenv-生成-bundle"></a>
<a id="25-探测与资格筛选的含义"></a>
<a id="1-任务执行与评测"></a>
<a id="2-准备任务与-bundle"></a>
<a id="3-配置运行环境agent-和沙箱"></a>
<a id="4-预览和启动训练"></a>
<a id="5-查看输出与评测结果"></a>
