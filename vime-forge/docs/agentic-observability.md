# Agent rollout trace 使用说明

Agent rollout 的执行过程记录在 VIME 的 `Sample.trace` 中。Trace 包含带开始和结束时间的阶段 span，以及表示状态变化或诊断结果的瞬时 event。
Trace 不包含提示词、命令正文、标准输出、标准错误或认证信息。

## 保存和查看 Trace

训练时使用 VIME 的 rollout debug dump 参数保存 Trace：

```bash
--save-debug-rollout-data '/path/to/debug/rollout_{rollout_id}.pt'
```

然后使用时间线查看器：

```bash
python tools/trace_timeline_viewer.py /path/to/debug/rollout_0.pt
```

查看器会生成结构化缓存和 HTML 页面，并默认启动本地服务。`--no-serve` 只生成文件，不启动服务。

一个 Agent trial 可以产生多个训练 Sample。Sample 表示训练片段，Trace 表示 trial 的执行过程；同一 trial 产生的 Sample 共享同一个 Trace carrier，时间线查看器按 Trace 只展示一个 trial。
Sample 的 `rollout_id` 用于训练侧关联，Trace 的 `trace_id` 用于时间线关联。

## Trace 的阶段结构

阶段名称沿用 VIME Trace API：

```text
coding_rollout
├── sandbox_boot_wait
│   └── sandbox_create → harness_install
│       └── sandbox_command
│   └── sandbox_cleanup       (启动或安装失败时)
├── sandbox_boot_backoff       (启动失败后才出现)
├── workspace_prepare
│   └── sandbox_command
├── agent_run
│   ├── agent_model_turn
│   │   ├── agent_tokenize
│   │   └── vllm_generate
│   └── sandbox_command
├── patch_extract
│   └── sandbox_command
├── sandbox_cleanup            (退出 Agent 沙箱上下文时)
├── evaluation
│   ├── evaluation_sandbox_create
│   ├── evaluation_prepare
│   │   └── sandbox_command
│   ├── evaluation_tests
│   │   └── sandbox_command
│   └── evaluation_sandbox_cleanup
├── trajectory_finalize
└── adapter_cleanup
```

`sandbox_boot_wait` 是一次完整沙箱启动尝试的外层 span，名称中的 “wait” 不代表它只记录等待时间。它包含等待启动并发槽位、`sandbox_create` 和 `harness_install`；其中执行顺序是先完成 `sandbox_create`，再执行 `harness_install`，安装完成后才进入 `workspace_prepare`。如果创建或安装失败，会先执行失败清理，再进入 `sandbox_boot_backoff`，随后重新尝试启动。

启动成功后，`workspace_prepare`、`agent_run` 和 `patch_extract` 都在同一个 Agent 沙箱上下文中执行。`patch_extract` 完成后退出这个上下文，随后执行正常的 `sandbox_cleanup`，再进入 `evaluation`。

`sandbox_command` 表示一次真实沙箱命令的执行阶段。它会挂在命令发生时的当前父阶段下，因此可能出现在 `harness_install`、`workspace_prepare`、`agent_run`、`patch_extract`、`evaluation_prepare` 或 `evaluation_tests` 下面。启动、清理和轮询等内部 RPC 不会单独生成这个 span。

`agent_model_turn` 表示一轮 Agent 模型请求；`agent_tokenize` 和 `vllm_generate` 是这一轮请求中的子阶段。`evaluation` 下的阶段用于准备评测环境、执行测试并记录评测沙箱的生命周期。

阶段 span 记录开始时间、结束时间、耗时和属性。模型请求阶段会记录 HTTP 状态、输入输出 token 数、缓存 token 数以及 vLLM 返回的请求指标（如果服务端提供）。

## Agent 诊断 event

Agent 相关的补充信息以 event 写入同一条 Trace：

| event | 用途 |
| --- | --- |
| `agentic.sandbox_boot_attempt_failed` | 沙箱启动或安装失败及是否重试 |
| `agentic.sandbox_ready` | 沙箱创建完成 |
| `agentic.sandbox_rpc_retry` / `agentic.sandbox_rpc_failed` | 沙箱 RPC 重试和最终失败 |
| `agentic.sandbox_destroy_failed` | 沙箱销毁失败 |
| `agentic.trial_aborted` | Agent 执行提前终止 |
| `agentic.streaming_error` | Adapter 流式响应处理失败 |
| `agentic.turn_recorded` | 一轮模型响应已写入轨迹树 |
| `agentic.turn_disposition` | 一轮响应的训练处置结果 |
| `agentic.assistant_rewrite` | 轨迹树合并了 assistant 重写 |
| `agentic.token_alignment` | 相邻 turn 的 token 对齐结果 |
| `agentic.trajectory_summary` | 轨迹整理完成后的统计摘要 |
| `agentic.verifier_diagnostic_error` | Verifier 结果解析诊断失败 |
| `agentic.grpo_group_reward_counts` | 一个 GRPO group 中 reward 为 1 和 0 的 rollout 数量 |

这些 event 是点事件，没有单独的开始和结束时间。事件属性只保存标量值。

## 训练片段诊断信息

轨迹整理完成后，首个输出 Sample 的 `metadata["trajectory_summary"]` 保存本次执行的汇总信息，包括 turn 数、叶节点数、生成的 Sample 数、token 对齐统计、处置结果统计、
总 token 数和有效 token 数。

每个输出 Sample 的 `metadata["trajectory_segment"]` 保存该训练片段的 turn 索引、截断前后 token 数、
`trainable_token_count` 和 `trainable_token_ratio`。
有效 token 比率定义为：

```text
trainable_token_ratio = sum(loss_mask) / len(tokens)
```

分母包含上下文、工具返回和模型输出，表示整个训练片段中实际参与损失计算的 token 占比。

## GRPO group reward 统计

GRPO rollout 汇总阶段会为每个 group 写入 `agentic.grpo_group_reward_counts` event：

```text
reward_1_count：该 group 中 reward == 1 的 rollout 数
reward_0_count：该 group 中 reward == 0 的 rollout 数
```

同时，TensorBoard 记录混合 group 数量。混合 group 指：

```text
1 <= reward_1_count <= VIME_GRPO_MAX_CG
```

并且至少包含一个 reward 为 0 的 rollout。`VIME_GRPO_MAX_CG` 默认取
`n_samples_per_prompt - 1`，用于调整统计上限。

## 直接使用 Trace API

自定义 rollout、工具调用或评测逻辑可以直接使用上游 Trace API：

```python
from vime.observability.trace_utils import trace_event, trace_span

with trace_span(sample, "custom_stage") as span:
    ...
    span.update({"status": "ok"})

trace_event(sample, "custom_event", attrs={"exit_code": 0})
```

通用 Trace API、时间线数据格式和查看器参数见 [`docs/zh/developer_guide/trace.md`](zh/developer_guide/trace.md)。
