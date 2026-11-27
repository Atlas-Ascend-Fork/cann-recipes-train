# CANN 原子化特性能力总览

本文档以首页 [README 能力入口](../README.md#能力入口) 为主索引，额外补充每项能力的开启方式、限制边界，以及 README 能力表之外但全仓已经落地的链路型 / 工程化能力。

---

## 一、预训练 / 续训练能力

本节覆盖 CPT / 预训练 / 续训练阶段的训练优化能力，主要由 TorchTitan-NPU、MindSpeed 与 CANN 底层能力协同提供。

| 能力项 | 适用范围 | 已覆盖样例 | 开启方式 | 限制与说明 |
|----|----|----|----|----|
| **训练入图（torch.compile + AutoFuse）** | TorchTitan-NPU 训练 / 续训练，尤其是 Python 调度和小算子开销明显的模型 | [DeepSeek-V4-Flash / Pro](../llm_pretrain/deepseekv4/README.md) | 按样例脚本使用 TorchTitan-NPU 路径开启 `torch.compile`，由 CANN AutoFuse 完成图优化与算子融合；优化说明见 [DeepSeek-V4 TorchTitan-NPU + AutoFuse](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/llm_pretrain/deepseek_v4/zh/deepseek-v4_torchtitan_npu_autofuse.md) | 首次编译存在 warmup 成本；动态 shape、未适配算子或框架侧图断点会影响入图范围 |
| **大规模并行训练（TP / PP / EP / CP / FSDP2）** | 超大模型多机多卡训练，包含长序列和 MoE 场景 | [DeepSeek-V4-Flash / Pro](../llm_pretrain/deepseekv4/README.md)、[DeepSeek-V3.2 32K 长序列预训练](../llm_pretrain/deepseekv32/README.md) | 在样例配置中组合张量并行、流水并行、专家并行、上下文并行和 FSDP2；TorchTitan-NPU 通过 DTensor / FSDP2 支撑权重切分与预取 | 并行度需结合模型结构、序列长度、显存和集群拓扑调参；通信收益依赖 A3 / A5 等硬件互联条件 |
| **低精度训练（MXFP8 / HiF8）** | Ascend 950 上的大模型低精度预训练 / 续训练 | [DeepSeek-V3 MXFP8 / HiF8](../llm_pretrain/DeepSeekV3/README.md)、[DeepSeek-V4-Flash / Pro](../llm_pretrain/deepseekv4/README.md) | DeepSeek-V3 使用 `run_pretrain_dsk3_A5_8P_hif8.sh`、`run_pretrain_dsk3_A5_8P_mxfp8.sh` 等低精度训练脚本；DeepSeek-V4-Flash 使用 A5 MXFP8 训练脚本；原理见 [DeepSeek-V3 MXFP8 / HiF8](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/llm_pretrain/deepseek_v3/zh/deepseek-v3_pre_train_hif8_mxfp8.md) | 低精度格式、缩放策略和算子覆盖会影响收敛与精度，需要结合任务指标验证 |
| **Delta Rule / DeltaNet 线性注意力算子** | 昇腾 NPU 上的线性 Transformer 与长序列 NLP 训练 | [Ascend-TLA DeltaNet / NLP](../llm_pretrain/ascend_tla_deltanet/README.md) | 使用 Ascend-TLA 基于 Triton 实现的 `delta_rule` 算子，并通过 Quick Start Notebook 或 NLP 训练脚本完成算子、训练与评测验证 | 当前样例聚焦 DeltaNet 与 NLP 语言建模，不包含通用 `linear_attn` 算子目录和 CV 任务；环境与完整工程以上游 Ascend-TLA 说明为准 |
| **Swap Optimizer / 优化器状态卸载** | FSDP2 场景下优化器状态显存占用较高的大模型训练 | [DeepSeek-V4-Flash / Pro](../llm_pretrain/deepseekv4/README.md)、[DeepSeek-V3.2 32K 长序列预训练](../llm_pretrain/deepseekv32/README.md) | TorchTitan-NPU 样例参考 MindSpeed Swap Optimizer 思路，将 AdamW 动量按需换入 / 换出 Host 内存 | 依赖 Host 内存和 H2D 带宽；收益与优化器状态规模、流水切片粒度和通信计算重叠情况相关 |

---

## 二、RL 训练与 Rollout 能力

本节覆盖 veRL + MindSpeed + vLLM-Ascend 的强化学习训练链路，重点解决长序列 rollout、MoE 负载、图模式推理和 On-Policy 数据分布抖动问题。

| 能力项 | 适用范围 | 已覆盖样例 | 开启方式 | 限制与说明 |
|----|----|----|----|----|
| **TorchAir / npugraph_ex 推理图优化** | RL rollout 推理阶段，尤其是长序列 decode 中 CPU 调度开销明显的场景 | [DeepSeek-R1 RL](../llm_rl/deepseek/README.md)、[Qwen3-235B-A22B / Qwen3-32B RL](../llm_rl/qwen3/verl-mindspeed/README.md) | DeepSeek-R1 路径通过 vLLM-Ascend TorchAir 图模式；Qwen3 路径通过 npugraph_ex 和多档位图配置，相关 patch 见 `llm_rl/qwen3/verl-mindspeed/patches/verl/0012-verl-feature-npugraph_ex_for_spec_decode.patch` | 图模式依赖静态或有限档位 shape；首次编译、动态 batch 和投机解码组合会影响收益 |
| **SAM 无损投机解码** | RL rollout decode，适合数学推理、代码生成等存在重复结构的长序列生成 | [Qwen3-235B-A22B / Qwen3-32B RL](../llm_rl/qwen3/verl-mindspeed/README.md) | veRL 启动参数配置 `actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method=sam` 与 `num_speculative_tokens`；可选 `VLLM_SPECULATIVE_BATCH_SIZE_THRE` 控制自适应开关 | 无需 draft model，结果保持无损；收益依赖 response 结构重复度、batch size、拒绝采样开销和长尾阶段占比 |
| **Rollout Rebalance 序列级均衡调度** | On-Policy rollout 中 response 长尾导致 DP 组间负载不均的场景 | [Qwen3-235B-A22B / Qwen3-32B RL](../llm_rl/qwen3/verl-mindspeed/README.md) | 应用 Qwen3 patch 后，通过 `ROLLOUT_REBALANCE_ENABLE=1` 启用；配置项见 `llm_rl/qwen3/verl-mindspeed/patches/verl/features/rollout_optimize/config.py` 和 [Rollout Rebalance 文档](https://gitcode.com/cann/cann-recipes-docs/blob/master/train/features/zh/rollout_rebalance.md) | 依赖 vLLM-Ascend 多档位编图能力；数据集本身已经均衡时收益有限；KV Cache 迁移会引入额外开销 |
| **Length-Aware Resampler** | RL On-Policy 数据采样，适合 response 长度分布长尾、step 时间抖动明显的数据集 | [Qwen3-235B-A22B / Qwen3-32B RL](../llm_rl/qwen3/verl-mindspeed/README.md) | Hydra 配置 `data.sampler.class_name=LengthAwareEpochSampler`、`bucket_size`、`ema_decay` 等；最小脚本为 `llm_rl/qwen3/verl-mindspeed/internal/train_grpo_qwen3_resampler_example.sh` | 需要上一轮 rollout 的 response 长度统计；建议 `data.shuffle=False` 且 `dataloader_num_workers=0`；长度分布集中时收益有限 |
| **EPLB 专家负载均衡** | MoE rollout / 训练中专家负载不均明显的场景 | [Qwen3-235B-A22B / Qwen3-32B RL](../llm_rl/qwen3/verl-mindspeed/README.md) | 应用 `llm_rl/qwen3/verl-mindspeed/patches/verl/0009-verl-feature-support_EPLB.patch` 后，通过 `VLLM_ENABLE_EPLB=1` 启用 | 当前 Qwen3 真实权重实验中负载较均衡时收益不明显；更适合构造或真实存在专家热点的场景 |
| **HDP 混合数据并行** | RL 训练侧长序列场景，适合 CP 通信成本高、样本长度差异大的 batch | [Qwen3-235B-A22B / Qwen3-32B RL](../llm_rl/qwen3/verl-mindspeed/README.md) | 应用 verl / megatron / mindspeed HDP patch 后，通过 `USE_HDP=1` 启用；可选 `VERL_HDP_GROUP_DIR` 输出分组信息 | 需要模型侧 RoPE、attention mask、loss mask 与通信组适配；收益依赖长度分布和 CP 通信占比 |
| **old_log_prob 免重算** | GRPO 等 RL 算法中可复用当前 actor log_prob 的场景 | [DeepSeek-R1 RL](../llm_rl/deepseek/README.md)、[Qwen3-235B-A22B / Qwen3-32B RL](../llm_rl/qwen3/verl-mindspeed/README.md) | Qwen3 patch 提供 `actor_rollout_ref.actor.recompute_old_log_prob` 配置；设置为 `False` 可跳过 old_log_prob 重新计算 | 需确认算法语义允许使用当前 log_prob 作为 old_log_prob；不适合必须严格重算旧策略概率的训练设置 |
| **零冗余 TP 转 EP 权重通信 / AllToAllV Reshard** | RL 训推切换中训练并行策略与 rollout 推理并行策略不同的 MoE 模型 | [DeepSeek-R1 RL](../llm_rl/deepseek/README.md)、[Qwen3-235B-A22B / Qwen3-32B RL](../llm_rl/qwen3/verl-mindspeed/README.md) | Qwen3 patch `0005-verl-feature-moe_alltoallv.patch` 支持 EP 参数按需 AllToAllV 重分发；DeepSeek-R1 文档中给出零冗余 TP 转 EP 通信方案 | 与训练 / 推理并行拓扑强相关；需要保证权重命名、分片元信息和 MoE dispatcher 配置一致 |
| **Chunk-MoE Prefill 显存优化** | MoE 大 EP rollout prefill 阶段，专家负载导致激活峰值挤占 KV Cache 的场景 | [Qwen3-235B-A22B / Qwen3-32B RL](../llm_rl/qwen3/verl-mindspeed/README.md) | Qwen3 vLLM-Ascend patch `0006-vllm_ascend-feature-chunk_moe.patch` 默认随样例 patch 集成 | 主要缓解 prefill 峰值显存，具体 `max_num_batched_tokens` 与 `gpu_memory_utilization` 仍需按模型和集群调整 |

---

## 三、多模态 RL 能力

本节覆盖文生图模型强化学习训练的 NPU 适配能力，包括训练框架、生成模型算子和奖励模型链路。

| 能力项 | 适用范围 | 已覆盖样例 | 开启方式 | 限制与说明 |
|----|----|----|----|----|
| **FLUX GRPO 文生图强化学习 NPU 适配** | Atlas A3 上的 FLUX 文生图 GRPO 训练与生成图片质量优化 | [FLUX GRPO](../multimodal_rl/flux_grpo/README.md) | 准备 DanceGRPO 和 diffusers 源码，分别应用样例提供的 `DanceGRPO.patch` 与 `diffusers.patch`；训练脚本默认使用 HPSv2 奖励模型 | 当前目录仅包含适配补丁，不包含上游框架源码；默认面向 Atlas A3 16 die；PickScore 尚未适配 NPU，开启 `--use_pickscore` 不会生效，建议保持使用 HPSv2 |

---

## 四、Agent / Code RL 能力

本节覆盖 Agentic RL 与代码任务训练链路，属于数据、工具执行和评测闭环层面的工程化能力。

| 能力项 | 适用范围 | 已覆盖样例 | 开启方式 | 限制与说明 |
|----|----|----|----|----|
| **工具调用训练链路** | Tool Agent / 多轮工具调用训练，模型需要在 rollout 中调用外部工具并根据反馈更新策略 | [Qwen3 Tool Agent RL](../agent_rl/qwen3_tool_agent/README.md)、[Qwen3 多轮工具调用 Code RL](../agent_rl/qwen3_code_toolcall/README.md) | 按样例脚本启动 SFT / DAPO / toolcall 训练；Qwen3 Tool Agent 使用 verl-retool / agent loop，Code ToolCall 样例提供 tool config 与多轮数据构造脚本 | 工具协议、奖励函数和数据格式与任务强绑定；外部工具稳定性会影响训练吞吐和 reward 质量 |
| **代码沙盒 Code RL** | 代码生成、长上下文代码任务和在线评测反馈训练 | [Code RL 长上下文代码生成](../agent_rl/qwen2_code_rl/README.md)、[Qwen3 多轮工具调用 Code RL](../agent_rl/qwen3_code_toolcall/README.md) | 通过 ScaleBox / sandbox 配置执行代码评测；样例提供数据构造、训练脚本和 LiveCodeBench 评测脚本 | 依赖沙盒资源、测试用例质量和 reward 设计；多轮工具调用会增加 rollout 调度复杂度 |
| **Code RL 投机解码扩展** | 多轮工具调用 / 代码任务 rollout，希望降低 decode 成本的场景 | [Qwen3 多轮工具调用 Code RL](../agent_rl/qwen3_code_toolcall/README.md) | 样例提供 `suffix_rl_run.sh`、`eagle3_rl_run.sh` 等脚本，配置 vLLM speculative decoding 方法和投机 token 数 | EAGLE3 依赖 draft model；suffix / SAM 类无模型方法收益依赖代码模式重复度和接受率 |

---

## 五、样例覆盖索引

| 场景 | 当前样例 |
|----|----|
| 预训练 / 续训练 | [DeepSeek-V4-Flash / Pro](../llm_pretrain/deepseekv4/README.md)、[DeepSeek-V3.2 32K 长序列预训练](../llm_pretrain/deepseekv32/README.md)、[DeepSeek-V3 MXFP8 / HiF8](../llm_pretrain/DeepSeekV3/README.md)、[Ascend-TLA DeltaNet / NLP](../llm_pretrain/ascend_tla_deltanet/README.md) |
| 监督微调 | [Qwen3-1.7B MindSpeed-LLM SFT](../llm_sft/qwen3/README.md)、[Qwen3-1.7B TorchTitan-NPU SFT](../llm_sft/qwen3_1.7B_torchtitan/README.md)、[Qwen3-30B-A3B 医学 SFT](../llm_sft/qwen3_30b_a3b/README.md) |
| 强化学习训练 | [DeepSeek-R1 RL](../llm_rl/deepseek/README.md)、[Qwen3-235B-A22B / Qwen3-32B RL](../llm_rl/qwen3/verl-mindspeed/README.md)、[Qwen3-30B-A3B TorchTitan RL](../llm_rl/qwen3/verl-torchtitan/README.md)、[Qwen3-1.7B Wordle 多轮 RL](../llm_rl/qwen3_wordle/README.md)、[Qwen2.5-1.5B RL 入门样例](../llm_rl/qwen2_5/verl_npu_demo/README.md) |
| 多模态强化学习 | [FLUX GRPO](../multimodal_rl/flux_grpo/README.md) |
| Agent / Code RL | [Qwen3 Tool Agent RL](../agent_rl/qwen3_tool_agent/README.md)、[Code RL 长上下文代码生成](../agent_rl/qwen2_code_rl/README.md)、[Qwen3 多轮工具调用 Code RL](../agent_rl/qwen3_code_toolcall/README.md) |

---

## 平台与版本参考

各样例所依赖的设备型号、CANN 版本、torch / torch_npu 版本与关键依赖，详见对应样例 README；共性优化原理可继续参考 [cann-recipes-docs/train](https://gitcode.com/cann/cann-recipes-docs/tree/master/train) 下的专题文档。
