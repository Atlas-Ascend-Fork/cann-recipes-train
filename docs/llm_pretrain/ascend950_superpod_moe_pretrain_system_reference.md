# 基于昇腾950超节点的万亿MoE模型预训练系统参考实践

**摘要**：本文针对昇腾 950 超节点架构，基于 PyTorch 原生训练框架 + TorchTitan-NPU 昇腾插件，提出了一套面向万亿 MoE 模型的高效预训练部署与优化的系统方案。为应对超大规模、超长序列、多模态、高可用训练及训练开发者效率提升等复杂场景的挑战，本实践系统性构建了涵盖并行策略、内存优化、长序列优化、通算掩盖、DSL算子接入、算子自动融合、分布式优化器、图编译优化及训练高可用等全栈关键技术，并在 DeepSeek-V4/V4.1 系列模型上完成了端到端的全流程验证。

本文涉及的所有代码已开源：https://gitcode.com/cann/torchtitan-npu

## Highlights

* **极致性能与显存优化**：系统构建了参数与优化器状态卸载、细粒度激活值重计算与 Swap、Profile-aware 图编译通算掩盖、Cube-Vector (CV) 流水重叠以及自研 FlexMuon 分布式优化器等技术。在昇腾 950 超节点实测中，DeepSeek-V4-Flash 训练未掩盖通信时间占比低于 5%，整网 MFU 突破 35%；轻量化部署方面，仅需 64P 即可跑通 DeepSeek-V4.1-Flash 550B 的全参 CPT/SFT 训练。
* **敏捷算子与前沿算法支持**：打通 TileLang DSL 算子接入通路与 Ascend C 算子自动融合能力，以零代码/低代码方式支撑新结构算子的快速原型验证与性能提升，在DeepSeek-V4.1-Flash模型上实测整网**开箱加速比超19%**；深度适配与优化 Engram 记忆表、Muon 及超长序列等新场景，在 128P 规模 950 超节点上成功完成 1M 超长序列的预训练验证。
* **大规模训练高可用与Agent维测**：面向千卡、万卡级大规模训练场景，构建了涵盖动态弹性恢复、静默数据错误（SDC）检测与 Loss Spike 自动回滚的全流程容错体系，优化故障恢复性能，DeepSeek-V4 Flash 128 卡进程恢复时间**小于 5 分钟**；同时引入 KPBot 与 CANNBot 等智能 Agent，实现了性能调优与训练精度异常诊断的端到端自动化。

本文所提出的参考实践表明，基于昇腾 950 超节点和 PyTorch 原生训练生态，可以形成从模型部署、性能优化、算子开发到训练稳定性保障的一体化训练能力，为万亿至十万亿参数规模模型的持续演进提供可扩展的工程基础。

## 1、DeepSeek 预训练支持

- **[PyTorch 原生训练框架下的 950 超节点并行部署策略](ascend950/deployment.md)**：介绍超节点内外的 FSDP、EP 与 CP 并行部署，结合 DeepSeek-V4-Flash 训练验证，分析 550B、1.6T 模型的切分策略及向 10T 规模外推的部署权衡。
- **[DeepSeek-V4.1 图文多模态训练实践](ascend950/multimodal_sft.md)**：复用官方 Encoding 逻辑，接入 LLaVA / ShareGPT 等数据格式，支持多轮对话中的图文交错、多图及纯文本 SFT 训练。

## 2、训练性能极致优化

- **[950 全局统一虚拟地址下的整网内存优化](ascend950/memory_optimization.md)**：结合 Tensor 粒度重计算、CPU Offload 与传输计算流水重叠，降低参数、梯度、优化器状态及激活的 NPU 显存占用。
- **[FlexMuon：分布式优化器的通信、内存与计算协同优化](ascend950/flexmuon.md)**：通过参数布局重组、层级流水、NovaSwap、HSDP 去冗余及 Newton–Schulz 融合计算，优化分布式 Muon 的通信、状态存储与更新开销。
- **[DeepSeek-V4.1 Engram：大容量记忆表的分布式训练](ascend950/engram.md)**：结合 Host 存储、EP × EFSDP 联合分片、MXFP8 查表与 CPU 稀疏更新，支持 DeepSeek-V4.1-Flash 大容量记忆表训练。
- **[GraphTrainer：Profile-aware 通算掩盖与图级 Cube–Vector 并行](ascend950/graphtrainer.md)**：基于静态图依赖与设备 Profile 编排 EP 通信和计算，并通过图级 Cube–Vector 并行减少串行等待。

## 3、基于DeepSeek的昇腾950低精度量化训练实践

- **[基于DeepSeek的昇腾950低精度量化训练实践](ascend950/low_precision_training.md)**：基于 TorchAO-NPU 接入低精度矩阵乘、Attention 融合算子与量化感知训练，并通过 FSDP 权重预量化复用通信载荷与计算输入，实现计算与通信协同优化。

## 4、1M长上下文训练

- **[FSDP + CP 协同：128P 昇腾 950 超节点的 1M 长序列训练实践](ascend950/long_context.md)**：采用 CP 域复用 FSDP 域的部署策略，结合 KVAllGather、ZigZag 负载均衡与注意力融合算子，完成 DeepSeek-V4/V4.1 的 1M 长序列预训练验证。

## 5、大规模训练高可用与故障恢复

- **[面向千卡、万卡集群的预训练高可用能力](ascend950/high_availability.md)**：通过故障恢复加速、DP 弹性、SDC 检测与 Loss spike 回滚，支持大规模集群的长周期训练。

## 6、开发者效率提升

- **[TileLang DSL 算子开发与训练接入](ascend950/tilelang_integration.md)**：通过 TileLang Python DSL 表达和调优算子，并在 TorchTitan / TorchTitan-NPU 中接入 DeepSeek-V4/V4.1 的 mhc_pre / mhc_post 相关算子。
- **[基于 torch.compile 与 AutoFuse 的 Ascend C 算子自动融合](ascend950/ascendc_autofuse.md)**：复用 torch.compile 编译入口，通过 Inductor 与 [Ascend C AutoFuse](https://gitcode.com/cann/graph-autofusion) 自动生成面向昇腾 NPU 的融合算子，减少手写开发与算子执行开销。
- **[训练 Agent 性能调优与精度异常诊断实践](ascend950/training_agents.md)**：通过 KPBot 分析和优化 Host-Bound 性能瓶颈，通过 CANNBot 定位数值偏差、NaN/Inf 与确定性不一致等训练精度问题。
