# 基于 SAM（后缀自动机） 的无损投机推理技术

## 1. 简介
在大语言模型（LLM）的强化学习后训练过程中，为了收集高质量的训练数据或评估模型策略性能，需要在训练过程中进行海量的交互式采样（Rollout）。这个采样过程占据了整个训练链条绝大部分的时间和计算开销，因此，如何高效执行采样已成为提升后训练效率的关键瓶颈。

投机解码（Speculative Decoding）技术是一种常见的推理加速方案。然而，当前主流的基于模型（Model-Based）的投机解码方法，如 EAGLE 或 MTP，需要依赖一个额外训练的小型辅助草稿模型（Draft Model）。在模型参数频繁更新的后训练场景中，保持草稿模型与主模型同步需要引入显著的额外训练和部署成本，严重限制了其应用效率。

对此，我们尝试引入一种无模型（Model-Free），基于 SAM（后缀自动机）的无损投机解码方案，旨在不引入额外模型参数的前提下加速 RL 采样过程。

对于 SAM 技术的详细介绍请参见文章：[SAM 投机推理：长序列强化学习训练加速利器](../llm_rl/sam_decoding.md)。

## 2. 使用说明

为了开启投机推理，需要配置以下变量。

| 配置                                                                                     | 类型                  | 说明                                              |
| ---------------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------- |
| `actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method`                 | veRL Hydra 命令行参数 | 投机推理的方法，例如 SAM，必须配置。              |
| `actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens` | veRL Hydra 命令行参数 | 投机推理的预测 token 数，必须配置。               |
| `VLLM_SPECULATIVE_BATCH_SIZE_THRE`                                                       | 环境变量              | batch_size 自适应开关阈值，可选配置，默认值为 32。 |

例如，Qwen3-32B 样例默认配置了 `method=sam` 和 `num_speculative_tokens=3`，开启 SAM 无损投机推理特性；如果希望关闭 SAM 无损投机推理特性，需要在启动 verl 训练任务时移除 internal/train_grpo_qwen3_32b_32die_true_weight.sh 中的这 2 个 Hydra 配置。示例如下：

```diff
...

python3 -m verl.trainer.main_ppo --config-path="${CONFIG_DIR}" \
     --config-name='ppo_megatron_trainer.yaml' \
     ...
     +actor_rollout_ref.actor.megatron.override_transformer_config.cp_window_size=1 \
-    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method=sam \
-    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens=3 \
     $@
```

## 3. 实现方案
本方案基于 vLLM 和 vLLM-Ascend 框架实现了基于 SAM 的投机推理。通过动态后缀自动机（D-SAM），系统能够在无需额外辅助模型训练开销的情况下，在线、增量、高效地编码当前序列的所有子串信息，在 $O(1)$ 的平均时间内从现有 Rollout 序列中检索出候选草稿，从而能够显著降低 Rollout 采样过程中的推理时延。相比传统检索模型如 N-gram，SAM 能够具有更高的运行效率和接受率。

同时，为了达成更好的加速效果，我们还进行了其他优化：
 - 拒绝采样加速：通过重写 vllm_ascend 中的`rejection_sampler`代码，优化了投机推理的整体表现。
 - batch_size 自适应开关：针对在 Rollout 的初始阶段，batch_size 巨大导致投机推理产生严重性能劣化的问题，检测 batch_size 超出阈值时关闭投机推理。在规避高 batch_size 下投机推理性能劣化的同时，为 SAM 自动机提供预热阶段。当 Rollout 进入长尾阶段，batch_size 数减少时，动态开启投机推理，从而达成更优的加速效果。

对于这些解决方案的详细介绍请参见文章：[SAM 投机推理：长序列强化学习训练加速利器](../llm_rl/sam_decoding.md)。

### 具体实现
#### `SAM` 类
SAM 无损投机推理能力的核心组件，实现了上文提到的动态 SAM 自动机本体，提供以下能力接口：

1. 状态更新
 - **`add_tokens(tokens)`**：将新的token添加到自动机中，同时扩展自动机的状态。

2. 序列预测
 - **`gen_draft(index, start_token)`**: 从当前输入的token生成指定数量个 token 的后续序列。

代码： [vllm_ascend/spec_decode/sam_proposer.py](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/spec_decode/sam_proposer.py)

#### `SAMProposer` 类
`vllm-ascend`后端调用`SAM`能力的适配接口。`vllm_ascend`通过`NPUModelRunner`中的`propose_draft_token_ids`调用此接口。

`SAMProposer`会调用`SAM`提供的状态更新与序列预测能力。

代码：[vllm_ascend/spec_decode/sam_proposer.py](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/spec_decode/sam_proposer.py)

#### 拒绝采样加速
用高效 tensor 操作替换了`vLLM_Ascend`的`AscendRejectionSampler`实现中的多个 for 循环，提升投机推理的整体效率。

代码：[vllm_ascend/0008-vllm_ascend-feature-tree_rejection_sampler.patch](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/0008-vllm_ascend-feature-tree_rejection_sampler.patch)

#### batch_size 自适应开关
在`vllm_ascend`的`NPUModelRunner`添加一个动态开关。每当`NPUModelRunner`取得输入请求，它会检测 batch_size 是否超过阈值，并通过 flag `speculative_decoding_active`来控制投机推理的开启与关闭。

代码：[0001-vllm_ascend-feature-bs_threshold_for_spec_decode.patch](../../llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend/0001-vllm_ascend-feature-bs_threshold_for_spec_decode.patch) 

## 4. 使能效果
我们在 Qwen3-32B Dense 模型上，于真实的 RL 后训练场景（DAPO，数学推理数据集上）进行了全面的端到端验证。
相关配置：

- 数据集： DAPO\-MATH\-17k
- 最大输出长度（`max_response_length`）：34816
- 训练 batch 大小（`train_batch_size`）：32
- 生成 batch 大小（`gen_batch_size`）：96
- Rollout 最大请求数（`max_num_seqs`）：128
- Rollout 模型张量并行（`tensor_model_parallel_size`）：8
- SAM 相关配置：自适应 batch size 开关阈值为 8，投机 token 个数为 3。
- 910C 卡数：2 机 16 卡 32die
- 运行模式：eager

1. 精度验证：
基于 Qwen3\-32B 模型在数学数据集 GSM8k 以及 AIME2024 进行了纯推理的测评（pass@1），采样参数为 temperature=0.6，repetition penalty=1，不设置 top\-k 和 top\-p。测试结果如下：

| 数据集   | 不开启 SAM | 开启 SAM |
| -------- | --------- | ------- |
| GSM8k    | 88.9      | 88.7    |
| AIME2024 | 66.5      | 66.8    |

2. 性能收益：

Qwen3-32B
|           | 单步总推理时间 / s | 单步总时间 / s |
| --------- | ------------------ | -------------- |
| 不开启 SAM | 9094               | 11870          |
| 开启 SAM   | 8223               | 11000          |
| 收益      | 10.59%             | 7.91%          |

Qwen3\-235B\-A22B
|           | 单轮平均推理时间 / s | 单步总推理时间 / s | 单步总时间 / s |
| --------- | -------------------- | ------------------ | -------------- |
| 不开启 SAM | 7102.92              | 14287.54           | 15441.45       |
| 开启 SAM   | 6467.52              | 12811.98           | 13960.98       |
| 收益      | 9.82%                | 11.52%             | 10.60%         |

更多实验及实验结论，请参见文章：[SAM 投机推理：长序列强化学习训练加速利器](../llm_rl/sam_decoding.md)。