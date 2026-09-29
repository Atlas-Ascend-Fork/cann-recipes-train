# 配置参考

本文覆盖 `config/` 下训练 recipe 使用的全部配置项，以及训练启动脚本和 NPU 内存诊断相关环境变量。配置由 Python 文件提供，通过 `train.py --config` 加载。

## 配置选择与覆盖

选择 recipe：

```bash
torchrun --nproc-per-node=8 train.py \
  --config config/dspark/dspark_deepseek_v4_flash.py
```

使用 `--opts key=value` 覆盖配置；每个覆盖项需单独传一次：

```bash
--opts train.lr=0.0003 \
--opts train.max_train_steps=1000 \
--opts model.tau_loss_alpha=0.05
```

值由 YAML 解析，因此布尔值、空值和列表可写为：

```text
true
false
null
[1, 9, 17, 25, 33]
```

约束：`--opts` 只能覆盖所选配置文件中已经存在的键，或训练流程集中提供的 profiling/checkpoint 保留键。其他不存在的键会报 `Unknown config key`。命令行覆盖完成后才执行 `finalize_cfg`，因此派生路径会使用覆盖后的 `project_name` 和 `exp_name`。

## Recipe 一览

| 系列 | Qwen3 | Gemma4 | DeepSeek V4 | trainer |
| --- | --- | --- | --- | --- |
| Eagle3 | 4B/8B/14B | 12B | 无 | `Qwen3Eagle3Trainer` / `Gemma4Eagle3Trainer` |
| DFlash | 4B/8B/14B | 12B | 无 | 当前复用 DSpark trainer |
| DSpark | 4B/8B/14B | 12B | Flash | `Qwen3DSparkTrainer` / `Gemma4DSparkTrainer` / `DeepSeekV4DSparkTrainer` |

DFlash recipe 使用 `block_size=7`、`num_draft_layers=5`，并关闭 Markov、confidence 和 L1 loss；DSpark Qwen/Gemma recipe 使用相同 block/layer 数，但开启 Markov、confidence 和 L1 loss。DeepSeek V4 DSpark 使用 `block_size=5`、`num_draft_layers=3`。

## 顶层配置

| 配置项 | 示例默认值 | 含义与约束 |
| --- | --- | --- |
| `project_name` | `"deepspec"` | 项目名，用于生成 TensorBoard 和 checkpoint 目录。 |
| `exp_name` | 各 recipe 不同 | 实验名；同一 checkpoint 根目录下应保持唯一，否则可能自动恢复旧实验。 |
| `seed` | DSpark/DFlash `42`；Eagle3 `0` | 全局随机种子，也控制 on-the-fly 训练/验证划分。恢复 checkpoint 时不要修改。 |
| `BASE_TB_DIR` | `~/tensorboard` | TensorBoard 根目录。DeepSeek V4 recipe 可由 `BASE_TB_DIR` 环境变量覆盖。 |
| `BASE_CKPT_DIR` | `~/checkpoints` | checkpoint 根目录。DeepSeek V4 recipe 可由 `BASE_CKPT_DIR` 环境变量覆盖。 |
| `finalize_cfg` | recipe 函数 | 派生 `logging.tensorboard_dir` 和 `logging.checkpoint_dir`，通常不应通过 `--opts` 覆盖。 |

派生路径通常为：

```text
logging.tensorboard_dir = BASE_TB_DIR/project_name/exp_name
logging.checkpoint_dir  = BASE_CKPT_DIR/project_name/exp_name
```

## `model` 通用配置

| 配置项 | 常见默认值 | 适用范围 | 含义与约束 |
| --- | --- | --- | --- |
| `target_model_name_or_path` | recipe 对应 Hugging Face 名称 | 全部 | Target model 名称或本地目录；训练时由 `scripts/train/offline.yaml` 的 `target_model_path` 提供。它可以与 cache manifest 中记录的生成路径不同。DeepSeek V4 默认可由 `TARGET_MODEL_PATH` 覆盖。 |
| `target_layer_ids` | 模型相关列表 | 全部 | 写入 cache 并输入 draft model 的 target hidden layers。必须严格递增且在 target 层范围内。Eagle3 必须恰好 5 个；DSpark 允许 `-1` 表示 embedding 输出。 |

Qwen3 4B/8B 默认 `[1, 9, 17, 25, 33]`；Qwen3 14B 默认 `[1, 10, 19, 28, 37]`；Gemma4 12B 默认 `[5, 17, 29, 41, 46]`；DeepSeek V4 默认 `[40, 41, 42]`。

Target cache 是根据 target model 的实际权重、`target_layer_ids` 和数据生成的。更换 target model 权重或 layer ids 后必须重新生成匹配的 cache，不能继续使用旧 cache。manifest 中的 `target_model_name_or_path` 只记录生成 cache 时使用的来源路径；训练运行时的本地路径由 `offline.yaml` 决定，不要求字符串相同。

## `model` DSpark/DFlash 配置

| 配置项 | DSpark 默认值 | DFlash 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `block_size` | Qwen/Gemma `7`；DeepSeek `5` | `7` | 每个 anchor 预测的 draft position 数；决定 `accept_rate@0..N-1` 数量。 |
| `num_draft_layers` | Qwen/Gemma `5`；DeepSeek `3` | `5` | Draft decoder 层数，必须为正整数。 |
| `mask_token_id` | Qwen `151669`；Gemma `4`；DeepSeek `128799` | 同模型 | DSpark 噪声/mask token id，必须属于 tokenizer vocabulary。 |
| `num_anchors` | `512` | `512` | 每个样本采样的 anchor 上限；增大会提高覆盖率，同时增加显存和计算量。 |
| `markov_rank` | `256` | `0` | Markov head rank，要求 `>= 0`。`0` 关闭 Markov head。 |
| `markov_head_type` | `"vanilla"` | 未配置 | `markov_rank > 0` 时必需；必须是实现支持的 head 类型。 |
| `confidence_head_alpha` | `1.0` | `0.0` | Confidence loss 权重，要求 `>= 0`。大于 0 时启用 confidence head。 |
| `confidence_head_with_markov` | `True` | 未配置 | `confidence_head_alpha > 0` 时必需；启用后需要 Markov 特征，因此应与正数 `markov_rank` 组合。 |
| `loss_decay_gamma` | `4.0` | `4.0` | 按 draft position 指数衰减 loss 权重；`None` 或非正数关闭衰减。 |
| `ce_loss_alpha` | `0.1` | `1.0` | Token cross-entropy loss 权重。 |
| `l1_loss_alpha` | `0.9` | `0.0` | Draft/target token 分布 L1 loss 权重；正数要求 forward 提供 aligned target logits。 |
| `tau_loss_alpha` | DeepSeek `0.05`；其他 recipe 隐式 `0.0` | 隐式 `0.0` | 期望接受长度辅助 loss 权重，要求 `>= 0`；正数要求 aligned target logits。只有包含该键的 recipe 才能用 `--opts` 覆盖。 |
| `n_routed_experts` | DeepSeek `None` | 不适用 | 仅 DeepSeek V4 MoE。`None` 继承 target config；正整数覆盖 target expert 数，且必须不小于 `num_experts_per_tok`。不包含 EP 分片语义。 |
| `sliding_window` | DeepSeek Flash `128` | 不适用 | 仅 DeepSeek V4 DSpark。`None` 表示每个 draft block 可见 anchor 前全部 context；正整数 `N` 表示只可见 `[anchor-N, anchor)`。布尔值、零、负数和非整数均非法。未配置时为 `None`，不会继承 target config。 |

重要组合：

- `markov_rank=0` 时不需要 `markov_head_type`，Markov head 关闭。
- `confidence_head_alpha=0` 时 confidence head 和 confidence loss 关闭，不需要 `confidence_head_with_markov`。
- `confidence_head_with_markov=True` 时应保证 `markov_rank>0`，否则模型没有可用的 Markov head。
- `ce_loss_alpha`、`l1_loss_alpha`、`tau_loss_alpha`、`confidence_head_alpha` 是加权相加关系；全部为零会得到无训练意义的目标。
- DFlash recipe 本质上通过 `markov_rank=0`、`confidence_head_alpha=0`、`l1_loss_alpha=0` 形成纯 CE 配置。

DeepSeek V4 Flash 标准 recipe 显式使用 `sliding_window=128`，与当前 CANN DSpark 配置一致。窗口会改变训练 attention 的计算流；修改该值或从旧实现升级后必须重新训练，旧 checkpoint 不能通过只改配置获得等价语义。当前 CANN DSpark 仅应部署使用相同正整数窗口训练的 checkpoint。

## `model` Eagle3 配置

| 配置项 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `ttt_length` | `7` | Test-time training/预测序列长度，要求 `>= 1`。 |
| `step_loss_decay` | `0.8` | 各预测 step 的 loss 衰减系数，要求 `> 0`。 |
| `draft_num_hidden_layers` | `1` | Eagle3 draft model 层数，要求 `>= 1`。 |

Eagle3 的 `target_layer_ids` 必须恰好包含 5 个严格递增的有效 target layer id。Eagle3 不读取 DSpark 的 `block_size`、Markov、confidence、L1 或 tau 配置。

## `train` 配置

| 配置项 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `trainer_cls` | recipe 决定 | Trainer 类，必须和模型系列及 target 架构匹配。不要仅靠 `--opts` 改成字符串。 |
| `lr` | `6.0e-4` | AdamW 学习率。 |
| `warmup_ratio` | `0.04` | 总 optimizer steps 中用于 warmup 的比例。应在 `[0, 1]` 内。 |
| `weight_decay` | `0.0` | AdamW 权重衰减。 |
| `precision` | `"bf16"` | 支持 `bf16`、`fp16`、`fp32`；必须是代码映射中的值。 |
| `local_batch_size` | `1` | 每个 rank 每个 micro-step 的样本数，必须为正整数。 |
| `global_batch_size` | `512` | 一个 optimizer step 的全局样本数，必须被 `world_size * local_batch_size` 整除。 |
| `num_train_epochs` | `10` | `max_train_steps=None` 时决定训练总步数。 |
| `max_train_steps` | `None` | 非空时覆盖 epoch 推导出的总步数；`num_train_epochs` 会按所需步数重新计算为向上取整值。 |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值。多卡使用 FSDP clip，单卡使用 PyTorch clip。 |
| `sharding_strategy` | 多数 recipe `"no_shard"`；DeepSeek `"full_shard"` | 支持 `full_shard`、`shard_grad_op`、`no_shard`、`hybrid_shard`、`hybrid_shard_zero2`、`_hybrid_shard_zero2`。单 rank 不进行 FSDP wrap。Hybrid 策略要求 world size 与节点设备布局兼容。 |
| `optimizer_cpu_offload` | DeepSeek `False`；其他 recipe 隐式 `False` | 将 FP32 master weights 和 AdamW state 放在 CPU。降低设备显存、增加主机内存和传输开销。 |
| `torch_compile` | Qwen DSpark/DFlash `True`；其余 `False` | 非 NPU 设备启用 `torch.compile(dynamic=True)`；NPU 上会打印提示并跳过。 |

Batch 与 epoch 约束：

```text
gradient_accumulation_steps = global_batch_size / (world_size * local_batch_size)
samples_per_epoch = floor(train_size / global_batch_size) * global_batch_size
steps_per_epoch = samples_per_epoch / global_batch_size
```

划分后的训练集至少要包含一个完整 `global_batch_size`，否则初始化失败。训练时不足一个 global batch 的尾部样本不参与该 epoch。

`max_train_steps` 与 `num_train_epochs` 的优先级：

- `max_train_steps=None`：总步数为 `num_train_epochs * steps_per_epoch`。
- `max_train_steps=N`：以 N 为准，`num_train_epochs` 仅被重新计算用于展示和 sampler 周期。

Activation checkpointing、optimizer CPU offload 和 FSDP 可组合使用，三者分别影响激活、optimizer state 和参数/梯度显存；组合能进一步降显存，但通常会增加计算或主机传输开销。

## 训练 NPU profiling 配置

训练 profiling 默认关闭，由训练流程集中提供默认值；无需在 recipe 中声明。开启示例：

```bash
--opts train.npu_profiling.enabled=true \
--opts train.npu_profiling.wait=2 \
--opts train.npu_profiling.warmup=1 \
--opts train.npu_profiling.active=5
```

`schedule` 的 `wait`、`warmup`、`active`、`repeat` 和 `skip_first` 以完成一次 optimizer step 计数，覆盖 forward、backward、梯度裁剪、BF16 master-gradient/AdamW/scheduler/参数回写。输出默认写入 `logging.checkpoint_dir/profiling/rank_N/`，每个 rank 独立生成原始 `torch_npu.profiler` trace 和 `metadata.json`，交给 Ascend profiling 工具分析，不依赖 TensorBoard。`output_dir` 可覆盖根目录；`with_stack`、`record_shapes`、`profile_memory` 默认关闭。

显式开启时若当前设备不是 NPU 或缺少 `torch_npu.profiler`，训练启动会报错。旧的 `ENABLE_PROFILE` 仍仅控制推理/通用 profiler，不控制训练 profiling。

## 训练验证配置

| 配置项 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `data.validation_fraction` | DeepSeek 显式 `0.1`；代码回退 `0.1` | 从同一个 target cache 中 on-the-fly 划为验证集的比例，要求 `0 <= value < 1`。 |
| `seed` | recipe 值 | 决定可复现的随机划分；恢复训练时不得修改。 |

验证与 checkpoint 保存绑定：每次满足 `logging.checkpointing_steps` 的周期保存时，先保存当前 step，再执行验证并更新最佳 checkpoint；训练结束时若最后一步尚未周期保存，还会保存最终 checkpoint 并执行一次验证；已在同一步周期保存时不会重复。挂起前的应急 checkpoint 不强制验证。

`validation_fraction=0` 会产生空验证集，从数据侧关闭验证。验证集至少需要 `global_batch_size` 个样本，否则无法形成完整分布式 batch，将打印提示并跳过。尾部不足一个 global batch 的样本不参与统计。

DSpark 输出 `validation/accept_rate@0..block_size-1` 和 `validation/tau`；指标跨 rank 聚合并写入 TensorBoard。Eagle3 当前未实现接受率验证钩子，命中验证周期时返回空指标。

### 接收率与接收长度

对每个 draft 位置，先将 draft logits 和 target logits 转为概率分布，按总变差距离计算概率接收率：

$$
a_i = \mathrm{clamp}\left(1 - 0.5 * \sum_v(|p_draft_i(v) - p_target_i(v)|), 0, 1\right)
$$

`eval_mask` 会屏蔽 padding、无效预测位置和无效 block。对于一个包含 `K` 个有效 draft 位置的 block，概率接收长度是前缀接收概率的期望：

$$
\mathrm{expected\_draft\_accepted} = a_0 + a_0*a_1 + ... + a_0*a_1*...*a_{K-1}\\
\mathrm{accept\_length} = \mathrm{expected\_draft\_accepted} + 1
$$

末尾的 `+1` 是每轮由 target 模型确认或产生的 token。因此，`validation/tau` 是所有有效 block 的 `tau` 总和除以有效 block 数，并不是通过 `argmax` 得到的实际连续命中长度。

训练指标 `train/tau_probabilistic` 使用完全相同的概率接收率、`eval_mask`、前缀连乘和 `+1` 公式；区别仅在于统计数据和记录范围：前者在训练 batch 上按训练指标周期聚合，后者在 held-out 验证集上按验证周期聚合。两者都以有效 block 为分母。相关实现位于 `deepspec/modeling/dspark/loss.py` 的 `compute_dspark_validation_stats` 和 `_compute_local_probabilistic_stats`。

不要将 `tau_probabilistic` 与 `tau_loss` 混淆：启用 tau loss 时，`tau_loss` 使用 `1 - expected_draft_accepted / valid_length`，不包含 target token 的 `+1`，表示归一化的未接收比例。

周期和最终 checkpoint 都会在保存后执行验证并更新最佳 checkpoint；只有挂起前的应急 checkpoint 不强制验证。

## `logging` 配置

| 配置项 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `logging_steps` | `10` | 每 N 个 optimizer step flush 并打印训练指标，要求正整数。 |
| `checkpointing_steps` | `3000` | 每 N 个 optimizer step 保存 checkpoint，要求正整数；训练结束还会保存一次。 |
| `keep_last_checkpoints` | `3` | 保留最近 N 个 `step_N` checkpoint，要求正整数。 |
| `keep_best_checkpoint` | `True` | 保留验证指标最优的 checkpoint，并通过 `step_best` 链接访问。 |
| `best_checkpoint_metric` | `"validation/tau"` | 用于选择最佳 checkpoint 的验证指标。 |
| `greater_is_better` | `True` | `True` 取指标最大值，`False` 取最小值。 |
| `tensorboard_dir` | `finalize_cfg` 派生 | TensorBoard 输出目录。 |
| `checkpoint_dir` | `finalize_cfg` 派生 | checkpoint 根目录；存在 `step_latest` 时启动会自动恢复。更换实验时必须更换目录或 `exp_name`。 |

`checkpointing_steps` 同时控制 checkpoint 保存和验证频率。每次周期保存都会在验证完成后更新 `step_best` 和 `best_checkpoint.json`。清理后保留最近 N 个 checkpoint，并额外保留一个不在最近 N 个之中的历史最优 checkpoint。checkpoint 保存后会执行 `gc.collect()` 和 accelerator cache 清理，但不改变 checkpoint 格式。

## `data` 配置

| 配置项 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `target_cache_path` | `None` | 训练必需的 target cache 目录。`None` 不能实际启动训练。 |
| `validation_fraction` | 代码回退 `0.1` | 同一 cache 的 held-out 比例，详见验证章节。 |
| `chat_template` | Qwen `"qwen"`；Gemma `"gemma4"`；DeepSeek `"deepseek"` | 数据准备/对话解析模板标识。对于当前 `CacheDataset` 训练路径，样本已预处理，训练 collator 不重新应用模板。 |
| `max_length` | `4096` | 最大序列长度。必须与 cache 生成策略一致；cache 样本自身携带实际长度。 |
| `num_workers` | `4` | 每个 rank 的 DataLoader worker 数。训练 loader 使用 persistent workers；验证 loader 不使用。 |

## 启动脚本变量

`scripts/train/train.sh` 支持以下环境变量或同名 shell 变量：

| 变量 | 默认值 | 映射 |
| --- | --- | --- |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 可见 NPU。 |
| `MASTER_ADDR` | `127.0.0.1` | torchrun rendezvous 地址。 |
| `MASTER_PORT` | `29500` | torchrun rendezvous 端口。 |
| `CONFIG_PATH` / `config_path` | DeepSeek V4 recipe | `--config`；使用 YAML 时以 `config_path` 为准，环境变量只作为缺省值。 |
| `TARGET_MODEL_PATH` / `target_model_path` | DeepSeek V4 Flash | `model.target_model_name_or_path`；使用 YAML 时以 `target_model_path` 为准，环境变量只作为缺省值。 |
| `BASE_TB_DIR` | `~/tensorboard` | TensorBoard 根目录。 |
| `BASE_CKPT_DIR` | `~/checkpoints` | checkpoint 根目录。 |
| `target_cache_dir` | `~/.cache/deepspec/deepseek_v4_target_cache` | `data.target_cache_path`。 |
| `target_layer_ids` | `[40, 41, 42]` | `model.target_layer_ids`；启动器同时兼容逗号分隔字符串。 |
| `global_batch_size` | `512` | `train.global_batch_size`。 |
| `num_train_epochs` | 空 | 非空时覆盖 `train.num_train_epochs`。 |
| `max_train_steps` | 空 | 非空时覆盖 `train.max_train_steps`。优先于 epochs。 |
| `data_max_length` | `4096` | `data.max_length`。 |
| `logging_steps` | `10` | `logging.logging_steps`。 |
| `checkpointing_steps` | `3000` | `logging.checkpointing_steps`。 |
| `exp_name` | 空 | 非空时覆盖顶层 `exp_name`。 |
| `n_routed_experts` | 空 | 非空时覆盖 DeepSeek `model.n_routed_experts`。 |
| `tau_loss_alpha` | 空 | 非空时覆盖 `model.tau_loss_alpha`。 |
| `sharding_strategy` | 空 | 非空时覆盖 `train.sharding_strategy`。 |
| `ENABLE_PROFILE` | 脚本强制 `False` | Trainer profiler 开关；仅字符串 `True` 启用。 |
| `WORK_DIR` | `.` | profiler 输出路径前缀。 |
| `RES_PATH` | 空 | profiler 输出路径后缀。 |

设备进程数 `NPROCS` 由脚本探测 NPU 或 CUDA device count，随后传给 `torchrun --nproc-per-node`。脚本当前不提供 validation 配置的同名快捷变量；可直接运行 `train.py --opts`，或在 recipe 中修改。

## DeepSeek V4 权重加载环境变量

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `DEEPSPEC_DSV4_WEIGHT_DIR` | `model.target_model_name_or_path` | embed/head safetensors shard 所在目录。 |
| `DEEPSPEC_DSV4_INDEX_PATH` | target 路径下 `model.safetensors.index.json` | safetensors 权重索引路径。 |

显式环境变量优先于从 target model 路径推导的值。两者必须指向同一套 checkpoint，否则 embedding/head shape 或 shard 查找会失败。

## 完整 DeepSeek V4 示例

```python
project_name = "deepspec"
exp_name = "dspark_block5_deepseek_v4_flash"
seed = 42

model = dict(
    target_model_name_or_path="deepseek-ai/DeepSeek-V4-Flash",
    block_size=5,
    num_draft_layers=3,
    sliding_window=128,
    n_routed_experts=None,
    target_layer_ids=[40, 41, 42],
    mask_token_id=128799,
    num_anchors=512,
    markov_rank=256,
    markov_head_type="vanilla",
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
    tau_loss_alpha=0.05,
)

train = dict(
    trainer_cls=DeepSeekV4DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    optimizer_cpu_offload=False,
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    validation_fraction=0.1,
    chat_template="deepseek",
    max_length=4096,
    num_workers=4,
)
```
