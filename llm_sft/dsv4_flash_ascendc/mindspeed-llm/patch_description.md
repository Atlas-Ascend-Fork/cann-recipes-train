`mindspeed-llm.patch` 是相对于基线
`0ebf60e35590f981bf86de42c453e283366e68c2` 生成的补丁，并可在该 commit 的干净工作树上直接应用。补丁共涉及 24 个路径：8 个新增文件、16 个修改项，其中 1 项为 MindSpeed 子模块指针。

### 1. 固定 MindSpeed 依赖版本

涉及 `3rdparty/MindSpeed`：

- 从 `7969e1c6b61c44fdb09d20607561824675bb216f` 调整为 `65634689bed75ee9b6df666d526f7f16051a3db5`。
- 补丁只记录子模块 commit，不复制 MindSpeed 源码。应用补丁后执行 `git submodule update --init` 获取该版本。
- Megatron-LM 继续使用基线固定的 `a845aa7e12b3a117e24c2352b9e3e60bad2e3a17`，没有修改其子模块指针。

应用方式使用 `git apply --index`，确保索引中的子模块指针也被更新。源码版本随仓库固定，不依赖子模块远端 `master` 的后续变化。

### 2. SwiGLU 截断与专家 LoRA 梯度同步

涉及：

- `mindspeed_llm/core/fusions/fused_bias_swiglu.py`
- `mindspeed_llm/tasks/posttrain/lora/moe/experts.py`

SwiGLU 包装函数在未显式传入 limit 时，每次从训练配置读取 `swiglu_limit`，修复首次调用未使用配置值的问题。训练示例显式设置 `--swiglu-limit 10.0`，使首次前向与后续调用、重计算使用同样的截断设置。

专家 LoRA 的 `weight1_lora_a`、`weight1_lora_b`、`weight2_lora_a`、`weight2_lora_b` 都保存本 EP rank 的局部专家参数。补齐原先两个参数缺失的 `allreduce` 属性，使四个参数一致使用 `not self.expert_parallel`。启用 EP 时，避免将不同 rank 上的不同专家误当作普通复制参数参与梯度同步。

### 3. CP 下的 next-token 标签配对

涉及：

- `mindspeed_llm/core/context_parallel/get_batch_utils.py`
- `mindspeed_llm/tasks/posttrain/sft/sft_trainer.py`
- `mindspeed_llm/core/models/deepseek4/deepseek4_model.py`
- `mindspeed_llm/core/transformer/multi_token_prediction.py`

当 `stage=sft`、`prompt_type=deepseek4` 且 CP 大于 1 时，先在完整序列上将 labels 和 loss mask 移位，再切分给各 CP rank。末尾分别补 `IGNORE_INDEX` 和 0；模型输出及 loss 路径跳过原来的再次移位。

这修复了 CP rank 持有不连续 token 块时，局部移位把前一块末尾与后一块开头错误配对的问题。修改作用于预测标签及 loss mask，不改变注意力可见范围。其他 SFT 模板和 CP=1 的路径保持原有移位方式。

### 4. Router bias 保存、导出与 FP32 状态加载

涉及：

- `mindspeed_llm/tasks/posttrain/lora/utils.py`
- `mindspeed_llm/tasks/checkpoint/convert_ckpt_deepseek4.py`
- `mindspeed_llm/tasks/models/transformer/deepseek4/mhc.py`

`--lora-ckpt-filter` 除 LoRA 参数外，也保留 `.expert_bias` buffer，普通 PP 和 VPP 保存分支均处理。DeepSeek-V4 的独立 LoRA 合并路径使用已有的 `_load_lora_override()`，让 adapter checkpoint 中的 router bias 等状态覆盖底座的同名状态。

FP16/BF16 模块初始化包装中，在加载 checkpoint 前将 `expert_bias`、`local_tokens_per_expert`、`attn_sink`、`ape` 的存储恢复为 FP32，避免先将原始 FP32 值舍入为 BF16，再在前向中转回 FP32。

两份训练示例默认冻结 router bias 更新，因此默认配置下不会产生动态更新后的 bias 差异。保存和导出修复仍保留；若其他配置启用 bias 更新，它负责让训练状态随 LoRA 保存和导出。

### 5. LoRA 断点续训

涉及：

- `mindspeed_llm/training/checkpointing.py`
- `mindspeed_llm/tasks/megatron_basic/requirements_basic.py`

加载 LoRA checkpoint 时，冻结的底座权重仍来自 `--load`，LoRA 参数以及 optimizer、RNG、iteration、样本进度等训练状态来自 `--lora-load`。只合并模型权重字典，避免把底座 checkpoint 的训练状态混入续训状态。

同时对齐包装函数与 Megatron 的加载接口；初始化 LoRA 时可以缺少 adapter 权重，续训时不能把缺少 adapter 的情况作为正常初始化放过。使用节点本地存储保存 legacy checkpoint 时，各节点 local rank 0 写入 tracker，并通过临时文件替换更新。

启动脚本默认保存 optimizer/RNG。初次训练使用 `--finetune --no-load-optim --no-load-rng`；指定 `RESUME_DIR` 后改用 `--lora-load` 和 checkpoint 中的 optimizer scheduler。该修改不能恢复历史 checkpoint 中未保存的状态，也不承诺跨环境重启后的逐位一致性。

### 6. 动态 padding 的 mHC 流水线通信

涉及 `mindspeed_llm/features_manager/transformer/mhc_feature.py`。

在启用 mHC、PP 大于 1，且使用 `no_pad_to_seq_lengths` 或 `variable_seq_lengths` 时，将流水线 P2P 通信及形状交换切换到固定 MindSpeed 版本提供的实现。原因是 mHC 传递四维激活，原生 Megatron 的对应形状交换按三维处理。

固定长度路径不启用这一新增条件分支。动态 padding 示例使用 `--no-pad-to-seq-lengths`，并按 `2 * CP * 128` 对齐；CP=16 时为 4096。此处没有新增跨样本 packing 启动脚本。

### 7. 0731 权重转换与反量化

涉及：

- `convert_ckpt_v2.py`
- `mindspeed_llm/tasks/checkpoint/convert_ckpt_deepseek4.py`
- `mindspeed_llm/tasks/checkpoint/convert_mg2hf.py`
- `tests/tools/ckpt_dequant/deepseekv4_ckpt_dequant.py`
- `examples/mcore/deepseek4_flash/ckpt_convert_deepseek4_flash_0731_hf2mcore.sh`
- `examples/mcore/deepseek4_flash/ckpt_convert_deepseek4_flash_lora_mcore2hf.sh`
- `examples/mcore/deepseek4_flash/ckpt_dequant_deepseek4_flash_0731_to_bf16.sh`

HF→MCore 的 `--deepseek4-backbone-only` 路径只转换 43 层模型主体，要求 MTP=0；识别附带的 dSPARK 元数据，将训练 checkpoint 中的 speculative-layer 信息清零，并保留主体对应的 `compress_ratios`。示例使用 `--noop-layers 43`，输出 44 个 MCore 层位置。

MCore→HF 从已加载的模型配置初始化 `num_layers`，修复未传 `--num-layers` 时可能对 `None` 做整数除法的问题。LoRA 合并后整理 `compress_ratios`、MTP/dSPARK 和量化配置，使导出配置与 BF16 主体权重一致，并按实际 safetensors payload 修正索引的 `total_size`。

反量化继续使用仓内已有的 FP8/packed FP4 算法，补充 MX scale 形状检查。`--strict` 检查 scale 配对、dtype、有限值及输入输出索引；缺失 scale 时失败，不将未反量化的 tensor 留在输出中。按索引选择输入分片；BF16 输出清理量化声明，保留原有 FP32 和整数 tensor。`torchao` 相关导入只在生成 MX 量化权重时执行，纯 BF16 反量化不需要它。

三个 SH 直接调用现有转换入口。新增的 `--validate-flash-0731` 默认关闭，脚本显式传入后执行 0731 输入输出检查；在 LoRA MCore→HF 路径还负责前述导出配置与索引整理。没有新增独立的 Python 转换包装文件。

### 8. No-think 数据准备与核对

新增：

- `examples/mcore/deepseek4_flash/prepare_data.sh`
- `mindspeed_llm/tasks/preprocess/prepare_deepseek4_data.py`
- `mindspeed_llm/tasks/preprocess/verify_deepseek4_data.py`

通过参数指定一个或多个 JSONL、tokenizer、输出目录、序列长度和 worker 数。沿用仓内 `deepseek4` 模板及 `--enable-thinking false`，没有修改模板定义或添加 reasoning-effort 前缀。

默认整条跳过加入模板后总 token 数大于等于 `seq_length` 的样本，保留样本不截断；也支持遇到超长样本时报错。核对 indexed dataset 的 token、label、mask 及样本数量，生成 `dataset_manifest.json`。多文件合并、manifest 维护和结果摘要放在独立 Python 工具中。

默认预处理长度为 32768，使用本示例的 64K 训练脚本时必须传入 `--seq-length 65536`。两份训练脚本读取 manifest 中的保留样本数来计算迭代数。

### 9. 两份 64K no-think LoRA 训练示例

新增：

- `examples/mcore/deepseek4_flash/lora_finetune_deepseek4_flash_0731_nothink_64k_A3_ptd.sh`
- `examples/mcore/deepseek4_flash/lora_finetune_deepseek4_flash_0731_nothink_dynamic_64k_A3_ptd.sh`

两份脚本均使用两节点、每节点 8 卡（共 16 卡），每节点启动 16 个训练进程，TP1/PP2/CP16/EP16、MBS=1、LR=`1e-5`、5 epochs。默认每 `ceil(样本数/GBS)` 个 iteration 保存一次，训练总 iteration 为其 5 倍。

两份训练脚本均默认 GBS=8，并与 LoRA 合并脚本统一使用 r32/alpha64。两份训练脚本均可通过环境变量覆盖 GBS，动态脚本还可覆盖 rank、alpha，完整用法见 [README](./README.md)。

两份脚本共同保留：

- `--swiglu-limit 10.0`。
- `--moe-router-load-balancing-type none --moe-aux-loss-coeff 0.0`，关闭受 padding 统计影响的均衡辅助目标。
- `--moe-router-bias-update-rate 0.0`，冻结路由 bias 动态更新。这里是配置选择，没有实现逐 token 屏蔽 padding 的 router/seq-aux 新算法。
- `--indexer-loss-coeff 1.0` 及原有融合算子路径，没有额外将 indexer 的 `wq_b` 加入 LoRA target。
- 七个 LoRA target：`linear_q`、`linear_kv`、`linear_fc1`、`linear_fc2`、`linear_q_up_proj`、`linear_o_down_proj`、`linear_o_up_proj`。
- 默认保存 optimizer/RNG，并通过 `RESUME_DIR` 接续训练。
