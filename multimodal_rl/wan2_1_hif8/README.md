# Wan2.1-T2V-14B HiF8 训练样例

## 概述

本样例基于 [MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) 的 Wan2.1-T2V-14B 训练案例,提供 **HiF8(8bit 浮点)训练**能力。HiF8 为计算量化方案(权重与主干保持 BF16),包含三条量化路径:

1. **量化 GEMM**:注意力 QKV/输出投影与 FFN 线性层切换为支持前向/反向双延迟缩放(dual delayed scaling)recipe 的量化线性层;
2. **量化 FlashAttention**(`FP8_FA` 环境变量开关,**默认关闭**):Q/K/V 按块统计 amax、延迟缩放量化为 hifloat8 后调用 NPU 量化融合注意力算子。该路径依赖 CANN 提供 hifloat8 量化融合注意力算子,**商发 CANN 9.2.0 暂不包含**,因此默认关闭;关闭后注意力回退到 BF16 融合算子,量化 GEMM 不受影响;
3. **FP8 all2all**(`FP8_ALL2ALL` 环境变量开关,预留):Ulysses 序列并行下先量化再 all_to_all。

训练形态:FSDP2(`--use-torch-fsdp2`)+ Ulysses CP(CP=8),单机 8 卡。

**默认配置为仅 GEMM 量化**(量化 FlashAttention 与 FP8 all2all 均默认关闭),该配置在商发 CANN 9.2.0 上完整支持。

**精度:训练 1000 步,HiF8 与 BF16 的 loss 相对差异小于 5‰。**

**性能:HiF8 相对 BF16 整网提速 1.38 倍**(实测数据对应 **GEMM + 量化 FlashAttention 全开**的配置;当前默认的仅 GEMM 量化配置性能数据待补)。

## 本样例目录

| 路径 | 作用 |
|---|---|
| `patches/mindspeed_mm_hif8.patch` | MindSpeed-MM 仓的 HiF8 改动(9 文件,基线 `8f5259bd`),**含全部代码改动与 t2v_fp8 启动配置** |
| `patches/mindspeed_hif8.patch` | MindSpeed 仓的 HiF8 改动(14 文件,基线 `9bbefa2d`):双 recipe 体系与量化线性层 |
| `configs/` | 启动配置副本(`pretrain_fsdp2.sh`、`pretrain_model_fsdp2.json`、`feature_data.json`、`fsdp2_config.yaml`),**仅供审阅**——patch 应用后已在 MM 树内生成同样文件,无需手工拷贝 |

## 硬件与软件要求

| 项 | 要求 |
|---|---|
| 硬件 | Ascend 950 A5 8卡 |
| CANN | **9.2.0(商发)**——默认配置(仅 GEMM 量化)所需的 hifloat8 能力已完整支持 |
| 驱动/固件 | 与 CANN 9.2.0 配套的版本 |
| (可选)量化 FlashAttention | 需 CANN 额外提供 `npu_quant_fusion_attention` / `npu_quant_fusion_attention_grad` 算子,商发 9.2.0 暂不包含,故 `FP8_FA` 默认关闭 |
| Python | ≥ 3.10(推荐 3.11) |
| torch / torch_npu | 2.7.1 / 2.7.1post4 |

## 权重下载(Wan2.1-T2V-14B,Diffusers 版)

| 渠道 | 链接 |
|---|---|
| HuggingFace | https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers |
| ModelScope(魔搭) | https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B-Diffusers |

```bash
# HuggingFace
pip install -U huggingface_hub
huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers --local-dir ./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers
# 或 ModelScope
pip install -U modelscope
modelscope download --model Wan-AI/Wan2.1-T2V-14B-Diffusers --local_dir ./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers
```

## 环境安装(具体操作)

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # CANN 9.2.0(商发)
conda create -n hif8_train python=3.11 -y && conda activate hif8_train
pip install torch==2.7.1 torch_npu==2.7.1.post4

# 创建工程目录
mkdir train_code && cd train_code

# 下载本样例(取 patches)
git clone https://gitcode.com/cann/cann-recipes-train.git

# 1) MindSpeed-MM:基线 + HiF8 补丁(补丁含全部代码改动与启动配置)
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
git checkout 8f5259bdf78bdc290020505978cccd103e7923d9
git apply ../cann-recipes-train/multimodal_rl/wan2_1_hif8/patches/mindspeed_mm_hif8.patch
cd ..

# 2) Megatron-LM:源码拷贝方式使用(不安装、不随仓分发)
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout a845aa7e12b3a117e24c2352b9e3e60bad2e3a17   # core_r0.12.0 线,已验证基线
cp -r megatron ../MindSpeed-MM/
cd ..

# 3) MindSpeed:基线 + HiF8 补丁,可编辑安装
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 9bbefa2d8a14de9bf9a5c73ab4bbc29c14862377
git apply ../cann-recipes-train/multimodal_rl/wan2_1_hif8/patches/mindspeed_hif8.patch
pip install -r requirements.txt && pip install -e .
cd ..

# 4) 安装 MindSpeed-MM
cd MindSpeed-MM
pip install -e .
```

> 完成后,**后续所有命令均在 `MindSpeed-MM/` 目录下执行**;启动脚本内全部为树内相对路径,无需其他改动即可运行。

## 调用关系(本样例如何跑起来)

```
examples/wan2.1/14b/t2v_fp8/pretrain_fsdp2.sh          ← 启动脚本(torchrun 8卡; FP8_FA=false; fp8 参数组)
  └─ torchrun … pretrain_sora.py $GPT_ARGS $MM_ARGS     ← 训练入口(MM 树根目录)
       ├─ GPT_ARGS: --use-torch-fsdp2 / --context-parallel-size 8 /
       │            --transformer-impl transformer_engine --fp8-format hif8
       │            --fp8-recipe delayed --dual-delayed-scaling …    ← megatron/mindspeed 参数体系
       ├─ MM_ARGS: --mm-model pretrain_model_fsdp2.json(模型结构+use_te+bridge_patch)
       │           --mm-data  feature_data.json(特征数据集路径)
       └─ mindspeed_mm/training.py::pretrain()          ← 训练主循环(megatron 式)
            ├─ initialize_megatron → merge_mm_args(json→args.mm) → PatchesManager(bridge_patch 等)
            ├─ SoRAModel(args.mm.model)                 ← 装配壳(特征模式下不建 VAE/文本编码器)
            │    └─ WanDiT(mindspeed_mm/models/predictor/dits/wan_dit.py)
            │         ├─ fp8 上下文: create_dual_delayed_scaling + get_fp8_context 包住逐层前向
            │         ├─ WanDiTBlock.ffn → fp8 时 TEFFN(DualRecipeColumnParallelLinear,来自 MindSpeed 补丁)
            │         │                     非 fp8 时 MYFFN(普通 ColumnParallelLinear,权重命名兼容)
            │         └─ attention(mindspeed_mm/models/common/attention.py)
            │              ├─ QKV/输出投影 → DualRecipeColumn/RowParallelLinear(MindSpeed 补丁)
            │              └─ FP8_FA=false(默认) → torch_npu.npu_fusion_attention(BF16)
            │                 FP8_FA=true(可选)  → attention_fp8.py::CustomQuantFlashAttentionFunc
            │                                      (torch_npu.npu_quant_fusion_attention, hifloat8)
            ├─ 权重: bridge_patch=true → bridge/ 在线加载 HF 权重(默认);或 mm-convert 离线转换
            └─ 数据: FeatureDataset ← ./sora_features/(由特征提取步骤产出)
```

## 使用流程

### 1. 权重准备(二选一)

- **在线加载(默认)**:权重下载到 `./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers` 即可,`pretrain_model_fsdp2.json` 已配 `"bridge_patch": true`,训练启动时直接读 HF 权重;
- **离线转换**:`mm-convert WanConverter hf_to_mm --cfg.mm_dir ./weights/Wan2.1-T2V-14B-Converted --cfg.hf_config.hf_dir ./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers`,并将 `pretrain_fsdp2.sh` 的 `LOAD_PATH` 指向转换结果。

### 2. 特征提取(离线预处理)

按 `examples/wan2.1/feature_extract/` 配置视频/文本路径后执行 `bash examples/wan2.1/feature_extract/feature_extraction.sh`,产出 `./sora_features/`。

### 3. HiF8 预训练

```bash
bash examples/wan2.1/14b/t2v_fp8/pretrain_fsdp2.sh
```

关键开关:

| 开关 | 位置 | 作用 |
|---|---|---|
| `--transformer-impl transformer_engine` | GPT_ARGS | 启用 TE 实现路径(fp8 前提) |
| `--fp8-format hif8` / `--fp8-recipe delayed` | GPT_ARGS | HiF8 格式 + 延迟缩放 |
| `--dual-delayed-scaling` | GPT_ARGS | 前向/反向不同延迟缩放配置(前向 most_recent/interval=10,反向 mean/history=128) |
| `--use-deter-comp --seed 42` | GPT_ARGS | 确定性计算,便于与 BF16 对比 |
| `export FP8_FA='false'` | 环境变量 | **默认关闭**。置 `'true'` 启用 HiF8 量化 FlashAttention(FA+FAG),需 CANN 提供量化融合注意力算子 |
| `export FP8_ALL2ALL='true'` | 环境变量 | (预留,默认关闭)FP8 all2all 通信 |
| `"use_te": true` | pretrain_model_fsdp2.json | 模型侧启用 TE 线性层路径 |

**BF16 对照**:去掉 GPT_ARGS 中 fp8 参数组(脚本内附注释掉的参数副本,一键切回),即为 BF16 基线训练,用于精度/性能对比。

### 4. 训练产物转回 HF 格式(可选)

训练保存的 checkpoint 为 MM 侧格式(`SAVE_PATH` 下,`--ckpt-format torch_dcp`),仅含 predictor(WanDiT)权重。如需用 Diffusers 推理或对外分发,转换回 HF 格式:

```bash
mm-convert WanConverter mm_to_hf \
  --cfg.save_hf_dir ./weights/Wan2.1-T2V-14B-HiF8-HF \
  --cfg.mm_dir ./save_dir \
  --cfg.hf_config.hf_dir ./weights/Wan-AI/Wan2.1-T2V-14B-Diffusers
```

> `save_hf_dir`=转换输出目录;`mm_dir`=训练 SAVE_PATH;`hf_config.hf_dir`=原始 HF 权重目录(提供 config 与目录结构模板)。转换只覆盖 transformer(DiT)部分,VAE/text_encoder 沿用原始权重。

## 修改点清单

- **MindSpeed-MM 侧(`patches/mindspeed_mm_hif8.patch`,9 文件,+1342/−92)**:`attention.py`(量化线性层分派+FA 延迟缩放,前/反向 recipe 各自持有独立 config)、`ffn.py`(MYFFN)、新增 `attention_fp8.py`(量化 FA autograd)、新增 `fp8_all2all.py`(FP8 all2all)、`wan_dit.py`(按 `config.fp8` 切换 TEFFN/MYFFN、fp8 上下文、te_checkpoint 重计算)、`fsdp2_config.yaml`(关闭重计算)、`t2v_fp8/` 启动配置三件套。
- **MindSpeed 侧(`patches/mindspeed_hif8.patch`,14 文件,+1422/−17)**:`te/pytorch/fp8/` 双 recipe 体系(forward/backward/dual delayed scaling recipe、DelayedScalingRecipeFA、ScaleDataFA、FP8DualMetadata)、amax 计算新增 `mean`(历史窗口未填满时仅对已填充部分求均值)、`ScaleData`/`ScaleDataFA` amax 历史窗口沿时间维(dim=0)滑动、`te/pytorch/module/` 新增 `DualRecipeColumn/RowParallelLinear` 与 `TEFFN`、`--dual-delayed-scaling` 参数注册与 hif8 + delayed 校验放开、`get_fp8_context` 支持 dual recipe。

## 声明

- Megatron-LM 版权归 NVIDIA,请从官方仓库获取,本样例不分发其源码;MindSpeed / MindSpeed-MM 请从官方仓库获取并应用本样例补丁。
