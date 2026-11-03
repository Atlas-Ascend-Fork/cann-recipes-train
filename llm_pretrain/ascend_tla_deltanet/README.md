# Ascend-TLA · DeltaNet / NLP 样例

本目录是一个基于 **DeltaNet（Delta Rule）线性注意力**、面向**昇腾 NPU** 的实践样例，配套 **NLP 语言建模**训练流程与 Quick Start Notebook。更多说明与完整工程见上游仓库：[https://gitcode.com/SMULL_Group/Ascend-TLA.git](https://gitcode.com/SMULL_Group/Ascend-TLA.git)

---

## 目录

- [概述](#概述)
- [特性](#特性)
- [相关模型与论文](#相关模型与论文)
- [环境与依赖](#环境与依赖)
- [本样例支持的任务](#本样例支持的任务)
- [本样例目录结构](#本样例目录结构)
- [演示配置 democonfig](#演示配置-democonfig)
- [快速开始](#快速开始)
- [评测与验证](#评测与验证)

---

## 概述

**Ascend-TLA** 在昇腾 NPU 上基于 **Triton** 实现高效注意力算子。本样例聚焦 **Delta Rule / DeltaNet** 路径及 **NLP** 侧训练与 Notebook，不包含通用 Linear Attention 算子目录（`linear_attn`）、视觉（CV）模型与相关文档。

---

## 特性

- **DeltaNet（Delta Rule）**：序列长度维度的并行化线性 Transformer 实现要点见下游论文与上游 `tla/ops/delta_rule`。
- **NLP 语言建模**：GPT-2 风格架构，可与 SlimPajama 等数据配合使用。
- **评测**：可与 `lm-eval` 等工具对接（详见上游与 Notebook 说明）。

---

## 相关模型与论文

### NLP（DeltaNet）


| Year | Venue   | Paper                                                                                                      | Code                                                                                                                      |
| ---- | ------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 2024 | NeurIPS | [Parallelizing Linear Transformers with Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) | [flash-linear-attention / delta_net](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py) |


NPU 侧实现与训练入口见本目录 `internal/tla/nlp/`。

---

## 环境与依赖

请在本样例根目录使用 `requirements.txt`（按团队规范锁定版本）。**完整环境搭建、CANN / PyTorch / Triton 等组合以上游仓库说明为准：** [Ascend-TLA](https://gitcode.com/SMULL_Group/Ascend-TLA.git)。

---

## 本样例支持的任务

- **NLP**
  - 语言建模（GPT、LLaMA 等量级可按上游配置裁剪）
  - 数据集示例：SlimPajama（见 Notebook 与 `preparation` 脚本）
  - 评测：`lm-eval` 等（见 `02_downstream_nlp.ipynb`）

本样例**不包含** CV 分类/检测等任务与通用 **Linear Attention（`linear_attn`）** 算子章节。

---

## 本样例目录结构

```text
ascend_tla_deltanet/
├── README.md
├── patches/                    # 框架补丁
└── internal/
    ├── quick_start/            # 概览、DeltaNet 算子、NLP 下游 Notebook
    └── tla/
        ├── utils.py
        ├── modules/
        ├── torch/              # 如 delta_net 相关参考实现
        └── nlp/
            └── tla-nlp-frame/  # 训练脚本、democonfig、DeltaNet attention 等
```

算子实现位于 `internal/tla/nlp/tla-nlp-frame/tla/ops/`（含 `delta_rule`、`common`、`utils` 等），**不包含** `linear_attn`。

---

## 演示配置 democonfig

为控制样例仓体积，`training/democonfig/` 在代码树中**仅保留占位**（`.gitkeep`），不含完整模型与分词器文件。训练与 Notebook 所需的演示配置已打包为 `**demo_config.zip`**，位于本仓库 `**demo-config-download**` 分支的 `[ascend_tla_deltanet/downloads/](https://gitcode.com/SMULL_Group/cann-recipes-train-smull-ascend-TLA/tree/demo-config-download/ascend_tla_deltanet/downloads)`，解压到固定路径后即可使用：

**命令行下载并解压**

```bash
cd ascend_tla_deltanet/internal/tla/nlp/tla-nlp-frame/training

wget -O demo_config.zip \
  'https://gitcode.com/SMULL_Group/cann-recipes-train-smull-ascend-TLA/raw/demo-config-download/ascend_tla_deltanet/downloads/demo_config.zip'

unzip -o demo_config.zip -d democonfig
```

也可在浏览器打开上述 [downloads 目录](https://gitcode.com/SMULL_Group/cann-recipes-train-smull-ascend-TLA/tree/demo-config-download/ascend_tla_deltanet/downloads) 手动下载 `demo_config.zip`，再执行 `unzip -o demo_config.zip -d democonfig`。

解压后应包含 `config.json`、`tokenizer.json`、`vocab.json`、`merges.txt` 等，供 `train.sh` 与 Notebook 中的 `model=`、`tokenizer=` 使用（例如 `.../training/democonfig/config.json`）。

---

## 快速开始

### NLP 语言建模

训练脚本与 DeepSpeed 配置见 `internal/tla/nlp/tla-nlp-frame/training/`。**请先按上一节解压 `democonfig`**，再启动训练：

- 配置示例：`internal/tla/nlp/tla-nlp-frame/training/democonfig/config.json`

建议先阅读 Notebook：

- `internal/quick_start/00_overview.ipynb` 
- `internal/quick_start/01_operator_deltanet.ipynb` — DeltaNet 算子实践
- `internal/quick_start/02_downstream_nlp.ipynb` — NLP 数据、训练与评测流程

更详细的命令行与数据准备说明见 **[Ascend-TLA 文档](https://gitcode.com/SMULL_Group/Ascend-TLA.git)**。

---

## 评测与验证

### 算子

### NLP（DeltaNet）

训练架构与配置见 `internal/tla/nlp/tla-nlp-frame/training/democonfig/`（需先从 `[demo-config-download` 分支 downloads](https://gitcode.com/SMULL_Group/cann-recipes-train-smull-ascend-TLA/tree/demo-config-download/ascend_tla_deltanet/downloads) 下载并解压 `demo_config.zip`，见上文）。


|      | PIQA (acc) | HellaSwag (acc_n) | WinoGrande (acc) | Arc_e (acc) | Arc_c (acc_n) |
| ---- | ---------- | ----------------- | ---------------- | ----------- | ------------- |
| CUDA | 63.8       | 32.2              | 52.9             | 46.1        | 20.4          |
| TLA  | 64.3       | 32.7              | 50.4             | 45.0        | 18.9          |


---

## 致谢与许可

算法与代码版权归上游 **[Ascend-TLA](https://gitcode.com/SMULL_Group/Ascend-TLA.git)** 项目及其许可证约束；使用本样例时请同时遵守上游仓库的 **License** 与引用要求。
