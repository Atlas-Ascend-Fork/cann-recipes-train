# Qwen3-1.7B SFT训练示例

## 硬件要求
硬件环境：Atlas A3。训练脚本默认使用 1 个 NPU 逻辑设备（`NPUS_PER_NODE=1`）。

## 基于Dockerfile构建环境

1. 在宿主机下载源码并进入样例目录。
   ```bash
   git clone https://gitcode.com/cann/cann-recipes-train.git
   cd cann-recipes-train/llm_sft/qwen3
   ```

2. 在宿主机创建镜像并启动容器。
   ```bash
   # 当前目录已包含 Dockerfile 和 build_docker.sh
   docker build -t qwen3_sft -f Dockerfile.cann.mindspeed.qwen3 .

   # 创建并进入容器；脚本默认使用容器名 qwen3_sft、镜像名 qwen3_sft:latest
   bash build_docker.sh
   ```

   `build_docker.sh` 默认按 A3 整机配置映射 `/dev/davinci0`～`/dev/davinci15` 共 16 个 NPU 设备节点。容器映射的设备数量不等于训练使用的设备数量：本样例训练脚本的 `NPUS_PER_NODE=1`，仅启动 1 个训练进程。

   如果运行环境只提供部分设备节点，请按实际可用节点修改 `build_docker.sh` 中的 `--device` 参数。例如，仅提供 `/dev/davinci0` 时，保留 `--device=/dev/davinci0` 及公共设备节点 `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`，移除 `davinci1`～`davinci15` 的映射。多设备训练还需同步调整训练进程数，以及训练和模型转换脚本中的并行配置。

   脚本会将宿主机上的本样例目录挂载到容器的 `/workspace/qwen3`。如需更改容器名或镜像名，请修改 `build_docker.sh` 中的 `container_name`、`image_name`，并与构建命令保持一致。

3. 在容器内准备依赖和训练脚本。
   ```bash
   cd /workspace/qwen3
   # 添加镜像中已经准备好的依赖文件
   cp -r /workspace/MindSpeed-LLM MindSpeed-LLM
   cp -r qwen3_dense MindSpeed-LLM/examples/mcore

   cd MindSpeed-LLM
   ```
## 一站式平台快速启动SFT训练示例

### 环境要求
一站式平台镜像选择：CANN-8.5.0-A3 或者 CANN-8.5.0-A2

### 项目及依赖构建
在当前目录下（cann-recipes-train/llm_sft/qwen3）执行:
```bash
bash build_project_platform.sh

# 进入对应目录
cd MindSpeed-LLM
```

## 数据集准备
本样例使用的alpaca数据集准备方法如下：
```bash
# 在本样例目录下创建dataset数据集目录
mkdir -p ./dataset
cd dataset/
# 从魔塔社区下载alpaca数据集的parquet文件
pip install modelscope

modelscope download --dataset OmniData/alpaca --local_dir ./alpaca
cd ../
```

## 模型权重准备
本样例使用的Qwen3-1.7B模型权重准备方法如下：
```bash
# 从魔塔社区下载模型的基础文件，存放在当前目录的 ./Qwen3-1.7B 目录下
mkdir ./Qwen3-1.7B
modelscope download --model Qwen/Qwen3-1.7B --local_dir ./Qwen3-1.7B
```

## Qwen3-1.7B
1. 数据转换，将原始数据转为训练数据格式：

   ```bash
   bash examples/mcore/qwen3_dense/data_convert_qwen3_instruction.sh
   ```

2. 模型转换，将HuggingFace格式的模型转换为Mcore训练格式：
   ```bash
   bash examples/mcore/qwen3_dense/ckpt_convert_qwen3_hf2mcore.sh
   ```

3. SFT训练启动脚本：
   ```bash
   bash examples/mcore/qwen3_dense/tune_qwen3_4K_full_A3_ptd.sh
   ```

4. 训练后模型转换回huggingface格式
   ```bash
   bash examples/mcore/qwen3_dense/ckpt_convert_qwen3_mcore2hf.sh
   ```

注：需要按照实际情况在以上脚本中修改 ascend-toolkit 路径，默认配置为：
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
一站式平台需要修改为（以CANN 8.5.0为例）：
```
source /home/developer/Ascend/cann-8.5.0/set_env.sh
```

## 附录 相关参数说明
1. 数据准备:

   | 参数 | 说明 |
   |------|------|
   | `--input` | 输入数据文件路径（可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持.parquet/.csv/.json/.jsonl/.txt/.arrow格式， 同一个文件夹下的数据格式需要保持一致） |
   | `--tokenizer-name-or-path` | tokenizer文件路径|
   | `--output-prefix` | 输出数据文件前缀 |
   | `--handler-name` | 微调数据预处理Alpaca风格数据集时，应指定为AlpacaStyleInstructionHandler，根据--map-keys参数提取对应数据的列。|
   | `--enable-thinking` | 是否添加推理tag，例如<think> </think>|
   | `--seq-length` | 序列长度，默认4096 |
   | `--prompt-type` | 指定模型模板，默认为qwen3 |

alpaca风格数据示例：
```
[
   {
      "instruction": "人类指令（必填）",
      "input": "人类输入（选填）",
      "output": "模型回答（必填）",
      "system": "系统提示词（选填）",
      "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
      ]
   }
]
```

2. 模型转换`hf2mcore`相关参数说明：

   | 参数 | 说明 |
   |------|------|
   | `--target-tensor-parallel-size` | 张量并行 |
   | `--target-pipeline-parallel-size` | 流水线并行 |
   | `--load-dir` | 输入模型路径 |
   | `--save-dir` | 输出模型路径 |
   | `--moe-grouped-gemm` | 启用MoE grouped GEMM优化 |
   | `--model-type-hf` | HuggingFace模型类型 |
   | `--expert-tensor-parallel-size` | 专家张量并行大小，需要显式指定为1 |

3. SFT训练：

- 路径配置

   | 参数 | 说明 |
   |------|------|
   | `CKPT_LOAD_DIR` | 加载预训练权重的路径 |
   | `CKPT_SAVE_DIR` | 保存微调模型的路径 |
   | `DATA_PATH` | 训练数据集路径 |
   | `TOKENIZER_PATH` | Tokenizer文件路径 |

- 并行配置，需要和模型转换的参数**保持一致**

   | 参数 | 说明 |
   |------|------|
   | `TP` | 张量并行大小 |
   | `PP` | 流水线并行大小 |
   | `CP` | 上下文并行大小 |
   | `CP_TYPE` | 上下文并行算法 |

- 训练配置

   | 参数 | 说明 |
   |------|------|
   | `SEQ_LENGTH` | 序列长度 |
   | `TRAIN_ITERS` | 训练迭代次数 |
   | `--micro-batch-size` | 单卡batch size |
   | `--global-batch-size` | 全局batch size |
   | `--lr` | 学习率 |
   | `--lr-decay-style` | 学习率衰减方式 |
   | `--min-lr` | 最小学习率 |
   | `--weight-decay` | 权重衰减 |
   | `--lr-warmup-fraction` | 预热比例 |
   | `--clip-grad` | 梯度裁剪 |

- 微调参数

   | 参数 | 说明 |
   |------|------|
   | `--finetune` | 启用微调模式 |
   | `--stage` | 训练阶段，sft表示监督微调 |
   | `--is-instruction-dataset` | 使用指令数据集 |
   | `--prompt-type` | 使用qwen3格式的prompt |

4. 模型转换 `mcore2hf`:

   | 参数 | 说明 |
   |------|------|
   | `--load-dir` | 输入模型路径，训练保存的模型路径 |
   | `--save-dir` | 输出模型路径 |
   | `--model-type-hf` | HuggingFace模型类型 |
