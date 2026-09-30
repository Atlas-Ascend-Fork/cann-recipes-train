# Qwen3-1.7B SFT训练示例

## 硬件要求
卡数：1张A2/1张A3


## 一站式平台快速启动SFT训练示例

### 环境要求
一站式平台模板选择：cann_9.0.0-py3.11-A2-arm-20260715/cann_9.0.0-py3.11-A3-arm-20260715

### 项目及依赖构建
在当前目录下（cann-recipes-train/llm_sft/qwen3_1.7B_torchtitan/）执行:
```bash
# 下载本样例所在代码仓，并安装依赖
bash build_project_platform.sh
# 进入目录
cd torchtitan-npu
# 快速验证环境是否正常
NGPU=1 bash scripts/run_train.sh
```

预计输出：
Running with configs: model and recipe resolved from the current master defaults
visible dies: 2
step: 1  loss: <finite>  grad_norm: <finite>
Training completed


注：需要按照实际情况在以上脚本中修改 ascend-toolkit 路径, 一站式平台需要修改（以CANN 9.0.0为例）：
```
/home/developer/Ascend/cann-9.0.0/set_env.sh
```

## 模型权重准备
本样例使用的Qwen3-1.7B模型权重准备方法如下：
```bash
# 从魔塔社区下载模型的基础文件，存放在当前目录的 ./assets/hf/Qwen3-1.7B 目录下
MODEL_DIR="./assets/hf/Qwen3-1.7B"
python3 -m pip install -U modelscope
modelscope download \
  --model PrimeIntellect/Qwen3-1.7B \
  --local_dir "$MODEL_DIR"

pwd
ls ./assets/hf/Qwen3-1.7B
```

## 数据集准备
本样例使用 willcb/V3-wordle 数据集：
```bash
# 1. 下载数据集（源文件已经是 Parquet 格式）
HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com hf download willcb/V3-wordle data/train-00000-of-00001.parquet --repo-type=dataset --local-dir ./assets/data/wordle_raw

# 2. 复制到 torchtitan 约定的数据路径
mkdir -p ./assets/data/wordle
cp ./assets/data/wordle_raw/data/train-00000-of-00001.parquet ./assets/data/wordle
```

Wordle 评测环境依赖 NLTK 语料库（words 和 averaged_perceptron_tagger）。以下脚本从 NLTK 官方源下载并自动解压到 $HOME/developer/nltk_data（约需 7 分钟）：

```bash
pip install nltk
python3 -c "
import nltk
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
"
```

查看数据集格式（从本地读取）：

```bash
python3 -c "
from datasets import load_dataset
import pprint
ds = load_dataset('./assets/data/wordle', split='train')
pprint.pprint(next(iter(ds)), width=100, depth=3)
"
```

`process_wordle_sample` processor 会将 `prompt` 和 `completion` 拼接为完整的 `[system, user, assistant, user, assistant, ...]` 消息列表。


## 基线 SFT 训练与 Wordle 评测

使用 TorchTitan-NPU 完成 Qwen3-1.7B Wordle 数据集 SFT 训练，并通过 `vf-eval` 对训练后的模型进行交互式 Wordle 评测。

### 1. 启动 SFT 训练

使用 TorchTitan-NPU 对 Qwen3-1.7B 进行 Wordle 数据集 SFT：

```bash
MODULE=torchtitan_npu.models.qwen3 \
CONFIG=sft_qwen3_1_7b_wordle \
NGPU=1 \
bash scripts/run_train.sh \
--hf_assets_path "assets/hf/Qwen3-1.7B" \
--checkpoint.folder checkpoint_wordle_sft \
--checkpoint.last_save_in_hf \
--checkpoint.enable \
--checkpoint.initial_load_in_hf \
dataloader:chat_data_loader_config \
--dataloader.dataset_path "assets/data/wordle"
```

参数说明：

| 参数 | 说明 |
|---|---|
| `MODULE` | 使用 Qwen3 TorchTitan-NPU 模型配置 |
| `CONFIG` | Wordle SFT 配置文件 |
| `NGPU` | 使用 GPU/NPU 数量 |
| `--checkpoint.folder` | checkpoint 保存目录 |
| `--checkpoint.last_save_in_hf` | 训练结束时保存 HuggingFace 格式 checkpoint |
| `--checkpoint.initial_load_in_hf` | 从 HuggingFace 格式模型初始化 |
| `chat_data_loader_config` | 使用对话格式数据加载器 |
| `--dataloader.dataset_path` | Wordle SFT 数据集路径 |

训练完成后，checkpoint 默认保存于：

```text
outputs/checkpoint_wordle_sft/
```

### 2. 准备 HuggingFace 推理 checkpoint

`infer_server.py` 使用 HuggingFace `AutoTokenizer` 加载 tokenizer。

部分 TorchTitan checkpoint 主要保存模型权重和训练状态，不包含完整 tokenizer 文件，因此需要将原始 Qwen3 模型中的 tokenizer 配置补充到 checkpoint 目录。

执行：

```bash
src=assets/hf/Qwen3-1.7B
dst=outputs/checkpoint_wordle_sft/step-20

test -d "$dst"

for name in \
config.json \
tokenizer.json \
tokenizer_config.json \
special_tokens_map.json
do
    if test -f "$src/$name"; then
        cp "$src/$name" "$dst/$name"
    fi
done

for name in \
generation_config.json \
vocab.json \
merges.txt
do
    if test -f "$src/$name"; then
        cp "$src/$name" "$dst/$name"
    fi
done

echo "Loadable checkpoint:"
find "$dst" -maxdepth 1 -type f -printf '%f\n' | sort
```

### 3. 启动推理服务
```bash
source /home/developer/Ascend/cann-9.0.0/set_env.sh
# 关闭已有推理服务（可选）
pkill -f infer_server.py || true

# 安装推理依赖
pip install transformers
pip install torchvision==0.27.0+cpu \
--index-url https://download.pytorch.org/whl/cpu
# 启动 inference server

python3 scripts/infer_server.py \
--model ./outputs/checkpoint_wordle_sft/step-20 \
--port 8000 \
> /tmp/infer_server.log 2>&1 &

sleep 10

curl http://localhost:8000/health
# {"status":"ok"}
```

### 4. 配置 vf-eval 评测环境

`vf-eval` 是基于 Prime-RL 的交互式评测工具，用于测试模型在 Wordle 环境中的推理能力。

评测流程如下：

1. 环境从预留评估词中选择秘密词；
2. 推理服务器接收系统提示和 Wordle 游戏规则；
3. 模型生成预测：

```xml
<think>
...
</think>
<guess>[word]</guess>
```

4. Wordle 环境解析 `<guess>` 内容；
5. 根据预测结果返回 G/Y/X 字母反馈；
6. 模型根据反馈继续预测；
7. 最多进行 6 轮交互；
8. 游戏结束后计算最终 reward。

Reward 由以下部分组成：

| Reward | 含义 |
|---|---|
| `correct_answer` | 是否猜中秘密词 |
| `partial_answer` | 根据 green/yellow 字母数量提供部分奖励 |
| `length_bonus` | 鼓励更少轮次完成任务 |
| `format_reward` | 检查 `<guess>` 输出格式是否正确 |

初始化 vf-eval 环境：

```bash
bash ../setup_prime_rl.sh
```

### 5. 运行 Wordle 评测
```bash
# 激活 vf-eval 环境：
PRIMERL_DIR=./prime-rl
source "$PRIMERL_DIR/.venv-wordle-legacy/bin/activate"

# 启动评测：
vf-eval wordle \
--num-examples 4 \
--rollouts-per-example 2 \
--api-base-url http://127.0.0.1:8000/v1 \
--max-concurrent 1 \
--verbose \
--temperature 0.6 \
--save-results
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--num-examples` | 评测样本数量 |
| `--rollouts-per-example` | 每个样本采样次数 |
| `--api-base-url` | inference server 地址 |
| `--max-concurrent` | 串行请求（避免单线程服务器过载）|
| `--temperature` | 生成采样温度 |
| `--save-results` | 保存评测结果 |

### 结果分析

基线模型与 SFT 后的参考性能对比：

| 指标 | Base Qwen3-1.7B | Wordle SFT | 变化 |
|------|----------------|------------|------|
| 平均 format_reward | 0.60 | 1.0 | +0.4 |
| 平均 correct_answer | 0.00 | 0.00 | 持平 |
| 平均 partial_answer | 0.00 | 0.25 | +0.25 |
| 平均 length_bonus | 0.00 | 0.00 | 持平 |
| 平均 reward | 0.22 | 0.4 | +0.18 |

格式正确性显著提升：format_reward 从 0.6 提升至 1.0，说明 SFT 阶段成功让模型学会了符合 Wordle 环境的 XML 交互格式，这与“SFT学习格式”的目标一致。

部分正确性有所改善：partial_answer 从 0 提高到 0.25，表明模型开始能够生成包含正确字母（G/Y 反馈）的猜测，逐步掌握游戏逻辑，但猜中率仍然为 0（correct_answer 未变）。

整体奖励上升：平均总奖励从 0.22 增至 0.4，增益主要来自格式和部分正确性的改进，但最终获胜率（length_bonus 仍为 0）未见提升，说明模型尚未能完整解出单词。

与官方基线对比： Qwen3-1.7B 基线（0.22）接近官方报告的 ~0.2，SFT 后（0.4）也基本符合预期。预计在 RL 阶段后平均奖励可达 ~1.5 且胜率约 60%，表明当前 SFT 模型是 RL 的基座，需要 RL 提升其策略能力。

