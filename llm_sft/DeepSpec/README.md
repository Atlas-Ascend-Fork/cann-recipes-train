# DeepSeek-V4 DSpark 草稿模型 NPU 训练样例

## 1. 概述

本样例基于 [DeepSpec](https://github.com/deepseek-ai/DeepSpec) 训练框架，在昇腾 NPU 上训练 DeepSeek-V4-Flash 的 DSpark 草稿模型（draft model）。草稿模型用于投机采样（speculative decoding）推理加速：推理时先由小规模草稿模型一次生成多个候选 token，再由 target 模型并行验证，在不改变输出结果的前提下提升推理吞吐。训练得到的草稿模型经 checkpoint 转换后，交付 `cann-recipes-infer` 仓进行 benchmark 验证或部署。

## 2. 使用的产品型号

| 项目 | 规格 |
| --- | --- |
| 产品型号 | Atlas A3 系列 |
| 卡数 | 8 卡 |
| 操作系统 | Linux ARM |
| CANN 版本 | 9.0.0 |
| Python | 3.11 |


## 3. 文件说明

| 文件/目录 | 说明 |
| --- | --- |
| `README.md` | 中文训练说明 |
| `patches/deepspec_npu_afdfa7c.patch` | 原版公共文件的 NPU 适配差异 |
| `config/dspark/dspark_deepseek_v4_flash.py` | DeepSeek-V4-Flash DSpark 训练配置 |
| `deepspec/modeling/dspark/deepseek_v4/` | DeepSeek-V4 DSpark 草稿模型实现 |
| `deepspec/utils/` | NPU 设备抽象、checkpoint 保留策略、profiling、NPU 显存诊断等新增工具 |
| `docs/configuration.md` | 训练配置项详细说明 |
| `scripts/data/` | 数据准备工具与流程说明 |
| `scripts/train/` | `offline.yaml` 启动配置与 `train_offline.sh` 启动脚本 |
| `scripts/tools/` | checkpoint 转换工具 |


## 4. 源码准备

```bash
# 1. 克隆 cann-recipes-train 仓库
git clone https://gitcode.com/cann/cann-recipes-train.git

# 2. 克隆 DeepSpec 原仓，并回退到指定版本
git clone https://github.com/deepseek-ai/DeepSpec.git
cd DeepSpec
git checkout afdfa7c
cd ..

# 3. 覆盖增量文件
cp -r cann-recipes-train/llm_sft/DeepSpec/. DeepSpec/

# 4. 应用 patch
cd DeepSpec
patch -p1 --fuzz=0 < patches/deepspec_npu_afdfa7c.patch
```

## 5. 环境准备

1. Atlas A3 Docker 部署

A3 使用预置 ARM 镜像，镜像中已包含本实践所需的 PyTorch、torch_npu、torchair 和 CANN 运行环境。从 [A3 ARM 镜像地址](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/DeepSeek/cann9.2.0.pt2.9.0_dsv4_aarch_a3_image_custom_20260827.tar) 下载 docker 镜像，上传到 A3 服务器的每个节点，并在每个节点执行：

```text
docker load -i cann9.2.0.pt2.9.0_dsv4_aarch_a3_image_custom_20260827.tar
```

2. 拉起 docker 容器

在各个节点上通过如下脚本拉起容器，默认容器名为 `deepspec_container`。请将权重路径和源码路径挂载到容器中。

```text
docker run -u root -itd --name deepspec_container --ulimit nproc=65535:65535 --ipc=host \
    --device=/dev/davinci0     --device=/dev/davinci1 \
    --device=/dev/davinci2     --device=/dev/davinci3 \
    --device=/dev/davinci4     --device=/dev/davinci5 \
    --device=/dev/davinci6     --device=/dev/davinci7 \
    --device=/dev/davinci8     --device=/dev/davinci9 \
    --device=/dev/davinci10    --device=/dev/davinci11 \
    --device=/dev/davinci12    --device=/dev/davinci13 \
    --device=/dev/davinci14    --device=/dev/davinci15 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /home/:/home \
    -v /data:/data \
    -v /etc/localtime:/etc/localtime \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip:/root/.pip \
    -v /etc/hosts:/etc/hosts -v /usr/bin/hostname:/usr/bin/hostname \
    --net=host --shm-size=128g --privileged \
    cann9.2.0.pt2.9.0_dsv4_aarch_a3_image_custom_20260827:latest /bin/bash
```

3. 安装其余 Python 依赖。

   ```bash
   pip install -r requirements.txt
   ```


## 6. 准备 target model 权重与 target cache

在开始 DSpark 草稿模型训练前，需要准备以下内容：

- target model 权重；
- 与该 target model 对应的 target cache。

以下示例以 [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) 为 target model，硬件环境为 Atlas A3。默认生成以下文件：

```text
train_datasets/perfectblend_train.jsonl
train_datasets/ds_v4/perfectblend_train_regen.jsonl
~/.cache/deepspec/target_cache
```

1. 下载并划分数据。

   源数据集为 `mlabonne/open-perfectblend`，样本数约1.4M，用户可自行修改`--dataset-name`，选择其他小数据集，例如`mlabonne/lmsys-arena-human-preference-55k-sharegpt`，样本数约57k。训练集写入 JSONL 文件，留出的用户轮次写入 `eval_datasets/`：

   ```bash
   python scripts/data/download_and_split.py \
     --dataset-name mlabonne/open-perfectblend \
     --test-size 0.05 \
     --train-output-path train_datasets/perfectblend_train.jsonl \
     --test-output-dir eval_datasets \
     --skip-existing
   ```

   该命令生成：

   ```text
   train_datasets/perfectblend_train.jsonl
   eval_datasets/perfectblend.jsonl
   ```

2. 基于 vLLM-Ascend 重生成回答。

   参考 [vLLM-Ascend v0.26.0](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/index.html) 和 [DeepSeek-V4-Flash 模型说明](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/tutorials/models/DeepSeek-V4-Flash.html)配置 vLLM-Ascend 环境，准备 DeepSeek-V4-Flash 权重并启动推理服务。服务启动后执行：

   ```bash
   python scripts/data/generate_train_data_vllm.py \
     --input-file-path train_datasets/perfectblend_train.jsonl \
     --output-file-path train_datasets/ds_v4/perfectblend_train_regen.jsonl
   ```

3. 提取 target cache。

   首先在 DeepSpec 仓库根目录拉取指定版本的 `cann-recipes-infer`，并复制特征提取模块：

   ```bash
   git clone https://gitcode.com/cann/cann-recipes-infer.git ../cann-recipes-infer
   git -C ../cann-recipes-infer checkout 6ee37ec6e5e709213de1d76c756f81c097970f9d
   cp -r scripts/data/dspark_data_preparation ../cann-recipes-infer/module/
   ```

   参考 `cann-recipes-infer/models/deepseek_v4/README.md` 配置环境并下载模型权重。将原始模型转换为适用于 Atlas A3 Pod 系列的 Hybrid INT8-INT4 权重，然后在 `module/dspark_data_preparation/config/extract_deepseek_v4.yaml` 中配置实际权重路径。配置项的详细说明见 [`scripts/data/dspark_data_preparation/config/README.md`](scripts/data/dspark_data_preparation/config/README.md)。

   进入 `cann-recipes-infer` 仓库并启动离线特征提取：

   ```bash
   cd ../cann-recipes-infer
   bash module/dspark_data_preparation/extract_features.sh \
     module/dspark_data_preparation/config/extract_deepseek_v4.yaml \
     ../DeepSpec/train_datasets/ds_v4/perfectblend_train_regen.jsonl \
     ~/.cache/deepspec/target_cache \
     "40 41 42" \
     --mean-hc-mult
   cd ../DeepSpec
   ```

   `"40 41 42"` 必须与训练配置中的 `target_layer_ids` 保持一致。特征提取完成后，将 `~/.cache/deepspec/target_cache` 配置为训练参数 `target_cache_dir`。

## 7. 配置训练参数

编辑：

```text
scripts/train/offline.yaml
```

详细参数说明可见[配置说明](docs/configuration.md)。

部分参数配置如下所示，`target_model_path` 和 `target_cache_dir` 需替换为上一步实际生成的量化权重与 target cache 路径：

```yaml
config_path: config/dspark/dspark_deepseek_v4_flash.py
target_model_path: /path/to/deepseek_v4_int8_w8a8
target_cache_dir: /path/to/deepseek_v4_target_cache
data_source: static
global_batch_size: 512
num_train_epochs: 10
validation_fraction: 0.1
```

关键字段说明：

| 配置项 | 说明 |
| --- | --- |
| `target_model_path` | target model 权重目录，使用第 6 步数据准备流程中转换得到的量化权重（Hybrid INT8-INT4）路径 |
| `target_cache_dir` | 第 6 步数据准备流程生成的 target cache 输出目录 |
| `ascend_rt_visible_devices` | 参与训练的 NPU 编号列表。多卡训练需设置为全部可见卡，例如 8 卡机器写 `0,1,2,3,4,5,6,7`；只配置一张卡时训练退化为单卡 |
| `base_ckpt_dir` | checkpoint 输出根目录 |
| `base_tb_dir` | TensorBoard 日志输出根目录 |
| `data_max_length` | 单条样本的最大 token 长度，训练时按该长度截断 |
| `target_layer_ids` | 草稿模型输入所使用的 target 模型层号，训练时从这些层提取隐藏状态作为草稿模型的输入特征，必须与提取 target cache 时使用的层号完全一致 |

训练参数：

| 配置项 | 作用 |
| --- | --- |
| `global_batch_size` | 一个 optimizer step 使用的全局样本数 |
| `num_train_epochs` | 训练 epoch 数 |
| `validation_fraction` | 验证集比例 |
| `logging_steps` | 日志打印间隔 |
| `num_workers` | 静态 cache DataLoader worker 数 |
| `sharding_strategy` | FSDP 分片策略 |

默认 `local_batch_size=1`、`global_batch_size=512`。8 卡时一个 optimizer step 需要 64 个 micro-step。`global_batch_size` 必须能被 `卡数 × local_batch_size` 整除。

实验名、总训练步数和 checkpoint 间隔由启动脚本根据时间戳和 cache 样本数计算。

## 8. 启动训练

在仓库根目录执行：

```bash
bash scripts/train/train_offline.sh
```

该脚本会读取 `scripts/train/offline.yaml`，根据可见 NPU 数量启动 `torchrun`，并把 target model 和 target cache 路径传入训练配置。

## 9. 输出位置

checkpoint 与 TensorBoard 日志分别输出到以下目录，根目录 `base_ckpt_dir`、`base_tb_dir` 在 `scripts/train/offline.yaml` 中配置（默认 `/workspace/checkpoints`、`/workspace/tensorboard`）：

```text
${base_ckpt_dir}/deepspec/<exp_name>/
${base_tb_dir}/deepspec/<exp_name>/
```

`exp_name` 由启动脚本（`train_offline.sh` 内部调用的 `train.sh`）按 `deepspec_<data_source>_<时间戳>` 格式自动生成，也可在 `offline.yaml` 中通过 `exp_name` 字段显式指定。每个 checkpoint 保存为 `step_<step数>` 目录，同时维护 `step_latest` 软链指向最新一份。

训练日志会打印 epoch、step、loss 和剩余时间；保存 checkpoint 时会在验证集上记录 DSpark 接受率指标。

## 10. 转换训练 checkpoint

训练得到的 DeepSpec checkpoint 不能直接作为 `cann-recipes-infer` 的 draft model，需要先转换为 CANN 推理使用的 DSpark 草稿模型格式。转换器仅支持 BF16 checkpoint，不执行量化或 dtype 转换。

1. 使用方法。

   在 DeepSpec 仓库根目录执行：

   ```bash
   python3 scripts/tools/convert_deepspec_dspark_checkpoint.py \
     --input-path /data/checkpoints/deepspec/my_experiment/step_latest \
     --output-path /data/checkpoints/deepspec/my_experiment/step_latest_cann \
     --target-root /data/models/deepseek_v4_int8_w8a8_dspark
   ```

   参数说明：

   | 参数 | 必填 | 说明 |
   | --- | --- | --- |
   | `--input-path` | 是 | 训练 checkpoint 目录或单个 `.safetensors` 文件。默认从 checkpoint 目录读取 `config.json` |
   | `--output-path` | 是 | 转换后的输出目录。该目录必须不存在，脚本不会覆盖已有目录 |
   | `--source-config` | 否 | 训练配置文件或其所在目录。未指定时使用 checkpoint 目录下的 `config.json` |
   | `--target-root` | 否 | target model 根目录。脚本从该目录读取 `config.json`，校验共享维度、补充推理配置字段，并复制 tokenizer 文件 |
   | `--num-stages` | 否 | 覆盖 DSpark stage 数量，通常由训练配置推断 |
   | `--target-num-layers` | 否 | 覆盖 target model 层数，通常由训练配置或 target model 配置推断 |
   | `--max-shard-size-gib` | 否 | 输出 safetensors 的近似分片大小，默认为 2 GiB |

   如果训练配置已经包含生成 CANN 配置所需的全部字段，可以省略 `--target-root`；否则建议传入 target model 根目录。该目录必须包含 `config.json`。

   转换完成后，将输出目录配置到 benchmark YAML 的 `draft_model_path`：

   ```yaml
   speculative_config:
     method: "dspark"
     draft_model_path: "/data/checkpoints/deepspec/my_experiment/step_latest_cann"
   ```

2. 权重命名和 MoE 转换。

   脚本会完成参数名转换、MoE 权重拆分、注意力输出投影布局转换以及推理配置同步。输入 checkpoint 必须为 BF16；脚本不会执行量化或 dtype 转换。

   训练 checkpoint 的共享 `embed_tokens.weight`、`lm_head.weight` 不会导出，CANN 草稿模型会复用 target model 的 embedding 和 LM head。每个 DSpark stage 的训练侧 `layers.N.mlp.*` 会转换为 CANN canonical 的 `mtp.N.ffn.*`：

   | 训练 checkpoint 键 | CANN 输出键 | 转换方式 |
   | --- | --- | --- |
   | `layers.N.mlp.gate.weight` | `mtp.N.ffn.gate.weight` | 仅重命名 |
   | `layers.N.mlp.gate.e_score_correction_bias` | `mtp.N.ffn.gate.bias` | 仅重命名 |
   | `layers.N.mlp.shared_experts.gate_up_proj.weight` | `mtp.N.ffn.shared_experts.w1.weight`、`w3.weight` | 沿第 0 维均分 |
   | `layers.N.mlp.shared_experts.down_proj.weight` | `mtp.N.ffn.shared_experts.w2.weight` | 仅重命名 |
   | `layers.N.mlp.w13_weight` | `mtp.N.ffn.experts.<id>.w1.weight`、`w3.weight` | 按 expert 拆分，随后转置 |
   | `layers.N.mlp.w2_weight` | `mtp.N.ffn.experts.<id>.w2.weight` | 按 expert 拆分，随后转置 |

   `w13_weight` 的布局为 `[expert, hidden, gate + up]`。其前半段 gate 输出为 `w1.weight`，后半段 up 输出为 `w3.weight`，不能交换：

   ```text
   w1.weight = w13_weight[expert, :, :intermediate].T
   w3.weight = w13_weight[expert, :, intermediate:].T
   w2.weight = w2_weight[expert].T
   ```

   当 `n_shared_experts > 0` 时，脚本要求每个 stage 都具备 canonical 的 `ffn.shared_experts.w1/w2/w3.weight`。所有 stage 还必须具备 `ffn.gate.weight`、`ffn.gate.bias` 以及每个 routed expert 的 `w1/w2/w3.weight`。缺失时转换会失败，不会产生不完整的输出目录。

3. 输出文件。

   输出目录包含以下文件：

   - BF16 safetensors 权重：单文件 `model.safetensors`，或多分片权重及 `model.safetensors.index.json`；
   - CANN 推理配置 `config.json`；
   - 转换记录 `conversion_info.json`，其中包含被跳过的共享权重、输出 dtype 统计、复制的 tokenizer 文件和 Confidence Head 兼容信息。

   传入 `--target-root` 时，脚本会从 target model 根目录复制 tokenizer 相关文件（例如
   `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、
   `tokenizer.model`、`vocab.json`、`merges.txt` 和 `chat_template.jinja`）到输出目录。
   target model 根目录至少需要包含其中一个 tokenizer 文件；否则转换会失败。未传入
   `--target-root` 时不会复制 tokenizer。

   转换后的目录可交给 `cann-recipes-infer` 的 benchmark 或部署流程使用。
