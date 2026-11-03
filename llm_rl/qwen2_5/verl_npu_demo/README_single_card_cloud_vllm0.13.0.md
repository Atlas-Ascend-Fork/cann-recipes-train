# README：Qwen2.5 GRPO 单卡云环境部署指南（vLLM 0.13.0 兼容路径）

> 适用场景：单卡昇腾 NPU 云环境部署  
> 目标模型：Qwen2.5-1.5B-Instruct  
> 训练框架：verl  
> 训练任务：MATH-lighteval 上的 GRPO 训练

## 说明

`llm_rl/qwen2_5/verl_npu_demo/README.md` 样例默认支持的是基于 **vLLM 0.9.1** 的依赖栈。本文档补充的是一条**单卡云环境**下、基于 **vLLM 0.13.0** 的兼容部署路径。  
也就是说，如果你使用的是官方 README 中的默认依赖栈，请优先参考 `llm_rl/qwen2_5/verl_npu_demo/README.md`；如果你希望在单卡云环境中按本文档这一路径完成部署、训练和验证，则需要按照下面的步骤完成额外适配。

------

## 1. 环境信息

本文档对应的实测环境如下：

- NPU：Ascend Atlas 910B2C（单卡）
- 部署方式：Docker
- CANN：8.2.RC1
- 模型：Qwen2.5-1.5B-Instruct
- 数据集：MATH-lighteval
- 训练算法：GRPO
- 兼容路径：vLLM 0.13.0

------

## 2. 启动 Docker 容器

先拉起单卡训练所需的运行环境。这里采用的是云环境中验证可用的 CANN 8.2.RC1 容器方案，并显式设置 `--shm-size`。

```bash
docker run -d -it \
--name qwen-grpo \
--shm-size=8g \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
ascendhub.huawei.com/public-ascendhub/cann:8.2.RC1-910b-openeuler22.03-py3.11 \
/bin/bash
```

进入容器：

```bash
docker exec -it qwen-grpo bash
```

------

## 3. 安装依赖

这一部分采用的是单卡云环境中验证过的一组依赖组合。  
**关键原则是：先安装项目自身 requirements，再在最后统一固定核心兼容依赖；其中 vLLM 相关建议按 `vllm -> vllm-ascend` 的顺序安装。**

这样做的原因是：

- `requirements.txt` 和 `requirements-npu.txt` 中的依赖解析，可能会覆盖你前面手动安装的核心版本；
- 本文档后续的兼容性修改，是建立在 **`vLLM 0.13.0 + transformers 4.57.6`** 这组版本组合上的；
- 因此更稳妥的方式，是先把项目通用依赖装齐，再在最后把核心依赖重新固定到目标版本；
- `vllm-ascend` 依赖 `vllm`，并会带出所需的 `torch_npu/torch` 依赖链，建议显式先装 `vllm` 再装 `vllm-ascend`；
- 根据审核意见，这里**不使用 `--force-reinstall`**，避免引入不必要的重复重装。

### 3.1 下载项目代码

先下载样例工程，再进入本地 `verl` 目录并固定到经过验证的提交版本。这样做的好处是，后续的兼容性适配和训练脚本修改都有明确的代码基线。

```bash
git clone https://gitcode.com/cann/cann-recipes-train.git
cd cann-recipes-train/llm_rl/qwen2_5/verl_npu_demo
```

如果当前 `verl_npu_demo/verl/` 目录下还没有 `verl` 源码，可执行：

```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 634bd935
cd ..
```

### 3.2 先安装项目 requirements

先安装项目依赖，让 `requirements` 先把通用依赖铺齐：

```bash
pip install -r requirements.txt
cd verl
pip install -r requirements-npu.txt
cd ..
```

### 3.3 最后统一固定核心兼容依赖

下面这一步放在最后执行，避免前面 `requirements` 再次覆盖这些关键版本。

```bash
# 配置 PyTorch CPU wheel 源
pip config set global.index-url https://download.pytorch.org/whl/cpu

# 固定核心兼容依赖
pip install torch==2.8.0+cpu
pip install torch_npu==2.8.0
pip install torchvision==0.23.0+cpu

pip install transformers==4.57.6
pip install ray==2.41.0
pip install sympy==1.13.3
pip install gguf==0.18.0
pip install compressed-tensors==0.12.2

# 卸载 uvloop，避免 asyncio 冲突
# 注意放到最后，防止前面的 requirements 又把它装回来
pip uninstall -y uvloop

# vLLM 相关按顺序安装：先 vllm，再 vllm-ascend
pip install vllm==0.13.0
pip install vllm-ascend==0.13.0
```

### 3.4 安装本地源码

基于当前环境安装本地 `verl` 源码：

```bash
cd verl
pip install -e . --no-deps
cd ..
```

这里保留 `--no-deps`，避免本地源码安装时再次触发依赖解析，影响上一步已经固定好的版本。

### 3.5 可选：使用 constraints 文件进一步锁定版本

如果你希望后续复现更稳，可以额外准备一个 `constraints.txt`，内容如下：

```txt
torch==2.8.0+cpu
torch_npu==2.8.0
torchvision==0.23.0+cpu
transformers==4.57.6
ray==2.41.0
sympy==1.13.3
gguf==0.18.0
compressed-tensors==0.12.2
vllm-ascend==0.13.0
```

然后安装命令为：

```bash
pip install -c constraints.txt -r requirements.txt
cd verl
pip install -c ../constraints.txt -r requirements-npu.txt
cd ..
```

这样可以在 `requirements` 解析阶段就限制核心版本。

------

## 4. 进行兼容性适配

由于本文档采用的是 **vLLM 0.13.0 + transformers 4.57.6** 的依赖组合，因此需要补充一些额外配置和兼容性改动。

### 4.1 补齐 NPU 工具函数文件

为避免与外层目录 `verl_npu_demo/verl/` 和内层源码包目录 `verl_npu_demo/verl/verl/` 混淆，**建议先进入明确目录再执行 `cat` 命令**。

```bash
cd cann-recipes-train/llm_rl/qwen2_5/verl_npu_demo/verl
```

然后新建文件：

`./verl/workers/engine/fsdp/npu_padding_utils.py`

```bash
cat > ./verl/workers/engine/fsdp/npu_padding_utils.py << 'EOF'
import torch
from einops import rearrange as einops_rearrange

def index_first_axis(x, indices):
    return x[indices]

def pad_input(hidden_states, indices, batch, seqlen):
    output = torch.zeros(
        batch * seqlen, *hidden_states.shape[1:],
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    output[indices] = hidden_states
    return output.view(batch, seqlen, *hidden_states.shape[1:])

def rearrange(tensor, pattern):
    return einops_rearrange(tensor, pattern)

def unpad_input(hidden_states, attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
    )
    return (
        index_first_axis(hidden_states.view(-1, *hidden_states.shape[2:]), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
EOF
```

### 4.2 调整 `transformers` 相关导入

以下命令默认在 `cann-recipes-train/llm_rl/qwen2_5/verl_npu_demo/verl` 目录下执行。

将以下 5 个文件中的导入路径替换为本地兼容实现：

- `./verl/workers/engine/fsdp/transformer_impl.py`
- `./verl/workers/actor/dp_actor.py`
- `./verl/trainer/fsdp_sft_trainer.py`
- `./verl/workers/critic/dp_critic.py`
- `./verl/workers/fsdp_workers.py`

执行命令：

```bash
sed -i 's/from transformers.integrations.npu_flash_attention import/from verl.workers.engine.fsdp.npu_padding_utils import/g' \
./verl/workers/engine/fsdp/transformer_impl.py \
./verl/workers/actor/dp_actor.py \
./verl/trainer/fsdp_sft_trainer.py \
./verl/workers/critic/dp_critic.py \
./verl/workers/fsdp_workers.py
```

### 4.3 调整 vLLM 0.13.0 API 路径

以下命令同样默认在 `cann-recipes-train/llm_rl/qwen2_5/verl_npu_demo/verl` 目录下执行：

```bash
sed -i 's/CompilationLevel/CompilationMode/g' \
./verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py

sed -i 's/from vllm.worker.worker_base/from vllm.v1.worker.worker_base/g' \
./verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py

sed -i 's/from vllm.lora.models/from vllm.lora.lora_model/g' \
./verl/utils/vllm/utils.py
```

------

## 5. 准备权重

在开始准备数据和启动训练之前，先下载模型权重。建议在 `verl_npu_demo/verl/` 目录下执行下面的命令：

```bash
cd cann-recipes-train/llm_rl/qwen2_5/verl_npu_demo/verl
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir model/Qwen2_5_1_5B_Instruct/
```

如果当前环境访问 Hugging Face 较慢，也可以先切换镜像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir model/Qwen2_5_1_5B_Instruct/
```

权重下载完成后，模型文件会位于：

`verl_npu_demo/verl/model/Qwen2_5_1_5B_Instruct/`

### 5.1 替换 Tokenizer 配置文件

根据审核意见，`tokenizer_config.json` 的替换步骤放到**模型权重下载完成后**执行：

```bash
cp ../tokenizer_config.json ./model/Qwen2_5_1_5B_Instruct/tokenizer_config.json
```

------

## 6. 准备数据

这一部分按照官方 README 的方式执行，使用数据预处理脚本**在线获取 MATH-lighteval 数据集并处理成 verl 所需格式**。

> 说明：此前文档中使用 `./data/math/raw/train.jsonl` 和 `./data/math/raw/test.jsonl` 作为输入参数，但未说明这两个文件的获取方式，容易引起歧义。这里统一改为与官方 README 一致的调用方式。

在 `verl_npu_demo/verl` 目录下执行：

```bash
cd cann-recipes-train/llm_rl/qwen2_5/verl_npu_demo/verl/examples/data_preprocess
python3 math_dataset.py --local_dir ../../data/math/data
```

执行完成后，会在 `verl_npu_demo/verl/data/math/data/` 目录下生成训练所需的：

- `train.parquet`
- `test.parquet`

### 6.1 无法联网时的离线准备方式

如果当前环境无法访问 Hugging Face，可先手动下载数据集：

```bash
cd cann-recipes-train/llm_rl/qwen2_5/verl_npu_demo/verl
hf download DigitalLearningGmbH/MATH-lighteval --repo-type=dataset --local-dir data/math
```

将数据集文件放到 `verl_npu_demo/verl/data/math/` 路径下后，再参考官方 README 附录中的方式，把 `examples/data_preprocess/math_dataset.py` 中的 `datasets.load_dataset(...)` 从在线加载改为本地 parquet 加载，然后重新执行：

```bash
cd examples/data_preprocess
python3 math_dataset.py --local_dir ../../data/math/data
```

------

## 7. 配置训练脚本

在启动训练前，先完成奖励函数替换，并将单卡运行参数和 NPU 相关配置正确写入 `run_qwen2_5_1_5b.sh`。

以下命令建议在 `cann-recipes-train/llm_rl/qwen2_5/verl_npu_demo/verl` 目录下执行。

### 7.1 替换奖励函数

```bash
cp ../new_math_reward.py ./verl/utils/reward_score/new_math_reward.py
```

### 7.2 将脚本移动到 `verl/` 目录

`run_qwen2_5_1_5b.sh` 需要放在 `qwen2_5/verl_npu_demo/verl/` 下执行。若当前脚本仍在上一层目录，可先移动：

```bash
mv ../run_qwen2_5_1_5b.sh ./run_qwen2_5_1_5b.sh
```

### 7.3 替换参数（不要直接用 `cat` 追加）

此前使用 `cat >> run_qwen2_5_1_5b.sh` 只是把新参数追加到文件末尾，**不会保证原有参数被替换**。如果脚本中前面已有同名参数，最终生效值可能不是预期。

建议改为“替换已有参数 + 补充缺失环境变量”：

```bash
# 替换单卡相关参数（按键名替换原值）
sed -i 's/^trainer\.n_gpus_per_node=.*/trainer.n_gpus_per_node=1 \\/' run_qwen2_5_1_5b.sh
sed -i 's/^actor_rollout_ref\.rollout\.tensor_model_parallel_size=.*/actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\/' run_qwen2_5_1_5b.sh
sed -i 's/^actor_rollout_ref\.model\.use_shm=.*/actor_rollout_ref.model.use_shm=False \\/' run_qwen2_5_1_5b.sh

# 若缺失则追加 NPU 环境变量
grep -q '^export VLLM_ASCEND_ENABLE_NZ=' run_qwen2_5_1_5b.sh || echo 'export VLLM_ASCEND_ENABLE_NZ=0' >> run_qwen2_5_1_5b.sh
grep -q '^export PYTORCH_CUDA_ALLOC_CONF=' run_qwen2_5_1_5b.sh || echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' >> run_qwen2_5_1_5b.sh
grep -q '^export OMP_NUM_THREADS=' run_qwen2_5_1_5b.sh || echo 'export OMP_NUM_THREADS=4' >> run_qwen2_5_1_5b.sh
```

这些配置的作用分别是：

- `VLLM_ASCEND_ENABLE_NZ=0`：关闭 FRACTAL_NZ 相关影响，降低 RL 场景下的精度风险
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`：优化内存分配行为
- `OMP_NUM_THREADS=4`：控制线程数量，降低资源争抢
- `trainer.n_gpus_per_node=1`：单卡训练
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1`：关闭多卡并行切分假设
- `actor_rollout_ref.model.use_shm=False`：规避共享内存相关问题

------

## 8. 启动训练

建议先确保当前目录位于 `cann-recipes-train/llm_rl/qwen2_5/verl_npu_demo/verl`：

```bash
mkdir -p run_log
bash run_qwen2_5_1_5b.sh
```

------

## 9. 观察训练状态

建议同时观察日志和 NPU 状态，方便确认训练是否已经稳定进入运行阶段。

### 9.1 查看训练日志

```bash
tail -f run_log/qwen2_5_1_5b_math.log
```

### 9.2 查看 NPU 状态

```bash
watch -n 1 npu-smi info
```

如果训练已经正常跑起来，一般可以看到：

- 日志持续滚动
- AICore 使用率较高
- HBM 占用持续上升并保持稳定
- 训练脚本能够进入持续执行状态

------

## 10. 训练完成后的验证建议

如果只看到训练启动成功，还不能算整条链路完全闭环。建议在训练完成后继续接入验证流程，例如使用 OpenCompass 对基模和强化学习后模型进行对比测试，确认最终效果提升是否符合预期。

如果你已经沿用了原样例中的验证流程，可以继续参考 `llm_rl/qwen2_5/verl_npu_demo/README.md` 中的 OpenCompass 相关说明。

------

## 11. 建议补充的版本检查命令

为了在安装结束后快速确认没有发生核心依赖漂移，建议额外执行：

```bash
python - << 'EOF'
import torch
import transformers
import sympy
import ray

print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('sympy:', sympy.__version__)
print('ray:', ray.__version__)
EOF

pip show torch_npu
pip show vllm-ascend
pip show compressed-tensors
pip show gguf
```

如果输出结果与本文档固定版本不一致，建议不要直接继续训练，而是先排查是否被 `requirements` 或其他安装步骤重新覆盖。
