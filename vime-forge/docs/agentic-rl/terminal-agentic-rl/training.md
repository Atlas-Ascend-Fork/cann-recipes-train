# Terminal／TMax AgenticRL 训练

本文说明如何使用已准备好的 TMax 数据 bundle，在 Harbor 或 AgentENV 沙箱上运行 Qwen3.5-4B AgenticRL 作为快速开始的训练样例。两种数据使用同一个 Terminal 入口，通过 `--bundle` 选择任务；训练后端为 [TorchTitan-NPU](https://gitcode.com/cann/torchtitan-npu)，推理服务为 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)。

本文所有命令均在 **NPU 服务器的 `vime-forge` 目录**执行。CPU 沙箱服务器上的服务、任务镜像和 Agent 安装包应已准备完成。本页以 TMax 的 2048 题子集为例，分别给出一次完整更新和完整训练的命令；实际题数以所选 bundle 为准。

## 1. 启动前需要准备什么

| 内容 | 所在位置与要求 | 操作指南 |
| --- | --- | --- |
| 训练与推理环境、模型 | NPU 服务器；使用与当前 Qwen3.5 recipe 配套的 Python、CANN 和 ATB 环境 | [机器环境与设备](../env-and-config/configuration.md#3-npu-服务器环境与设备) |
| Terminal／TMax bundle | NPU 服务器；已完成任务转换、筛选及环境探测 | [Terminal／TMax 数据准备](data-preparation.md) |
| 沙箱服务与任务镜像 | CPU 沙箱服务器；镜像摘要、工作目录及依赖与 bundle 一致 | [沙箱部署与配置](../env-and-config/sandboxes.md) |
| Agent 安装包 | 在 CPU 沙箱服务器打包和验证，按分发方式复制并配置 | [Agent 安装与初始化](../env-and-config/harness-preparation.md) |

NPU 到 CPU 的控制连接、沙箱到 NPU 的模型回调，以及可选安装包下载连接需要分别可用。bundle 准备完成后，需逐台检查 CPU 沙箱服务器上的镜像；缺失镜像需在训练前构建或下载。

入口为 [run-qwen35-4B-tbench-torchtitan.sh](../../../examples/terminal_agent_rl/run-qwen35-4B-tbench-torchtitan.sh)。它选择 `terminalbench` 任务，复用 [run-qwen35-4B.sh](../../../scripts/torchtitan/run-qwen35-4B.sh) 和 TorchTitan 公共 launcher，最终提交 `train_async.py --train-backend torchtitan`。TMax 不需要另外的模型启动脚本，也不需要 SWE-Gym 配套源码。

## 2. 填写外部配置

### 2.1 NPU 服务器配置

首次使用时，将模板复制到仓库外；已有外部配置时直接更新对应文件：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
mkdir -p "$HOME/.config/vime"
cp -n examples/common/agenticrl.local.sh.example "$HOME/.config/vime/tmax-harbor.local.sh"
cp -n examples/common/sandboxes.json.example "$HOME/.config/vime/sandboxes.json"
```

编辑 `tmax-harbor.local.sh`，填写实际的 `VIME_TRAIN_ENV`、`VIME_VLLM_ENV`、`VIME_CANN_HOME`、`VIME_NNAL_HOME`、`MODEL_PATH`、`MASTER_ADDR`、Agent 安装包和 `ADAPTER_*` 地址。公共启动器会加载 CANN 和 ATB 的 `set_env.sh`，请填写有效目录。公共字段及默认值见[配置索引](../env-and-config/configuration.md)。

本页示例使用四个 NPU 训练、另外四个 NPU 推理，在该文件中确认以下设置：

```bash
# 配置位置：NPU 服务器上的外部 tmax-harbor.local.sh。
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 替换为本次空闲设备编号
export TRAIN_TP=4 TRAIN_DP=1 TRAIN_CP=1 ROLLOUT_GPUS=4 VLLM_RANKS_PER_ENGINE=1
export VIME_AGENT_SANDBOX_BACKEND=harbor
export VIME_AGENT_SANDBOX_CONFIG="$HOME/.config/vime/sandboxes.json"

# TMax 沿用任务镜像的 PATH，避免继承 SWE 的 Conda 设置。
export VIME_AGENT_TASK_CONDA_SH=''
export VIME_AGENT_TASK_CONDA_PREFIX=''
```

Terminal 默认不激活 Conda；上面显式留空便于从 SWE 配置切换。只有任务镜像确实需要激活 Conda 时才填写这两项，路径指向 CPU 服务器上的**任务沙箱内部**；NPU 服务器的训推环境单独配置。

通过 `--local` 加载配置，不必提前 `source`。实验参数放在命令行的 `--` 后，优先于 `.local.sh` 中的 `OVERRIDE_ARGS`；文件中的普通 `export` 会覆盖调用 Shell 中同名变量，因此后端、设备和路径应在所选 `.local.sh` 中统一设置。

本页使用 CP=1。开启 CP=4 时，将外部配置中的训练并行度改为 `TRAIN_TP=1 TRAIN_DP=1 TRAIN_CP=4`，并按[公共配置说明](../env-and-config/configuration.md#32-npu-分配)选择 recipe；推理与沙箱配置保持原值。CP=4 的单轮验证记录见 [Qwen3.5 文档](../../rl/qwen35.md#24-上下文并行cp)，本任务的 CP 训练另行验证。

### 2.2 Harbor 容量与启动并发

在 NPU 服务器的 `sandboxes.json` 中填写两台 CPU 沙箱服务器的实际 SSH 地址、Python、worker 和状态目录。本页命令的总轨迹并发上限为 32，示例容量设置如下：

| JSON 字段 | 示例值 | 含义 |
| --- | --- | --- |
| `common.lifetime_sec` | `9000` | 单个沙箱生命周期，单位为秒 |
| `common.boot_concurrency`、`common.boot_retries` | `8`、`3` | 每个生成进程同时进行的启动准备流程上限、启动最多尝试次数 |
| `harbor.hosts` | 两台 CPU 沙箱服务器 | 填写各自连接与工作路径 |
| 每个 `hosts[]` 的 `slots` | `16` | 每台最多占用 16 个沙箱名额 |
| `harbor.prepare_concurrency` | `4` | 每台最多同时创建、安装 CLI 和准备工作区的沙箱数 |
| `harbor.prepare_acquire_timeout`、`harbor.prepare_queue_limit` | `1800`、`32` | 每台准备队列的等待时间上限（秒）及等待请求数上限 |

将模板中的 `slots=64`、`prepare_concurrency=8` 调整为表中的值，并按 CPU 沙箱服务器的容量设置容器 CPU 和内存。请求按到达顺序等待准备名额；准备完成后，沙箱可继续运行，不再占用准备名额。

TMax 任务的资源需求随题目变化，实际运行速度由任务组合、服务器资源和并发设置共同决定；本节提供启动配置示例。镜像应保留 bundle 指定的 digest，已有私网复制流程可继续使用。字段与容量限制详见[沙箱指南](../env-and-config/sandboxes.md)。

### 2.3 上下文与训练 token 预算

这两个参数控制不同的限制，可在外部 `.local.sh` 中按需设置：

| 参数 | 作用与默认值 |
| --- | --- |
| `VIME_AGENT_MAX_CONTEXT_TOKENS` | Agent 的上下文上限，默认 `32768`；启动器据此设置训练序列长度、推理最大模型长度及 rollout 上下文限制 |
| `MAX_TOKENS_PER_GPU` | 动态训练微批的 token 容量，未设置时沿用上述上下文长度；也可通过命令行 `--max-tokens-per-gpu` 覆盖 |

上下文上限决定单条序列可以有多长；训练微批容量用于安排每次前向、反向计算处理的 token 数，轨迹数量由 batch 与每题采样数决定，超长样本保留原长度。两者不要求始终相等；当前 CP=1 的动态组批中，单个样本超过微批容量时仍会单独成批，实际显存占用还受样本长度、模型和算子影响。

## 3. 启动前确认配置

确认外部配置中的 Python 环境、CANN/ATB、模型和 bundle 路径可用，并按[沙箱指南](../env-and-config/sandboxes.md#6-启动前检查与验证范围)检查连接、镜像和回调。查看任务解析结果可使用：

```bash
# 执行位置：NPU 服务器的 vime-forge 目录；只加载可信的配置文件。
source "$HOME/.config/vime/tmax-harbor.local.sh"
VIME_RL_PREPARED_BUNDLE=/path/to/prepared/bundle \
python3 vime/agent/config.py shell --task terminalbench --repo-root .
```

该命令只输出任务配置，不激活环境、不读取密钥、不启动 Ray 或训练，也不构造完整训练命令。最终训练参数由下文启动命令决定。

共享服务器上需确认设备和端口空闲；不同任务使用独立的 `VIME_SLOT`、Ray 临时目录及服务端口。槽位不会自动分配空闲设备或隔离所有端口。

## 4. 使用 Harbor 启动训练

### 4.1 先完成一次 4×8 训练更新

本节的 `4×8` 用于小批量验证。正式训练的 batch 和训练长度按实验目标设置，第 4.2 节提供 `32×8` 示例。

在 NPU 服务器创建独立 tmux 会话：

```bash
# 执行位置：NPU 服务器。
tmux new-session -s tmax-harbor
```

进入新会话后执行，仓库路径和 bundle 路径按部署替换：

```bash
# 执行位置：NPU 服务器的上述 tmux 会话。
cd /path/to/cann-recipes-train-rl/vime-forge
bash examples/terminal_agent_rl/run-qwen35-4B-tbench-torchtitan.sh \
  --local "$HOME/.config/vime/tmax-harbor.local.sh" \
  --bundle /path/on/npu/prepared/tmax-2048 -- \
  --rollout-batch-size 4 --n-samples-per-prompt 8 --num-rollout 1 \
  --vllm-server-concurrency 8 --router-policy consistent_hash
```

本次抽取 4 题，每题生成 8 条轨迹，共 32 条；在当前每批一次更新的设置下，global batch 为 32。从头运行时，`--num-rollout 1` 使任务在一次训练更新及权重同步后结束，不会遍历全部 2048 题。验证训练更新时请省略 `--debug-rollout-only`，以执行完整训练流程。

若本次不检查 checkpoint 保存，在所用 `.local.sh` 中设 `VIME_RL_SAVE_INTERVAL=off`；保留默认保存设置时，最后一次更新仍会触发保存。关闭保存的验证不覆盖 checkpoint 保存流程。小批量验证覆盖本次抽取的任务，其他任务环境按实际训练范围继续检查。

### 4.2 运行完整训练

小批量验证结束后，使用新的输出目录，并确认 checkpoint 保存策略：若此前在 `.local.sh` 中设置了 `VIME_RL_SAVE_INTERVAL=off`，需要将其改为所需保存间隔，例如 `6`，才能恢复保存。下面以 `32×8`、128 个 rollout 批次为例，实际值按实验目标调整：

```bash
# 执行位置：NPU 服务器（vime-forge 目录，可在独立 tmux 中执行）。
bash examples/terminal_agent_rl/run-qwen35-4B-tbench-torchtitan.sh \
  --local "$HOME/.config/vime/tmax-harbor.local.sh" \
  --bundle /path/on/npu/prepared/tmax-2048 -- \
  --rollout-batch-size 32 --n-samples-per-prompt 8 --num-rollout 128 \
  --vllm-server-concurrency 8 --router-policy consistent_hash
```

本例每批 256 条轨迹，`--num-rollout 128` 显式指定训练总目标为 128 个 rollout 批次。从头运行、每批一次更新时，共执行 128 次训练更新。对于 2048 题、每批 32 题，这相当于两遍数据的题目数量；该换算只帮助理解数据遍历量，不需要另设 epoch 参数。`32×8` 不表示只运行一批。恢复训练时，`--num-rollout` 仍表示训练的总目标，程序从已完成的位置继续执行。

四个 TP=1 推理引擎配合 `--vllm-server-concurrency 8`，总轨迹并发上限为 `8×4=32`，与每批 256 条轨迹分别控制。沙箱启动、主机容量和准备并发仍受 JSON 限制。两台 CPU 沙箱服务器通过同一模型适配器和路由器访问推理引擎，不与某个引擎固定绑定；`consistent_hash` 按请求的哈希映射分配引擎，瞬时负载随请求变化。

显式传入 `--vllm-server-concurrency` 时，以该参数为准，覆盖启动器根据 `VIME_RL_TRIAL_CONCURRENCY` 推导的值；本页样例无需再设置后者。每批 256 条轨迹可以分批执行，不要求同时运行 256 个沙箱。

未覆盖的参数保持 recipe 默认：学习率 `1e-6`、上下文 `32768`、温度 `1.0`、开启 prefix cache、关闭 KL-Cov；算法使用 GRPO 和 TIS，训练采用异步流程，每批更新后同步推理权重。需要改变上下文时使用 `VIME_AGENT_MAX_CONTEXT_TOKENS`，其余训练参数见[实验参数](../env-and-config/configuration.md#42-训练参数优先使用命令行)。

## 5. 使用 AgentENV 进行AgenticRL训练

使用 AgentENV 时，准备另一份外部 `tmax-agentenv.local.sh`，在文件中设置 `VIME_AGENT_SANDBOX_BACKEND=agentenv`。JSON 的 `agentenv` 区填写服务地址、NPU 服务器上的凭据文件和 SDK 配置，`selector_mode` 使用 `template`；模型回调和安装包地址也要适用于 AgentENV 沙箱。

选用包含当前服务有效模板 ID 的 Terminal／TMax bundle。切换 AgentENV 前，需为 `template_id` 为空的 Harbor bundle 补齐有效模板 ID。AgentENV 不读取 Harbor 的 `slots` 或准备队列设置，服务容量需单独确认。

下面给出一次更新的 AgentENV 命令，可在相应的独立 tmux 会话内执行：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
bash examples/terminal_agent_rl/run-qwen35-4B-tbench-torchtitan.sh \
  --local "$HOME/.config/vime/tmax-agentenv.local.sh" \
  --bundle /path/on/npu/prepared/tmax-agentenv -- \
  --rollout-batch-size 4 --n-samples-per-prompt 8 --num-rollout 1 \
  --vllm-server-concurrency 8 --router-policy consistent_hash
```

这里的 `4×8` 同样仅用于验证。正式训练时参考第 4.2 节，按实验目标设置 batch、`--num-rollout` 和 checkpoint 保存间隔。

使用 Terminal-Bench 数据时保留同一个入口，将 `--bundle` 换成其已准备的数据包，并核对任务镜像和 Conda 设置。选择 Claude Code 时，在 `--` 之前加入 `--agent claude_code`，并按 [Harness 指南](../env-and-config/harness-preparation.md)配置对应安装包和回调协议。

## 6. 查看进度与输出

默认输出目录为 `outputs/<RUN_NAME>/`，`RUN_NAME` 自动包含 `terminalbench` 和时间戳，TMax 也沿用该任务名。可以在外部配置中指定运行名和 `OUTPUT_DIR`，但每次运行必须使用新的目录，不要提前创建它。小批量验证与完整训练是分别启动的任务，不会因使用同一个 bundle 自动接续。

| 路径（相对输出目录） | 内容 |
| --- | --- |
| `tensorboard/<RUN_NAME>/<时间戳>/train.log` | Ray job 运行日志，包含轨迹生成、训练指标和报错 |
| `tensorboard/<RUN_NAME>/<时间戳>/rerun.sh` | 归档的启动命令及配置 |
| `tensorboard/` | TensorBoard 事件文件，默认启用 |
| `rollout_dumps/<rollout_id>.pt` | 默认保存的轨迹和元数据 |
| `checkpoints/` | 启用保存时的检查点；默认每 6 次更新及最后一次更新保存，保留最近 2 份 |

一次完整更新需要看到 `train metrics @ rollout_id=0`，随后对应的 `Timer update_weights end`，最后任务正常退出。推理响应和轨迹文件用于检查生成阶段，初始化权重同步用于检查启动阶段。

重点观察 `rollout/raw_reward`、`train/entropy_loss` 和 `train/grad_norm`，分别对应原始奖励、训练计算的熵和梯度范数。奖励为 0 与基础设施失败应结合日志区分；不要仅凭某个 batch 的奖励或梯度范数判断整个任务是否正常。

在 NPU 服务器另开终端，用实际输出路径查看日志：

```bash
# 执行位置：NPU 服务器。
tail -f /path/on/npu/outputs/run-name/tensorboard/run-name/timestamp/train.log
```