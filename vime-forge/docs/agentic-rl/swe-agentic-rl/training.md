# SWE-Gym AgenticRL 训练

本文说明如何使用已准备好的 SWE-Gym bundle，在 Harbor 或 AgentENV 沙箱上运行 AgenticRL 训练，并使用 Qwen3.5-4B 作为快速开始的训练模型。训练后端为 TorchTitan-NPU，推理服务为 vLLM-Ascend；默认 Agent 为 Codex，也可选择 Claude Code。

本文所有命令均在 **NPU 服务器的 `vime-forge` 目录**执行。CPU 沙箱服务器上的服务、镜像和安装包应已准备完成；训练过程中，框架会按需创建沙箱、初始化 Agent 并收集轨迹。

## 1. 启动前需要准备什么

| 内容 | 所在位置与要求 | 操作指南 |
| --- | --- | --- |
| 训练与推理环境、模型 | NPU 服务器；使用与当前 Qwen3.5 recipe 配套的 Python、CANN 和 ATB 环境 | [机器环境与设备](../env-and-config/configuration.md#3-npu-服务器环境与设备) |
| SWE-Gym bundle | NPU 服务器；包含完整的训练输入、环境映射、任务检查规格和 manifest | [SWE 数据准备](data-preparation.md) |
| SWE-Gym 配套源码及依赖 | NPU 服务器；版本须与 bundle 的准备流程一致，Python 依赖须安装在训练环境中 | [固定依赖](data-preparation.md#31-源数据与固定依赖) |
| 沙箱服务与任务镜像 | CPU 沙箱服务器；Harbor 使用 bundle 指定 digest 的镜像，AgentENV 使用对应模板 | [沙箱部署与配置](../env-and-config/sandboxes.md) |
| Agent 安装包 | 在 CPU 沙箱服务器打包和验证，按分发方式复制并配置 | [Agent 安装与初始化](../env-and-config/harness-preparation.md) |

`VIME_SWEGYM_SOURCE_PATH` 只指定评测器源码位置，不会安装它的 Python 依赖。重建训练环境后，需要同时补齐对应源码声明的运行依赖。

同时确认 NPU 到 CPU 的控制连接、沙箱到 NPU 的模型回调，以及沙箱到安装包服务的下载连接分别可用。启动脚本不负责建立隧道、下载全量任务镜像或部署安装包服务。

SWE 任务 (`swegym`) 的样例训练入口为 [run-qwen35-4B-swegym-torchtitan.sh](../../../examples/swe_agent_rl/run-qwen35-4B-swegym-torchtitan.sh)。然后复用 [run-qwen35-4B.sh](../../../scripts/torchtitan/run-qwen35-4B.sh) 和 TorchTitan 公共 launcher。

## 2. AgenticRL 外部配置

### 2.1 NPU 服务器配置

首次使用时，将模板复制到仓库外；已有外部配置时直接更新对应文件：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
mkdir -p "$HOME/.config/vime"
cp -n examples/common/agenticrl.local.sh.example "$HOME/.config/vime/swe-harbor.local.sh"
cp -n examples/common/sandboxes.json.example "$HOME/.config/vime/sandboxes.json"
```

编辑 `swe-harbor.local.sh`，填写实际的 `VIME_TRAIN_ENV`、`VIME_VLLM_ENV`、`VIME_CANN_HOME`、`VIME_NNAL_HOME`、`MODEL_PATH`、`MASTER_ADDR`、Agent 安装包和 `ADAPTER_*` 地址。全部路径按 NPU 服务器上的可见路径填写；公共启动器会加载 CANN 和 ATB 的 `set_env.sh`，请填写有效目录。变量定义见[配置索引](../env-and-config/configuration.md)。

本页 Harbor 示例使用四个 NPU 训练、另外四个 NPU 推理，在该文件中确认以下设置：

```bash
# 配置位置：NPU 服务器上的外部 swe-harbor.local.sh。
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 替换为本次空闲设备编号
export TRAIN_TP=4 TRAIN_DP=1 TRAIN_CP=1 ROLLOUT_GPUS=4 VLLM_RANKS_PER_ENGINE=1
export VIME_AGENT_SANDBOX_BACKEND=harbor
export VIME_AGENT_SANDBOX_CONFIG="$HOME/.config/vime/sandboxes.json"
export VIME_SWEGYM_SOURCE_PATH=/path/on/npu/SWE-Gym-source

# 沿用 SWE 训练示例的工作区准备与状态查询预算。
export VIME_AGENT_WORKSPACE_PREP_TIMEOUT_SEC=300
export VIME_AGENT_MARKER_TIMEOUT_SEC=15
export VIME_AGENT_MARKER_TIMEOUT_RETRIES=3
```

SWE 默认激活**任务沙箱内**的 `/opt/miniconda3/etc/profile.d/conda.sh` 和 `/opt/miniconda3/envs/testbed`。镜像使用其他布局时，调整 `VIME_AGENT_TASK_CONDA_SH` 与 `VIME_AGENT_TASK_CONDA_PREFIX`；两项同时为空表示沿用沙箱中的 `PATH`。这些路径与 NPU 服务器的训推 Python 环境无关。

通过 `--local` 加载外部配置和环境变量。命令行 `--` 后的训练参数优先于 `.local.sh` 中的 `OVERRIDE_ARGS`；文件中的普通 `export` 则会覆盖调用 Shell 中同名变量，因此后端、设备和路径应在所选 `.local.sh` 中统一设置。

本页使用 CP=1。开启 CP=4 时，将外部配置中的训练并行度改为 `TRAIN_TP=1 TRAIN_DP=1 TRAIN_CP=4`，并按[公共配置说明](../env-and-config/configuration.md#32-npu-分配)选择 recipe；推理与沙箱配置保持原值。CP=4 的单轮验证记录见 [Qwen3.5 文档](../../rl/qwen35.md#24-上下文并行cp)，本任务的 CP 训练另行验证。

### 2.2 Harbor 沙盒配置

在 NPU 服务器的 `sandboxes.json` 中填写两台 CPU 沙箱服务器的实际 SSH 地址、Python、worker 和状态目录。本页沿用 SWE 的 `32×8`、轨迹并发 `32` 示例，相关 JSON 配置如下；完整配置说明和字段见[沙箱指南](../env-and-config/sandboxes.md)。

| JSON 字段 | 示例值 | 含义 |
| --- | --- | --- |
| `common.lifetime_sec` | `9000` | 单个沙箱的生命周期，单位为秒；这是显式覆盖，SWE 默认值为 `4500` |
| `common.boot_concurrency`、`common.boot_retries` | `8`、`3` | 每个生成进程最多同时进行 8 个启动准备流程；启动最多尝试 3 次 |
| `harbor.hosts` | 两台 CPU 沙箱服务器 | 填写各自连接与工作路径 |
| 每个 `hosts[]` 的 `slots` | `16` | 每台最多占用 16 个沙箱名额，包含 SWE 的独立任务检查沙箱 |
| `harbor.prepare_concurrency` | `4` | 每台最多同时创建、安装 CLI 和准备工作区的沙箱数 |
| `harbor.prepare_acquire_timeout`、`harbor.prepare_queue_limit` | `1800`、`32` | 每台准备队列的等待时间上限（秒）及等待请求数上限 |

模板中的 `slots=64`、`prepare_concurrency=8` 与本例不同，需要改为表中的值。容器的 `cpus`、`memory_mb` 根据 CPU 沙箱服务器容量设置；增加沙箱名额不会增加物理资源。镜像与已有私网复制流程保持不变。

请求按到达顺序等待准备名额。完成创建、CLI 安装和工作区设置后释放准备名额，沙箱本身继续运行，直到销毁时才释放沙箱名额。轨迹并发与这些限制分别生效。

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
source "$HOME/.config/vime/swe-harbor.local.sh"
VIME_RL_PREPARED_BUNDLE=/path/to/prepared/bundle \
python3 vime/agent/config.py shell --task swegym --repo-root .
```

该命令只输出任务配置，不激活环境、不读取密钥、不启动 Ray 或训练，也不构造完整训练命令。最终训练参数由下文启动命令决定。

共享服务器上需确认设备和端口空闲；不同任务使用独立的 `VIME_SLOT`、Ray 临时目录及服务端口。槽位不会自动分配空闲设备或隔离所有端口。

## 4. 使用 Harbor 启动训练

### 4.1 完整的 32×8 训练

在 NPU 服务器创建独立 tmux 会话：

```bash
# 执行位置：NPU 服务器。
tmux new-session -s swe-harbor
```

进入新会话后执行，仓库路径和 bundle 路径按部署替换：

```bash
# 执行位置：NPU 服务器的上述 tmux 会话。
cd /path/to/cann-recipes-train-rl/vime-forge
bash examples/swe_agent_rl/run-qwen35-4B-swegym-torchtitan.sh \
  --local "$HOME/.config/vime/swe-harbor.local.sh" \
  --bundle /path/on/npu/prepared/swegym-bundle -- \
  --rollout-batch-size 32 --n-samples-per-prompt 8 --num-rollout 18 \
  --vllm-server-concurrency 8 --router-policy consistent_hash
```

这组参数的含义是：

- 每个batch包含 32 个tasks，每个task 采样 8 条轨迹，使用[GRPO](https://arxiv.org/abs/2402.03300)算法训练；当前每批进行一次训练更新，global batch 由这些参数自动推导。
- 推理使用四个 TP=1 引擎，`--vllm-server-concurrency 8` 对应总轨迹并发上限 `8×4=32`。其余轨迹排队，不会一次创建 256 个沙箱。
- `--num-rollout 18` 显式指定训练总目标为 18 个 rollout 批次。从头运行、每批一次更新时，共执行 18 次训练更新。对于 288 题、每批 32 题，这相当于两遍数据的题目数量；该换算只帮助理解数据遍历量，不需要另设 epoch 参数。`32×8` 表示每批 32 题、每题 8 条轨迹。
- 其余参数保持 recipe 默认：学习率 `1e-6`、上下文 `32768`、温度 `1.0`、开启 prefix cache、关闭 KL-Cov。算法使用 GRPO 和 TIS，训练采用异步流程，每批更新后同步推理权重。

显式传入 `--vllm-server-concurrency` 时，以该参数为准，覆盖启动器根据 `VIME_RL_TRIAL_CONCURRENCY` 推导的值；本页样例无需再设置后者。每批 256 条轨迹不要求同时运行 256 个沙箱，沙箱启动、主机容量和准备并发仍受 JSON 配置限制。

两台 CPU 沙箱服务器上的沙箱均经同一模型适配器和路由器请求推理服务，不与某个引擎固定绑定。`consistent_hash` 按请求的哈希映射分配引擎，四个引擎的瞬时负载随请求变化。

恢复训练时，`--num-rollout` 仍表示训练的总目标，程序从已完成的位置继续执行。正式训练前应确认 checkpoint 保存策略：若此前验证时在 `.local.sh` 中设置了 `VIME_RL_SAVE_INTERVAL=off`，需要将其改为所需保存间隔，例如 `6`，才能恢复保存。

### 4.2 只验证一次完整训练更新

需要先验证流程时，使用同一外部配置，将训练长度显式改为 `--num-rollout 1`。下面的 `4×8` 用于小批量验证；正式训练的 batch 和训练长度按实验目标设置：

```bash
# 执行位置：NPU 服务器（vime-forge 目录，可在独立 tmux 中执行）。
bash examples/swe_agent_rl/run-qwen35-4B-swegym-torchtitan.sh \
  --local "$HOME/.config/vime/swe-harbor.local.sh" \
  --bundle /path/on/npu/prepared/swegym-bundle -- \
  --rollout-batch-size 4 --n-samples-per-prompt 8 --num-rollout 1 \
  --vllm-server-concurrency 8 --router-policy consistent_hash
```

从头运行且保持当前每批一次更新的设置时，该命令生成 32 条轨迹，执行一次训练更新及权重同步后结束。验证训练更新时请省略 `--debug-rollout-only`，以执行完整训练流程。若本次不检查 checkpoint 保存，在所用 `.local.sh` 中设 `VIME_RL_SAVE_INTERVAL=off`；保留默认保存设置时，最后一次更新仍会触发保存。关闭保存的验证不覆盖 checkpoint 保存流程。

完整训练和单次验证是两种独立运行方式，按需要选择；每次使用新的输出目录。

## 5. 使用 AgentENV 开始训练

准备另一份外部 `swe-agentenv.local.sh`，沿用同一机器环境和实验参数，并在文件中设置 `VIME_AGENT_SANDBOX_BACKEND=agentenv`。JSON 的 `agentenv` 区填写服务地址、NPU 服务器上的凭据文件和 SDK 配置，`selector_mode` 使用 `template`；模型回调和安装包地址必须能从 AgentENV 沙箱访问。

使用包含当前 AgentENV 服务有效模板 ID 的 SWE bundle。仅复制 bundle 不会迁移模板；Harbor 的 `slots` 和准备队列设置也不会限制 AgentENV 服务，应单独确认其容量。下面命令在相应的独立 tmux 会话内执行：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
bash examples/swe_agent_rl/run-qwen35-4B-swegym-torchtitan.sh \
  --local "$HOME/.config/vime/swe-agentenv.local.sh" \
  --bundle /path/on/npu/prepared/swegym-agentenv -- \
  --rollout-batch-size 32 --n-samples-per-prompt 8 --num-rollout 18 \
  --vllm-server-concurrency 8 --router-policy consistent_hash
```

选择 Claude Code 时，在 `--` 之前加入 `--agent claude_code`，并按 [Harness 指南](../env-and-config/harness-preparation.md)配置对应安装包和回调协议。

## 6. 查看进度与输出

默认输出目录为 `outputs/<RUN_NAME>/`，`RUN_NAME` 自动包含任务名和时间戳，也可在外部配置中指定。启动器拒绝已存在的 `OUTPUT_DIR`，不要提前创建它；修改固定运行名后还需同步修改固定输出路径。

| 路径（相对输出目录） | 内容 |
| --- | --- |
| `tensorboard/<RUN_NAME>/<时间戳>/train.log` | Ray job 运行日志，包含轨迹生成、训练指标和报错 |
| `tensorboard/<RUN_NAME>/<时间戳>/rerun.sh` | 归档的启动命令及配置 |
| `tensorboard/` | TensorBoard 事件文件，默认启用 |
| `rollout_dumps/<rollout_id>.pt` | 默认保存的轨迹和元数据 |
| `checkpoints/` | 启用保存时的检查点；默认每 6 次更新及最后一次更新保存，保留最近 2 份 |

在日志中先确认引擎注册和沙箱执行，再查看 `train metrics @ rollout_id=...`。训练指标产生后，还应确认该轮的 `Timer update_weights end`。HTTP 响应和轨迹文件用于检查生成阶段，初始化权重同步用于检查启动阶段。单次验证应看到 `rollout_id=0` 的训练指标、随后权重同步完成及任务正常退出。

重点观察 `rollout/raw_reward`、`train/entropy_loss` 和 `train/grad_norm`，分别对应原始奖励、训练计算的熵和梯度范数。指标缺失可能表示尚未进入对应阶段；`grad_norm=0` 应结合奖励、有效 token 和错误日志判断，据此定位训练状态。

在 NPU 服务器另开终端，用实际输出路径查看日志：

```bash
# 执行位置：NPU 服务器。
tail -f /path/on/npu/outputs/run-name/tensorboard/run-name/timestamp/train.log
```