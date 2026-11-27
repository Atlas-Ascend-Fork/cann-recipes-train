# AgenticRL 配置说明

本文说明 SWE-Gym 与 Terminal／TMax 共用的配置入口、环境变量和参数覆盖规则。两个任务的训练入口如下。

- [SWE-Gym 训练入口](../../../examples/swe_agent_rl/run-qwen35-4B-swegym-torchtitan.sh)
- [Terminal／TMax 训练入口](../../../examples/terminal_agent_rl/run-qwen35-4B-tbench-torchtitan.sh)

下文列出**通过这两个任务入口启动**时的默认值；单轮 RL 和独立导入 Python 模块的默认值见各自启动脚本。

本文默认已有 NPU 服务器和 CPU 沙箱服务器。下文的配置编辑、文件复制及启动检查命令均在 **NPU 服务器**执行，`.local.sh`、`sandboxes.json`、模型和训练用 bundle 也保存在 NPU 服务器；JSON 中的 `hosts[].python`、`worker`、`root` 等字段指向 CPU 沙箱服务器上的路径。

按需查阅：[配置职责](#1-配置放在哪里)、[覆盖顺序](#2-覆盖顺序与空值)、[机器与设备](#3-npu-服务器环境与设备)、[数据与实验参数](#4-数据包与实验参数)、[Agent 参数](#5-agent-环境变量)、[沙箱入口](#6-沙箱配置入口)、[模型回调](#7-模型回调连接)、[输出与端口](#8-输出端口与可选机器配置)、[配置检查](#9-检查最终配置)。

## 1. 配置放在哪里

日常使用只需一份外部 `.local.sh` 和它引用的一份沙箱 JSON。常用训练参数可以放在命令行，也可以保存到 `.local.sh` 的 `OVERRIDE_ARGS` 数组中。

| 配置位置 | 负责的内容 |
| --- | --- |
| 外部 `.local.sh` | Python、CANN、ATB、模型路径、NPU 分配、Agent 安装包、模型回调地址，以及确有需要的机器配置覆盖 |
| 外部 `sandboxes.json` | 沙箱服务地址、凭据文件路径、CPU 沙箱服务器连接、资源限制、容量和准备并发 |
| 入口选项 `--bundle`、`--agent` | 选择本次任务的数据包和智能体 |
| `--` 后的原生训练参数 | Batch、采样数、训练长度、学习率、路由策略和推理请求容量 |

配置模板分别为 [agenticrl.local.sh.example](../../../examples/common/agenticrl.local.sh.example) 和 [sandboxes.json.example](../../../examples/common/sandboxes.json.example)。以下命令在 **NPU 服务器的 `vime-forge` 目录**执行：

```bash
# 执行位置：NPU 服务器。
mkdir -p "$HOME/.config/vime"
cp -n examples/common/agenticrl.local.sh.example "$HOME/.config/vime/agenticrl.local.sh"
cp -n examples/common/sandboxes.json.example "$HOME/.config/vime/sandboxes.json"
```

复制后，先替换文件中的路径、地址和资源数量。`.local.sh` 是 Bash 配置文件，变量使用 `export`；`OVERRIDE_ARGS` 是 Bash 数组，无需 `export`。不要把它当作 dotenv 文件解析。通过 `--local` 加载即可，无需提前 `source` 另一份环境文件，也无需添加启动包装脚本。

没有路径前缀的 `--local` 名称会相对于 `scripts/torchtitan/common/` 查找；外部文件建议始终使用绝对路径。不指定 `--local` 时，启动器尝试读取该目录的 `local.sh`，文件不存在则使用默认配置。

## 2. 覆盖顺序与空值

不同配置采用不同的覆盖规则：

| 类型 | 生效规则 |
| --- | --- |
| 机器和 Agent 环境变量 | 先继承调用进程环境，再执行 `.local.sh` 中的赋值，最后补充未设置项的默认值。因此，文件内的直接赋值可以覆盖命令前设置的同名变量 |
| 数据包和 Agent 选择 | 显式 `--bundle`、`--agent` 优先于 `VIME_RL_PREPARED_BUNDLE`、`VIME_RL_AGENT` |
| 原生训练参数 | 脚本默认参数 → `OVERRIDE_ARGS` → `--` 后的参数；同名标量参数最后一次显式赋值生效 |
| 沙箱部署字段 | 对应环境变量的显式值 → JSON 中的字段 → 程序默认值。建议把部署字段集中在 JSON，不重复填写两处 |

例如，在 `.local.sh` 中设置常用参数：

```bash
# 配置位置：NPU 服务器上的外部 .local.sh。
OVERRIDE_ARGS=(
  --rollout-batch-size 16
  --n-samples-per-prompt 8
  --router-policy consistent_hash
)
```

命令行追加 `-- --rollout-batch-size 32` 后，本次使用 `32×8` 采样。若文件写了 `export VIME_RL_SAVE_INTERVAL=6`，则命令前的 `VIME_RL_SAVE_INTERVAL=off` 会被它覆盖；需要允许外部覆盖时，可以省略该赋值，或使用 `export VIME_RL_SAVE_INTERVAL="${VIME_RL_SAVE_INTERVAL:-6}"`。

沙箱内的 `VIME_AGENT_TASK_CONDA_SH` 和 `VIME_AGENT_TASK_CONDA_PREFIX` 同时留空时，任务使用镜像已有的 `PATH`。关闭入口默认的 checkpoint 或轨迹保存时，将对应的 `VIME_RL_SAVE_INTERVAL` 或 `VIME_RL_DUMP_MODE` 设为 `off`。

启动配置会传给 Ray 工作进程；沙箱配置在入口解析后固定，运行中编辑 JSON 不会改变已有任务。JSON 位于 NPU 服务器，CPU 沙箱服务器无需读取该文件。

## 3. NPU 服务器环境与设备

### 3.1 环境目录与模型

在第 1 节创建的 `.local.sh` 中填写训练和推理 Python 环境、CANN 和 ATB 安装目录，以及模型权重路径 `MODEL_PATH`。容器部署使用容器内可见的路径。单轮 RL 与 AgenticRL 共用这些配置，填写方式见[公共机器配置样例](../../../scripts/torchtitan/common/local.sh.example)。机器默认值见 [default.sh](../../../scripts/torchtitan/common/default.sh)，Python、Ray 和 vLLM 的执行路径由[公共启动器](../../../scripts/torchtitan/common/launcher.sh)推导。

AgenticRL 使用 `python3` 解析任务配置和沙箱 JSON。需要指定解析器时，在同一份 `.local.sh` 中设置 `VIME_JSON_PYTHON`。

### 3.2 NPU 分配

SWE-Gym 和 Terminal／TMax 使用异步训练，训练与推理分开占用 NPU。在 `.local.sh` 中用 `ASCEND_RT_VISIBLE_DEVICES` 列出本次可用的设备编号。训练占用 `TRAIN_TP × TRAIN_DP × TRAIN_CP` 个设备，推理占用 `ROLLOUT_GPUS` 个设备。每个推理引擎使用 `VLLM_RANKS_PER_ENGINE` 个设备，因此 `ROLLOUT_GPUS` 应为它的整数倍。

例如，使用 4 个 NPU 训练、另用 4 个 NPU 运行四个 TP=1 推理引擎时，设置 `TRAIN_TP=4 TRAIN_DP=1 TRAIN_CP=1 ROLLOUT_GPUS=4 VLLM_RANKS_PER_ENGINE=1`，并选取 8 个空闲设备。可直接参考 [AgenticRL 配置样例](../../../examples/common/agenticrl.local.sh.example)。

SWE-Gym 和 Terminal／TMax 的训练示例均使用 CP=1。模型支持的 CP 配置和单轮验证记录见[上下文并行说明](../../rl/qwen35.md#24-上下文并行cp)。

## 4. 数据包与实验参数

### 4.1 数据包和任务选择

| 配置入口 | 默认值 | 说明 |
| --- | --- | --- |
| `--bundle` / `VIME_RL_PREPARED_BUNDLE` | SWE：`prepared/swegym-293`；Terminal：`prepared/terminalbench`，均位于仓库 `vime-forge` 下 | 选择完整数据包；TMax 使用 Terminal 入口，并指定对应 bundle |
| `--agent` / `VIME_RL_AGENT` | `codex` | 支持 `codex`、`claude_code`，两个任务使用相同选择方式 |
| `VIME_SWEGYM_SOURCE_PATH` | `/home/developer/dependencies/SWE-Bench-Package-16dd480c` | 仅 SWE-Gym 需要的配套源码路径；应填写实际安装位置，Terminal／TMax 不使用 |
| `VIME_RL_PROMPT` | 使用各任务自带提示词 | 直接填写驱动 Agent 的提示词文本，bundle 中的任务数据保持原值 |

从 bundle 根目录自动推导 `rollout/train.no_gold.jsonl`、`environment/agentenv_templates.jsonl`、`verifier/verifier_specs.jsonl` 和 `preparation_manifest.json`。无需再为每个文件设置环境变量；入口会拒绝与 bundle 派生路径冲突的单文件覆盖。

### 4.2 训练参数优先使用命令行

下表参数放在入口命令的 `--` 后，也可以写入 `OVERRIDE_ARGS`。这些参数与单轮 RL 共享，下表列出当前两个任务入口的默认值。

| 用途 | 参数 | 当前默认值 |
| --- | --- | --- |
| 每批题目数 | `--rollout-batch-size` | `16` |
| 每题采样数 | `--n-samples-per-prompt` | `8` |
| 训练长度 | `--num-rollout` | `2`；统一使用该参数表达训练长度，按 epoch 折算后填入更新次数 |
| 学习率 | `--optimizer.param-groups.0.optimizer-kwargs.lr` | `1e-6`，来自 TorchTitan recipe |
| 采样温度 | `--rollout-temperature` | `1.0` |
| Top-p / Top-k | `--rollout-top-p` / `--rollout-top-k` | `1.0` / `-1`，其中 `-1` 表示不启用 Top-k 筛选 |
| 路由策略 | `--router-policy` | `cache_aware` |
| 推理容量系数 | `--vllm-server-concurrency` | 由 `VIME_RL_TRIAL_CONCURRENCY` 按推理引擎数推导；显式设置 `SERVER_CONCURRENCY` 或传入该参数即以其为准 |
| 单个引擎的调度容量 | `--vllm-max-num-seqs` | `64`；不同于轨迹并发和沙箱容量 |
| 推理显存比例 | `--vllm-gpu-memory-utilization` | `0.7` |
| 每卡训练 token 容量 | `--max-tokens-per-gpu` | 跟随上下文长度，与之分别配置时在 `--` 后传入 |

设备分配（`TRAIN_TP`、`TRAIN_DP`、`TRAIN_CP`、`ROLLOUT_GPUS`、`VLLM_RANKS_PER_ENGINE`）和训练 recipe（`TITAN_CONFIG`）是模型入口的 shell 变量，在 local 文件中设置，不使用 `VIME_RL_*` 名称。上下文长度是 AgenticRL 专有配置，只通过 `VIME_AGENT_MAX_CONTEXT_TOKENS` 调整，训练、推理与 Agent 三处限制随之一致。已退役的 `VIME_RL_TRAIN_TP`、`VIME_RL_ROLLOUT_BATCH_SIZE`、`VIME_RL_NUM_EPOCHS`、`VIME_RL_MAX_TOKENS_PER_GPU`、`VIME_RL_TITAN_CONFIG`、`VIME_RL_LEARNING_RATE` 等变量（含 `SWE_`、`TB_` 前缀的旧名）在启动时报错并提示替代写法，不会被静默忽略。

默认 `--num-steps-per-rollout 1`，global batch 根据最终采样参数推导：`16×8=128`，`32×8=256`。不必再独立填写相同的 global batch。训练长度统一由 `--num-rollout` 表达：模型入口已传入该参数，按 epoch 折算时，每个 epoch 的 rollout 数为数据量除以 batch size 后向下取整，乘以 epoch 数后填入 `--num-rollout`。上游的 `--num-epoch` 只在未设置 `--num-rollout` 时生效，与之同时出现会被忽略，因此两类入口都不使用它。

默认启用 prefix caching。需要关闭其默认配置时可设置 `VIME_RL_ENABLE_PREFIX_CACHING=0`。`TITAN_CONFIG` 可显式覆盖 recipe，所选配置需与模型及并行度匹配。

KL-Cov 默认关闭；需要时通过原生参数启用，例如 `--use-kl-cov --keep-old-actor --kl-cov-ratio 2e-3 --kl-cov-coef 1.0`。当前实现要求 CP=1。

轨迹并发按以下关系计算：

```text
总轨迹并发上限 = --vllm-server-concurrency × ROLLOUT_GPUS ÷ VLLM_RANKS_PER_ENGINE
```

四个 TP=1 引擎配置 `--vllm-server-concurrency 8` 时，总轨迹并发上限为 32。该值控制整体请求容量；每个引擎的实际负载由路由策略与请求分布决定。沙箱启动并发、每台主机容量和准备并发由 JSON 另行限制。

## 5. Agent 环境变量

### 5.1 上下文与任务执行预算

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VIME_AGENT_MAX_CONTEXT_TOKENS` | `32768` | Agent、训练序列和推理上下文长度的共同配置来源；应为正整数 |
| `VIME_AGENT_MAX_TURNS_PER_SESSION` | `72` | 每个会话通过模型适配器发起的请求次数上限；终端命令按 Agent 的工具调用执行 |
| `VIME_FORK_MERGE_MAX_RESPONSE_TOKENS` | `1024` | 轨迹处理时尝试合并短响应分支的 token 阈值；模型单次生成长度另行控制，通常保留此阈值 |
| `VIME_RL_AGENT_TIME_BUDGET_SEC` | `3600` | Agent 执行阶段的时间预算 |
| `VIME_RL_EVAL_TIMEOUT_SEC` | `3600` | 任务结果检查阶段的时间上限；任务数据中的预算仍可能更小 |
| `VIME_RL_ROLLOUT_GUARD_SEC` | `8400` | 整条轨迹的总时间限制，不会自动延长沙箱生命周期 |
| `VIME_AGENT_TASK_CONDA_SH` | SWE：`/opt/miniconda3/etc/profile.d/conda.sh`；Terminal：空 | 沙箱内的 Conda 初始化脚本 |
| `VIME_AGENT_TASK_CONDA_PREFIX` | SWE：`/opt/miniconda3/envs/testbed`；Terminal：空 | 沙箱内的任务环境目录；与上一项一起填写或一起设为空 |

修改上下文长度时使用 `VIME_AGENT_MAX_CONTEXT_TOKENS`，不要分别修改 `--training.seq_len`、`--vllm-max-model-len`、`--rollout-max-context-len`；入口会拒绝不一致的覆盖。

Conda 两项指向的是**沙箱内部路径**，与 NPU 服务器上的 Python 环境无关。TMax 默认使用镜像已有 `PATH`。从 SWE 配置切换到 TMax 时，如果 `.local.sh` 显式设置了 Conda 两项，需要根据 TMax 镜像调整。

### 5.2 安装包与准备过程

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VIME_AGENT_NODE_TARBALL` | `/mnt/workspace/agentenv-artifacts/node-v22.20.0-linux-x64.tar.xz` | NPU 服务器上的 Node 安装包路径；架构应与沙箱 CPU 一致 |
| `VIME_AGENT_CODEX_TARBALL` | `/mnt/workspace/agentenv-artifacts/vime-codex-offline-0.153.0.tgz` | Codex 安装包路径；按实际安装包位置填写 |
| `VIME_AGENT_CC_TARBALL` | 无 | Claude Code 安装包路径；选择该 Agent 且镜像未预装时需要提供 |
| `VIME_AGENT_CODEX_VERSION` | 从安装包文件名的 `-版本号.tgz` 后缀推导，无法推导时不固定版本 | 预装 Codex 复用和安装后检查所要求的版本；自定义包名建议显式填写 |
| `VIME_AGENT_ARTIFACT_BASE_URL` | 空 | 沙箱可访问的安装包 HTTP 服务地址；按 NPU 服务器配置的安装包文件名拼接下载地址。未配置时由 NPU 服务器上传到沙箱 |
| `VIME_AGENT_NPM_ONLINE` | 未设置，按离线模式安装 | 仅值为 `1` 时允许 npm 在线获取依赖 |
| `VIME_AGENT_NPM_INSTALL_TIMEOUT_SEC` | `300` | 每次 npm 安装尝试的执行预算，不包含准备队列等待时间 |
| `VIME_AGENT_WORKSPACE_PREP_TIMEOUT_SEC` | `60` | 创建 Agent 用户、设置工作区所有权等操作的超时；整个沙箱准备阶段另有总预算 |
| `VIME_AGENT_MARKER_TIMEOUT_SEC` | `15` | 单次查询后台命令完成标记的超时 |
| `VIME_AGENT_MARKER_TIMEOUT_RETRIES` | `3` | 只读完成标记查询连续遇到明确命令超时时，最多追加的重试次数；成功查询后计数重置 |

安装包路径可按实际存放位置设置。安装包架构以沙箱 CPU 架构为准。配置制品地址不会自动启动 HTTP 服务；安装器遇到明确的下载失败时可以回退到上传，因此通常仍应保留 NPU 服务器上的安装包。

标记查询重试只重新查询完成状态，不重新执行原命令，也不会延长上层时间预算。完整安装、初始化和分发流程见 [Agent 准备指南](harness-preparation.md)。

### 5.3 Codex 高级配置

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VIME_AGENT_CODEX_MODEL_PROFILE` | 由任务入口指定 | 写入 Codex 配置的模型标识；不负责下载或选择训练权重 |
| `VIME_AGENT_CODEX_CONTEXT_RESERVE_TOKENS` | `min(1024, max(1, 上下文长度 // 10))` | 从实际上下文中预留的 token 数；Codex 可见窗口为上下文长度减去该值 |
| `VIME_AGENT_CODEX_AUTO_COMPACT_TOKENS` | Codex 可见窗口的 80%，向下取整且至少为 1 | Codex 自动压缩上下文的阈值，必须小于其可见窗口 |
| `VIME_AGENT_CODEX_EXTRA_ARGS` | 空 | 追加给 `codex exec` 的命令行参数字符串 |
| `VIME_AGENT_CODEX_EXTRA_ENVS` | 空 | 注入沙箱内 Codex 进程的额外环境变量，格式为 JSON 对象 |

额外参数按命令行字符串拼接，应自行保证引用和转义正确。额外环境变量最后合并，会覆盖 Harness 注入的同名字段；不要无意覆盖模型回调地址和会话凭据。

### 5.4 Claude Code 高级配置

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VIME_AGENT_CC_TOOLS` | `Read,Edit,Write,Bash,Agent` | 传给 Claude Code 的可用工具列表；空值也使用默认列表 |
| `VIME_AGENT_CC_MAX_OUTPUT_TOKENS` | 不额外设置 | 注入 Claude Code 的最大输出 token 配置 |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | 跟随 Agent 上下文长度 | Claude Code 自动压缩使用的窗口 |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | `80` | 自动压缩触发比例 |
| `VIME_AGENT_CC_EXTRA_ARGS` | 空 | 追加给 Claude Code 的命令行参数字符串 |
| `VIME_AGENT_CC_EXTRA_ENVS` | 空 | 注入 Claude Code 进程的额外环境变量，格式为 JSON 对象，最后合并 |

两个任务都会传递所选 Agent 的高级配置，无需分别设置一套 SWE 或 TMax 参数。Agent 的模型回调和会话凭据由框架提供。

## 6. 沙箱配置入口

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VIME_AGENT_SANDBOX_BACKEND` | `agentenv` | 选择 `agentenv` 或 `harbor`；一次运行使用一个后端 |
| `VIME_AGENT_SANDBOX_CONFIG` | 无 | NPU 服务器上的沙箱 JSON 路径；推荐通过该文件集中管理部署信息 |
| `VIME_AGENT_SANDBOX_LIFETIME_SEC` | SWE：`4500`；Terminal：轨迹总预算加 `600`，默认 `9000` | 覆盖 JSON 的 `common.lifetime_sec`；生命周期与各阶段预算分别配置 |

JSON 使用 `common` 和所选后端的区域；可以同时保存两种后端配置，但启动时只解析所选后端。Harbor 不要求安装 AgentENV SDK 或读取其密钥。显式指定的 JSON 不可读、格式错误或缺少所选后端时会报错。

以下配置放在 JSON 中即可，无需在 `.local.sh` 重复声明：

| JSON 区域 | 内容 |
| --- | --- |
| `common` | 生命周期 `lifetime_sec`、启动准备并发 `boot_concurrency`、创建尝试次数 `boot_retries` |
| `agentenv` | API 与沙箱访问地址、凭据文件、SDK 路径、模板选择和资源配置 |
| `harbor` | CPU 沙箱服务器连接、Docker 资源、镜像仓库、容量与准备队列配置 |

`common.boot_concurrency` 默认 `8`，`common.boot_retries` 默认 `3`，后者包含首次尝试。示例 JSON 显式填写 `lifetime_sec=9000`，会覆盖任务默认生命周期；示例中的主机容量和准备并发同样按该部署填写，程序默认值见字段说明。

AgentENV 凭据以 `agentenv.api_key_file` 指向文件，正式启动时加载；不要把密钥内容写进 JSON。CPU 沙箱服务器上的 Python、工作进程和状态目录在 Harbor 的 `hosts` 中填写，不使用 NPU 服务器的 `VIME_TRAIN_ENV`。

完整字段、部署前提、容量与排队规则见[沙箱部署与配置](sandboxes.md)，可参考[沙箱 JSON 示例](../../../examples/common/sandboxes.json.example)填写。本页维护公共环境变量入口，沙箱部署参数统一由该指南解释。

## 7. 模型回调连接

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `ADAPTER_BIND_HOST` | `0.0.0.0` | 模型适配器在 NPU 服务器上监听的地址 |
| `ADAPTER_PORT` | `18001` | 模型适配器在 NPU 服务器上监听的端口 |
| `ADAPTER_PUBLIC_HOST` | `192.0.2.1` | 沙箱访问适配器所用的主机名或 IP；默认值是占位地址，必须按实际网络填写 |
| `ADAPTER_PUBLIC_PORT` | `18080` | 沙箱访问入口的端口；直连时应与实际监听端口一致，使用转发时填写转发入口端口 |

这四项分别配置适配器监听端点与沙箱访问端点，两端需配套设置。直连示例：

```bash
# 配置位置：NPU 服务器上的外部 .local.sh。
export ADAPTER_BIND_HOST=0.0.0.0
export ADAPTER_PORT=18001
export ADAPTER_PUBLIC_HOST=model-callback.example  # 替换为沙箱可达的 NPU 服务器地址
export ADAPTER_PUBLIC_PORT=18001
```

沙箱控制连接、模型回调和安装包下载是三条不同的路径。分别检查 NPU 到 Harbor CPU 服务器的 SSH 连接、容器到模型适配器的访问，以及安装包下载地址 `VIME_AGENT_ARTIFACT_BASE_URL`。启动器不会自动建立隧道或启动制品服务。

## 8. 输出、端口与可选机器配置

### 8.1 输出与保存

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `RUN_NAME` | `<入口前缀>-<任务名>-<时间戳>` | 任务名为 `swegym` 或 `terminalbench`，TMax 使用后者 |
| `OUTPUT_DIR` | `<vime-forge>/outputs/$RUN_NAME` | 本次运行输出目录；正式启动时创建，已存在则拒绝复用 |
| `VIME_RL_SAVE_INTERVAL` | `6` | checkpoint 保存间隔，单位为 rollout；`off` 关闭入口默认添加的保存参数 |
| `VIME_RL_DUMP_MODE` | `rollout` | 默认保存轨迹；`off` 不添加默认轨迹保存参数 |

checkpoint 默认位于 `$OUTPUT_DIR/checkpoints`，启用保存时保留最近 2 份；最后一批也会触发保存。日志、命令归档和 TensorBoard 位于 `$OUTPUT_DIR/tensorboard/$RUN_NAME/<时间戳>`。轨迹默认位于 `$OUTPUT_DIR/rollout_dumps`，可以通过入口选项 `--dump-dir` 改变位置；`--replay-dir` 用于读取已保存轨迹，与 `--dump-dir` 互斥。

`off` 控制入口是否添加保存参数；用户在 `OVERRIDE_ARGS` 或命令行中添加的保存参数仍会生效。`VIME_WORKSPACE` 不会改变 AgenticRL 的 `OUTPUT_DIR`，输出根目录应直接用 `OUTPUT_DIR` 设置。

### 8.2 通信端口与缓存

同一台 NPU 服务器运行多个任务时，为每个任务设置独立的 `VIME_SLOT` 和输出目录，另行分配可见设备、Ray worker 端口范围和模型回调端口。Ray 服务端口、HCCL 端口范围和缓存目录的推导规则见 [default.sh](../../../scripts/torchtitan/common/default.sh)。

`MASTER_ADDR` 指向 Ray head。沙箱访问模型使用 `ADAPTER_PUBLIC_HOST` 和 `ADAPTER_PUBLIC_PORT`，按[模型回调连接](#7-模型回调连接)填写沙箱可达的地址和端口。

## 9. 检查最终配置

先填写外部 local 文件和沙箱 JSON。启动时通过 `--local` 读取机器配置、通过 `--bundle` 选择数据包；实验参数写在 `--` 后。

如需单独查看任务配置，在 vime-forge 目录执行：

```bash
source /path/to/agenticrl.local.sh
VIME_RL_PREPARED_BUNDLE=/path/to/prepared/bundle \
python3 vime/agent/config.py shell --task swegym --repo-root .
```

Terminal 任务将 `--task` 改为 `terminalbench`。该命令输出配置赋值，不激活环境、不创建运行目录、不读取密钥、不访问服务器；它不包含 local 文件中的非导出变量或 `OVERRIDE_ARGS`，也不替代最终训练命令及实际连通性验证。
