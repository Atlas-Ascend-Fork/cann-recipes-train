# 沙箱部署与配置

SWE-Gym 与 Terminal／TMax 共用沙箱配置，一次训练选择 AgentENV 或 Harbor 中的一个后端。本工作默认配置基于 NPU 服务器和分离的 CPU 沙箱服务器：外部配置、SDK 和控制命令位于 NPU 服务器；沙箱服务、任务镜像和实际沙箱位于 CPU 沙箱服务器。本文说明服务部署前提、JSON 字段、并发限制及网络连接；NPU 服务器环境变量见 [configuration.md](configuration.md) Agent Harness的安装和准备见 [Agent 安装与初始化](harness-preparation.md)。

## 1. 选择后端并保存配置

在 **NPU 服务器**上、仓库外的 `.local.sh` 中填写：

```bash
# 配置位置：NPU 服务器上的外部 .local.sh。
export VIME_AGENT_SANDBOX_BACKEND=harbor
export VIME_AGENT_SANDBOX_CONFIG="$HOME/.config/vime/sandboxes.json"
```

`VIME_AGENT_SANDBOX_BACKEND` 接受 `harbor` 或 `agentenv`，默认值为 `agentenv`。`VIME_AGENT_SANDBOX_CONFIG` 指向 NPU 服务器上的 JSON 文件；CPU 沙箱服务器不需要读取这个文件。

JSON 分为 `common`、`agentenv` 和 `harbor` 三个区域。可以同时保存两种后端，也可以只保留所选后端。启动器只解析公共区域和所选后端，选择 Harbor 时不会加载 AgentENV SDK 或读取其密钥。显式指定的 JSON 不可读、格式错误或缺少所选后端时会报错。

配置在启动时解析并传给工作进程；运行中编辑 JSON 不会改变已启动任务的配置。字段对应的沙箱环境变量若已显式设置，其值优先于 JSON；日常使用建议集中在 JSON 中填写，避免重复配置。`.local.sh` 的加载方式与覆盖顺序见[配置职责](configuration.md#1-配置放在哪里)。

下面是保存在 **NPU 服务器 `sandboxes.json`** 中的配置示例，描述两台 Harbor CPU 沙箱服务器，同时保留 AgentENV 配置供切换使用：

```json
{
  "common": {
    "lifetime_sec": 9000,
    "boot_concurrency": 8,
    "boot_retries": 3
  },
  "agentenv": {
    "api_url": "http://agentenv.example:18000",
    "sandbox_url": "http://agentenv.example:18000",
    "api_key_file": "/path/on/npu/agentenv-api-key",
    "sdk_path": "",
    "selector_mode": "template",
    "size": "md",
    "rpc_retries": 6
  },
  "harbor": {
    "environment": "remote_docker",
    "hosts": [
      {
        "ssh": "sandbox@cpu-a.example",
        "ssh_args": ["-p", "22", "-i", "/path/on/npu/sandbox-key", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes"],
        "python": "/opt/vime-harbor/venv/bin/python",
        "worker": "/opt/vime-harbor/harbor_worker.py",
        "root": "/mnt/harbor-data/vime",
        "slots": 16,
        "docker_compose": []
      },
      {
        "ssh": "sandbox@cpu-b.example",
        "ssh_args": ["-p", "22", "-i", "/path/on/npu/sandbox-key", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes"],
        "python": "/opt/vime-harbor/venv/bin/python",
        "worker": "/opt/vime-harbor/harbor_worker.py",
        "root": "/mnt/harbor-data/vime",
        "slots": 16,
        "docker_compose": []
      }
    ],
    "cpus": 16,
    "memory_mb": 65536,
    "registry": "",
    "startup_timeout": 600,
    "acquire_timeout": 600,
    "marker_cancel_grace": 30,
    "prepare_concurrency": 4,
    "prepare_acquire_timeout": 1800,
    "prepare_queue_limit": 32
  }
}
```

本例每台主机设置 16 个沙箱名额、4 个准备名额。仓库中的 [sandboxes.json.example](../../../examples/common/sandboxes.json.example) 使用每台 64 个沙箱名额、8 个准备名额；两者都是部署示例，并非程序默认值或固定的资源要求。SSH 端口由实际服务决定，使用其他端口时修改 `ssh_args` 中 `-p` 后的值。

## 2. 部署前提与自动执行范围

### 2.1 AgentENV

NPU 服务器上的轨迹生成进程通过兼容的 [E2B SDK 接口](../../../examples/swe_agent_rl/sandbox.py) 访问AgentENV服务。从零开始的AgentENV 准备请参考 [AgentENV 部署步骤](agentenv-deployment.md#1-部署-agentenv-服务)。

两个训练入口使用 数据集的 bundle 中抽取每个任务镜像模板，因此 `selector_mode` 应保持为 `template`。模板 ID 必须在当前 AgentENV 部署中有效。

API 密钥保存在 NPU 服务器上的单独文件中，JSON 只记录路径。正式启动时由启动器读取，交给 Ray 主进程继承，不写入配置快照或配置查看输出。

### 2.2 Harbor

本仓库的 Harbor 适配器使用 `remote_docker`：NPU 服务器通过 SSH 启动 CPU 沙箱服务器上的 [harbor_worker.py](../../../vime/agent/harbor_worker.py)，由该进程调用 Harbor 管理本机 Docker 容器。它不要求额外部署一个对外监听的 Harbor API 服务，也不是同名的容器镜像仓库产品。

每台 CPU 沙箱服务器需要准备：

- Docker Engine、Docker Compose v2，以及与任务镜像匹配的 CPU 架构。
- 独立的 Harbor Python 环境。当前工作进程明确检查 `harbor==0.24.0`，不能任意替换版本。
- 与训练端代码配套的 `harbor_worker.py`，以及 SSH 账号可写、支持文件锁的本机状态目录。
- 可供该账号使用的 Docker 权限，以及 NPU 服务器到该主机的非交互式 SSH 登录。
- bundle 指定的任务镜像，以及所选 Agent 所需的容器运行配置。

例如，在已经创建的 CPU 沙箱服务器 Python 环境中安装固定版本：

```bash
# 执行位置：CPU 沙箱服务器宿主环境。
/opt/vime-harbor/venv/bin/python -m pip install 'harbor==0.24.0'
```

`hosts[].python` 使用 CPU 沙箱服务器上的环境路径，与 NPU 服务器的 `VIME_TRAIN_ENV`、`VIME_VLLM_ENV` 无关。NPU 服务器只需具备 SSH 客户端和本仓库的适配器，不需要为了连接远端 Docker 安装 Harbor 包。

容器运行参数可通过 `docker_compose` 引用主机上的覆盖文件，例如配置所选 CLI 需要的运行环境。启动器不会安装这些文件或修改主机的容器安全策略。验证时应运行实际使用的 Agent 命令；仅能启动容器、执行普通 Shell 命令，不足以证明 Agent 的运行模式可用。

### 2.3 镜像与 bundle

两个后端继续使用数据 bundle 中的 `environment/agentenv_templates.jsonl`。该映射同时包含template ID 和源镜像信息。

- AgentENV 选择映射中的 `template_id`。
- Harbor 选择 `source_image`，要求使用 `镜像仓库/镜像名@sha256:摘要`，摘要为 64 位小写十六进制字符串；仅有镜像 tag 不满足要求。

切换后端前，需要确认映射中具备相应信息，并且目标服务中确实存在模板或镜像。Harbor CPU 沙箱服务器应提前准备各自可能执行的任务镜像；当前主机选择逻辑不按镜像是否缓存分配任务。

`registry` 只改变镜像仓库地址，保留镜像路径和 digest，不生成新 bundle，也不建立镜像加速或复制服务。目标仓库必须能够提供同一摘要对应的镜像。CPU 沙箱服务器之间已有的私网镜像复制可以继续使用，与沙箱控制连接分别配置。

启动器会按需创建任务沙箱、调用 Harness 准备 CLI 和工作区，并在使用结束后释放沙箱。Docker、SSH、镜像分发、网络转发和安装包服务需要预先准备，不由训练入口部署。

## 3. sandbox 配置文件的字段说明

### 3.1 公共配置 `common`

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `lifetime_sec` | SWE-Gym：`4500`；Terminal／TMax：轨迹总预算加 `600`，默认 `9000` | 单个沙箱会话的生命周期，与 Agent 执行和轨迹总预算分别配置。Harbor 从远端工作进程开始运行时计时，可能包含排队时间；不是从 Agent 开始执行时才计时 |
| `boot_concurrency` | `8` | 每个轨迹生成进程同时进入沙箱启动准备流程的上限，包括等待 Harbor 准备队列、创建沙箱、安装 CLI 和设置工作区；它不是各 CPU 沙箱服务器共享的总上限 |
| `boot_retries` | `3` | 沙箱启动准备失败后的最多尝试次数，包含首次尝试。重试使用新沙箱，不重放旧沙箱中执行状态未知的命令；准备排队超时直接上报，不重新入队刷新等待预算 |

`VIME_AGENT_SANDBOX_LIFETIME_SEC` 可覆盖 `common.lifetime_sec`。上面的示例显式设置 `9000`，因此两个任务均使用该值；它不会随其他预算的调整而自动变化。

### 3.2 AgentENV 的配置字段

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `api_url` | `http://127.0.0.1:18000` | 管理 API 地址，用于创建等沙箱管理操作；应从 NPU 服务器可达 |
| `sandbox_url` | `http://127.0.0.1:18000` | SDK 访问沙箱命令和文件接口的地址；可与 `api_url` 相同，以服务实际部署为准 |
| `api_key_file` | `/home/developer/.config/agentenv/api-key` | NPU 服务器上的 API 密钥文件路径；文件中保存密钥，JSON 中只填写路径 |
| `sdk_path` | `/home/developer/.venvs/agentenv-harness/lib/python3.11/site-packages` | SDK 的额外 Python 包搜索目录，不是 Python 可执行文件。SDK 已安装在训练环境时设为 `""`；填写目录时应保证依赖与当前 Python 兼容 |
| `selector_mode` | `template` | `template` 将环境标识作为模板 ID；`metadata` 将环境标识放进请求元数据。当前两个 bundle 训练入口要求使用 `template` |
| `image_metadata_key` | 无 | `metadata` 模式使用的环境选择字段名，必须与服务端约定一致；`template` 模式不使用 |
| `size` | `md` | `metadata` 模式传给服务端的规格名称，具体 CPU、内存由服务端定义；`template` 模式不传此字段，资源由模板决定 |
| `rpc_retries` | `6` | 可重试远程调用的最多尝试次数，包含首次尝试；仅适用于允许重试的操作和可恢复的通信错误，不会让所有失败命令自动重跑 |

表中的本机地址和绝对路径是当前入口默认值，不是部署要求，应按实际服务填写。AgentENV 的资源规格与 Harbor 的 CPU、内存配置各自解释，不做自动换算；其 RPC 重试设置也不作用于 Harbor。

### 3.3 Harbor 的配置字段

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `environment` | `remote_docker` | 当前适配器唯一支持的运行方式，通过 SSH 控制远端 Docker |
| `hosts` | 无，必填 | CPU 沙箱服务器数组，至少一项；每项字段见下一节 |
| `cpus` | `16` | 每个沙箱容器的 CPU 资源配置，适用于列表中的所有主机 |
| `memory_mb` | `65536` | 每个沙箱容器的内存限制，单位为 MB，适用于列表中的所有主机 |
| `registry` | `""` | 可选镜像仓库，格式为主机名或 `主机名:端口`，不含协议或路径。为空时使用 bundle 中的原始地址；非空时替换仓库地址并保留镜像路径、摘要 |
| `startup_timeout` | `600` | 等待远端处理一次沙箱创建及 Docker 启动请求的时间；不包括后续 CLI 安装和工作区准备 |
| `acquire_timeout` | `600` | 未使用准备队列时，反复尝试获取空闲沙箱名额的等待预算。单次启动请求另受 `startup_timeout` 限制，所以不能将其视为整个创建过程的严格总时限 |
| `marker_cancel_grace` | `30` | 上层取消命令完成状态查询时，额外等待该只读查询返回的时间，用于收完已有响应，避免下一次请求读到上一条响应。它不延长 Agent 执行预算，也不重新执行原命令 |
| `prepare_concurrency` | `0` | 每台主机同时处于准备阶段的沙箱数；准备包括创建容器、安装 CLI 和设置工作区。`0` 表示关闭额外的准备队列限制，不表示禁止创建沙箱 |
| `prepare_acquire_timeout` | `600` | 启用准备队列后，从选择主机到获准开始准备的等待预算，包含加入队列和排队；不包括获准后的创建、安装和工作区准备时间 |
| `prepare_queue_limit` | `64` | 每台主机准备队列允许的等待请求数，不包含已开始准备的沙箱。队列满时尝试其他主机或继续等待，仍受 `prepare_acquire_timeout` 限制 |

Harbor 的 CPU、内存和各超时字段应填写正整数；`prepare_concurrency` 可以为 `0`，`prepare_queue_limit` 必须为正整数。资源和容量需结合 CPU 沙箱服务器的实际负载设置，增加 `slots` 不会增加物理 CPU、内存或磁盘带宽。

`marker_cancel_grace` 只适用于特定的只读完成状态查询。SSH 断开或命令执行状态未知时，适配器不会自动重放命令；调大该字段也不能修复 SSH 连接。

### 3.4 Harbor CPU 沙箱服务器配置 `hosts[]`

| 字段 | 必填／默认值 | 含义 |
| --- | --- | --- |
| `ssh` | 必填 | 从 NPU 服务器使用的 SSH 目标，例如 `sandbox@cpu-a.example`，也可使用已配置的 SSH 主机别名 |
| `ssh_args` | 可选，`[]` | SSH 参数数组，每个参数和参数值分开填写。私钥、SSH 配置和主机指纹文件位于 NPU 服务器；端口应与 CPU 沙箱服务器的 SSH 服务一致 |
| `python` | 必填 | CPU 沙箱服务器上运行 Harbor 工作进程的 Python 绝对路径，所在环境须安装配套 Harbor 版本 |
| `worker` | 必填 | CPU 沙箱服务器上 `harbor_worker.py` 的绝对路径；需与训练端适配器配套 |
| `root` | 必填 | CPU 沙箱服务器上的状态目录绝对路径，保存会话、上传临时文件、容量记录及准备队列。它不是 Docker 镜像存储目录 |
| `slots` | 必填，正整数 | 该主机、该 `root` 下允许同时占用的沙箱名额数，包含正在准备和已经运行的沙箱 |
| `docker_compose` | 可选，`[]` | Docker Compose 覆盖文件的绝对路径数组；文件必须存在于对应 CPU 沙箱服务器，按列表顺序传给 Harbor |

适配器以非交互方式建立 SSH 连接，固定设置 `BatchMode=yes`、`ConnectTimeout=10`、`ServerAliveInterval=15`、`ServerAliveCountMax=3`。应预先配置免交互认证并核对主机指纹，不要依赖启动时手工输入密码。准备队列的超时参数不会改变这些 SSH 设置。

同一主机、同一 `root` 下的工作进程共享容量记录和准备队列，应使用一致的 `slots`、准备并发和队列配置。不同 `root` 对应独立的计数范围，不能用各自的上限代替整台机器的资源规划。

## 4. 理解容量、并发和排队

这些限制作用于不同阶段：

| 配置 | 控制什么 |
| --- | --- |
| 轨迹并发 | 同时处理多少条轨迹，其中可能有轨迹正在等待沙箱 |
| `common.boot_concurrency` | 单个生成进程同时进行多少个启动准备流程 |
| `harbor.hosts[].slots` | 每台 CPU 沙箱服务器允许占用多少个沙箱名额 |
| `harbor.prepare_concurrency` | 每台 CPU 沙箱服务器允许多少个沙箱同时创建容器、安装 CLI 和准备工作区 |

以本页 JSON 为例，两台主机各有 16 个沙箱名额，合计最多占用 32 个；每台最多有 4 个沙箱同时准备。准备完成后，沙箱继续占用沙箱名额，但立即让出准备名额，后续请求便可开始准备。生成进程的 `boot_concurrency=8` 还会限制进入这一流程的请求数量。

启用准备队列后，每台主机按照请求进入队列的先后顺序处理：先到的先开始，后到的继续等待。轮到某个请求时，还必须同时有空闲的沙箱名额和准备名额，才会开始创建容器。等待期间不提前占住其中一种名额。

轨迹并发由RL rollout 的采样力度决定。当前入口根据推理引擎数和 `--vllm-server-concurrency` 计算它；例如 4 个引擎、每个并发 8，对应总并发 32。`--rollout-batch-size 32 --n-samples-per-prompt 8` 则表示一批共 256 条轨迹，可在并发 32 下分批完成，不要求同时创建 256 个沙箱。

排队超时与执行超时分开处理：`prepare_acquire_timeout` 限制获准准备之前的等待，`startup_timeout` 限制容器启动；CLI 安装和工作区准备另有 [Agent 准备参数](configuration.md#52-安装包与准备过程)。延长排队预算不会加快安装，也不会延长一次 SSH 连接或命令执行的超时。

## 5. 沙盒环境，Agent Harness，和 vime 引擎之间的网络互联

| 路径 | 方向与用途 | 配置位置 |
| --- | --- | --- |
| 沙箱控制 | NPU 服务器 → AgentENV 服务，或 NPU 服务器 → Harbor CPU 沙箱服务器；负责创建沙箱、执行命令和读写文件 | `agentenv.api_url`、`agentenv.sandbox_url` 或 `harbor.hosts` |
| Agent 调用模型请求 | CPU 沙箱服务器上的沙箱内 Agent → NPU 服务器上的模型适配器，再由适配器经路由器请求 vLLM 引擎 | `ADAPTER_*`，放在 `.local.sh` |
| 安装包下载 | 沙箱 → HTTP 制品服务；未配置服务时，由轨迹生成工作进程通过沙箱文件接口上传 | `VIME_AGENT_ARTIFACT_BASE_URL` 与安装包路径，放在 `.local.sh` |

三条路径要分别可用。SSH 登录成功只证明控制连接可用；必须从实际沙箱容器内确认模型回调及安装包下载，不能用 CPU 沙箱服务器宿主环境上的连通性代替。

例如，沙箱可以直接访问 NPU 服务器的适配器时，在 `.local.sh` 中设置：

```bash
# 配置位置：NPU 服务器上的外部 .local.sh。
export ADAPTER_BIND_HOST=0.0.0.0
export ADAPTER_PORT=18001
export ADAPTER_PUBLIC_HOST=model-callback.example
export ADAPTER_PUBLIC_PORT=18001
```

`ADAPTER_BIND_HOST`、`ADAPTER_PORT` 表示 adapter 的监听位置；`ADAPTER_PUBLIC_HOST`、`ADAPTER_PUBLIC_PORT` 表示沙箱环境调用 NPU 上rollout 引擎的地址和端口，实际调用关系如下:

```
沙箱内 Codex / Claude Code
    → http://ADAPTER_PUBLIC_HOST:ADAPTER_PUBLIC_PORT
    → NPU 上的 Adapter
    → vLLM Router
    → 推理引擎
```

## 6. 相关代码路径和功能

| 内容 | 代码 |
| --- | --- |
| JSON 解析、默认值与配置快照 | [config.py](../../../vime/agent/config.py) |
| 沙盒选择 | 各任务的 [swe_agent_rl/sandbox.py](../../../examples/swe_agent_rl/sandbox.py)、[terminal_agent_rl/sandbox.py](../../../examples/terminal_agent_rl/sandbox.py)；后端名称与校验在 [config.py](../../../vime/agent/config.py) |
| AgentENV 适配器 | [sandbox.py](../../../vime/agent/sandbox.py) |
| Harbor 连接、排队与取消处理 | [harbor_sandbox.py](../../../vime/agent/harbor_sandbox.py) |
| CPU 沙箱服务器上的队列、容量和 Docker 操作 | [harbor_worker.py](../../../vime/agent/harbor_worker.py) |

任务数据准备分别见 [SWE-Gym 数据准备](../swe-agentic-rl/data-preparation.md)和 [Terminal／TMax 数据准备](../terminal-agentic-rl/data-preparation.md)。启动步骤分别见 [SWE-Gym 训练指南](../swe-agentic-rl/training.md)与 [Terminal／TMax 训练指南](../terminal-agentic-rl/training.md)。
