# Terminal／TMax 训练数据bundle准备

本文说明如何使用当前仓库的 `prepare_terminalbench.py` 将 Terminal-Bench 或 TMax 原始数据转换、筛选并导出为 AgenticRL 数据包（bundle），分别介绍 AgentENV 与 Harbor 的准备要求。

已有完整 bundle 时，可以直接阅读[复用与后续训练](#4-复用与后续训练)。沙箱服务部署见[沙箱指南](../env-and-config/sandboxes.md)，Agent 安装见 [Harness 指南](../env-and-config/harness-preparation.md)。数据准备不需要启动策略模型、安装 Agent CLI 或建立模型回调。

本文默认已具备 NPU 服务器和 CPU 沙箱服务器。下文的数据转换、筛选和 bundle 导出命令均在 **NPU 服务器的 `vime-forge` 目录**执行；任务镜像保存在 CPU 沙箱服务器，沙箱及任务检查也在那里运行。

## 1. 先选择准备路径

| 目标后端 | 当前支持的准备方式 | 预先需要什么 |
| --- | --- | --- |
| AgentENV | 转换和筛选任务，再通过 `gen-bundle` 发现或指定模板、探测并导出 | 对应任务镜像及 AgentENV 模板、服务地址与凭据 |
| Harbor | 转换和筛选任务，再由 `gen-bundle` 直接创建 Harbor 沙箱进行探测并导出 | 对应任务 OCI 镜像、Harbor CPU 沙箱服务器配置；不需要 AgentENV 模板、SDK 或凭据 |

Terminal 准备器支持两种后端，但**不构建任务 OCI 镜像**。`--image-template` 只生成镜像引用，不会执行 TMax 的 `container_def`、下载缺失外部文件或构建任务环境。镜像与所依赖资源应先准备好，并与沙箱 CPU 架构匹配。

## 2. Bundle 结构与准备环境

### 2.1 目录与文件

```text
<bundle>/
├── rollout/train.no_gold.jsonl
├── environment/agentenv_templates.jsonl
├── verifier/verifier_specs.jsonl
├── preparation_manifest.json
├── qualification/                       # 准备检查记录，按具体流程生成
└── state/                               # 模板发现缓存，按具体流程生成
```

| 文件 | 内容 |
| --- | --- |
| `rollout/train.no_gold.jsonl` | 任务指令和环境元数据，不包含隐藏测试正文 |
| `environment/agentenv_templates.jsonl` | 按 `task_id` 记录镜像摘要、工作目录及可选模板 ID |
| `verifier/verifier_specs.jsonl` | 保存任务检查命令、测试正文、测试摘要、时间预算和环境绑定 |
| `preparation_manifest.json` | 记录来源、源文件摘要、任务数量与准备状态；字段按后端有所不同 |

环境映射文件名对两个后端保持不变。AgentENV 通过有效模板 ID 创建实例，Harbor 通过 `source_image` 中的 `镜像名@sha256:摘要` 创建容器。

仅供 Harbor 使用的 bundle 可以包含 `template_id: null`，这种 bundle 不能直接切换到 AgentENV。已有模板 ID 时，也应保持环境映射与检查规格中的绑定一致。

### 2.2 命令执行位置与依赖

NPU 服务器保存仓库代码、源数据和导出的 bundle，并运行 `prepare_terminalbench.py`。使用 AgentENV 时，脚本通过服务 API 控制 CPU 沙箱服务器上的沙箱；使用 Harbor 时，通过 SSH 调用 CPU 沙箱服务器上的工作进程。工作目录、测试入口和原始环境检查均在沙箱内执行，结果返回 NPU 服务器用于导出 bundle。

`python` 填写 NPU 服务器上满足下列依赖的 Python 路径；现有训练环境满足要求时可直接使用。

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
python -m examples.terminal_agent_rl.prepare_terminalbench --help
```

NPU 服务器上的数据准备环境建议使用 Python 3.12，并按要执行的阶段准备依赖：

| 阶段 | NPU 服务器依赖 |
| --- | --- |
| JSONL 转换、筛选和抽样 | 准备脚本在这些路径主要使用 Python 标准库 |
| `from-tasks` | 额外需要 PyYAML，读取 `task.yaml` |
| TMax Parquet 输入 | 额外需要 `pyarrow`；JSONL 输入不需要 |
| AgentENV 模板探测 | 兼容的 E2B SDK；已有部署参考使用 `e2b==2.46.4` |
| Harbor 探测 | SSH 客户端、本仓库沙箱适配器和外部 JSON；Harbor Python 包与 Docker 安装在 CPU 沙箱服务器 |

当前 Terminal 准备器有一处帮助文本包含未转义的 `%files`。Python 3.14 会在创建参数解析器时因此报错，暂时不要用该版本运行 Terminal 准备命令。

`--help` 能返回不代表在线依赖已齐全。服务首次部署与连通性步骤见 [AgentENV 部署指南](../env-and-config/agentenv-deployment.md)和 [Harbor 部署要求](../env-and-config/sandboxes.md#22-harbor)。

## 3. 准备 Terminal／TMax bundle

### 3.1 转换原始任务

> 本工作使用 [TMax-15K](https://huggingface.co/datasets/allenai/TMax-15K) 数据集进行 AgenticRL训练

[prepare_terminalbench.py](../../../examples/terminal_agent_rl/prepare_terminalbench.py) 提供两条转换路径，按数据来源选择其中一条。

Terminal-Bench 输入应是包含 `task.yaml` 与 `tests/run-tests.sh` 的任务目录树：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
export TMAX_PREP_SOURCE=/path/to/preparation/terminal-source.jsonl

python -m examples.terminal_agent_rl.prepare_terminalbench from-tasks \
  --tasks-root /path/to/terminalbench/tasks \
  --output "$TMAX_PREP_SOURCE" \
  --image-template 'registry.example/terminalbench/{task_id}:x86_64'
```

该命令读取任务指令、工作目录和测试目录中的文本文件。任务未在 YAML 中声明镜像时才使用 `--image-template`。它不接受任意其他版本的任务目录格式，也不会构建镜像。

TMax 输入可以是原始 Parquet 或 JSONL，需要包含 `task_id`、`container_def`、`description` 和 `test_final_state`：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
export TMAX_PREP_SOURCE=/path/to/preparation/tmax-source.jsonl

python -m examples.terminal_agent_rl.prepare_terminalbench from-tmax \
  --source /path/to/data/tmax-data.parquet \
  --output "$TMAX_PREP_SOURCE" \
  --image-template 'registry.example/tmax/{task_id}:x86_64' \
  --keep-arch-pinned
```

转换结果包含 `task_id`、`instruction`、`image`、`workdir`、`run_tests_cmd`、`timeout_sec` 和测试正文；TMax 分类信息保存在 `metadata`。`from-tmax` 默认会过滤不完整、重复、依赖 `%files` 外部文件或固定二进制的记录。

### 3.2 选择任务子集

结合实际需求，需要复现一个数据集子集时，应使用原来的任务 ID 列表，不能仅用相同数量重新抽样。对于新子集，可在转换后的 source JSONL 上抽样，并同时保存 ID 列表 (此处以筛选2048条数据为例)：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
export TMAX_PREP_SUBSET=/path/to/preparation/tmax-subset.jsonl
export TMAX_PREP_LABELS=/path/to/preparation/tmax-task-ids.txt

python -m examples.terminal_agent_rl.prepare_terminalbench sample-tasks \
  --source "$TMAX_PREP_SOURCE" --output "$TMAX_PREP_SUBSET" \
  --labels "$TMAX_PREP_LABELS" --total 2048 --seed 42 --strategy proportional
```

`proportional` 按 `metadata.domain` 的数量比例分配样本，缺少完整领域信息时回退为均匀抽样。源记录不足时输出全部可用记录，不保证凑足 2048 条；任务后续还可能在准备检查中被排除。固定源文件、参数和随机种子，并保存输出列表，才能复现选择结果。

若已有目标task列表，可改用以下命令，与随机抽样二选一：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
python -m examples.terminal_agent_rl.prepare_terminalbench filter-tasks \
  --source "$TMAX_PREP_SOURCE" --output "$TMAX_PREP_SUBSET" \
  --include-task-ids /path/to/existing-task-ids.txt \
  --labels "$TMAX_PREP_LABELS"
```

`filter-tasks` 也支持 `--exclude-task-ids` 和 `--where metadata.domain=...` 等等值条件。筛选后应核对输出 ID 与目标列表；该命令不会因某个目标 ID 不在 source 中而自动补齐。不要只改最终 bundle 的一份 JSONL，应在准备阶段筛选后重新导出相互对应的文件。

### 3.3 使用 Harbor 生成 bundle

先在 NPU 服务器配置后端和可用的沙箱 JSON，再对所选 source 生成新的 bundle。JSON 和输出目录均位于 NPU 服务器，JSON 中的 `hosts` 指向 CPU 沙箱服务器：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
export VIME_AGENT_SANDBOX_BACKEND=harbor
export VIME_AGENT_SANDBOX_CONFIG="$HOME/.config/vime/sandboxes.json"
export TMAX_PREP_BUNDLE=/path/to/prepared/tmax-harbor

python -m examples.terminal_agent_rl.prepare_terminalbench gen-bundle \
  --source "$TMAX_PREP_SUBSET" --bundle "$TMAX_PREP_BUNDLE" \
  --instances "$TMAX_PREP_LABELS" --qualify
```

此路径读取所选任务的镜像引用，解析 tag 为 digest，使用 Harbor 临时沙箱检查工作目录、测试入口和测试内容摘要，然后导出同一套 bundle 文件。source 中已有 `image@sha256:...` 时无需再从 tag 解析摘要。也可以不传 `--instances`，处理 source 中全部任务。

Harbor 准备路径当前按任务顺序执行，`--probe-concurrency` 和 `--qualify-concurrency` 不会将其变为并行处理；这些参数作用于 AgentENV 分支。Harbor 连接、资源与生命周期读取沙箱 JSON，`--probe-sandbox-timeout` 不控制 Harbor 会话生命周期。该准备路径直接创建沙箱，不经过训练生成器的 CLI 安装准备队列。

出现镜像、命令或连接异常时，Harbor 分支会中止，正常结束后才写出本轮 bundle；该分支没有逐阶段恢复记录。处理异常后可重跑，但不要把失败后目录里可能留存的旧产物当作本轮成功结果。为每轮准备使用新的输出目录更容易核对。

### 3.4 使用 AgentENV 生成 bundle

AgentENV 需要先准备对应任务模板。可以用任务 ID 列表加镜像规则自动发现模板，也可以直接提供模板映射。下面采用明确映射，减少对模板别名规则的依赖：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
export VIME_AGENT_SANDBOX_BACKEND=agentenv
unset VIME_AGENT_SANDBOX_CONFIG
export TMAX_PREP_MAP=/path/to/preparation/task-template-map.tsv

python -m examples.terminal_agent_rl.prepare_terminalbench gen-bundle \
  --source "$TMAX_PREP_SUBSET" \
  --bundle /path/to/prepared/tmax-agentenv \
  --instance-map "$TMAX_PREP_MAP" \
  --api-url http://agentenv.example:18000 \
  --sandbox-url http://agentenv.example:18000 \
  --api-key-file /path/on/npu/agentenv-api-key \
  --probe-concurrency 2 --qualify-concurrency 2 --qualify
```

映射文件按任务列出以下四列，使用实际值并以制表符分隔：

```text
task_id    template_id    image_digest    image_tag
```

模板 ID 应是当前 AgentENV 服务中的有效 UUID，`image_digest` 为 `sha256:` 加 64 位十六进制摘要，`image_tag` 为对应的完整镜像引用。第四列只有在同时提供 `--image-template` 时才可省略；仅 source 中有镜像字段不能替代这个要求。映射里的每个任务都应存在于所选 source 中。

使用发现模式时，以 `--instances "$TMAX_PREP_LABELS" --image-template 'registry.example/tmax/{task_id}:x86_64'` 替换 `--instance-map`。默认只发现已有模板；显式 `--auto-pull` 才会调用已配置的 `aenv` CLI 导入缺失模板，仍不会替你构建 OCI 镜像。

AgentENV 准备分支使用自己的 `--api-url`、`--sandbox-url`、`--api-key-file` 参数，不像训练入口那样从沙箱 JSON 自动填充这些连接字段。凭据通过文件读取，不放入 source、bundle 或命令正文。

该分支会排除未发现模板、探测失败或未通过所选资格检查的任务；不是所有错误都会阻止其他任务导出。结束后须核对日志、资格记录和最终 ID 集合，不能仅凭进程成功退出认定请求的全部任务已准备完成。

### 3.5 如何理解 Terminal 的准备检查

默认 `gen-bundle` 会创建临时沙箱进行探测，并记录测试内容摘要。`--no-probe` 可以跳过探测，但这种产物不能视为已经验证环境可用。若同时传入 `--qualify`，仍会执行原始环境检查；Harbor 分支还会在该路径完成必要探测并写入摘要，AgentENV 分支不会补回被跳过的探测摘要。

`--qualify` 排除在 Agent 尚未修改的原始环境中就能通过测试的任务。原始环境测试失败只能说明当前检查未通过，**不等于已经证明存在正确解答或依赖完整**；需要结合失败原因确认可训练性。

测试正文保存在 `verifier_specs.jsonl`，不应加入训练 prompt。`--timeout-sec` 在准备时覆盖任务规格预算；后续只增大训练侧超时不会改写已保存的 bundle。本文只说明数据准备所需的检查，不展开训练中的奖励计算流程。

## 4. 复用与后续训练

准备完成后，至少核对以下内容：

1. 四个训练所需文件齐全，训练行的 `label` 与对应 `task_id` 一致；三份 JSONL 的任务集合、镜像 digest、模板及工作目录绑定相符。
2. 查看 manifest、准备日志和资格记录，确认最终题数及被排除任务。抽样 2048 题不代表最终一定导出 2048 题。
3. AgentENV 模板属于实际连接的服务；Harbor CPU 沙箱服务器可获取并运行同一 digest 的镜像。复制 bundle 不会迁移模板、镜像或任务外部资源。
4. 保留源文件及其摘要、子集任务 ID 列表、准备参数和输出记录。保持现有 bundle 不变，将新的筛选结果写入独立目录。

Terminal 训练入口通过 `--bundle` 选择完整目录，由启动器推导内部文件路径。TMax 使用同一个入口，并指定自己的 bundle。以下命令只预览配置：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
source "$HOME/.config/vime/agenticrl.local.sh"
VIME_RL_PREPARED_BUNDLE=/path/to/prepared/tmax-bundle \
python3 vime/agent/config.py shell --task terminalbench --repo-root .
```

配置查看命令不验证模板、镜像或任务可执行性，不能替代本页的准备检查。正式训练前还需按 [Harness 指南](../env-and-config/harness-preparation.md)准备 Agent，并按[沙箱指南](../env-and-config/sandboxes.md)验证控制连接、模型回调和安装包分发。

训练操作见 [Terminal／TMax 训练指南](training.md)。
