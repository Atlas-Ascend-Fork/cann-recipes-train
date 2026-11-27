# SWE-Gym 数据Bundle准备

本文说明如何使用当前仓库的 `prepare_swegym.py` 准备 SWE-Gym 数据包（bundle），包括源数据检查、AgentENV 模板准备、资格检查和训练文件导出，以及如何将已完成的 bundle 用于 Harbor。

已有完整 bundle 时，可以直接阅读[复用与后续训练](#4-复用与后续训练)。沙箱服务部署见[沙箱指南](../env-and-config/sandboxes.md)，Agent 安装见 [Harness 指南](../env-and-config/harness-preparation.md)。数据准备不需要启动策略模型、安装 Agent CLI 或建立模型回调。

本文默认已具备 NPU 服务器和 CPU 沙箱服务器。数据处理及 bundle 导出命令在 **NPU 服务器的 `vime-forge` 目录**执行；第 3.4 节的镜像清点和下载命令在 **CPU 沙箱服务器**执行。

## 1. 先选择准备路径

| 目标后端 | 当前支持的准备方式 | 预先需要什么 |
| --- | --- | --- |
| AgentENV | `prepare_swegym.py` 执行 `inspect → provision → qualify → finalize`，创建或复用模板并导出 bundle | 固定版本源数据、任务 OCI 镜像、可用的 AgentENV 服务与配套 SDK、固定版本 SWE-Gym 依赖 |
| Harbor | 复用上述流程生成的合格 bundle，在 CPU 沙箱服务器准备其中引用的镜像 | 完整 bundle、已配置 Harbor 和 Docker 的 CPU 沙箱服务器、相同 digest 的任务镜像 |

> 注 #1: `prepare_swegym.py` 仅面向 AgentENV 沙盒的数据准备。

> 注 #2: 本工作已经支持AgentENV沙箱在鲲鹏CPU环境上运行，SWE-Gym的数据集准备可以参照鲲鹏SDK工作流 [**[link](https://gitcode.com/KunpengSDK/agentENV/tree/main/A%2BK-Ecosystem%20-Co-Building/swegym-image-build-workflow/)**]

## 2. Bundle 结构与准备环境

### 2.1 目录与文件

```text
<bundle>/
├── rollout/train.no_gold.jsonl
├── environment/agentenv_templates.jsonl
├── verifier/verifier_specs.jsonl
├── preparation_manifest.json
├── qualification/                       # 资格检查记录
└── state/                               # 准备阶段的中间状态
```

| 文件 | 内容 |
| --- | --- |
| `rollout/train.no_gold.jsonl` | 问题描述和环境原始，移除golden patch |
| `environment/agentenv_templates.jsonl` | 按 `instance_id` 记录模板、镜像摘要、代码仓库及起始提交 |
| `verifier/verifier_specs.jsonl` | 保存任务检查所需脚本、测试补丁、版本和环境绑定 |
| `preparation_manifest.json` | 记录数据来源、任务筛选结果及输出校验值 |

环境映射文件名对两个后端保持不变。AgentENV 通过模板 ID 创建实例，Harbor 通过 `source_image` 中的 `镜像名@sha256:摘要` 创建容器。

SWE bundle 中的模板 ID 还参与现有记录绑定，Harbor 训练时也应保留，不要因不再使用 AgentENV 服务而删除。

### 2.2 命令执行位置与依赖

NPU 服务器保存仓库代码、源数据和导出的 bundle，并运行 `prepare_swegym.py`。脚本通过 AgentENV API 请求 CPU 沙箱服务器创建模板和沙箱，再收集检查结果；任务测试在 CPU 沙箱服务器上的沙箱内执行。

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
python -m examples.swe_agent_rl.prepare_swegym --help
```

NPU 服务器上的数据准备环境建议使用 Python 3.12。依赖按执行位置区分如下：

| 阶段 | 所需环境与依赖 |
| --- | --- |
| `inspect`、`finalize` | NPU 服务器上的 Python；这些路径主要使用标准库 |
| AgentENV 模板准备与探测 | NPU 服务器上的兼容 E2B SDK；已有部署参考使用 `e2b==2.46.4` |
| `qualify` | NPU 服务器上的本项目任务层依赖，包括 PyTorch，以及指定提交的 SWE-Gym 源码和依赖；不会进行 NPU 训练计算 |
| Harbor 镜像清点与下载 | CPU 沙箱服务器上的 Python、Docker，以及可访问的镜像仓库；具体操作见第 3.4 节 |

`--help` 能返回不代表在线依赖已齐全。SDK 连通性检查环境也不能代替完整的资格检查环境。服务首次部署与连通性步骤见 [AgentENV 部署指南](../env-and-config/agentenv-deployment.md)和 [Harbor 部署要求](../env-and-config/sandboxes.md#22-harbor)。

## 3. 准备 SWE-Gym bundle

准备流程从 **NPU 服务器**发起：`prepare_swegym.py` 读取 SWE 任务，按任务 ID 确定对应镜像，再使用 AgentENV 服务地址和 API 密钥，通过兼容的 E2B SDK 在 **CPU 沙箱服务器**创建或复用模板，并从模板启动临时沙箱。

通过连接AgentENV对数据进行校准和验证：检查仓库起始提交、工作区和所需工具，并运行测试，确认空补丁不能解决任务、参考补丁可以解决任务。检查通过后，脚本在 NPU 服务器导出 bundle，记录题目与模板 ID、镜像摘要及测试规格的对应关系，供后续训练复用。整个准备过程不需要调用模型rollout。

### 3.1 源数据与固定依赖

当前 [prepare_swegym.py](../../../examples/swe_agent_rl/prepare_swegym.py) 针对仓库使用的 293 题源文件进行验证，其固定信息如下：

| 项目 | 当前值 |
| --- | --- |
| 数据来源 | `NovaSky-AI/SkyRL-v0-293-data`，`train` split |
| 数据 revision | `25f0e2f3b982d8dd0c1e59979341e5297e5aee31` |
| 源文件 | `examples/swe_agent_rl/swegym_train_293.jsonl` |
| 预期记录数 | `293`，任务 ID 不重复 |
| 源文件 SHA256 | `8af4d6aa43b0cd4325e77ae712301443141f311d324397e53cfd6cd899f17cb1` |
| SWE-Gym 源码提交 | `16dd480cce9b27bf111a362d280881c6def5d2a7` |

每行需要包含单条 user 消息形式的 `prompt`、`metadata.instance_id`、`metadata.split` 和完整的 `metadata.instance`。`instance` 中包含仓库、40 位起始提交、问题描述、参考补丁、测试补丁及测试列表；ID 和问题描述应与外层记录一致。任意 Hugging Face 导出格式不能直接替代此源文件。

源文件缺失时，应取得上述固定来源对应的已转换文件并核对格式与摘要。保留原始参考补丁供准备检查使用；准备器会另行生成不含参考答案的训练输入，不要提前从源文件删去 `patch`。

准备配置示例：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
export SWE_PREP_SOURCE="$PWD/examples/swe_agent_rl/swegym_train_293.jsonl"
export SWE_PREP_REGISTRY=https://docker.1ms.run
export VIME_SWEGYM_SOURCE_PATH=/path/to/pinned/SWE-Gym-source
export VIME_AGENT_SANDBOX_BACKEND=agentenv

# 本准备流程显式使用下面的 API 参数，不加载训练用沙箱 JSON。
unset VIME_AGENT_SANDBOX_CONFIG
export E2B_API_URL=http://agentenv.example:18000
export E2B_SANDBOX_URL="$E2B_API_URL"
export VIME_AGENTENV_API_KEY_FILE=/path/on/npu/agentenv-api-key
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

git -C "$VIME_SWEGYM_SOURCE_PATH" rev-parse HEAD
sha256sum "$SWE_PREP_SOURCE"
```

`SWE_PREP_REGISTRY` 沿用准备器当前默认的镜像仓库入口示例，不表示必须使用该服务。替换时，NPU 服务器和 AgentENV 服务端都应能获取脚本生成的任务镜像路径。它使用带协议的 registry URL，与 Harbor JSON 的 `registry` 主机名字段格式不同。

SWE-Gym 源码目录必须提供 `swegym.harness`，提交号会在生成任务检查规格时核对。不能用任意版本的 `swebench` 包替代。

### 3.2 快速开始样例

先从已校验的源文件抽取一题，并使用新的输出目录：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
export SWE_PREP_RUN="$PWD/prepared/swegym-canary-$(date +%Y%m%d-%H%M%S)"
export SWE_PREP_CANARY="$SWE_PREP_RUN/source/canary.jsonl"
export SWE_PREP_BUNDLE="$SWE_PREP_RUN/bundle"

python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

raw = Path(os.environ['SWE_PREP_SOURCE']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == '8af4d6aa43b0cd4325e77ae712301443141f311d324397e53cfd6cd899f17cb1'
rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
assert len(rows) == 293
selected = [row for row in rows if row['metadata']['instance_id'] == 'getmoto__moto-7365']
assert len(selected) == 1
target = Path(os.environ['SWE_PREP_CANARY'])
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(selected[0], ensure_ascii=False) + '\n')
PY

SWE_PREP_CANARY_SHA="$(sha256sum "$SWE_PREP_CANARY" | awk '{print $1}')"
python -m examples.swe_agent_rl.prepare_swegym inspect \
  --source "$SWE_PREP_CANARY" --bundle "$SWE_PREP_BUNDLE" \
  --expected-rows 1 --expected-source-sha256 "$SWE_PREP_CANARY_SHA" \
  --registry-base "$SWE_PREP_REGISTRY"
```

`inspect` 只检查 NPU 服务器上的源数据并写出候选任务、来源记录和镜像计划，不创建沙箱，也不查询镜像仓库。显式覆盖记录数与摘要用于这个从完整源文件提取的子集，不应作为绕过未知数据差异的方式。

随后仍在 NPU 服务器执行在线阶段。命令会远程创建沙箱，任务检查由 CPU 沙箱服务器上的沙箱执行：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
python -m examples.swe_agent_rl.prepare_swegym provision \
  --bundle "$SWE_PREP_BUNDLE" --registry-base "$SWE_PREP_REGISTRY" \
  --api-url "$E2B_API_URL" --sandbox-url "$E2B_SANDBOX_URL" \
  --api-key-file "$VIME_AGENTENV_API_KEY_FILE" \
  --concurrency 1 --retries 3 --cpu-count 16 --memory-mib 65536

python -m examples.swe_agent_rl.prepare_swegym qualify \
  --source "$SWE_PREP_CANARY" --bundle "$SWE_PREP_BUNDLE" \
  --api-url "$E2B_API_URL" --sandbox-url "$E2B_SANDBOX_URL" \
  --api-key-file "$VIME_AGENTENV_API_KEY_FILE" \
  --swegym-source "$VIME_SWEGYM_SOURCE_PATH" \
  --concurrency 1 --timeout-sec 1800

python -m examples.swe_agent_rl.prepare_swegym finalize \
  --source "$SWE_PREP_CANARY" --bundle "$SWE_PREP_BUNDLE"
```

| 阶段 | 作用及输出边界 |
| --- | --- |
| `provision` | 将镜像标签解析为 digest，创建或复用 AgentENV 模板，并用临时沙箱检查工具、仓库起始提交和工作区状态 |
| `qualify` | 生成任务检查规格，验证空补丁与参考补丁的预期结果；记录 `qualified`、`rejected` 或 `retryable` 状态 |
| `finalize` | 检查记录完整性和绑定关系，只将合格任务导出到训练文件；存在未完成、过期或需重试的记录时拒绝导出 |

示例中的 16 vCPU、65536 MiB 是显式资源配置；脚本默认是 4 vCPU、8192 MiB。根据任务及主机容量决定资源和并发，不需要同时创建全部任务沙箱。`provision` 使用现有任务镜像，不会根据问题描述重新安装整个项目。

### 3.3 准备完整数据集与恢复失败任务

单任务通过后，为完整源文件选择新的 bundle 目录：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
export SWE_PREP_FULL_BUNDLE="$PWD/prepared/swegym-293"

python -m examples.swe_agent_rl.prepare_swegym prepare-all \
  --source "$SWE_PREP_SOURCE" --bundle "$SWE_PREP_FULL_BUNDLE" \
  --registry-base "$SWE_PREP_REGISTRY" \
  --api-url "$E2B_API_URL" --sandbox-url "$E2B_SANDBOX_URL" \
  --api-key-file "$VIME_AGENTENV_API_KEY_FILE" \
  --swegym-source "$VIME_SWEGYM_SOURCE_PATH" \
  --provision-concurrency 2 --qualify-concurrency 2 \
  --retries 3 --cpu-count 16 --memory-mib 65536 --timeout-sec 1800
```

`prepare-all` 依次运行四个阶段，保存中间状态。可以在排查失败原因后用相同输入重跑，或对在线阶段使用 `--instance-id` 定位任务。仅对部分任务执行在线阶段后，完整候选集合仍需具备有效结果才能最终导出。

模板或检查规格改变后不能直接沿用旧资格结果。`rejected` 任务被排除，未解决的 `retryable` 任务会阻止导出；不要手改状态伪装成功。已有 bundle 若包含 `qualification/quarantined_instances.jsonl`，准备器会按其中的明确排除记录处理并在 manifest 中记录，不应忽略这些记录计算训练样本数。

最终题数以导出的训练文件为准，不保证等于 293。SWE 默认任务规格预算通常为 900 秒，MONAI 为 1800 秒；准备检查采用调用预算与规格预算的较小值，因此命令中的 `--timeout-sec 1800` 不会将所有规格都改成 1800 秒。

### 3.4 将已完成的 SWE bundle 用于 Harbor

**Harbor 无需导入 AgentENV 模板，也无需转换 bundle 格式。** AgentENV 准备流程已经在 `environment/agentenv_templates.jsonl` 中同时记录了模板 ID 和创建模板所用的源镜像。训练时，代码根据所选后端决定用哪一个字段创建沙箱。

#### 同一份 bundle 如何支持两个后端

| 环境映射字段 | 使用 AgentENV 时 | 使用 Harbor 时 |
| --- | --- | --- |
| `instance_id` | 按题目 ID 查找对应环境记录 | 相同 |
| `template_id` | 传给 AgentENV，用该模板创建沙箱 | 保留在任务元数据中，用于与任务检查规格核对；不传给 Harbor 创建容器 |
| `source_image` | 记录创建模板所用的源镜像 | 传给 Harbor，按 `镜像名@sha256:摘要` 创建 Docker 容器 |
| `image_digest`、`repo`、`base_commit`、`workdir` | 记录并核对任务与运行环境的对应关系 | 相同 |

具体到一条训练任务，流程如下：

```text
NPU 服务器读取 bundle 中的题目
    → 按 instance_id 查找环境映射
    → 选择 Harbor：取 source_image，而不是 template_id
    → 通过 SSH 将镜像地址交给 CPU 沙箱服务器上的 Harbor worker
    → Harbor 用该镜像创建 Docker 容器
    → 框架在容器中安装或复用 Agent，准备工作区并执行任务
```

字段选择由 [apply_template_mapping()](../../../examples/swe_agent_rl/template_mapping.py) 完成；[任务的沙箱选择](../../../examples/swe_agent_rl/sandbox.py)返回 Harbor 适配器，CPU 端的 [harbor_worker.py](../../../vime/agent/harbor_worker.py) 将镜像地址作为 `EnvironmentConfig.docker_image` 交给 Harbor。训练使用的 bundle 始终由 NPU 服务器读取，Harbor worker 不需要读取整份 bundle。

因此，环境映射文件仍叫 `agentenv_templates.jsonl`，其中的 `template_id` 也必须原样保留：当前 SWE 读取器要求该字段存在，`verifier_specs.jsonl` 还会核对它与镜像摘要等信息。**使用 Harbor 时保留这个字段，不意味着运行时还要连接 AgentENV 服务。**

#### 用户需要完成的两步

**第一步：在 CPU 沙箱服务器准备源镜像。** 保持 NPU 服务器上的完整 bundle 不变，在每台可能执行任务的 CPU 沙箱服务器准备 `source_image` 指定的镜像。复制 bundle 只复制任务记录，不会复制 Docker 镜像。

使用仓库的 [harbor_images.py](../../../tools/harbor_images.py) 下载时，先将完整 bundle 原样复制到 CPU 沙箱服务器，供该工具读取镜像清单。这个副本只用于镜像准备，不是交给 Harbor 导入的数据包。随后在各台 CPU 沙箱服务器执行：

```bash
# 执行位置：CPU 沙箱服务器宿主环境（vime-forge 目录）。
export VIME_SWEGYM_PREPARED_BUNDLE=/path/on/cpu/swegym-bundle
export VIME_HARBOR_IMAGE_REPORT=/path/on/cpu/reports/swegym-images.json

# 只查询远端镜像清单并写报告；仍会访问镜像仓库。
python3 tools/harbor_images.py
# 确认 Docker 存储目录空间足够后，才执行下载。
python3 tools/harbor_images.py --pull
```

报告必须放在 bundle 目录之外。上面的两个变量是镜像工具的独立输入，训练仍通过 `--bundle` 选择目录。该工具默认检查至少 100 GiB 剩余空间，但这不是全量镜像所需空间的估算；压缩层总大小也不等于解压后的占用。需要使用镜像仓库替换时，工具单独读取 `VIME_HARBOR_IMAGE_REGISTRY`，不会自动读取训练用沙箱 JSON，应与训练实际使用的仓库一致。

已有私网镜像复制流程时，可以下载后复制到其他 CPU 沙箱服务器，不必每台重复下载；各台最终都要能按相同的镜像摘要创建容器。

**第二步：在 NPU 服务器选择 Harbor，并继续使用原 bundle。** 在外部 `.local.sh` 中设置以下两项，JSON 的 `harbor` 区填写 CPU 沙箱服务器连接与资源配置：

```bash
# 配置位置：NPU 服务器上的外部 swe-harbor.local.sh。
export VIME_AGENT_SANDBOX_BACKEND=harbor
export VIME_AGENT_SANDBOX_CONFIG="$HOME/.config/vime/sandboxes.json"
```

查看任务配置，数据包仍指向原来在 NPU 服务器上导出的目录：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
source "$HOME/.config/vime/swe-harbor.local.sh"
VIME_RL_PREPARED_BUNDLE=/path/on/npu/prepared/swegym-bundle \
python3 vime/agent/config.py shell --task swegym --repo-root .
```

这一步不需要重新运行 `provision`、`qualify` 或 `finalize`，也不修改四个训练文件及已有准备记录。SWE 的补丁收集和独立任务检查沙箱仍沿用现有流程，选择 Harbor 后两类沙箱均由 Harbor 创建。

复用的前提是 **Harbor 源镜像中的仓库起始提交、工作目录和任务依赖与 bundle 相符**。AgentENV 模板构建时附加的修改不会随 bundle 自动搬到 Harbor；原有 AgentENV 资格记录也不等于已验证 Harbor 的容器权限、Agent 安装和网络。镜像准备完成后，按[沙箱指南](../env-and-config/sandboxes.md#6-启动前检查与验证范围)检查部署，再按[训练指南](training.md#42-只验证一次完整训练更新)验证完整流程；上面的命令仅显示任务配置。

## 4. 复用与后续训练

准备完成后，至少核对以下内容：

1. 四个训练所需文件齐全，训练行的 `label` 与对应 `instance_id` 一致；三份 JSONL 的任务集合、镜像 digest、模板、代码起始提交及工作目录绑定相符。
2. 查看 manifest、准备日志和资格记录，确认最终题数及被排除任务。最终导出数量不一定是 293。
3. AgentENV 模板属于实际连接的服务；Harbor CPU 沙箱服务器可获取并运行同一 digest 的镜像。复制 bundle 不会迁移模板、镜像或任务外部资源。
4. 保留源文件及其摘要、准备参数和输出记录。保持现有 bundle 不变，将新的筛选结果写入独立目录。

SWE 训练入口通过 `--bundle` 选择完整目录，由启动器推导内部文件路径。以下命令只预览配置：

```bash
# 执行位置：NPU 服务器（vime-forge 目录）。
source "$HOME/.config/vime/agenticrl.local.sh"
VIME_RL_PREPARED_BUNDLE=/path/to/prepared/swegym-bundle \
python3 vime/agent/config.py shell --task swegym --repo-root .
```

配置查看命令不验证模板、镜像或任务可执行性，不能替代本页的准备检查。正式训练前还需按 [Harness 指南](../env-and-config/harness-preparation.md)准备 Agent，并按[沙箱指南](../env-and-config/sandboxes.md)验证控制连接、模型回调和安装包分发。

训练操作见 [SWE-Gym 训练指南](training.md)。
