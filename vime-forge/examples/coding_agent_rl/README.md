# SWE-Gym AgenticRL

## 1. 部署 AgentENV 与准备数据集

本节从独立的 AgentENV 服务器开始，将 [NovaSky-AI/SkyRL-v0-293-data](https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-293-data) 转换为训练可用的 prepared bundle。准备过程不启动策略模型、Codex、vLLM 或 RL 训练，也不需要模型回调 relay。

Get Started: 下载 `NovaSky-AI/SkyRL-v0-293-data` 到 `examples/coding_agent_rl`. 

| 位置 | 职责 |
| --- | --- |
| 沙盒服务器 | 运行 AgentENV，保存任务模板，创建用于探测和评测的 Firecracker microVM |
| 准备机 | 保存本仓库和源数据，通过 E2B SDK 调用 AgentENV，生成模板映射与评测清单；可与训练机共用 |
| 镜像仓库 | 提供包含任务仓库和依赖的 SWE-Gym OCI 镜像 |

JSONL 描述任务，OCI 镜像提供运行环境，AgentENV template 是可复用模板，sandbox 是从模板创建的一次性实例。无需把整个数据集复制进 sandbox，也不必同时启动 293 个 sandbox。

### 1.1 部署 AgentENV 服务

以下命令在**沙盒服务器**执行，主机应具备可用的 KVM、TUN、ublk 和 cgroup v2，本文以 Linux 6.8+ 为部署基线。OCI 镜像由 AgentENV 转换为 microVM 环境，无需额外安装 Docker daemon。

先检查主机并验证 KVM 确实可以创建虚拟机：

```bash
uname -m
uname -r
free -h
df -hT
stat -fc %T /sys/fs/cgroup

sudo modprobe kvm
sudo modprobe tun
sudo modprobe ublk_drv
ls -l /dev/kvm /dev/net/tun /dev/ublk-control

sudo python3 - <<'PY'
import fcntl
import os

fd = os.open('/dev/kvm', os.O_RDWR)
try:
    assert fcntl.ioctl(fd, 0xAE00, 0) == 12
    vm = fcntl.ioctl(fd, 0xAE01, 0)
    os.close(vm)
    print('KVM API and VM creation: OK')
finally:
    os.close(fd)
PY
```

若云实例未开放 KVM，需要更换实例规格或启用平台支持的嵌套虚拟化。磁盘需同时容纳镜像、模板和 sandbox 写层；本文为每个任务实例配置 16 vCPU、64 GiB 内存（65536 MiB），准备并发需按主机余量设置。

安装系统依赖，获取已验证的 AgentENV 源码版本：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential clang libclang-dev libprotobuf-dev protobuf-compiler \
  pkg-config libssl-dev cmake ca-certificates curl git e2fsprogs \
  iproute2 iptables jq sudo umoci zstd python3 python3-venv

git clone https://github.com/kvcache-ai/AgentENV.git
cd AgentENV
git checkout --detach 2d766447261a7252447fe8bccef2e87d48ece5a6
```

先按该版本的工具链要求安装 Rust/Cargo，再构建。记录 `rustc --version` 和 `cargo --version`，保留源码中的 lockfile：

```bash
cargo build --release --locked \
  -p agentenv --bin server \
  -p uvm-ublk-daemon --bin uvm-ublk-daemon
cargo build --release --locked -p aenv
```

创建服务账号并安装文件。已有部署应先核对现有账号和配置，不要覆盖：

```bash
sudo groupadd --system aenv
sudo useradd --system --gid aenv --home-dir /var/lib/aenv \
  --shell /usr/sbin/nologin aenv
sudo usermod -aG kvm aenv

sudo install -d -o aenv -g aenv -m 0750 \
  /var/lib/aenv /var/lib/aenv/config /var/lib/aenv/ublk
sudo install -m 0755 target/release/server /usr/local/bin/server
sudo install -m 0755 target/release/aenv /usr/local/bin/aenv
sudo install -m 0755 target/release/uvm-ublk-daemon /var/lib/aenv/ublk/uvm-ublk-daemon
sudo install -o root -g aenv -m 0640 config/default.toml /var/lib/aenv/config/config.toml

sudo env AENV_CONFIG_PATH=/var/lib/aenv/config/config.toml \
  AENV_HOME_PATH=/var/lib/aenv AENV_VIRTUALIZATION_MODE=kvm \
  /usr/local/bin/server --setup-only

sudo env AENV_CONFIG_PATH=/var/lib/aenv/config/config.toml \
  AENV_HOME_PATH=/var/lib/aenv AENV_VIRTUALIZATION_MODE=kvm \
  /usr/local/bin/server --setup-host --runtime-user aenv --runtime-group aenv

sudo chown -R aenv:aenv /var/lib/aenv
sudo chown root:aenv /var/lib/aenv/config /var/lib/aenv/config/config.toml
sudo chmod 0750 /var/lib/aenv/config
sudo chmod 0640 /var/lib/aenv/config/config.toml
```

`--setup-only` 准备 Firecracker、guest kernel、镜像工具等 runtime；`--setup-host` 配置宿主机设备权限与网络。离线部署需按该 commit 的依赖清单准备并校验制品，再执行完整性检查。

创建 `/etc/default/aenv`：

```ini
API_ADDR="127.0.0.1:8000"
AENV_CONFIG_PATH="/var/lib/aenv/config/config.toml"
AENV_HOME_PATH="/var/lib/aenv"
AENV_RUNTIME_PATH="/run/aenv"
AENV_VIRTUALIZATION_MODE="kvm"
```

创建 `/etc/systemd/system/aenv.service`：

```ini
[Unit]
Description=AgentENV Server
After=network.target

[Service]
User=aenv
Group=aenv
SupplementaryGroups=kvm
EnvironmentFile=/etc/default/aenv
ExecStart=/usr/local/bin/server
RuntimeDirectory=aenv
RuntimeDirectoryMode=0750
AmbientCapabilities=CAP_NET_ADMIN CAP_SYS_ADMIN
CapabilityBoundingSet=CAP_NET_ADMIN CAP_SYS_ADMIN
NoNewPrivileges=true
UMask=0027
LimitNOFILE=1048576
LimitMEMLOCK=infinity
Restart=on-failure
RestartSec=5
KillMode=process
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

启动并检查服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now aenv
sudo systemctl status aenv --no-pager
sudo journalctl -u aenv -n 100 --no-pager
curl -i --max-time 10 http://127.0.0.1:8000/health
```

预期 health 返回 HTTP 204。首次启动生成的 API key 位于 `/var/lib/aenv/secrets/api-key`。运行 `aenv auth`，填写本机服务地址和实际 API key；随后验证 `aenv template list`、`aenv ls` 可以正常返回。API key 不应提交到仓库或写入数据集。

### 1.2 验证本机 sandbox 与远程 SDK

在**沙盒服务器**创建最小模板，等待状态为 `ready`：

```bash
aenv pull docker.1ms.run/library/ubuntu:22.04 \
  --name ubuntu-smoke --cpu 16 --memory 65536
aenv template list
aenv start ubuntu-smoke --detach

# 使用上一步返回的 sandbox ID；不要填 template ID。
SB_ID='<sandbox ID>'
aenv exec "$SB_ID" sh -lc \
  'uname -m; printf "smoke-ok\n" > /tmp/smoke.txt; cat /tmp/smoke.txt'
aenv delete "$SB_ID"
aenv ls
```

模板保留供后续复用，删除 sandbox 不等于删除模板。示例镜像代理需在部署网络中可访问；也可以使用具有相同镜像的 registry。

在**准备机**使用已配置密钥认证、允许 TCP forwarding 的 SSH 账号建立控制隧道，并保持该进程运行：

```bash
ssh -N -T \
  -i '<SSH 私钥绝对路径>' \
  -o IdentitiesOnly=yes -o BatchMode=yes \
  -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18000:127.0.0.1:8000 \
  '<SSH 用户>@<AgentENV 服务器地址>'
```

首次使用前核对并保存 SSH 主机指纹。控制 API 保持 loopback 监听，无需向公网开放 8000 端口。隧道可在独立 tmux 会话中运行；SSH 保活不会自动重连。

将服务器 API key 安全保存到准备机 `~/.config/agentenv/api-key`，文件权限设为 `0600`。在另一终端创建独立客户端环境并检查控制链路：

```bash
python3 -m venv "$HOME/.venvs/agentenv-client"
"$HOME/.venvs/agentenv-client/bin/python" -m pip install 'e2b==2.46.4'

export E2B_API_URL=http://127.0.0.1:18000
export E2B_SANDBOX_URL="$E2B_API_URL"
export VIME_AGENTENV_API_KEY_FILE="$HOME/.config/agentenv/api-key"
export SMOKE_TEMPLATE_ID='<ubuntu-smoke 的 template ID>'
curl -i --max-time 10 "$E2B_API_URL/health"

"$HOME/.venvs/agentenv-client/bin/python" - <<'PY'
import os
from pathlib import Path
from e2b import Sandbox

key = Path(os.environ['VIME_AGENTENV_API_KEY_FILE']).read_text().strip()
sandbox = Sandbox.create(
    template=os.environ['SMOKE_TEMPLATE_ID'],
    api_key=key, api_url=os.environ['E2B_API_URL'],
    sandbox_url=os.environ['E2B_SANDBOX_URL'],
)
try:
    result = sandbox.commands.run("printf 'remote-sdk-ok'")
    assert result.exit_code == 0 and result.stdout == 'remote-sdk-ok'
    sandbox.files.write('/tmp/sdk-smoke.txt', 'file-roundtrip-ok')
    assert sandbox.files.read('/tmp/sdk-smoke.txt') == 'file-roundtrip-ok'
    print('远程命令与文件读写验证通过')
finally:
    sandbox.kill()
PY
```

### 1.3 准备数据处理环境

后续命令均在**准备机、本仓库根目录**执行。选择 Python 3.11+ 环境，安装本项目所需的 Python 依赖、`e2b==2.46.4`，并准备官方 SWE-Gym evaluator 源码及其依赖。Evaluator checkout 必须保持以下固定 revision：

```text
16dd480cce9b27bf111a362d280881c6def5d2a7
```

数据准备不执行加速器计算，但 `qualify` 会导入本项目的 `swe.py`、PyTorch 等模块；仅安装 E2B SDK 的 smoke 环境并不足够。Evaluator 必须可导入 `swegym.harness`，不能用任意版本的 `swebench` 包替代。

```bash
export SWE_PREP_PY='<已安装上述依赖的 Python 绝对路径>'
export VIME_SWEGYM_SOURCE_PATH='<固定 revision 的 SWE-Gym 源码目录>'
export SWE_PREP_SOURCE="$PWD/examples/coding_agent_rl/swegym_train_293.jsonl"
export SWE_PREP_REGISTRY=https://docker.1ms.run
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export VIME_AGENT_E2B_SELECTOR_MODE=template

# 在准备环境中安装 E2B；无需借用其他 Python 环境的 site-packages。
"$SWE_PREP_PY" -m pip install 'e2b==2.46.4'
git -C "$VIME_SWEGYM_SOURCE_PATH" rev-parse HEAD
"$SWE_PREP_PY" -m examples.coding_agent_rl.prepare_swegym --help
```

`--help` 只检查命令入口；evaluator revision 会在 `qualify` 时检查。若必须使用另一个兼容环境中的 SDK，可通过 `--sdk-path` 指定路径。

源文件包含 293 个唯一任务，记录数据来源 `NovaSky-AI/SkyRL-v0-293-data`，revision 为 `25f0e2f3b982d8dd0c1e59979341e5297e5aee31`。每行包含 `prompt`、`label` 和 `metadata.instance`；后者保存仓库、起始提交、问题描述、参考补丁、测试补丁及测试列表。准备器默认校验文件 SHA256：

```text
8af4d6aa43b0cd4325e77ae712301443141f311d324397e53cfd6cd899f17cb1
```

保留原始文件及其中的 gold `patch`，资格验证需要它。最终训练输入由准备器另行导出。

### 1.4 先验证一个真实任务

[prepare_swegym.py](prepare_swegym.py) 按 `inspect → provision → qualify → finalize` 执行。先为一个任务创建独立 bundle，验证镜像、模板和 grader 全链路后再扩大范围。

```bash
export SWE_PREP_RUN="$PWD/prepared/swegym-canary-$(date +%Y%m%d-%H%M%S)"
export SWE_PREP_CANARY="$SWE_PREP_RUN/source/canary.jsonl"
export SWE_PREP_BUNDLE="$SWE_PREP_RUN/bundle"

"$SWE_PREP_PY" - <<'PY'
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
"$SWE_PREP_PY" -m examples.coding_agent_rl.prepare_swegym inspect \
  --source "$SWE_PREP_CANARY" --bundle "$SWE_PREP_BUNDLE" \
  --expected-rows 1 --expected-source-sha256 "$SWE_PREP_CANARY_SHA" \
  --registry-base "$SWE_PREP_REGISTRY"

"$SWE_PREP_PY" -m examples.coding_agent_rl.prepare_swegym provision \
  --bundle "$SWE_PREP_BUNDLE" --registry-base "$SWE_PREP_REGISTRY" \
  --api-url "$E2B_API_URL" --sandbox-url "$E2B_SANDBOX_URL" \
  --api-key-file "$VIME_AGENTENV_API_KEY_FILE" \
  --concurrency 1 --retries 3 --cpu-count 16 --memory-mib 65536

"$SWE_PREP_PY" -m examples.coding_agent_rl.prepare_swegym qualify \
  --source "$SWE_PREP_CANARY" --bundle "$SWE_PREP_BUNDLE" \
  --api-url "$E2B_API_URL" --sandbox-url "$E2B_SANDBOX_URL" \
  --api-key-file "$VIME_AGENTENV_API_KEY_FILE" \
  --swegym-source "$VIME_SWEGYM_SOURCE_PATH" \
  --concurrency 1 --timeout-sec 1800

cat "$SWE_PREP_BUNDLE/qualification/qualification_results.jsonl"
```

各阶段的验收条件如下：

| 阶段 | 行为与验收条件 |
| --- | --- |
| `inspect` | 校验源文件、任务字段和唯一 ID，生成候选数据与镜像计划；不创建远端资源 |
| `provision` | 将任务镜像 tag 固定为 digest，构建或复用模板；创建临时 sandbox，探测工具并检查 `/testbed` 的 HEAD 与 `base_commit` 一致、工作区干净 |
| `qualify` | 生成 VerifierSpec，在不同的干净 sandbox 中评测空补丁与参考补丁；要求无基础设施错误、补丁可应用、空补丁 reward=0、参考补丁 reward=1 |
| `finalize` | 检查资格结果，导出合格任务对应的训练输入、模板映射和评测清单 |

`provision` 使用已有的任务 OCI 镜像，设置 Git 安全目录、重置仓库并配置 `/testbed` 工作目录，不从 JSONL 重新安装所有项目依赖。默认 registry 必须提供脚本推导出的任务镜像路径，准备机与 AgentENV 服务器均需能够访问。

确认结果为 `qualification_status: qualified` 后，完成导出：

```bash
"$SWE_PREP_PY" -m examples.coding_agent_rl.prepare_swegym finalize \
  --source "$SWE_PREP_CANARY" --bundle "$SWE_PREP_BUNDLE"
```

### 1.5 准备完整数据集

单任务通过后，为完整源文件选择新的 bundle 目录。以下命令会创建远端模板并执行全量资格验证；根据 ECS 内存、磁盘与测试负载调整并发：

```bash
export SWE_PREP_FULL_BUNDLE="$PWD/prepared/swegym-293"

"$SWE_PREP_PY" -m examples.coding_agent_rl.prepare_swegym prepare-all \
  --source "$SWE_PREP_SOURCE" --bundle "$SWE_PREP_FULL_BUNDLE" \
  --registry-base "$SWE_PREP_REGISTRY" \
  --api-url "$E2B_API_URL" --sandbox-url "$E2B_SANDBOX_URL" \
  --api-key-file "$VIME_AGENTENV_API_KEY_FILE" \
  --swegym-source "$VIME_SWEGYM_SOURCE_PATH" \
  --provision-concurrency 2 --qualify-concurrency 2 \
  --retries 3 --cpu-count 16 --memory-mib 65536 --timeout-sec 1800
```

无需提供已有 template map。准备器会保存中间状态；排查失败原因后，可对同一 bundle 重跑恢复。`qualified` 表示通过，`rejected` 表示不满足资格条件，`retryable` 表示需处理的执行或基础设施问题。存在未完成或 `retryable` 结果时，最终导出会被阻止；不要为绕过错误而手动改写资格状态。更换模板或评分契约后需重新验证，不能直接沿用旧结果。

最终样本数由资格结果决定。本文统一设置 `--timeout-sec 1800`。评分实际超时仍为调用参数与 `VerifierSpec.timeout_sec` 的较小值；当前准备器生成的 spec 通常为 900 秒、MONAI 为 1800 秒，因此仅修改命令参数不会使所有任务的实际超时都达到 1800 秒。

### 1.6 Bundle 与后续训练的衔接

```text
prepared/swegym-293/
├── rollout/train.no_gold.jsonl
├── environment/agentenv_templates.jsonl
├── verifier/verifier_specs.jsonl
├── preparation_manifest.json
├── qualification/
└── state/
```

| 产物 | 用途 |
| --- | --- |
| `train.no_gold.jsonl` | 训练任务及 metadata，移除参考答案补丁；评测 metadata 不应作为模型上下文 |
| `agentenv_templates.jsonl` | 按 `instance_id` 绑定模板 ID、镜像 digest、仓库及 base commit |
| `verifier_specs.jsonl` | 记录评测脚本、测试补丁、F2P/P2P 测试、grader 版本及环境绑定 |
| `preparation_manifest.json` | 记录来源、合格与排除的任务、计数及产物校验值 |

训练侧按 `instance_id` 关联前三份文件。一次 trial 先从模板创建 Agent sandbox，提取候选补丁并销毁；再创建独立的 Verifier sandbox 执行测试。模板可复用，可写的运行实例不共享。

Bundle 中的 template ID 属于本次 AgentENV 部署。换服务器时需要恢复模板状态，或重新准备映射；仅复制 JSONL 无法迁移模板。后续训练使用本次导出的 bundle，并核对其实际样本数。准备完成只证明环境与 grader 通过资格验证；sandbox 到模型 Adapter 的回调网络、Agent CLI 和训练环境仍需在启动训练前配置。

## 2. 准备 Codex Harness

运行 Codex Agent rollout 前，需要在 rollout worker 所在机器上准备 Node 与 Codex 安装包。当前 Harness **不会自动下载**这些文件；它会在每次创建 Agent sandbox 后上传、安装并检查版本。上一节的模板准备和资格验证不需要这些制品。

| 制品 | 用途 | 目标平台 |
| --- | --- | --- |
| `node-v22.20.0-linux-x64.tar.xz` | sandbox 内的 Node.js 与 npm runtime | Linux x86_64 |
| `vime-codex-offline-0.153.0.tgz` | Codex CLI 启动器及已打包的原生平台依赖 | Linux x86_64 |

平台选择取决于 **AgentENV sandbox**，与训练机是否采用 ARM/NPU 无关。以下准备与版本验收命令在可联网的 Linux x86_64 机器执行，不安装全局 CLI，也不启动训练。

### 2.1 下载并校验 Node.js

从 [Node.js v22.20.0 官方归档](https://nodejs.org/dist/v22.20.0/) 下载，校验值来自该版本的 [SHASUMS256.txt](https://nodejs.org/dist/v22.20.0/SHASUMS256.txt)：

```bash
set -euo pipefail
export HARNESS_ARTIFACTS="$PWD/agentenv-artifacts"
mkdir -p "$HARNESS_ARTIFACTS"

curl -fL --retry 3 \
  https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz \
  -o "$HARNESS_ARTIFACTS/node-v22.20.0-linux-x64.tar.xz"

printf '%s  %s\n' \
  00bbd05e306ea68b6e13e17360d0e2f680b493ef95f2fea1c4296ff7437530bc \
  "$HARNESS_ARTIFACTS/node-v22.20.0-linux-x64.tar.xz" | sha256sum -c -

# 仅在准备机的临时目录解压，用于后续打包与验收。
export HARNESS_BUILD_DIR="$(mktemp -d)"
tar -xJf "$HARNESS_ARTIFACTS/node-v22.20.0-linux-x64.tar.xz" \
  -C "$HARNESS_BUILD_DIR"
export PATH="$HARNESS_BUILD_DIR/node-v22.20.0-linux-x64/bin:$PATH"
node --version
npm --version
```

`node --version` 应输出 `v22.20.0`。保留原始 `.tar.xz` 文件供 Harness 上传；无需把解压目录复制到训练机。

### 2.2 制作 Codex 离线安装包

[Codex CLI](https://developers.openai.com/codex/cli/) 通过 npm 分发。这里固定使用 `0.153.0`，不使用 `latest`。`vime-codex-offline` 是本部署重新打包后的名称，没有对应的官方同名下载地址。

普通 `@openai/codex` 包通过 optional dependency 引入平台二进制；仅下载主包不能保证离线安装。以下步骤同时下载 [主包元数据](https://registry.npmjs.org/@openai%2fcodex/0.153.0) 对应的包和 [Linux x64 平台包](https://registry.npmjs.org/@openai%2fcodex/0.153.0-linux-x64)，将平台依赖直接纳入最终 tgz：

```bash
(
  set -euo pipefail
  cd "$HARNESS_BUILD_DIR"
  npm pack @openai/codex@0.153.0 \
    --registry=https://registry.npmjs.org --ignore-scripts
  npm pack @openai/codex@0.153.0-linux-x64 \
    --registry=https://registry.npmjs.org --ignore-scripts

  mkdir cli platform
  tar -xzf openai-codex-0.153.0.tgz -C cli
  tar -xzf openai-codex-0.153.0-linux-x64.tgz -C platform
  mkdir -p cli/package/node_modules/@openai
  mv platform/package cli/package/node_modules/@openai/codex-linux-x64

  python3 - <<'PY'
import json
from pathlib import Path

path = Path('cli/package/package.json')
package = json.loads(path.read_text())
assert package['version'] == '0.153.0'
platform = Path('cli/package/node_modules/@openai/codex-linux-x64/package.json')
assert json.loads(platform.read_text())['version'] == '0.153.0-linux-x64'
package['name'] = 'vime-codex-offline'
package['optionalDependencies'] = {
    '@openai/codex-linux-x64': 'npm:@openai/codex@0.153.0-linux-x64'
}
package['bundledDependencies'] = ['@openai/codex-linux-x64']
path.write_text(json.dumps(package, indent=2) + '\n')
PY

  cd cli/package
  npm pack --offline --ignore-scripts --pack-destination "$HARNESS_ARTIFACTS"
)
```

该操作修改包名和依赖打包方式，保留官方启动器与平台二进制，不编译或修改 Codex 源码。最终文件应为 `vime-codex-offline-0.153.0.tgz`。重新打包后的 SHA256 以实际产物为准，不直接套用其他部署的离线包校验值。

使用独立、空的 npm cache 做离线安装验收，并记录制品校验值：

```bash
HARNESS_VERIFY_DIR="$(mktemp -d)"
npm install -g --prefix="$HARNESS_VERIFY_DIR/install" \
  --cache="$HARNESS_VERIFY_DIR/cache" --offline --no-audit --no-fund \
  "$HARNESS_ARTIFACTS/vime-codex-offline-0.153.0.tgz"
"$HARNESS_VERIFY_DIR/install/bin/codex" --version

(
  cd "$HARNESS_ARTIFACTS"
  sha256sum node-v22.20.0-linux-x64.tar.xz vime-codex-offline-0.153.0.tgz \
    > harness-sha256.txt
)
```

版本输出应为 `codex-cli 0.153.0`。空 cache 下安装失败时，应检查平台依赖是否确实进入 tgz，不应依赖 sandbox 临时联网补齐。当前 Harness 安装命令本身没有 `--offline`；这里的验收用于确认安装包无需额外获取依赖。

### 2.3 配置 rollout worker

将两个安装包及 `harness-sha256.txt` 复制到 rollout worker 可读取的持久目录。在目标目录执行 `sha256sum -c harness-sha256.txt`，然后在启动训练前设置：

```bash
export VIME_AGENT_NODE_TARBALL='<制品目录>/node-v22.20.0-linux-x64.tar.xz'
export VIME_AGENT_CODEX_TARBALL='<制品目录>/vime-codex-offline-0.153.0.tgz'
test -r "$VIME_AGENT_NODE_TARBALL"
test -r "$VIME_AGENT_CODEX_TARBALL"
```

使用绝对路径；多节点运行时，每个执行 rollout 的 worker 都需要能读取这些文件。无需在 NPU 主机运行 Linux Codex 二进制，也无需手工登录每个 sandbox 安装。

[CodexHarness](../../vime/agent/harness/codex.py) 调用共享安装逻辑：Node 压缩包在 worker 侧解压为 tar 后上传到 sandbox，安装到 `/opt/node22`；Codex tgz 上传后通过 npm 安装到 `/usr/local`，再执行版本检查。该安装不修改已保存的 AgentENV template。

Harness 随后写入 Vime Responses Adapter 的地址，并将当前 trial 的 session ID 用作请求凭据。本训练路径调用 Vime 管理的策略模型，不要求在 sandbox 中执行 `codex login`；AgentENV API key 仍只用于环境控制。模型回调网络必须在运行 Agent 前另行打通。

## 3. 启动 Qwen3.5-4B SWE-Gym AgenticRL 训练

完成前两节的 bundle 和 Harness 准备后，在 **NPU 训练机、本仓库根目录**启动训练。主入口是 [run-qwen35-4B-swegym-torchtitan.sh](run-qwen35-4B-swegym-torchtitan.sh)：环境路径、Agent 配置和 `TRAIN_CMD` 参数均在该文件中，明确调用 `train_async.py --train-backend torchtitan`。它使用 Codex 与 prepared SWE-Gym verifier，不使用 normal RL 的数学题 reward function。

[swegym-env.sh](swegym-env.sh) 负责训练环境激活、bundle 检查和 Ray worker 环境传递。本目录的主脚本是唯一启动入口，任务数从 bundle 读取。该入口只接收 `--help` 或 `--dry-run`；其他配置通过环境变量或主脚本中的参数设置，不接受任意追加的训练 CLI 参数。

### 3.1 确认训练与推理运行环境

当前脚本针对**单节点 Ascend NPU、训练与推理分离**的部署。默认使用 16 个 NPU dies：训练占 8 个（TP=4、DP shard=2），rollout 占 8 个（8 个 TP=1 的 vLLM engine），不启用 colocate。需要事先安装能够运行 Qwen3.5 的 TorchTitan/TorchTitan-NPU、PyTorch/torch-npu、vLLM/vLLM-Ascend 及其匹配的 CANN/ATB；脚本不会安装这些依赖。

| 变量 | 含义 |
| --- | --- |
| `TRAIN_ENV` | 训练 Python 3.12 环境，需有 `bin/python`、`bin/ray`、`bin/activate` 和本项目依赖 |
| `VIME_TRAIN_PYTHON` | 实际训练解释器，默认 `${TRAIN_ENV}/bin/python`；通常不必单独覆盖 |
| `ROLLOUT_ENV` | 推理 Python 3.11 环境，提供 `bin/python` 和 `bin/vllm`；脚本不激活该环境 |
| `VIME_CANN_HOME` | 提供 `set_env.sh` 的 CANN 目录，本运行配置使用 CANN 9.2 |
| `VIME_AGENT_TASK_CONDA_*` | **sandbox 内**执行任务代码和测试的 Python 环境，与 NPU 主机的 Python 分开 |

当前训练环境激活逻辑要求 `bin/activate`，不等价于自动支持任意 Conda 安装方式。环境辅助文件仍使用 `/opt/buildtools/python-3.12.9/lib` 和 `/usr/local/Ascend/driver/lib64*` 等运行库路径；部署布局不同时需核对并适配该文件，不能只修改环境目录名。`npu-smi` 也需已安装并可从启动 shell 的 PATH 找到。

建议先进入专用 tmux 会话，再在该会话内执行后续 `export` 和启动命令，避免已有 tmux server 使用旧环境：

```bash
tmux new-session -s qwen35-swegym
# 在新会话内进入实际 checkout；以下占位路径必须替换。
cd '<本仓库绝对路径>'
```

### 3.2 配置本机路径、bundle 和 AgentENV

以下值是配置示例，必须指向训练机上**已经存在**的安装和数据。脚本中的默认路径不是安装步骤；部分默认值仍来自特定部署，请显式覆盖。修改前缀不会搬动文件。

```bash
# NPU 主机上的训练、推理环境和模型。
export TRAIN_ENV=/home/developer/envs/qwen35-rl-py312
export VIME_TRAIN_PYTHON="$TRAIN_ENV/bin/python"
export ROLLOUT_ENV=/home/developer/envs/qwen35-rollout-py311
export VIME_CANN_HOME=/home/developer/Ascend/cann92-isolated/cann
export MODEL_PATH=/home/developer/models/Qwen3.5-4B

# 第 1 节 finalize 的产物；如果在别的机器准备，先复制整个 bundle。
export VIME_SWEGYM_PREPARED_BUNDLE="$PWD/prepared/swegym-293"
export VIME_SWEGYM_SOURCE_PATH='<固定 revision 的 SWE-Gym evaluator 源码绝对路径>'

# 第 2 节准备并校验过的制品，必须能从 NPU rollout worker 读取。
export VIME_AGENT_NODE_TARBALL=/home/developer/agentenv-artifacts/node-v22.20.0-linux-x64.tar.xz
export VIME_AGENT_CODEX_TARBALL=/home/developer/agentenv-artifacts/vime-codex-offline-0.153.0.tgz

# 将兼容的 E2B SDK 安装到训练 Python 环境后，无需借用其他 site-packages。
export VIME_AGENT_E2B_SDK_PATH=''
# 如确需使用外部兼容 SDK，则将该变量改为其 site-packages 绝对路径。

# NPU -> AgentENV：沿用第 1 节的控制隧道，API key 文件位于 NPU 主机。
export E2B_API_URL=http://127.0.0.1:18000
export E2B_SANDBOX_URL="$E2B_API_URL"
export VIME_AGENTENV_API_KEY_FILE="$HOME/.config/agentenv/api-key"

# Sandbox -> Vime Responses Adapter：不是上面的 AgentENV API 地址。
export ADAPTER_BIND_HOST=0.0.0.0
export ADAPTER_PORT=18001
export ADAPTER_PUBLIC_HOST=192.0.2.1  # 文档占位符；正式启动前替换为实际可达主机。
export ADAPTER_PUBLIC_PORT=18080    # 已部署 relay 的端口；按实际拓扑调整。

# 以下路径存在于 sandbox 内。也可以同时设为空，使用 sandbox 原有 PATH。
export VIME_AGENT_TASK_CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh
export VIME_AGENT_TASK_CONDA_PREFIX=/opt/miniconda3/envs/testbed
```

设置 `VIME_SWEGYM_PREPARED_BUNDLE` 后，脚本从该目录统一选取 `rollout/train.no_gold.jsonl`、`environment/agentenv_templates.jsonl`、`verifier/verifier_specs.jsonl` 和 `preparation_manifest.json`。正式启动前检查任务 ID 唯一性、三份映射的一致性和 evaluator 可用性；任务数从数据读取，不要求恰好 288。当前要求任务数能被 `SWE_ROLLOUT_BATCH_SIZE` 整除。

训练时有两个独立的网络方向：

```text
NPU worker -> 127.0.0.1:18000 -> 控制隧道 -> AgentENV API / sandbox service
Sandbox Codex -> ADAPTER_PUBLIC_HOST:ADAPTER_PUBLIC_PORT
              -> 已有 relay/转发 -> NPU 上的 ADAPTER_PORT -> vLLM
```

若 sandbox 可直接访问 NPU adapter，可将 public host 设为可达的 NPU 地址，并让 public port 与 `ADAPTER_PORT` 一致。使用 relay 时，需要事先将请求转发到 NPU adapter，并保留请求的认证头和请求体；本脚本不创建 relay 或 SSH 隧道。`127.0.0.1:18000/health` 成功只验证控制方向，不能证明模型回调方向已连通。Adapter 在 rollout 初始化时启动，启动前该端口尚未监听是正常现象。

`192.0.2.1` 只用于文档，正式启动会直接拒绝，`--dry-run` 允许保留。真实地址通过当前 shell 或不提交的个人配置提供。脚本从 key 文件读取 `E2B_API_KEY`，不需要把 key 内容写入命令；此 key 用于 AgentENV 控制，模型请求使用 Harness 分配的 trial/session 凭据。

### 3.3 查看参数并进行 dry-run

先指定当前空闲的物理 NPU IDs。下面的 `0..15` 仅为示例，必须根据实际占用调整；`VIME_SLOT` 选择独立的 Ray/HCCL 端口组，但不会自动避免 NPU 或 adapter 端口重叠。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VIME_SLOT=31

bash examples/coding_agent_rl/run-qwen35-4B-swegym-torchtitan.sh --help
bash examples/coding_agent_rl/run-qwen35-4B-swegym-torchtitan.sh --dry-run
```

Dry-run 输出 JSON，其中 `command` 是完整的 Python 入口和训练参数。它不激活环境、不读取 key、不创建输出目录、不访问 NPU，也不启动 Ray。若 prompt 文件本地可读，会检查 ID 并计算任务数；不可读时 `prompt_count` 和 `planned_updates` 为 `null`，表示未知。Dry-run 不检查 template/verifier 映射、运行时依赖或网络连通性，因此通过不代表可以直接完成训练。

| 配置 | 默认值 / 行为 |
| --- | --- |
| `SWE_ROLLOUT_BATCH_SIZE` / `SWE_N_SAMPLES_PER_PROMPT` | 16 个 prompt × 每个 8 次尝试；global batch size 自动设为 128 |
| `SWE_NUM_EPOCHS` | 2；更新次数 = 任务数 ÷ prompt batch size × epochs |
| 采样 | temperature=1.0、Top-P=1.0、Top-K=-1（禁用） |
| context / response / tokens per GPU | 32768 / 32767 / 32768；Adapter 将单次请求限制在剩余上下文内 |
| `VIME_AGENT_MAX_TURNS_PER_SESSION` | 72 |
| `SWE_TRIAL_CONCURRENCY` / `SWE_BOOT_CONCURRENCY` | 8 / 8；trial concurrency 需能被 rollout engine 数整除 |
| Agent / verifier 时间预算 | 各 3600 秒；verifier 实际预算还受 bundle 中 `VerifierSpec.timeout_sec` 限制 |
| `SWE_LEARNING_RATE` | `2e-6`；设为空保留 TorchTitan recipe 的学习率 |
| 异步权重同步 | `train_async.py`，`--update-weights-interval 1`；TIS 开启 |
| Reward / loss | GRPO 路径，reward normalization 关闭；使用有效 token 加权的 loss |
| 推理 | prefix cache 开启，`FULL_DECODE_ONLY` 图模式；`--no-vllm-async-scheduling` 不会关闭训练侧的异步流水线 |
| 保存 | 每 6 次更新保存，保留最近 2 个 TorchTitan checkpoint；`SWE_SAVE_INTERVAL=off` 关闭保存 |
| Rollout dump / tracing | `SWE_DUMP_MODE=rollout`；详细 tracing 关闭。Dump 仍会占用磁盘，可设 `off` 关闭 |

若只有 12 个可用 dies，可保留训练 TP=4、DP=2，设置 `SWE_ROLLOUT_GPUS=4` 并列出 12 个空闲物理 IDs；这仍需满足实际模型的显存需求。仅改变可见设备列表不会自动缩小训练或 rollout 的分配。

### 3.4 启动 2×2 试跑或默认训练

完成路径配置并替换 `ADAPTER_PUBLIC_HOST` 后，在同一个 tmux 会话中启动 2 个 prompt × 每个 2 次尝试的试跑：

```bash
RUN_NAME="qwen35-4B-swegym-2x2-$(date +%Y%m%d-%H%M%S)-async" \
SWE_ROLLOUT_BATCH_SIZE=2 \
SWE_N_SAMPLES_PER_PROMPT=2 \
SWE_SAVE_INTERVAL=off \
SWE_DUMP_MODE=rollout \
bash examples/coding_agent_rl/run-qwen35-4B-swegym-torchtitan.sh
```

这只缩小每批尝试数并关闭 checkpoint，不缩短 context、max tokens per GPU 或 session 轮数，**也没有设置试跑步数上限**。仍按 `SWE_NUM_EPOCHS` 遍历数据：例如 288 个任务、2 epochs 时会运行 288 次更新，而不是只运行一批。需要结束试跑时，在运行命令的 tmux pane 中按 `Ctrl-C`。

默认 16×8 配置可用下面的命令启动；先结束同一 slot/设备上的试跑，再运行新任务：

```bash
RUN_NAME="qwen35-4B-swegym-16x8-$(date +%Y%m%d-%H%M%S)-async" \
SWE_ROLLOUT_BATCH_SIZE=16 \
SWE_N_SAMPLES_PER_PROMPT=8 \
SWE_SAVE_INTERVAL=6 \
SWE_DUMP_MODE=rollout \
bash examples/coding_agent_rl/run-qwen35-4B-swegym-torchtitan.sh
```

脚本激活训练 Python/CANN，并通过 `VIME_VLLM_PYTHON` 选择推理解释器；NPU engine 强制使用 `spawn`。推理进程继承 Ray worker 的环境，脚本不再单独激活推理 CANN/ATB、重设库路径或清除训练专用变量。仅选择不同 Python 不等于完整环境隔离；使用不同运行库的部署需先适配环境，再验证训练。

### 3.5 查看输出与确认训练进度

输出默认位于 `outputs/${RUN_NAME}/`，也可以通过 `OUTPUT_DIR` 指定。脚本会拒绝已存在的输出根目录；无需预先 `mkdir`，也无需额外 `tee` 或日志重定向。

| 路径（相对于输出根目录） | 内容 |
| --- | --- |
| `tensorboard/<RUN_NAME>/<时间戳>/train.log` | Ray job 的运行日志 |
| `tensorboard/<RUN_NAME>/<时间戳>/rerun.sh` | 归档的 Ray 启动和训练提交命令，包含本机配置，分享前需检查 |
| `tensorboard/` | TensorBoard 日志 |
| `rollout_dumps/<rollout_id>.pt` | 开启 dump 时保存的 rollout 样本及 metadata |
| `checkpoints/` | 开启保存时写入的 checkpoint |

依次确认 vLLM worker 注册、Agent sandbox 执行、verifier reward、`train metrics @ rollout_id=...` 和 `Timer update_weights end`。HTTP 200 或 rollout 完成不等于已经完成训练更新。Reward=0 本身不是运行错误；当前关闭 reward normalization，整批 reward=0 时可能出现 `loss=0`、`grad_norm=0`。非零 `agent_exit_code` 则应结合 Agent 结束原因和 verifier 结果另行检查。

使用 `Ctrl-B` 再按 `D` 可离开 tmux 而不停止训练；返回时执行 `tmux attach -t qwen35-swegym`。本脚本不安装依赖、不准备模板，也不会修复断开的控制隧道或模型回调网络。
