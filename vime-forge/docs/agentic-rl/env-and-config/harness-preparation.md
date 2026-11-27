# Agent 安装、初始化与安装包分发

本文适用于 Codex、Claude Code 两种 Agent，以及 AgentENV、Harbor 两种沙盒环境，面向 SWEGym 和 TMax 两种任务。

## 1. Agent Harness 的安装与准备

两个任务入口都接受 `--agent codex` 或 `--agent claude_code`，默认使用 Codex。也可在 `.local.sh` 中设置 `VIME_RL_AGENT`；显式 `--agent` 优先。

环境变量的定义和 harness 版本

| Agent | 预装检查与复用条件 | 需要安装时使用的文件 |
| --- | --- | --- |
| Codex | 为了方便不同版本 harness 实现AgenticRL训练，`VIME_AGENT_CODEX_VERSION` 显式定义Codex 版本，从安装包名的 `-版本号.tgz` 后缀推导。运行前执行 `codex --version` 对Codex harness 版本进行验证 | `VIME_AGENT_NODE_TARBALL` 与 `VIME_AGENT_CODEX_TARBALL`  |
| Claude Code | `/usr/local/bin/claude --version` 成功即复用；当前实现不比较预装 CLI 与安装包的版本 | `VIME_AGENT_NODE_TARBALL` 与 `VIME_AGENT_CC_TARBALL` |


## 2. 准备安装包

正式Agentic RL训练开始前，请使用目标CPU服务器对 Agent Harness (Codex 或 Claude) 进行打包。

以 Codex 为例，默认 Codex 配置使用以下固定版本如下:

| 文件 | 用途 |
| --- | --- |
| `node-v22.20.0-linux-x64.tar.xz` | Node.js 与 npm 运行环境，Codex 和 Claude Code 共用 |
| `vime-codex-offline-0.153.0.tgz` | 包含 Linux x64 平台依赖的 Codex npm 安装包 |

### 2.1 下载并校验 Node.js

下载地址与校验值来自 [Node.js v22.20.0 官方归档](https://nodejs.org/dist/v22.20.0/)和该版本的 [SHASUMS256.txt](https://nodejs.org/dist/v22.20.0/SHASUMS256.txt)。

```bash
# 执行位置：CPU 沙箱服务器宿主环境。
set -euo pipefail
export HARNESS_ARTIFACTS=/path/on/cpu/agent-artifacts
mkdir -p "$HARNESS_ARTIFACTS"

curl -fL --retry 3 \
  https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz \
  -o "$HARNESS_ARTIFACTS/node-v22.20.0-linux-x64.tar.xz"

printf '%s  %s\n' \
  00bbd05e306ea68b6e13e17360d0e2f680b493ef95f2fea1c4296ff7437530bc \
  "$HARNESS_ARTIFACTS/node-v22.20.0-linux-x64.tar.xz" | sha256sum -c -

# 解压到 CPU 沙箱服务器的临时目录，为下面的打包与检查提供 Node/npm。
export HARNESS_BUILD_DIR="$(mktemp -d)"
tar -xJf "$HARNESS_ARTIFACTS/node-v22.20.0-linux-x64.tar.xz" \
  -C "$HARNESS_BUILD_DIR"
export PATH="$HARNESS_BUILD_DIR/node-v22.20.0-linux-x64/bin:$PATH"
node --version
npm --version
```

`node --version` 应输出 `v22.20.0`。保留原始 `.tar.xz` 用于分发。这里的 `PATH` 修改只服务于 CPU 沙箱服务器打包，不应照搬到 NPU 服务器的 `.local.sh`。

### 2.2 制作 Codex 离线安装包

Codex `0.153.0` 的主包通过可选依赖引用平台二进制，仅保存主包不能保证离线安装。下面将 [主包](https://registry.npmjs.org/@openai%2fcodex/0.153.0)与 [Linux x64 平台包](https://registry.npmjs.org/@openai%2fcodex/0.153.0-linux-x64)一起打入 tgz。`vime-codex-offline` 是这里重新打包的名称，不是另一个官方软件包。

```bash
# 执行位置：CPU 沙箱服务器宿主环境。
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

该步骤修改包名和依赖打包方式，保留原有启动器及平台二进制。输出文件为 `vime-codex-offline-0.153.0.tgz`。使用空 npm 缓存检查安装，并记录生成文件的校验值：

```bash
# 执行位置：CPU 沙箱服务器宿主环境。
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

应输出 `codex-cli 0.153.0`。重新打包后的 SHA256 以实际产物为准。空缓存安装失败时，应检查平台依赖是否已包含在包内，不要依赖 CPU 沙箱服务器原有 npm 缓存掩盖缺失依赖。

## 3. 外部配置与安装包分发 (以 Codex 为例)

### 3.1 将安装包复制到 NPU 服务器并配置路径

CPU 沙箱服务器上制作的安装包不会自动出现在 NPU 服务器。先把所选包和校验清单复制到 NPU 服务器上，放在轨迹生成进程可读取的持久目录。下面以 Codex 为例，在 **NPU 服务器**执行；`cpu-sandbox-a` 表示已在该 NPU 服务器的 SSH 配置中设置了地址、端口和密钥的 CPU 沙箱服务器别名：

```bash
# 执行位置：NPU 服务器。
export HARNESS_ARTIFACTS=/path/on/npu/agent-artifacts
mkdir -p "$HARNESS_ARTIFACTS"
scp cpu-sandbox-a:/path/on/cpu/agent-artifacts/node-v22.20.0-linux-x64.tar.xz \
    cpu-sandbox-a:/path/on/cpu/agent-artifacts/vime-codex-offline-0.153.0.tgz \
    cpu-sandbox-a:/path/on/cpu/agent-artifacts/harness-sha256.txt \
    "$HARNESS_ARTIFACTS/"
(
  cd "$HARNESS_ARTIFACTS"
  sha256sum -c harness-sha256.txt
)
```

选择 Claude Code 时，替换上面的 CLI 文件名，并使用包含 Node 和所选 CLI 文件的校验清单。多节点运行时，每台执行生成器的 NPU 服务器都需要能读取这些文件。这里的 `HARNESS_ARTIFACTS` 是 NPU 服务器路径，与第 2 节 CPU 服务器上的同名变量分别设置。

校验通过后，在 **NPU 服务器的外部 `.local.sh`** 中填写：

```bash
# 配置位置：NPU 服务器上的外部 .local.sh。
export VIME_RL_AGENT=codex
export VIME_AGENT_NODE_TARBALL=/path/on/npu/agent-artifacts/node-v22.20.0-linux-x64.tar.xz
export VIME_AGENT_CODEX_TARBALL=/path/on/npu/agent-artifacts/vime-codex-offline-0.153.0.tgz
# 包名无法推导版本时，显式填写这一项。
# export VIME_AGENT_CODEX_VERSION=0.153.0
```

使用 Claude Code 时，保留 Node 路径，改用：

```bash
# 配置位置：NPU 服务器上的外部 .local.sh。
export VIME_RL_AGENT=claude_code
export VIME_AGENT_CC_TARBALL=/path/on/npu/agent-artifacts/claude-code-offline.tgz
```

这些是 NPU 服务器路径，不是 CPU 沙箱服务器或沙箱内的路径。安装器需要安装时才使用对应文件；它不会去官方站点自动寻找缺失的 tarball。预装模式可以跳过安装，但部署中保留回退文件有助于处理预装检查未通过的沙箱。

完整参数及默认值见[安装包与准备过程](configuration.md#52-安装包与准备过程)。NPU 服务器的 Python、CANN 和 ATB 环境仍由公共 launcher 配置，不在本页增加另一层启动脚本。

### 3.2 由 NPU 服务器上传到沙箱

未设置 `VIME_AGENT_ARTIFACT_BASE_URL` 时，共享安装器通过沙箱文件接口上传文件：

1. NPU 服务器上的轨迹生成进程将 Node `.tar.xz` 解压为 tar，在 NPU 服务器的临时目录缓存，然后上传到 CPU 沙箱服务器上的任务沙箱。
2. 在任务沙箱内解压到 `/opt/node`，将 `node`、`npm`、`npx` 链接到 `/usr/local/bin`。
3. 将所选 CLI 的 tgz 上传到任务沙箱的 `/tmp/harness-cli.tgz`，在沙箱内通过 npm 安装到 `/usr/local`，执行版本检查。

该路径使用 AgentENV 的文件接口或 Harbor 的 SSH 工作进程，不要求沙箱访问外部包仓库。上传模式会占用 NPU 服务器到沙箱的网络带宽。因此，本工作也支持 Agent Harness 在远端CPU服务器的预加载。

### 3.3 使用 CPU服务器 本地服务加载Harness

设置 `VIME_AGENT_ARTIFACT_BASE_URL` 后，沙箱优先从 HTTP 服务下载 Node 和 CLI 安装包。
可在每台 CPU 沙箱服务器保存同一份制品并提供本地下载入口，减少安装包在 NPU 到 CPU 的 SSH 链路上的重复传输。

下载 URL 按以下规则生成：

```text
VIME_AGENT_ARTIFACT_BASE_URL + "/" + 安装包路径的文件名
```

因此，服务目录中的文件名必须与 `VIME_AGENT_NODE_TARBALL` 以及所选 CLI tarball 的文件名一致。先将同一批安装包和校验清单复制到每台提供下载服务的 CPU 沙箱服务器；只在其中一台完成打包，不代表其他服务器已经拥有这些文件。下面在**每台提供下载服务的 CPU 沙箱服务器宿主环境**执行，路径及监听地址按部署替换：

```bash
# 执行位置：CPU 沙箱服务器宿主环境。
export HARNESS_ARTIFACTS=/path/on/cpu/agent-artifacts
(
  cd "$HARNESS_ARTIFACTS"
  sha256sum -c harness-sha256.txt
)

python3 -m http.server 18123 --bind 172.17.0.1 \
  --directory "$HARNESS_ARTIFACTS"
```

服务目录只放用于分发的制品。`172.17.0.1` 是示例 Docker 网桥地址，必须先核对该 CPU 沙箱服务器的网桥地址及容器可达性；长期运行可用独立服务或 tmux 管理。训练入口不会自动启动 HTTP 服务。

在 NPU 服务器的 `.local.sh` 中增加：

```bash
# 配置位置：NPU 服务器上的外部 .local.sh。
export VIME_AGENT_ARTIFACT_BASE_URL=http://172.17.0.1:18123
```

当两台 CPU 沙箱服务器的容器都能通过该地址访问各自的本地服务时，可以使用同一个配置值；否则应使用所有目标沙箱均可达的共同入口。AgentENV 同样支持 HTTP 分发，但需填写其沙箱网络中可达的地址，不能直接照搬 Docker 网桥地址。

HTTP 模式直接传输原始 Node `.tar.xz`，在沙箱中解压。沙箱需要至少提供 curl、wget、Python 3 中的一种下载工具；解压 `.xz` 需要 xz，或带 `lzma` 的 Python 3。下载成功后不再上传同一个文件。
