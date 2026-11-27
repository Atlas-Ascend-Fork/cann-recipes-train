# AgentENV 服务部署与连接检查

本文默认已具备 NPU 服务器和 CPU 沙箱服务器，保留原 SWE 数据准备指南中的 AgentENV 部署参考，固定使用 `2d766447261a7252447fe8bccef2e87d48ece5a6` 提交及 E2B SDK `2.46.4`。已有可用服务时无需重复部署，只需确认 NPU 服务器能访问服务、创建模板沙箱并读写文件。

服务安装、配置和沙箱运行检查在 **CPU 沙箱服务器**执行；控制隧道和远程 SDK 检查命令在 **NPU 服务器**执行。SDK 发出的任务命令由 CPU 沙箱服务器上的沙箱运行。这些步骤不启动策略模型或 RL 训练；路径、资源和镜像代理均为部署示例，应按实际部署调整。

AgentENV 与 Harbor 的选择和配置见[沙箱部署与配置](sandboxes.md)。本页不适用于 运行 Harbor Docker 的 CPU 沙箱服务器。任务 bundle 准备分别见 [SWE-Gym 数据准备](../swe-agentic-rl/data-preparation.md)和 [Terminal／TMax 数据准备](../terminal-agentic-rl/data-preparation.md)。

## 1. 部署 AgentENV 服务

以下命令在**CPU 沙箱服务器**执行，主机应具备可用的 KVM、TUN、ublk 和 cgroup v2，本文以 Linux 6.8+ 为部署基线。OCI 容器镜像由 AgentENV 转换为轻量虚拟机环境，无需额外安装 Docker 守护进程。CPU沙盒服务器的配置要求可以参见 [AgentENV 要求和解读](https://github.com/kvcache-ai/AgentENV/issues/117)。

先检查主机并验证 KVM 确实可以创建虚拟机：

```bash
# 执行位置：CPU 沙箱服务器宿主环境。
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

若云实例未开放 KVM，需要更换实例规格或启用平台支持的嵌套虚拟化。磁盘需同时容纳镜像、模板和沙箱写层；本文为每个任务实例配置 16 vCPU、64 GiB 内存（65536 MiB），准备并发需按主机余量设置。

安装系统依赖，获取已验证的 AgentENV 源码版本：

```bash
# 执行位置：CPU 沙箱服务器宿主环境。
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
# 执行位置：CPU 沙箱服务器宿主环境。
cargo build --release --locked \
  -p agentenv --bin server \
  -p uvm-ublk-daemon --bin uvm-ublk-daemon
cargo build --release --locked -p aenv
```

创建服务账号并安装文件。已有部署应先核对现有账号和配置，不要覆盖：

```bash
# 执行位置：CPU 沙箱服务器宿主环境。
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

`--setup-only` 准备 Firecracker、guest kernel、镜像工具等运行环境；`--setup-host` 配置宿主机设备权限与网络。离线部署需按该 commit 的依赖清单准备并校验制品，再执行完整性检查。

在 **CPU 沙箱服务器**创建 `/etc/default/aenv`：

```ini
API_ADDR="127.0.0.1:8000"
AENV_CONFIG_PATH="/var/lib/aenv/config/config.toml"
AENV_HOME_PATH="/var/lib/aenv"
AENV_RUNTIME_PATH="/run/aenv"
AENV_VIRTUALIZATION_MODE="kvm"
```

在 **CPU 沙箱服务器**创建 `/etc/systemd/system/aenv.service`：

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
# 执行位置：CPU 沙箱服务器宿主环境。
sudo systemctl daemon-reload
sudo systemctl enable --now aenv
sudo systemctl status aenv --no-pager
sudo journalctl -u aenv -n 100 --no-pager
curl -i --max-time 10 http://127.0.0.1:8000/health
```

预期健康检查返回 HTTP 204。首次启动生成的 API 密钥位于 CPU 沙箱服务器的 `/var/lib/aenv/secrets/api-key`。仍在该服务器运行 `aenv auth`，填写其本机服务地址和实际 API 密钥；随后验证 `aenv template list`、`aenv ls` 可以正常返回。API 密钥不应提交到仓库或写入数据集。

## 2. 验证沙箱运行与 NPU 服务器连接

在**CPU 沙箱服务器**创建最小模板，等待状态为 `ready`：

```bash
# 执行位置：CPU 沙箱服务器宿主环境。
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

模板保留供后续复用，删除沙箱不等于删除模板。示例镜像代理需在部署网络中可访问；也可以使用具有相同镜像的 registry。

在**NPU 服务器**使用已配置密钥认证、允许 TCP forwarding 的 SSH 账号建立控制隧道，并保持该进程运行：

```bash
# 执行位置：NPU 服务器。
ssh -N -T \
  -i '<SSH 私钥绝对路径>' \
  -o IdentitiesOnly=yes -o BatchMode=yes \
  -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18000:127.0.0.1:8000 \
  '<SSH 用户>@<AgentENV 服务器地址>'
```

首次使用前核对并保存 SSH 主机指纹。控制 API 保持 loopback 监听，无需向公网开放 8000 端口。隧道可在独立 tmux 会话中运行；SSH 保活不会自动重连。

将 CPU 沙箱服务器生成的 API 密钥安全复制到 NPU 服务器的 `~/.config/agentenv/api-key`，文件权限设为 `0600`。在 **NPU 服务器的另一终端**创建独立客户端环境并检查控制链路；示例中的 `127.0.0.1:18000` 指向前一步在 NPU 服务器建立的隧道：

```bash
# 执行位置：NPU 服务器。
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

完成连接检查后，按任务阅读 [SWE-Gym 数据准备](../swe-agentic-rl/data-preparation.md)或 [Terminal／TMax 数据准备](../terminal-agentic-rl/data-preparation.md)。此处的最小沙箱检查不替代具体任务镜像与 bundle 的准备检查。
