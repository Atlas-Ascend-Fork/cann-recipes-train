#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""按 LOCAL_RANK 为每个训练 rank 注入一个 vLLM endpoint（Qwen3.8-27B）。

由 torchrun 启动（替代 ``-m speculators.train``），其余参数透传给
``speculators.train``。仅生成请求走路由：mooncake 下 hidden-states 按
req_id 经 store 直连拉取，与 endpoint 绑定无关；file backend 下训练侧
与 vLLM 实例须共享文件系统。

本样例训练为单机（torchrun --standalone），``LOCAL_RANK`` 即全局 rank；
多机训练各节点 ``LOCAL_RANK`` 重复，需改用 ``RANK`` 路由。

端点表取环境变量 ``ENGINES``（空格或逗号分隔的 ``host:port``）；逗号分隔的
完整 URL 列表 ``ENDPOINT_ROUTES`` 同样受支持。rank r 使用第
``r % len(routes)`` 条。

用法：
    export ENGINES="10.0.0.1:8010 10.0.0.2:8010"
    torchrun --standalone --nproc-per-node 4 scripts/route_rank_endpoints_qwen3_8_27b.py \
        --verifier-name-or-path ... --hidden-states-backend mooncake ...
"""
import logging
import os
import runpy
import sys
from pathlib import Path

# 共享的端点表解析在 speculators/utils/：本仓库布局下位于脚本上两级，
# 拷入 speculators 源码仓后位于脚本上一级（scripts/ 的同级），向上找到即挂上 sys.path
for _base in Path(__file__).resolve().parents:
    if (_base / "utils" / "endpoint_routes.py").is_file():
        sys.path.insert(0, str(_base / "utils"))
        break
from endpoint_routes import parse_routes  # noqa: E402

LOG = logging.getLogger("route_rank_endpoints_qwen3_8_27b")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(message)s"))
LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)
LOG.propagate = False


def main():
    raw = os.environ.get("ENDPOINT_ROUTES", "") or os.environ.get("ENGINES", "")
    if not raw:
        sys.exit(
            "route_rank_endpoints_qwen3_8_27b: 请通过环境变量 ENGINES（host:port 列表）"
            "或 ENDPOINT_ROUTES（URL 列表）提供端点，"
            "例如 ENGINES=\"10.0.0.1:8010 10.0.0.2:8010\""
        )
    routes = parse_routes(raw)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    vllm_url = routes[local_rank % len(routes)]

    # 注入后再交棒：speculators.train 看到的仍是单端点的普通命令行。
    sys.argv.extend(["--vllm-endpoint", vllm_url])
    LOG.info(
        f"[route_rank_endpoints_qwen3_8_27b] rank {local_rank}/{len(routes)} routes -> "
        f"vllm={vllm_url}"
    )

    runpy.run_module("speculators.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
