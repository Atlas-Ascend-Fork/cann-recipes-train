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

"""按 LOCAL_RANK 为每个训练 rank 注入一个 HS 引擎 endpoint 与 mooncake master（GLM-5.2）。

由 torchrun 启动（替代 ``-m speculators.train``），其余参数透传给
``speculators.train``。仅生成请求走路由：hidden-states 始终经 mooncake
store 按 req_id 拉取，与 endpoint 绑定无关。

端点表取环境变量 ``ENGINES``（空格或逗号分隔的 ``host:port``）；逗号分隔的
完整 URL 列表 ``ENDPOINT_ROUTES`` 同样受支持。rank r 使用第
``r % len(routes)`` 条。

可选的 ``MOONCAKE_MASTER_ROUTES``（逗号分隔，与端点表等长同序）按 rank
注入对应的 ``--mooncake-master``，用于双 master 拓扑（每台引擎机器一个
master）；未设置时使用 ``--mooncake-master`` 单 master。

本样例训练为单机（torchrun --standalone），``LOCAL_RANK`` 即全局 rank；
多机训练各节点 ``LOCAL_RANK`` 重复，需改用 ``RANK`` 路由。

用法：
    export ENGINES="10.0.0.1:6700 10.0.0.1:6701 ... 10.0.0.2:6703"
    export MOONCAKE_MASTER_ROUTES="10.0.1.1:50051,10.0.1.1:50051,...,10.0.1.2:50051"
    torchrun --standalone --nproc-per-node 8 scripts/route_rank_endpoints_glm5_2.py \
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
from endpoint_routes import parse_masters, parse_routes  # noqa: E402

LOG = logging.getLogger("route_rank_endpoints_glm5_2")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(message)s"))
LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)
LOG.propagate = False


def main():
    raw = os.environ.get("ENDPOINT_ROUTES", "") or os.environ.get("ENGINES", "")
    if not raw:
        sys.exit(
            "route_rank_endpoints_glm5_2: 请通过环境变量 ENGINES（host:port 列表）"
            "或 ENDPOINT_ROUTES（URL 列表）提供端点，"
            "例如 ENGINES=\"10.0.0.1:6700 10.0.0.1:6701\""
        )
    routes = parse_routes(raw)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    vllm_url = routes[local_rank % len(routes)]

    # 注入后再交棒：speculators.train 看到的仍是单端点的普通命令行。
    argv_extra = ["--vllm-endpoint", vllm_url]
    masters_raw = os.environ.get("MOONCAKE_MASTER_ROUTES", "")
    master_note = ""
    if masters_raw:
        masters = parse_masters(masters_raw, len(routes))
        # argparse 后出现者覆盖：这里追加的按 rank 路由值，会盖掉命令行里
        # 训练脚本透传的单 master 值（--mooncake-master）
        argv_extra += ["--mooncake-master", masters[local_rank % len(masters)]]
        master_note = f"  mooncake-master={masters[local_rank % len(masters)]}"
    sys.argv.extend(argv_extra)
    LOG.info(
        f"[route_rank_endpoints_glm5_2] rank {local_rank}/{len(routes)} routes -> "
        f"vllm={vllm_url}{master_note}"
    )

    runpy.run_module("speculators.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
