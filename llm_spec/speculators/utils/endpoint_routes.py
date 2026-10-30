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

"""两个样例训练 wrapper（route_rank_endpoints_*.py）共用的端点表解析。"""

from urllib.parse import urlparse


def parse_routes(raw):
    """端点表 -> URL 列表。

    接受空格或逗号分隔的 ``host:port``（自动补 ``http://`` 前缀与 ``/v1``
    后缀），也接受逗号分隔的完整 URL；条目缺 host 或端口时报错。
    """
    routes = []
    for entry in raw.replace(",", " ").split():
        if "://" not in entry:
            entry = f"http://{entry}"
        if not entry.rstrip("/").endswith("/v1"):
            entry = entry.rstrip("/") + "/v1"
        parsed = urlparse(entry)
        if not parsed.hostname or parsed.port is None:
            raise ValueError(f"端点条目缺少 host:port: {entry!r}")
        routes.append(entry)
    if not routes:
        raise ValueError("端点列表为空")
    return routes


def parse_masters(raw, n_routes):
    """mooncake master 表 -> 列表：条数须与端点数一致（双 master 拓扑）。"""
    masters = [m for m in raw.split(",") if m]
    if len(masters) != n_routes:
        raise ValueError(
            f"MOONCAKE_MASTER_ROUTES 条数({len(masters)})须与端点数({n_routes})一致"
        )
    return masters
