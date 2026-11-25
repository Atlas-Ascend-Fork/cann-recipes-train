# Copyright (c) 2026 SMULL_Group, Harbin Institute of Technology, Shenzhen.
# SPDX-License-Identifier: MIT

from huggingface_hub import snapshot_download

# 前 1/4 的 train shards: 00000 ~ 00062，共 63 个
allow_patterns = [
    f"data/train-{i:05d}-of-00250.parquet"
    for i in range(63)
]

local_path = snapshot_download(
    repo_id="gmongaras/SlimPajama-627B_Reupload",
    repo_type="dataset",
    allow_patterns=allow_patterns,
    max_workers=24,
    # local_dir="./SlimPajama_part1",   # 想指定下载目录就取消注释
)

import logging

logger = logging.getLogger(__name__)

logger.info("文件夹已下载到: %s", local_path)
