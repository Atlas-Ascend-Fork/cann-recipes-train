# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Build the fixed validation subset both precisions validate on.

The rule is fixed before any run: from the upstream test.parquet, keep the rows
whose extra_info index is a multiple of 25. Run this from the verl source tree.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

TEST_PARQUET = Path("./data/math/data/test.parquet")
VAL_SUBSET_PARQUET = Path("./data/math/data/val_subset.parquet")
STRIDE = 25

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if not TEST_PARQUET.is_file():
        raise SystemExit(f"not found: {TEST_PARQUET.resolve()}; run the upstream preprocessing first")

    frame = pd.read_parquet(TEST_PARQUET)
    subset = frame[frame["extra_info"].map(lambda info: info["index"] % STRIDE == 0)]
    if subset.empty:
        raise SystemExit(f"no row has an extra_info index that is a multiple of {STRIDE}")

    # index=False keeps the column set identical to test.parquet.
    subset.to_parquet(VAL_SUBSET_PARQUET, index=False)

    logger.info("rule   : keep extra_info index %% %d == 0", STRIDE)
    logger.info("input  : %s (%d rows)", TEST_PARQUET.resolve(), len(frame))
    logger.info("output : %s (%d rows)", VAL_SUBSET_PARQUET.resolve(), len(subset))
    logger.info("The run drops prompts over data.max_prompt_length, so training reports fewer.")


if __name__ == "__main__":
    main()
