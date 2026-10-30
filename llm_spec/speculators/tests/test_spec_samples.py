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

"""样例自带脚本的边界用例（不依赖 NPU / vLLM / speculators 运行时）。

覆盖四处最容易改坏、又最难靠人工跑流程发现的逻辑：

* ``utils/endpoint_routes.py``（两个 ``route_rank_endpoints_*.py`` wrapper
  共用的解析）：端点表与 mooncake master 表的解析；
* ``glm5.2/utils/opb_regen_to_tokenized_jsonl.py``：对话角色白名单
  （未识别角色整条丢弃）与 loss_mask 只覆盖 assistant 段；
* ``utils/select_cmb_chinese_prompts.py``（speculators/utils/ 共用脚本）：
  中文 prompt 的取轮、中文占比过滤与降序取前 N 条；
* ``qwen3.8_27b/utils/sample_opb_by_source.py``：英文源池（先抽再重生成）
  的 source 归组、比例分配（守恒 + 确定性）与分层抽样。

运行：

    python -m unittest discover -s llm_spec/speculators/tests -v

数据转换脚本依赖 numpy / pyarrow / transformers，缺失时相关用例自动跳过。
"""

import importlib.util
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path


SAMPLES_DIR = Path(__file__).resolve().parents[1]


def _load(path, name):
    """按路径加载样例脚本（文件名含小数点，不能直接按模块名 import）。"""
    spec = importlib.util.spec_from_file_location(name, SAMPLES_DIR / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 两个 wrapper 共用的端点表解析（speculators/utils/）；先于 wrapper 加载，
# wrapper 里的 `from endpoint_routes import ...` 会复用这里注册的同一模块实例
ENDPOINT_ROUTES = _load("utils/endpoint_routes.py", "endpoint_routes")

ROUTE_WRAPPERS = {
    "qwen": _load(
        "qwen3.8_27b/scripts/route_rank_endpoints_qwen3_8_27b.py",
        "route_rank_endpoints_qwen",
    ),
    "glm": _load(
        "glm5.2/scripts/route_rank_endpoints_glm5_2.py",
        "route_rank_endpoints_glm",
    ),
}

try:
    OPB = _load(
        "glm5.2/utils/opb_regen_to_tokenized_jsonl.py", "opb_regen_to_tokenized_jsonl"
    )
except ImportError as exc:  # 缺 numpy / pyarrow / transformers 时跳过相关用例
    OPB = None
    OPB_IMPORT_ERROR = exc

# 纯标准库，任何环境都能加载（speculators/utils/ 共用脚本）
CMB_SELECT = _load(
    "utils/select_cmb_chinese_prompts.py", "select_cmb_chinese_prompts"
)
OPB_SAMPLE = _load(
    "qwen3.8_27b/utils/sample_opb_by_source.py", "sample_opb_by_source"
)


class TestEndpointRoutes(unittest.TestCase):
    """共享的端点表解析（两个样例的 wrapper 都走 speculators/utils/ 这一套）。"""

    def test_adds_scheme_and_v1_suffix(self):
        self.assertEqual(
            ENDPOINT_ROUTES.parse_routes("10.0.0.1:8010"),
            ["http://10.0.0.1:8010/v1"],
        )

    def test_keeps_existing_scheme_and_v1(self):
        self.assertEqual(
            ENDPOINT_ROUTES.parse_routes("http://10.0.0.1:8010/v1"),
            ["http://10.0.0.1:8010/v1"],
        )

    def test_accepts_space_and_comma_separated(self):
        self.assertEqual(
            ENDPOINT_ROUTES.parse_routes("10.0.0.1:8010,10.0.0.2:8010"),
            ["http://10.0.0.1:8010/v1", "http://10.0.0.2:8010/v1"],
        )
        self.assertEqual(
            ENDPOINT_ROUTES.parse_routes("10.0.0.1:8010 10.0.0.2:8010"),
            ["http://10.0.0.1:8010/v1", "http://10.0.0.2:8010/v1"],
        )

    def test_missing_port_raises(self):
        with self.assertRaises(ValueError):
            ENDPOINT_ROUTES.parse_routes("http://10.0.0.1")

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            ENDPOINT_ROUTES.parse_routes("")


class TestRouteWrappers(unittest.TestCase):
    """wrapper 本体可加载，且端点解析直接复用 speculators/utils/ 的共享实现。"""

    def test_wrappers_reuse_shared_parser(self):
        for name, mod in ROUTE_WRAPPERS.items():
            with self.subTest(sample=name):
                self.assertTrue(callable(mod.main))
                self.assertIs(mod.parse_routes, ENDPOINT_ROUTES.parse_routes)


class TestParseMasters(unittest.TestCase):
    """双 master 拓扑：master 条数必须与端点条数一致。"""

    def test_splits_on_comma(self):
        self.assertEqual(
            ENDPOINT_ROUTES.parse_masters("10.0.1.1:50051,10.0.1.2:50051", 2),
            ["10.0.1.1:50051", "10.0.1.2:50051"],
        )

    def test_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ENDPOINT_ROUTES.parse_masters("10.0.1.1:50051,10.0.1.2:50051", 3)


@unittest.skipIf(OPB is None, "缺少 numpy / pyarrow / transformers，跳过数据转换用例")
class TestNormalizeConversation(unittest.TestCase):
    def test_maps_sharegpt_roles(self):
        conv = [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ]
        self.assertEqual(
            OPB.normalize_conversation(conv),
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )

    def test_keeps_already_normalized_roles(self):
        conv = [{"role": "user", "content": "hi"}]
        self.assertEqual(
            OPB.normalize_conversation(conv),
            [{"role": "user", "content": "hi"}],
        )

    def test_unknown_role_drops_whole_sample(self):
        # function_call / observation 不能当成 assistant：否则工具调用与工具
        # 返回会被写进监督区间
        conv = [
            {"from": "human", "value": "hi"},
            {"from": "function_call", "value": "ls"},
            {"from": "observation", "value": "README"},
        ]
        self.assertIsNone(OPB.normalize_conversation(conv))


@unittest.skipIf(OPB is None, "缺少 numpy / pyarrow / transformers，跳过数据转换用例")
class TestLossMaskFromOffsets(unittest.TestCase):
    def test_masks_only_assistant_span(self):
        text = "user: hi assistant: hello"
        # token 边界： "user: hi " / "assistant: " / "hello"
        offsets = [(0, 9), (9, 19), (19, 24)]
        pattern = r"assistant: (hello)"
        self.assertEqual(OPB.loss_mask_from_offsets(text, offsets, pattern), [0, 0, 1])

    def test_no_match_gives_all_zero(self):
        text = "user: hi"
        self.assertEqual(OPB.loss_mask_from_offsets(text, [(0, 9)], r"(nope)"), [0])


class TestSelectCmbChinesePrompts(unittest.TestCase):
    """CMB 中文 prompt 的取轮与筛选规则（glm5.2 §6.3 / qwen3.8_27b §6.2.3 共用脚本）。"""

    def test_takes_first_human_turn_only(self):
        record = {
            "conversations": [
                {"from": "human", "value": "第一个问题"},
                {"from": "assistant", "value": "回答"},
                {"from": "human", "value": "第二个问题"},
            ]
        }
        self.assertEqual(CMB_SELECT.first_human_text(record), "第一个问题")

    def test_accepts_messages_field_with_role_content(self):
        record = {"messages": [{"role": "user", "content": "你好"}]}
        self.assertEqual(CMB_SELECT.first_human_text(record), "你好")

    def test_skips_empty_human_value(self):
        record = {
            "conversations": [
                {"from": "human", "value": ""},
                {"from": "assistant", "value": "回答"},
            ]
        }
        self.assertIsNone(CMB_SELECT.first_human_text(record))

    def test_no_human_turn_returns_none(self):
        record = {"conversations": [{"from": "assistant", "value": "只有回答"}]}
        self.assertIsNone(CMB_SELECT.first_human_text(record))

    def test_zh_ratio_counts_cjk_chars(self):
        self.assertAlmostEqual(CMB_SELECT.zh_ratio("中文abc"), 2 / 5)
        self.assertEqual(CMB_SELECT.zh_ratio(""), 0.0)

    def test_filters_by_ratio_and_sorts_desc(self):
        records = [
            {"conversations": [{"from": "human", "value": "中文问题"}]},          # ratio 1.0
            {"conversations": [{"from": "human", "value": "中文abc"}]},        # ratio 0.4
            {"conversations": [{"from": "human", "value": "english only"}]},   # ratio 0.0
        ]
        self.assertEqual(
            self._run(records, 0.24, 10),
            ["中文问题", "中文abc"],
        )

    def test_num_prompts_keeps_top_n_only(self):
        records = [
            {"conversations": [{"from": "human", "value": "中文问题"}]},
            {"conversations": [{"from": "human", "value": "中文abc"}]},
        ]
        self.assertEqual(self._run(records, 0.24, 1), ["中文问题"])

    def test_no_num_prompts_keeps_all_passing(self):
        # 不传 --num-prompts：过阈值的全部保留，不取前 N
        records = [
            {"conversations": [{"from": "human", "value": "中文问题"}]},
            {"conversations": [{"from": "human", "value": "中文abc"}]},
            {"conversations": [{"from": "human", "value": "english only"}]},
        ]
        self.assertEqual(self._run(records, 0.24, None), ["中文问题", "中文abc"])

    def _run(self, records, min_zh_ratio, num_prompts):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "cmb.jsonl"
            out = Path(tmp) / "out.jsonl"
            with open(src, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            argv = [
                "select_cmb_chinese_prompts.py",
                "--src", str(src),
                "--out", str(out),
                "--min-zh-ratio", str(min_zh_ratio),
            ]
            if num_prompts is not None:
                argv += ["--num-prompts", str(num_prompts)]
            old_argv = sys.argv
            sys.argv = argv
            try:
                logging.disable(logging.INFO)
                CMB_SELECT.main()
            finally:
                logging.disable(logging.NOTSET)
                sys.argv = old_argv
            with open(out, encoding="utf-8") as f:
                return [json.loads(line)["prompt"] for line in f if line.strip()]


class TestSampleOpbBySource(unittest.TestCase):
    """英文源池的按 source 分层抽样（qwen 样例 §6.2.2，先抽 65w 再重生成）。"""

    def test_source_of_strips_shard_suffix(self):
        cases = {
            "ultrachat.jsonl": "ultrachat",
            "ultrachat_s0of10.jsonl": "ultrachat",
            "ultrachat_s1of10.jsonl": "ultrachat",
            "ultrachat.shard3.jsonl": "ultrachat",
            "ultrachat.part2.jsonl": "ultrachat",
        }
        for name, want in cases.items():
            with self.subTest(name=name):
                self.assertEqual(OPB_SAMPLE.source_of(name), want)

    def test_largest_remainder_conserves_total(self):
        shares = {"a": 60, "b": 40}
        alloc = OPB_SAMPLE.largest_remainder(shares, 10)
        self.assertEqual(sum(alloc.values()), 10)
        self.assertEqual(alloc, {"a": 6, "b": 4})

    def test_largest_remainder_is_deterministic(self):
        shares = {"a": 1, "b": 1, "c": 1}
        first = OPB_SAMPLE.largest_remainder(shares, 10)
        second = OPB_SAMPLE.largest_remainder(shares, 10)
        self.assertEqual(first, second)
        # 份额相同，多出来的余数只能给一个：各 3 条 + 一个 4 条
        self.assertEqual(sum(first.values()), 10)
        self.assertEqual(sorted(first.values()), [3, 3, 4])

    def test_largest_remainder_keeps_proportion(self):
        # 真实 open-perfectblend 的两个 source 条数（1.42M 源池）
        shares = {"a": 653754, "b": 767138}
        alloc = OPB_SAMPLE.largest_remainder(shares, 650000)
        self.assertEqual(sum(alloc.values()), 650000)
        share_a = shares["a"] / sum(shares.values())
        self.assertAlmostEqual(alloc["a"] / 650000, share_a, places=3)

    def test_main_preserves_source_proportion(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "regen"
            out = Path(tmp) / "out.jsonl"
            self._write(src / "math.jsonl", "math", 30)
            self._write(src / "chat.jsonl", "chat", 10)
            rows, _ = self._run(src, out, 20)
            self.assertEqual(len(rows), 20)
            tags = [r["id"].split("-")[0] for r in rows]
            self.assertEqual(tags.count("math"), 15)
            self.assertEqual(tags.count("chat"), 5)
            # 抽中的行不能重复
            self.assertEqual(len({r["id"] for r in rows}), 20)

    def test_main_merges_shards_of_same_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "regen"
            out = Path(tmp) / "out.jsonl"
            self._write(src / "chat_s0of2.jsonl", "chat", 5)
            self._write(src / "chat_s1of2.jsonl", "chat", 5)
            self._write(src / "math.jsonl", "math", 10)
            rows, _ = self._run(src, out, 20)
            tags = [r["id"].split("-")[0] for r in rows]
            self.assertEqual(len(rows), 20)
            self.assertEqual(tags.count("chat"), 10)
            self.assertEqual(tags.count("math"), 10)

    def test_main_takes_all_when_pool_smaller(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "regen"
            out = Path(tmp) / "out.jsonl"
            self._write(src / "math.jsonl", "math", 3)
            self._write(src / "chat.jsonl", "chat", 2)
            rows, log = self._run(src, out, 100)
            self.assertEqual(len(rows), 5)
            self.assertIn("池子不足", log)

    def test_main_same_seed_same_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "regen"
            self._write(src / "math.jsonl", "math", 30)
            self._write(src / "chat.jsonl", "chat", 10)
            first, _ = self._run(src, Path(tmp) / "a.jsonl", 20, seed=42)
            second, _ = self._run(src, Path(tmp) / "b.jsonl", 20, seed=42)
            self.assertEqual(
                [r["id"] for r in first], [r["id"] for r in second]
            )

    def _write(self, path, tag, n):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({"id": f"{tag}-{i}"}) + "\n")

    def _run(self, src_dir, out, num_samples, seed=42):
        argv = [
            "sample_opb_by_source.py",
            "--src-dir",
            str(src_dir),
            "--out",
            str(out),
            "--num-samples",
            str(num_samples),
            "--seed",
            str(seed),
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            with self.assertLogs(OPB_SAMPLE.LOG, level="INFO") as cm:
                OPB_SAMPLE.main()
        finally:
            sys.argv = old_argv
        with open(out, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return rows, "\n".join(rec.getMessage() for rec in cm.records)


if __name__ == "__main__":
    unittest.main()
