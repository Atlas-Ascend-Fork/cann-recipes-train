#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Build standalone HTML and Chrome Trace/Perfetto files from Vime logs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREFIX = "VIME_PERF_TRACE "
STREAM_ORDER = {"rollout": 0, "train": 1, "system": 2, "driver": 3}
COLORS = {
    "rollout": "#60a5fa",
    "train": "#fb923c",
    "system": "#a78bfa",
    "driver": "#94a3b8",
    "request": "#38bdf8",
}
TRAIN_SPAN_NAMES = {
    "train_phase", "actor_train", "critic_train", "log_probs", "ref_log_probs",
    "teacher_log_probs",
    "actor_compute_advantages", "critic_compute_advantages", "optimizer_step",
    "backup_weights", "fill_routing_replay", "data_preprocess",
}
ROLLOUT_SPAN_NAMES = {
    "get_rollout_data", "_convert_samples_to_train_data", "_split_train_data_by_dp",
    "vllm_generate", "vllm_mm_generate", "vllm_inference_generate_stream",
    "reward_model", "group_reward_model", "generate_and_rm", "generate_and_rm_group",
}


def event_stream(event: dict[str, Any]) -> str:
    """Derive the logical stream from the stable event name."""
    name = str(event.get("name") or "")
    if name in TRAIN_SPAN_NAMES:
        return "train"
    if name in ROLLOUT_SPAN_NAMES or name.startswith("request_"):
        return "rollout"
    if name in {"weight_sync", "policy_version"}:
        return "system"
    if name == "rl_step":
        return "driver"
    return "other"


@dataclass
class ParseResult:
    events: list[dict[str, Any]]
    malformed_lines: int
    duplicate_events: int


def parse_logs(paths: Iterable[Path]) -> ParseResult:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    malformed = 0
    duplicates = 0
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                search_from = 0
                while True:
                    marker = line.find(PREFIX, search_from)
                    if marker < 0:
                        break
                    payload_start = marker + len(PREFIX)
                    next_marker = line.find(PREFIX, payload_start)
                    if next_marker < 0:
                        payload = line[payload_start:].strip()
                    else:
                        payload = line[payload_start:next_marker].strip()
                    search_from = next_marker if next_marker >= 0 else len(line)
                    try:
                        decoder = json.JSONDecoder()
                        event, _ = decoder.raw_decode(payload)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    event_id = event.get("event_id")
                    if event_id and event_id in seen:
                        duplicates += 1
                        continue
                    if event_id:
                        seen.add(event_id)
                    events.append(event)
    events.sort(
        key=lambda event: (
            event_time_ns(event),
            str(event.get("source_id", "")),
            int(event.get("source_seq", 0)),
        )
    )
    return ParseResult(events, malformed, duplicates)


def dedupe_train_consume_relations(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep one copy of each semantically identical consume relation."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    duplicates = 0

    for event in events:
        if not (
            event.get("type") == "relation"
            and event.get("name") == "train_consumes_rollout"
        ):
            result.append(event)
            continue

        source = event.get("source", {})
        target = event.get("target", {})
        key = (
            target.get("rl_cycle_id"),
            target.get("dp_rank"),
            target.get("actor_policy_version"),
            tuple(source.get("rollout_ids", [])),
            tuple(source.get("sample_indices", [])),
        )
        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        result.append(event)

    return result, duplicates


def event_time_ns(event: dict[str, Any]) -> int:
    return int(
        event.get("start_wall_ns")
        or event.get("wall_ns")
        or event.get("end_wall_ns")
        or 0
    )


def select_run(
    events: list[dict[str, Any]], requested: str | None
) -> tuple[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("perf_run_id", "unknown"))].append(event)
    if requested is not None:
        if requested not in grouped:
            raise ValueError(
                f"run {requested!r} not found; available={sorted(grouped)}"
            )
        return requested, grouped[requested]
    if len(grouped) != 1:
        raise ValueError(
            f"expected one run, found {sorted(grouped)}; pass --run-id"
        )
    return next(iter(grouped.items()))


def stable_int(*parts: object) -> int:
    digest = hashlib.sha1(
        "\x1f".join(map(str, parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _safe_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _event_in_step_range(e: dict, keep_cycles: set[int], sample_to_cycle: dict[str, int]) -> bool:
    """Check if an event belongs to one of the keep_cycles."""
    # Direct rl_cycle_id
    cycle = _safe_int(e.get("rl_cycle_id"))
    if cycle is not None:
        return cycle in keep_cycles
    # attrs.rl_cycle_id
    cycle = _safe_int(e.get("attrs", {}).get("rl_cycle_id"))
    if cycle is not None:
        return cycle in keep_cycles
    # relation: target.rl_cycle_id
    cycle = _safe_int(e.get("target", {}).get("rl_cycle_id"))
    if cycle is not None:
        return cycle in keep_cycles
    # request span: check sample_id via sample_to_cycle
    sample_id = e.get("sample_id")
    if sample_id is not None:
        cycle = sample_to_cycle.get(str(sample_id))
        if cycle is not None:
            return cycle in keep_cycles
        # sample not consumed by any step in range — keep if it's a vllm_generate/request span
        # (it might be incomplete or consumed by a step outside range)
        if e.get("type") == "span" and e.get("name") in ("vllm_generate", "vllm_mm_generate", "vllm_inference_generate_stream", "generate_and_rm", "generate_and_rm_group", "reward_model"):
            return False  # consumed by a step outside range
    # request_processing_started: check sample_indices
    if e.get("type") == "instant" and e.get("name") == "request_processing_started":
        indices = e.get("attrs", {}).get("sample_indices", [])
        if indices:
            # Keep if any sample is consumed by a step in range
            return any(str(idx) in sample_to_cycle and sample_to_cycle[str(idx)] in keep_cycles for idx in indices)
        return False
    # No cycle info — keep (e.g., counters, perf_writer)
    return True


def chrome_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    process_names: set[int] = set()
    thread_names: set[tuple[int, int]] = set()

    rollout_source_map: dict[int, str] = {}
    rollout_complete_ns: dict[int, int] = {}
    for event in events:
        if event.get("type") == "span" and event_stream(event) == "rollout":
            batch_id = event.get("rl_cycle_id")
            if batch_id is not None:
                rollout_source_map[int(batch_id)] = str(event.get("source_id", "unknown"))
                end_ns = event.get("end_wall_ns")
                if end_ns is not None:
                    rollout_complete_ns[int(batch_id)] = int(end_ns)

    # Build request_id → (source_id, end_wall_ns, pid, tid) map from vllm_generate spans
    vllm_generate_map: dict[str, tuple[str, int, int, int]] = {}
    for event in events:
        if event.get("type") == "span" and event.get("name") in ("vllm_generate", "vllm_mm_generate", "vllm_inference_generate_stream"):
            tid = event.get("request_id") or event.get("attrs", {}).get("request_id") or event.get("trace_id") or event.get("attrs", {}).get("trace_id") or event.get("sample_id") or event.get("attrs", {}).get("sample_id")
            if tid:
                end_ns = int(event.get("start_wall_ns", 0)) + int(event.get("duration_ns", 0))
                src = str(event.get("source_id", "rollout"))
                pid_val = stable_int(src)
                vg_stream = event_stream(event)
                vg_rank = event.get("rank")
                sample_id = event.get("sample_id") or event.get("attrs", {}).get("sample_id")
                group_id = event.get("group_id") or event.get("attrs", {}).get("group_id")
                sub_key = sample_id if sample_id is not None else group_id
                if sub_key is not None:
                    vg_tid = stable_int(src, vg_stream, vg_rank, sub_key)
                else:
                    vg_tid = stable_int(src, vg_stream, vg_rank)
                vllm_generate_map[str(tid)] = (src, end_ns, pid_val, vg_tid)
                # Also index by sample_id for relation-based flow matching
                if sample_id is not None and str(sample_id) not in vllm_generate_map:
                    vllm_generate_map[str(sample_id)] = (src, end_ns, pid_val, vg_tid)
    train_phase_by_key: dict[tuple[str, int], tuple[int, int, int]] = {}
    for event in events:
        if event.get("type") == "span" and event.get("name") == "train_phase":
            cycle = event.get("rl_cycle_id")
            start_ns = event.get("start_wall_ns")
            src = str(event.get("source_id", "unknown"))
            pid_val = stable_int(src)
            tid_val = (pid_val + 0) & 0x7FFFFFFF
            if cycle is not None and start_ns is not None:
                key = (src, int(cycle))
                train_phase_by_key[key] = (pid_val, tid_val, int(start_ns))

    emitted_spans: set[tuple[int, int, float, float, str]] = set()

    pid_tid_map: dict[int, int] = {}

    # Build dynamic train span name ordering per pid (by first start time)
    train_name_order_by_pid: dict[int, list[str]] = {}
    train_first_start_by_pid: dict[int, dict[str, int]] = {}
    for event in events:
        if event.get("type") != "span" or event_stream(event) != "train":
            continue
        src = str(event.get("source_id", "unknown"))
        pid_val = stable_int(src)
        name = event.get("name", "unknown")
        start_ns = int(event.get("start_wall_ns", 0))
        if pid_val not in train_first_start_by_pid:
            train_first_start_by_pid[pid_val] = {}
        if name not in train_first_start_by_pid[pid_val] or start_ns < train_first_start_by_pid[pid_val][name]:
            train_first_start_by_pid[pid_val][name] = start_ns

    for pid_val, name_starts in train_first_start_by_pid.items():
        # train_phase always first, then others by start time
        sorted_names = sorted(name_starts.keys(), key=lambda n: (0 if n == "train_phase" else 1, name_starts[n]))
        train_name_order_by_pid[pid_val] = sorted_names

    def _train_tid_for(pid_val: int, name: str, source: str) -> int:
        order_list = train_name_order_by_pid.get(pid_val, [])
        if name in order_list:
            idx = order_list.index(name)
        elif name.endswith("log_probs"):
            # Match ref_log_probs, teacher_log_probs etc
            for i, n in enumerate(order_list):
                if n.endswith("log_probs"):
                    idx = i
                    break
            else:
                idx = len(order_list)
        else:
            idx = len(order_list)
        return (pid_val + idx) & 0x7FFFFFFF

    for event in events:
        source = event.get("source_id", "unknown")
        stream = event_stream(event)
        rank = event.get("rank")
        pid = stable_int(source)
        tid = stable_int(source, stream, rank)

        if pid not in process_names:
            process_names.add(pid)
            rank_str = f"rank{rank}" if rank is not None else "driver"
            node_str = str(event.get("node", "node"))[:12]
            output.append(
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": pid,
                    "tid": 0,
                    "args": {
                        "name": f"{node_str}/{rank_str}"
                    },
                }
            )
        if (pid, tid) not in thread_names:
            thread_names.add((pid, tid))
            stream_label = str(stream) if stream and stream != "other" else "misc"
            parts = [stream_label]
            if rank is not None:
                parts.append(f"rank{rank}")
            output.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": pid,
                    "tid": tid,
                    "args": {
                        "name": "/".join(parts)
                    },
                }
            )

        event_type = event.get("type")
        args = {
            key: value
            for key, value in event.items()
            if key
            not in {
                "type",
                "name",
                "start_wall_ns",
                "end_wall_ns",
                "duration_ns",
                "wall_ns",
            }
        }
        if event_type == "span":
            start_ns = event.get("start_wall_ns")
            duration_ns = event.get("duration_ns")
            if start_ns is None or duration_ns is None:
                continue
            name = event.get("name", "span")
            ts_us = int(start_ns) / 1000
            dur_us = int(duration_ns) / 1000

            # Assign deterministic tid per train span name, ordered by first start time per pid
            if stream == "train":
                tid = _train_tid_for(pid, name, source)

            if name in ("generate_and_rm_group", "generate_and_rm", "vllm_generate", "vllm_mm_generate", "vllm_inference_generate_stream"):
                sample_id = event.get("sample_id")
                group_id = event.get("group_id")
                sub_key = sample_id if sample_id is not None else group_id
                if sub_key is not None:
                    tid = stable_int(source, stream, rank, sub_key)
                else:
                    tid = stable_int(source, stream, rank)
                if (pid, tid) not in thread_names:
                    thread_names.add((pid, tid))
                    req_label = f"/s={sub_key}" if sub_key is not None else ""
                    rank_label = f"/rank{rank}" if rank is not None else ""
                    output.append(
                        {
                            "ph": "M",
                            "name": "thread_name",
                            "pid": pid,
                            "tid": tid,
                            "args": {
                                "name": f"rollout{rank_label}{req_label}"
                            },
                        }
                    )

            span_key = (pid, tid, ts_us, dur_us, name)
            if span_key in emitted_spans:
                continue
            emitted_spans.add(span_key)
            if pid not in pid_tid_map:
                pid_tid_map[pid] = tid
            output.append(
                {
                    "ph": "X",
                    "name": event.get("name", "span"),
                    "cat": stream,
                    "ts": ts_us,
                    "dur": dur_us,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
        elif event_type == "instant":
            output.append(
                {
                    "ph": "i",
                    "s": "g",
                    "name": event.get("name", "instant"),
                    "ts": int(event.get("wall_ns", 0)) / 1000,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
        elif event_type == "counter":
            output.append(
                {
                    "ph": "C",
                    "name": event.get("name", "counter"),
                    "ts": int(event.get("wall_ns", 0)) / 1000,
                    "pid": pid,
                    "tid": tid,
                    "args": event.get("values", {}),
                }
            )
        elif event_type == "relation":
            rel_source = event.get("source", {})
            rel_target = event.get("target", {})
            rollout_ids = rel_source.get("rollout_ids", [])
            train_cycle = rel_target.get("rl_cycle_id")
            train_pid = stable_int(event.get("source_id", "unknown"))
            f_ts_ns = int(event.get("wall_ns", 0))
            f_tid = None
            f_pid = train_pid
            rel_src_id = str(event.get("source_id", "unknown"))
            tp_key = (rel_src_id, int(train_cycle)) if train_cycle is not None else None
            if tp_key and tp_key in train_phase_by_key:
                f_pid, f_tid, f_ts_ns = train_phase_by_key[tp_key]
            if f_tid is None:
                f_tid = pid_tid_map.get(train_pid, 0)
            batch_to_samples: dict[str, list[Any]] = defaultdict(list)
            for rollout_id, sample_index in zip(
                rollout_ids,
                rel_source.get("sample_indices", []),
            ):
                batch_to_samples[str(rollout_id)].append(sample_index)
            for batch_id in dict.fromkeys(rollout_ids):
                bid = int(batch_id)
                rollout_src_id = rollout_source_map.get(bid, f"rollout-batch-{bid}")
                s_ts_ns = rollout_complete_ns.get(bid, int(event.get("wall_ns", 0)))
                samples = list(dict.fromkeys(batch_to_samples.get(str(bid), [])))
                rollout_pid = stable_int(rollout_src_id)
                s_tid = pid_tid_map.get(rollout_pid, 0)
                # Generate one flow per unique request (capped at 32 per batch).
                for sample_index in samples[:32]:
                    request_info = vllm_generate_map.get(str(sample_index))
                    if request_info:
                        _, request_complete_ns, s_pid, s_tid_request = request_info
                        s_ts = request_complete_ns
                    else:
                        continue
                    flow_id = stable_int(
                        event.get("perf_run_id"),
                        bid,
                        sample_index,
                        train_cycle,
                        rel_target.get("dp_rank"),
                    )
                    output.append(
                        {
                            "ph": "s",
                            "id": flow_id,
                            "name": "request→train",
                            "ts": s_ts / 1000,
                            "pid": s_pid,
                            "tid": s_tid_request,
                            "cat": "relation",
                            "args": {
                                "rollout_id": bid,
                                "sample_index": str(sample_index),
                            },
                        }
                    )
                    output.append(
                        {
                            "ph": "f",
                            "id": flow_id,
                            "name": "request→train",
                            "ts": f_ts_ns / 1000,
                            "pid": f_pid,
                            "tid": f_tid,
                            "cat": "relation",
                            "bp": "e",
                            "args": {
                                "train_cycle": train_cycle,
                                "dp_rank": rel_target.get("dp_rank"),
                            },
                        }
                    )
                # If more than 32 requests, add a summary flow for the batch.
                if len(samples) > 32:
                    flow_id = stable_int(
                        event.get("perf_run_id"),
                        bid,
                        "summary",
                        train_cycle,
                        rel_target.get("dp_rank"),
                    )
                    output.append(
                        {
                            "ph": "s",
                            "id": flow_id,
                            "name": f"batch{bid}→train ({len(samples)} requests)",
                            "ts": s_ts_ns / 1000,
                            "pid": rollout_pid,
                            "tid": s_tid,
                            "cat": "relation",
                            "args": {
                                "rollout_id": bid,
                                "request_count": len(samples),
                            },
                        }
                    )
                    output.append(
                        {
                            "ph": "f",
                            "id": flow_id,
                            "name": f"batch{bid}→train ({len(samples)} requests)",
                            "ts": f_ts_ns / 1000,
                            "pid": train_pid,
                            "tid": f_tid,
                            "cat": "relation",
                            "bp": "e",
                            "args": {
                                "train_cycle": train_cycle,
                                "dp_rank": rel_target.get("dp_rank"),
                            },
                        }
                    )
    return {"traceEvents": output, "displayTimeUnit": "ms"}


@dataclass
class _Lane:
    label: str
    stream: str
    source_id: str
    rank: int | None
    node: str
    sub_rows: list[list[dict[str, Any]]]


def span_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        event
        for event in events
        if event.get("type") == "span"
        and event.get("start_wall_ns") is not None
        and event.get("duration_ns") is not None
    ]
    rows.sort(
        key=lambda event: (
            STREAM_ORDER.get(event_stream(event), 99),
            int(event["start_wall_ns"]),
        )
    )
    return rows


def _lane_label(event: dict[str, Any]) -> str:
    stream = event_stream(event)
    rank = event.get("rank")
    if rank is not None:
        return f"{stream} / rank{rank}"
    node = str(event.get("node", ""))[:12]
    return f"{stream} / {node}"


def _build_lanes(spans: list[dict[str, Any]]) -> list[_Lane]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        source_id = str(span.get("source_id", "unknown"))
        stream = event_stream(span)
        grouped[(source_id, stream)].append(span)

    lanes: list[_Lane] = []
    for (source_id, stream), group_spans in grouped.items():
        group_spans.sort(key=lambda e: int(e["start_wall_ns"]))
        # For train stream: put train_phase spans in row 0, sub-phases in later rows
        if stream == "train":
            phase_spans = [s for s in group_spans if s.get("name") == "train_phase"]
            other_spans = [s for s in group_spans if s.get("name") != "train_phase"]
            sub_rows: list[list[dict[str, Any]]] = []
            if phase_spans:
                sub_rows.append(phase_spans)
            for span in other_spans:
                start = int(span["start_wall_ns"])
                placed = False
                for sub_row in sub_rows:
                    last_end = int(sub_row[-1]["start_wall_ns"]) + int(sub_row[-1]["duration_ns"])
                    if start >= last_end:
                        sub_row.append(span)
                        placed = True
                        break
                if not placed:
                    sub_rows.append([span])
        else:
            sub_rows: list[list[dict[str, Any]]] = []
            for span in group_spans:
                start = int(span["start_wall_ns"])
                placed = False
                for sub_row in sub_rows:
                    last_end = int(sub_row[-1]["start_wall_ns"]) + int(sub_row[-1]["duration_ns"])
                    if start >= last_end:
                        sub_row.append(span)
                        placed = True
                        break
                if not placed:
                    sub_rows.append([span])
        first = group_spans[0]
        lanes.append(
            _Lane(
                label=_lane_label(first),
                stream=stream,
                source_id=source_id,
                rank=first.get("rank"),
                node=str(first.get("node", "")),
                sub_rows=sub_rows,
            )
        )

    def lane_sort_key(lane: _Lane) -> tuple:
        rank_val = lane.rank if lane.rank is not None else 999
        return (
            STREAM_ORDER.get(lane.stream, 99),
            rank_val,
            lane.node,
            lane.source_id,
        )

    lanes.sort(key=lane_sort_key)
    return lanes


def _build_step_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate spans by rl_cycle_id, computing rollout/train/sync durations and overlap."""
    steps: dict[int, dict[str, Any]] = defaultdict(lambda: {
        "rl_cycle_id": None,
        "rollout_spans": [],
        "train_spans": [],
        "system_spans": [],
        "driver_spans": [],
        "rollout_batch_spans": [],
    })

    # Collect rl_step spans for time-based matching of get_rollout_data
    rl_step_spans: list[tuple[int, int, int]] = []  # (start, end, cycle)
    rollout_batch_raw: list[dict[str, Any]] = []
    request_gen_raw: list[dict[str, Any]] = []  # vllm_generate spans for async overlap

    for event in events:
        if event.get("type") != "span":
            continue
        name = event.get("name", "")
        start_ns = int(event.get("start_wall_ns", 0))
        end_ns = start_ns + int(event.get("duration_ns", 0))
        if name == "rl_step":
            cycle = event.get("rl_cycle_id")
            if cycle is not None:
                rl_step_spans.append((start_ns, end_ns, int(cycle)))
        if name in ("vllm_generate", "vllm_mm_generate", "vllm_inference_generate_stream"):
            request_gen_raw.append({"start": start_ns, "end": end_ns})
        if name == "get_rollout_data":
            rollout_batch_raw.append({"start": start_ns, "end": end_ns})
            continue
        cycle = event.get("rl_cycle_id")
        if cycle is None:
            continue
        cycle = int(cycle)
        steps[cycle]["rl_cycle_id"] = cycle
        stream = event_stream(event)
        interval = (start_ns, end_ns)
        if stream == "rollout" or name == "get_rollout_data":
            steps[cycle]["rollout_spans"].append(interval)
        elif stream == "train":
            steps[cycle]["train_spans"].append(interval)
        elif stream == "system" or name == "weight_sync":
            steps[cycle]["system_spans"].append(interval)
        elif stream == "driver":
            steps[cycle]["driver_spans"].append(interval)

    # Match get_rollout_data spans to steps by time overlap with rl_step
    rl_step_spans.sort()
    for rb in rollout_batch_raw:
        matched_cycle = None
        for s_start, s_end, s_cycle in rl_step_spans:
            if rb["start"] >= s_start and rb["start"] < s_end:
                matched_cycle = s_cycle
                break
        if matched_cycle is not None:
            steps[matched_cycle]["rollout_batch_spans"].append((rb["start"], rb["end"]))

    # Build timeline end for incomplete request synthesis
    _timeline_end_ns = max(
        [
            int(e.get("end_wall_ns") or (int(e.get("start_wall_ns", 0)) + int(e.get("duration_ns", 0))) or e.get("wall_ns") or 0)
            for e in events if e.get("type") in ("instant", "span")
        ] or [0]
    )

    # Match vllm_generate spans to steps for async overlap calculation
    for rg in request_gen_raw:
        for s_start, s_end, s_cycle in rl_step_spans:
            if rg["end"] >= s_start and rg["start"] <= s_end:
                steps[s_cycle]["rollout_spans"].append((rg["start"], rg["end"]))

    # Also include incomplete requests (processing_started but no vllm_generate) in overlap
    _known_gen_samples = {str(e.get("sample_id")) for e in events
                          if e.get("type") == "span" and e.get("name") in ("vllm_generate", "vllm_mm_generate", "vllm_inference_generate_stream")
                          and e.get("sample_id") is not None}
    for event in events:
        if event.get("type") != "instant" or event.get("name") != "request_processing_started":
            continue
        ts = int(event.get("wall_ns", 0))
        for idx in event.get("attrs", {}).get("sample_indices", []):
            if str(idx) in _known_gen_samples:
                continue
            for s_start, s_end, s_cycle in rl_step_spans:
                if _timeline_end_ns >= s_start and ts <= s_end:
                    steps[s_cycle]["rollout_spans"].append((ts, _timeline_end_ns))

    def union_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        sorted_iv = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_iv[0]]
        for start, end in sorted_iv[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def intersect_duration(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
        ua = union_intervals(a)
        ub = union_intervals(b)
        total = 0
        for s_a, e_a in ua:
            for s_b, e_b in ub:
                overlap_start = max(s_a, s_b)
                overlap_end = min(e_a, e_b)
                if overlap_end > overlap_start:
                    total += overlap_end - overlap_start
        return total

    result = []
    for cycle in sorted(steps):
        s = steps[cycle]
        sync_union = union_intervals(s["system_spans"])
        sync_dur = sum(e - start for start, e in sync_union)
        driver_dur = sum(e - start for start, e in s["driver_spans"])

        rollout_union = union_intervals(s["rollout_spans"])
        train_union = union_intervals(s["train_spans"])
        rollout_busy = sum(e - start for start, e in rollout_union)
        rollout_batch_dur = sum(e - start for start, e in s["rollout_batch_spans"])
        train_busy = sum(e - start for start, e in train_union)
        rollout_display = rollout_batch_dur if rollout_batch_dur > 0 else rollout_busy
        overlap = intersect_duration(s["rollout_spans"], s["train_spans"])

        if driver_dur > 0:
            step_total = driver_dur
        else:
            all_starts_list = [start for start, e in s["rollout_spans"] + s["train_spans"] + s["system_spans"] if start > 0]
            all_ends_list = [e for start, e in s["rollout_spans"] + s["train_spans"] + s["system_spans"] if e > 0]
            step_total = (max(all_ends_list) - min(all_starts_list)) if all_starts_list and all_ends_list else 0

        unaccounted = max(0, step_total - rollout_display - train_busy - sync_dur)

        result.append({
            "rl_cycle_id": cycle,
            "step_time_ms": step_total / 1_000_000,
            "rollout_busy_ms": rollout_display / 1_000_000,
            "train_busy_ms": train_busy / 1_000_000,
            "sync_ms": sync_dur / 1_000_000,
            "overlap_ms": overlap / 1_000_000,
            "overlap_ratio": overlap / step_total if step_total > 0 else 0.0,
            "unaccounted_ms": unaccounted / 1_000_000,
            "rollout_only_ms": (rollout_busy - overlap) / 1_000_000,
            "train_only_ms": (train_busy - overlap) / 1_000_000,
            "rollout_span_count": len(s["rollout_spans"]),
            "train_span_count": len(s["train_spans"]),
        })

    return result


def render_html(
    run_id: str,
    events: list[dict[str, Any]],
    malformed: int,
    duplicates: int,
) -> str:
    spans = span_rows(events)
    # Assign logical streams offline so producers only need stable event names.
    for s in spans:
        s["stream"] = event_stream(s)
    if spans:
        origin = min(int(event["start_wall_ns"]) for event in spans)
        end = max(
            int(event["start_wall_ns"]) + int(event["duration_ns"])
            for event in spans
        )
    else:
        origin, end = 0, 1

    # Also consider instant events for timeline bounds
    for event in events:
        if event.get("type") == "instant":
            ts = event.get("wall_ns")
            if ts is not None:
                origin = min(origin, int(ts))
                end = max(end, int(ts))

    # Use vllm_generate spans directly for per-request rollout visualization
    request_spans = [
        e for e in spans
        if e.get("name") in ("vllm_generate", "vllm_mm_generate", "vllm_inference_generate_stream")
        or e.get("request_id") is not None
        or e.get("attrs", {}).get("request_id") is not None
    ]
    processing_start_by_sample: dict[str, int] = {}
    group_id_by_sample: dict[str, Any] = {}
    for event in events:
        if event.get("type") != "instant" or event.get("name") != "request_processing_started":
            continue
        start_ns = event.get("wall_ns")
        indices = event.get("attrs", {}).get("sample_indices", [])
        group_ids = event.get("attrs", {}).get("group_ids", [])
        if start_ns is None:
            continue
        for i, sample_index in enumerate(indices):
            processing_start_by_sample[str(sample_index)] = int(start_ns)
            if i < len(group_ids) and group_ids[i] is not None:
                group_id_by_sample[str(sample_index)] = group_ids[i]
    for span in request_spans:
        sample_id = span.get("sample_id")
        start_ns = span.get("start_wall_ns")
        if sample_id is not None and start_ns is not None:
            processing_start_by_sample.setdefault(str(sample_id), int(start_ns))
        if sample_id is not None and str(sample_id) in processing_start_by_sample:
            span.setdefault("attrs", {})["processing_start_wall_ns"] = processing_start_by_sample[str(sample_id)]
    known_request_samples = {
        str(span.get("sample_id"))
        for span in request_spans
        if span.get("sample_id") is not None
    }
    timeline_end_ns = max(
        [
            int(e.get("end_wall_ns") or (int(e.get("start_wall_ns", 0)) + int(e.get("duration_ns", 0))) or e.get("wall_ns") or 0)
            for e in events if e.get("type") in ("instant", "span")
        ] or [0]
    )
    for sample_id, start_ns in processing_start_by_sample.items():
        if sample_id in known_request_samples or timeline_end_ns <= start_ns:
            continue
        request_spans.append(
            {
                "type": "span",
                "name": "request",
                "sample_id": sample_id,
                "start_wall_ns": start_ns,
                "end_wall_ns": timeline_end_ns,
                "duration_ns": timeline_end_ns - start_ns,
                "status": "incomplete",
                "attrs": {"processing_start_wall_ns": start_ns},
            }
        )
    non_request_spans = [e for e in spans if e.get("name") not in (
        "vllm_generate", "generate_and_rm", "generate_and_rm_group", "reward_model",
        "vllm_mm_render", "vllm_mm_generate", "vllm_inference_generate_stream",
    )]

    display_spans = list(non_request_spans)

    # Assign rl_cycle_id to request spans using the relation events' sample_indices
    # mapping (accurate) instead of time-based heuristics
    sample_to_batch: dict[str, int] = {}
    sample_to_rank: dict[str, int] = {}
    traj_to_staleness: dict[str, dict] = {}
    for event in events:
        if event.get("type") == "relation" and event.get("name") == "train_consumes_rollout":
            source = event.get("source", {})
            target = event.get("target", {})
            dp_rank = target.get("dp_rank")
            rl_cycle_id = target.get("rl_cycle_id")
            for idx in source.get("sample_indices", []):
                sample_to_batch[str(idx)] = rl_cycle_id
                if dp_rank is not None:
                    sample_to_rank[str(idx)] = int(dp_rank)

    # Match each request to the policy version published at its start/end.
    version_timeline: list[tuple[int, int]] = sorted(
        (int(e.get("wall_ns", 0)), int(e.get("attrs", {}).get("version")))
        for e in events
        if e.get("type") == "instant"
        and e.get("name") == "policy_version"
        and _safe_int(e.get("attrs", {}).get("version")) is not None
    )

    # Build traj → rollout_policy_version and actor_policy_version from relation events
    traj_actor_version: dict[str, Any] = {}
    for event in events:
        if event.get("type") == "relation" and event.get("name") == "train_consumes_rollout":
            source = event.get("source", {})
            target = event.get("target", {})
            actor_ver = target.get("actor_policy_version")
            for idx in source.get("sample_indices", []):
                if actor_ver is not None:
                    traj_actor_version[str(idx)] = actor_ver

    # Build sample → completion time mapping from request spans.
    traj_completed_ns: dict[str, int] = {}
    for s in request_spans:
        sample_id = s.get("sample_id")
        end_ns = s.get("end_wall_ns")
        if sample_id is not None and end_ns is not None:
            traj_completed_ns[str(sample_id)] = int(end_ns)

    import bisect
    version_end_times = [vt[0] for vt in version_timeline]
    for tid, proc_start in processing_start_by_sample.items():
        actor_ver = traj_actor_version.get(tid)
        start_version = None
        if version_timeline:
            idx = bisect.bisect_right(version_end_times, proc_start) - 1
            start_version = version_timeline[idx][1] if idx >= 0 else version_timeline[0][1]

        end_version = start_version
        proc_end = traj_completed_ns.get(tid)
        if proc_end is not None and version_timeline:
            idx_end = bisect.bisect_right(version_end_times, proc_end) - 1
            end_version = version_timeline[idx_end][1] if idx_end >= 0 else version_timeline[0][1]

        if start_version is not None:
            start_v = int(start_version)
            end_v = int(end_version) if end_version is not None else start_v
            info = {
                "start_rollout_version": start_v,
                "end_rollout_version": end_v,
            }
            try:
                actor_v = int(actor_ver)
                staleness_from_start = actor_v - start_v
                staleness_from_end = actor_v - end_v
                info.update({
                    "staleness": staleness_from_start,
                    "staleness_from_start": staleness_from_start,
                    "staleness_from_end": staleness_from_end,
                    "actor_policy_version": actor_v,
                })
            except (TypeError, ValueError):
                pass
            traj_to_staleness[tid] = info

    req_attrs_by_traj: dict[str, dict] = {}
    finish_reason_by_sample: dict[str, str] = {}
    for s in request_spans:
        sample_id = s.get("sample_id")
        sid_key = str(sample_id) if sample_id is not None else ""
        s["rl_cycle_id"] = sample_to_batch.get(sid_key)
        attrs = s.get("attrs", {})
        if sid_key and attrs:
            req_attrs_by_traj[sid_key] = attrs
        fr = attrs.get("finish_reason")
        if sid_key and fr:
            finish_reason_by_sample[sid_key] = str(fr)

    traj_spans = []
    # Include requests that have a processing start but no completed span.
    existing_samples = {str(t.get("sample_id")) for t in traj_spans if t.get("sample_id") is not None}
    for request in request_spans:
        if request.get("status") != "incomplete":
            continue
        sample_id = request.get("sample_id")
        if sample_id is None or str(sample_id) in existing_samples:
            continue
        sid_key = str(sample_id)
        inc_gid = group_id_by_sample.get(sid_key, "incomplete")
        traj_spans.append({
            "start_wall_ns": int(request["start_wall_ns"]),
            "end_wall_ns": int(request["end_wall_ns"]),
            "duration_ns": int(request["duration_ns"]),
            "request_id": None,
            "rl_cycle_id": None,
            "source_id": "rollout",
            "group_id": inc_gid,
            "sample_id": sample_id,
            "status": "incomplete",
        })

    # Include request spans in the main activity timeline. Current trace export
    # provides sample_id directly, while older logs may provide trajectory spans.
    existing_samples = {str(t.get("sample_id")) for t in traj_spans if t.get("sample_id") is not None}
    for request in request_spans:
        sample_id = request.get("sample_id")
        if sample_id is None or str(sample_id) in existing_samples:
            continue
        traj_spans.append({
            "start_wall_ns": int(request["start_wall_ns"]),
            "end_wall_ns": int(request["end_wall_ns"]),
            "duration_ns": int(request["duration_ns"]),
            "request_id": request.get("request_id"),
            "rl_cycle_id": request.get("rl_cycle_id"),
            "source_id": request.get("source_id", "rollout"),
            "group_id": request.get("group_id"),
            "sample_id": sample_id,
            "status": request.get("status", "ok"),
            "name": request.get("name", "request"),
            "attrs": request.get("attrs", {}),
        })
        existing_samples.add(str(sample_id))

    # Sort by start time
    traj_spans.sort(key=lambda t: int(t["start_wall_ns"]))

    # Match None-cycle traj_spans to step using weight_sync completion time
    # (merged_ws is built later, so we build it here too)
    ws_periods_for_match: list[tuple[int, int]] = []
    for event in events:
        if event.get("type") == "span" and event.get("name") == "weight_sync":
            s_ns = int(event.get("start_wall_ns", 0))
            e_ns = s_ns + int(event.get("duration_ns", 0))
            if e_ns > s_ns:
                ws_periods_for_match.append((s_ns, e_ns))
    ws_periods_for_match.sort()
    merged_ws_for_match: list[tuple[int, int]] = []
    for s, e in ws_periods_for_match:
        if merged_ws_for_match and s <= merged_ws_for_match[-1][1]:
            merged_ws_for_match[-1] = (merged_ws_for_match[-1][0], max(merged_ws_for_match[-1][1], e))
        else:
            merged_ws_for_match.append((s, e))
    def _match_step_ws(end_ns):
        if not merged_ws_for_match:
            return None
        count = sum(1 for _, ws_e in merged_ws_for_match if ws_e <= end_ns)
        return count
    for t in traj_spans:
        pass  # rl_cycle_id already set from sample_to_batch; unconsumed requests stay None

    # Group vllm_generate by group_id for group-level folding
    groups_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in traj_spans:
        gid = t.get("group_id")
        if gid is None:
            continue
        groups_map[str(gid)].append(t)
    # Build group-level summary spans
    group_spans: list[dict[str, Any]] = []
    for gid, members in sorted(groups_map.items(), key=lambda x: min(m["start_wall_ns"] for m in x[1])):
        members.sort(key=lambda x: x["start_wall_ns"])
        g_start = members[0]["start_wall_ns"]
        g_end = max(m["end_wall_ns"] for m in members)
        g_cycle = members[0].get("rl_cycle_id") if members else None
        group_spans.append({
            "group_id": gid,
            "start_wall_ns": g_start,
            "end_wall_ns": g_end,
            "duration_ns": g_end - g_start,
            "sample_count": len(members),
            "rl_cycle_id": g_cycle,
        })

    lanes = _build_lanes(display_spans)

    # Step boundaries for vertical grid lines
    step_boundaries: list[tuple[int, int]] = []
    for event in events:
        if event.get("type") == "span" and event.get("name") == "rl_step":
            cycle = event.get("rl_cycle_id")
            if cycle is not None:
                step_boundaries.append((int(event.get("start_wall_ns", 0)), int(cycle)))
    step_boundaries.sort()

    # Weight sync periods (for red bands)
    ws_periods: list[tuple[int, int]] = []
    for event in events:
        if event.get("type") == "span" and event.get("name") == "weight_sync":
            s_ns = int(event.get("start_wall_ns", 0))
            e_ns = s_ns + int(event.get("duration_ns", 0))
            if e_ns > s_ns:
                ws_periods.append((s_ns, e_ns))
    # Merge overlapping sync periods (multiple ranks)
    ws_periods.sort()
    merged_ws: list[tuple[int, int]] = []
    for s, e in ws_periods:
        if merged_ws and s <= merged_ws[-1][1]:
            merged_ws[-1] = (merged_ws[-1][0], max(merged_ws[-1][1], e))
        else:
            merged_ws.append((s, e))

    # Rollout batch → train phase arrows
    rollout_batch_spans = sorted(
        [e for e in display_spans if e.get("name") == "rollout_batch"],
        key=lambda e: int(e["start_wall_ns"]),
    )
    train_phase_spans = sorted(
        [e for e in display_spans if e.get("name") == "train_phase"],
        key=lambda e: int(e["start_wall_ns"]),
    )
    # If train_phase spans lack rl_cycle_id, match by time using relation events
    # (each relation's wall_ns is the train consume time, target.rl_cycle_id is the step)
    rel_cycles = sorted(
        (int(e.get("wall_ns", 0)), int(e.get("target", {}).get("rl_cycle_id", 0)))
        for e in events
        if e.get("type") == "relation" and e.get("name") == "train_consumes_rollout"
        and e.get("target", {}).get("rl_cycle_id") is not None
    )
    if rel_cycles and any(tp.get("rl_cycle_id") is None for tp in train_phase_spans):
        import bisect as _bisect_tp
        rel_times = [r[0] for r in rel_cycles]
        for tp in train_phase_spans:
            if tp.get("rl_cycle_id") is not None:
                continue
            tp_end = int(tp.get("start_wall_ns", 0)) + int(tp.get("duration_ns", 0))
            idx = _bisect_tp.bisect_right(rel_times, tp_end) - 1
            if 0 <= idx < len(rel_cycles):
                tp["rl_cycle_id"] = rel_cycles[idx][1]
    # Match by rl_cycle_id
    tp_by_cycle = {}
    for tp in train_phase_spans:
        c = tp.get("rl_cycle_id")
        if c is not None and c not in tp_by_cycle:
            tp_by_cycle[c] = tp

    width = 1400
    label_width = 220
    plot_width = width - label_width - 20
    duration = max(1, end - origin)
    bar_height = 18
    row_height = 24
    lane_gap = 8
    header_height = 24
    group_row_height = 18
    group_bar_height = 14
    req_row_height = 14
    req_bar_height = 10

    # Assign sequential indices
    sorted_traj = sorted(traj_spans, key=lambda x: (x["source_id"], x["start_wall_ns"]))
    for traj_seq, t in enumerate(sorted_traj):
        t["_seq"] = traj_seq

    # Group sub_row layout: greedy non-overlapping placement
    group_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in traj_spans:
        gid = t.get("group_id")
        if gid is None:
            continue
        group_members[str(gid)].append(t)

    group_sub_rows: list[list[tuple[int, int]]] = []
    group_row_assignment: dict[str, int] = {}
    for g in sorted(group_spans, key=lambda x: x["start_wall_ns"]):
        g_start = g["start_wall_ns"]
        g_end = g["end_wall_ns"]
        placed = False
        for ri, row in enumerate(group_sub_rows):
            if row[-1][1] <= g_start:
                row.append((g_start, g_end))
                group_row_assignment[g["group_id"]] = ri
                placed = True
                break
        if not placed:
            group_sub_rows.append([(g_start, g_end)])
            group_row_assignment[g["group_id"]] = len(group_sub_rows) - 1

    max_members = max((len(group_members.get(str(g["group_id"]), [])) for g in group_spans if g["group_id"] != "incomplete"), default=1)
    group_expand_height = max_members * req_row_height + 4

    group_lane_y = header_height
    group_lane_height_collapsed = max(40, len(group_sub_rows) * group_row_height + 20)
    group_lane_height = group_lane_height_collapsed

    y_cursor = group_lane_y + group_lane_height + lane_gap

    # Other lanes
    lane_y_ranges: list[tuple[int, int, _Lane]] = []
    train_rank_y: dict[int, int] = {}
    for lane in lanes:
        lane_height = len(lane.sub_rows) * row_height + lane_gap
        lane_y_ranges.append((y_cursor, y_cursor + lane_height, lane))
        if lane.stream == "train" and lane.rank is not None:
            train_rank_y[int(lane.rank)] = y_cursor + 3
        y_cursor += lane_height
    svg_height = max(100, y_cursor + 10)

    # Time axis ticks
    tick_count = 10
    tick_labels: list[str] = []
    for i in range(tick_count + 1):
        tick_ns = origin + duration * i // tick_count
        tick_s = (tick_ns - origin) / 1_000_000_000
        tick_labels.append(f"{tick_s:.0f}s")

    # Step colors: use HSL color ring for unlimited steps
    def step_color(cycle: int | None) -> str:
        if cycle is None:
            return "#3b82f6"
        hue = (cycle * 47) % 360
        return f"hsl({hue},65%,55%)"

    # Build SVG content
    svg_bars: list[str] = []

    # Time axis
    for i in range(tick_count + 1):
        x = label_width + plot_width * i / tick_count
        svg_bars.append(f'<line class="tick-line" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{svg_height:.0f}" />')
        svg_bars.append(f'<text class="tick-label" x="{x - 15:.1f}" y="14">{tick_labels[i]}</text>')

    # Step boundary lines
    for step_ns, cycle in step_boundaries:
        x = label_width + (step_ns - origin) / duration * plot_width
        if label_width <= x <= label_width + plot_width:
            svg_bars.append(f'<line class="step-line" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{svg_height:.0f}" />')
            svg_bars.append(f'<text class="step-label" x="{x + 3:.1f}" y="14">step {cycle}</text>')

    # Group lane with sub_row layout
    svg_bars.append(f'<rect class="lane-bg group-lane-bg" x="0" y="{group_lane_y:.0f}" width="{width}" height="{group_lane_height:.0f}" rx="2" />')
    svg_bars.append(f'<text class="lane-label" x="4" y="{group_lane_y + 13:.0f}">rollout groups ({len(group_spans)})</text>')

    for g in sorted(group_spans, key=lambda x: x["start_wall_ns"]):
        gid = g["group_id"]
        ri = group_row_assignment.get(gid, 0)
        g_start = g["start_wall_ns"]
        g_end = g["end_wall_ns"]
        row_y = group_lane_y + 18 + ri * group_row_height
        x = label_width + (g_start - origin) / duration * plot_width
        bar_w = max(1.0, (g_end - g_start) / duration * plot_width)
        cycle = g.get("rl_cycle_id")
        members = group_members.get(str(gid), [])
        all_incomplete = all(m.get("status") == "incomplete" for m in members) if members else False
        all_aborted = all(m.get("status") == "error" for m in members) if members else False
        if all_incomplete:
            color = "#fbbf24"
        elif all_aborted:
            color = "#ef4444"
        else:
            color = step_color(cycle)
        group_dash = 'stroke-dasharray="3,2" stroke-width="1"' if all_incomplete or all_aborted else ''
        dur_s = (g_end - g_start) / 1_000_000_000
        g_expand_height = min(len(members), 50) * req_row_height + 4
        tooltip = html.escape(json.dumps({"group": gid, "samples": g["sample_count"], "duration": f"{dur_s:.1f}s", "batch": cycle}, ensure_ascii=False))
        svg_bars.append(f'<rect class="traj-bar group-bar" data-cycle="{cycle}" data-group="{gid}" data-row="{ri}" x="{x:.2f}" y="{row_y:.0f}" width="{bar_w:.2f}" height="{group_bar_height}" rx="2" fill="{color}" {group_dash}><title>{tooltip}</title></rect>')
        if bar_w > 40:
            svg_bars.append(f'<text class="bar-text group-text" data-group="{gid}" data-row="{ri}" x="{x + 2:.1f}" y="{row_y + 10:.0f}">{html.escape(f"group {gid}"[:int(bar_w / 6)])}</text>')
        # Request bars in a <g> container for this group, hidden by default
        svg_bars.append(f'<g class="req-group" data-group="{gid}" data-row="{ri}" data-expand-height="{g_expand_height}" style="display:none">')
        # Limit members shown for very large groups
        display_members = members if len(members) <= 50 else members[:50]
        for _mi, m in enumerate(display_members):
            m_start = m["start_wall_ns"]
            m_end = m["end_wall_ns"]
            m_x = label_width + (m_start - origin) / duration * plot_width
            m_w = max(1.0, (m_end - m_start) / duration * plot_width)
            m_cycle = m.get("rl_cycle_id")
            m_color = step_color(m_cycle)
            m_dur = (m_end - m_start) / 1_000_000_000
            m_idx = m.get("_seq", -1)
            m_tid = str(m.get("sample_id", m.get("request_id", "")))
            m_staleness = traj_to_staleness.get(m_tid, {})
            m_tooltip_parts = [f"req {m_idx}", f"duration: {m_dur:.1f}s", f"batch: {m_cycle}"]
            if m_staleness:
                s_start = m_staleness.get("staleness_from_start")
                s_end = m_staleness.get("staleness_from_end")
                start_v = m_staleness.get("start_rollout_version")
                end_v = m_staleness.get("end_rollout_version")
                actor_ver = m_staleness.get("actor_policy_version")
                if s_start is not None:
                    m_tooltip_parts.append(f"staleness_start: {s_start}")
                if s_end is not None and s_end != s_start:
                    m_tooltip_parts.append(f"staleness_end: {s_end}")
                if start_v is not None:
                    m_tooltip_parts.append(f"start_version: v{start_v}")
                if end_v is not None and end_v != start_v:
                    m_tooltip_parts.append(f"end_version: v{end_v}")
                if actor_ver is not None:
                    m_tooltip_parts.append(f"actor_version: v{actor_ver}")
            m_req_attrs = req_attrs_by_traj.get(m_tid, {})
            pl = m_req_attrs.get("prompt_length")
            rl = m_req_attrs.get("response_length")
            if pl is not None:
                m_tooltip_parts.append(f"prompt_length: {pl}")
            if rl is not None:
                m_tooltip_parts.append(f"response_length: {rl}")
            m_tooltip = html.escape("\n".join(m_tooltip_parts))
            m_y = row_y + group_bar_height + 2 + _mi * req_row_height
            m_incomplete = m.get("status") == "incomplete"
            m_aborted = m.get("status") == "error"
            if m_incomplete:
                m_fill = "#fbbf24"
                m_opacity = "0.4"
                m_dash = 'stroke-dasharray="3,2" stroke="#fbbf24" stroke-width="1"'
            elif m_aborted:
                m_fill = "#ef4444"
                m_opacity = "0.6"
                m_dash = 'stroke-dasharray="2,2" stroke="#ef4444" stroke-width="1"'
            else:
                m_fill = m_color
                m_opacity = "0.7"
                m_dash = ""
            svg_bars.append(f'<rect class="traj-bar req-bar" data-cycle="{m_cycle}" data-group="{gid}" x="{m_x:.2f}" y="{m_y:.0f}" width="{m_w:.2f}" height="{req_bar_height}" rx="2" fill="{m_fill}" opacity="{m_opacity}" {m_dash}><title>{m_tooltip}</title></rect>')
            if m_w > 30:
                svg_bars.append(f'<text class="bar-text" x="{m_x + 1:.1f}" y="{m_y + 8:.0f}">req {m_idx}</text>')
        svg_bars.append('</g>')

    # Collect batch-level arrows only (no group arrows)
    # Collect group-level and req-level arrows
    collected_arrows: list[str] = []
    for g in sorted(group_spans, key=lambda x: x["start_wall_ns"]):
        gid = g["group_id"]
        ri = group_row_assignment.get(gid, 0)
        g_start = g["start_wall_ns"]
        g_end = g["end_wall_ns"]
        row_y = group_lane_y + 18 + ri * group_row_height
        cycle = g.get("rl_cycle_id")
        color = step_color(cycle)
        members = group_members.get(str(gid), [])
        tp = tp_by_cycle.get(cycle)
        if tp is not None and g_end < int(tp["start_wall_ns"]):
            tp_start = int(tp["start_wall_ns"])
            ax1 = label_width + (g_end - origin) / duration * plot_width
            ax2 = label_width + (tp_start - origin) / duration * plot_width
            if abs(ax2 - ax1) > 1:
                group_ranks = set()
                for m in members:
                    m_tid = str(m.get("sample_id", m.get("request_id", "")))
                    m_rank = sample_to_rank.get(m_tid)
                    if m_rank is not None:
                        group_ranks.add(m_rank)
                if not group_ranks:
                    group_ranks.add(0)
                for rank in sorted(group_ranks):
                    if rank in train_rank_y:
                        tp_y_arr = train_rank_y[rank]
                    else:
                        continue
                    g_mid_y = (row_y + group_bar_height + tp_y_arr) / 2
                    collected_arrows.append(f'<path class="arrow traj-arrow group-arrow" data-cycle="{cycle}" data-group="{gid}" data-row="{ri}" fill="none" style="stroke:{color}" marker-end="url(#arrowhead-sm)" d="M {ax1:.1f} {row_y + group_bar_height:.0f} C {ax1:.1f} {g_mid_y:.0f}, {ax2:.1f} {g_mid_y:.0f}, {ax2:.1f} {tp_y_arr:.0f}" />')
        for m in members:
            m_end = m["end_wall_ns"]
            m_cycle = m.get("rl_cycle_id")
            m_color = step_color(m_cycle)
            m_y = row_y + group_bar_height + 2 + members.index(m) * req_row_height
            tp_m = tp_by_cycle.get(m_cycle)
            if tp_m is not None and m_end < int(tp_m["start_wall_ns"]):
                tp_m_start = int(tp_m["start_wall_ns"])
                m_tid = str(m.get("sample_id", m.get("request_id", "")))
                m_rank = sample_to_rank.get(m_tid)
                if m_rank is not None and m_rank in train_rank_y:
                    tp_m_y = train_rank_y[m_rank]
                else:
                    tp_m_y = None
                    for lane_y, _, lane in lane_y_ranges:
                        if lane.stream == "train":
                            tp_m_y = lane_y + 3
                            break
                if tp_m_y is not None:
                    rax1 = label_width + (m_end - origin) / duration * plot_width
                    rax2 = label_width + (tp_m_start - origin) / duration * plot_width
                    if abs(rax2 - rax1) > 1:
                        r_mid_y = (m_y + req_bar_height + tp_m_y) / 2
                        collected_arrows.append(f'<path class="traj-arrow req-arrow" data-cycle="{m_cycle}" data-group="{gid}" data-row="{ri}" fill="none" style="stroke:{m_color};stroke-width:0.6;opacity:0.5;display:none" marker-end="url(#arrowhead-sm)" d="M {rax1:.1f} {m_y + req_bar_height:.0f} C {rax1:.1f} {r_mid_y:.0f}, {rax2:.1f} {r_mid_y:.0f}, {rax2:.1f} {tp_m_y:.0f}" />')

    # Weight sync red bands — drawn on top of group lane, behind train lanes
    # Start below the time axis (header_height), cover group lane + below-group area
    for ws_start, ws_end in merged_ws:
        x1 = label_width + (ws_start - origin) / duration * plot_width
        x2 = label_width + (ws_end - origin) / duration * plot_width
        if x2 > label_width and x1 < label_width + plot_width:
            x1 = max(x1, label_width)
            x2 = min(x2, label_width + plot_width)
            svg_bars.append(f'<rect class="ws-band" x="{x1:.1f}" y="{header_height:.0f}" width="{x2 - x1:.1f}" height="{group_lane_height:.0f}" />')

    # Everything below group lane — lane-bg, arrows, bars all rendered in order
    # NO parent <g> transform; recalcLayout updates y coordinates directly

    # Draw lane backgrounds
    for lane_y, lane_h, lane in lane_y_ranges:
        svg_bars.append(f'<rect class="lane-bg below-group-elem" data-orig-y="{lane_y:.0f}" x="0" y="{lane_y:.0f}" width="{width}" height="{lane_h:.0f}" rx="2" />')
        svg_bars.append(f'<text class="lane-label below-group-elem" data-orig-y="{lane_y + 13:.0f}" x="4" y="{lane_y + 13:.0f}">{html.escape(lane.label)}</text>')

    # Draw collected arrows (group + req) above lane-bg, below bars
    for arrow_svg in collected_arrows:
        svg_bars.append(arrow_svg)

    # Batch-level arrows: from rollout_batch end to train_phase start
    rb_y = tp_y = None
    for lane_y, _, lane in lane_y_ranges:
        if lane.stream == "rollout":
            rb_y = lane_y + 3
        if lane.stream == "train" and tp_y is None:
            tp_y = lane_y + 3
    for rb in rollout_batch_spans:
        c = rb.get("rl_cycle_id")
        if c is None:
            continue
        tp = tp_by_cycle.get(c)
        if tp is None:
            continue
        rb_end = int(rb["start_wall_ns"]) + int(rb["duration_ns"])
        tp_start = int(tp["start_wall_ns"])
        if rb_y is None or tp_y is None:
            continue
        x1 = label_width + (rb_end - origin) / duration * plot_width
        x2 = label_width + (tp_start - origin) / duration * plot_width
        mid_y = (rb_y + tp_y) / 2
        svg_bars.append(f'<path class="arrow traj-arrow batch-arrow below-group-elem" data-cycle="{c}" data-row="0" style="display:none;stroke:{step_color(c)};stroke-width:1.5;fill:none" d="M {x1:.1f} {rb_y + bar_height:.0f} C {x1:.1f} {mid_y:.0f}, {x2:.1f} {mid_y:.0f}, {x2:.1f} {tp_y:.0f}" />')
        svg_bars.append(f'<text class="step-label traj-arrow below-group-elem" data-cycle="{c}" data-row="0" style="display:none" data-orig-y="{mid_y - 2:.0f}" x="{(x1 + x2) / 2 - 30:.1f}" y="{mid_y - 2:.0f}">batch {c} → step {c}</text>')

    # Draw lane bars (on top of arrows)
    for lane_y, _, lane in lane_y_ranges:
        stream = lane.stream
        color = COLORS.get(stream, "#64748b")
        for sub_idx, sub_row in enumerate(lane.sub_rows):
            for event in sub_row:
                y = lane_y + sub_idx * row_height + 3
                start = int(event["start_wall_ns"])
                span_duration = int(event["duration_ns"])
                x = label_width + (start - origin) / duration * plot_width
                bar_w = max(1.5, span_duration / duration * plot_width)
                name = str(event.get("name", "span"))
                cycle = event.get("rl_cycle_id")
                dur_s = span_duration / 1_000_000_000
                dur_str = f"{dur_s:.1f}s" if dur_s >= 1 else f"{span_duration / 1_000_000:.0f}ms"
                tooltip_parts = [f"name: {name}", f"duration: {dur_str}", f"rl_cycle_id: {cycle}"]
                if event.get("status"):
                    tooltip_parts.append(f"status: {event['status']}")
                if event.get("rank") is not None:
                    tooltip_parts.append(f"rank: {event['rank']}")
                tooltip = html.escape("\n".join(str(p) for p in tooltip_parts))
                svg_bars.append(f'<rect class="bar below-group-elem" data-cycle="{cycle}" data-orig-y="{y:.0f}" x="{x:.2f}" y="{y:.0f}" width="{bar_w:.2f}" height="{bar_height}" rx="3" fill="{color}"><title>{tooltip}</title></rect>')
                if bar_w > 50:
                    label_text = f"{name} {dur_str}" if bar_w > 100 else name
                    max_chars = int(bar_w / 6)
                    svg_bars.append(f'<text class="bar-text below-group-elem" data-orig-y="{y + 13:.0f}" x="{x + 4:.1f}" y="{y + 13:.0f}">{html.escape(label_text[:max_chars])}</text>')

    svg_content = "".join(svg_bars)

    # Compact overview SVG: GPU stream + CPU stream
    ov_label_w = 90
    ov_plot_w = 1200
    ov_bar_h = 6
    ov_lane_h = 14
    ov_x_scale = ov_plot_w / duration

    def _ov_x(ns_val):
        return ov_label_w + (ns_val - origin) * ov_x_scale

    def _ov_bar(x_ns, dur_ns, y, color, title, opacity=1.0, dashed=False):
        x1 = _ov_x(x_ns)
        w = max(0.5, dur_ns * ov_x_scale)
        dash_attr = ' stroke-dasharray="3,2" stroke="#fbbf24" stroke-width="0.5"' if dashed else ''
        return f'<rect x="{x1:.1f}" y="{y}" width="{w:.1f}" height="{ov_bar_h}" fill="{color}" opacity="{opacity}" rx="1"{dash_attr}><title>{html.escape(title)}</title></rect>'

    def _split_by_ws(t_start, t_end):
        cut_points = [t_start]
        for ws_s, ws_e in merged_ws:
            if ws_s > t_start and ws_s < t_end:
                cut_points.append(ws_s)
            if ws_e > t_start and ws_e < t_end:
                cut_points.append(ws_e)
        cut_points.append(t_end)
        return sorted(set(cut_points))

    ov_parts = []
    ov_y = 22

    # Requests (global optimal merge to fit max_rows, no overlap, split by weight_sync)
    ov_max_rows = 20

    # Sort all requests by start time then duration
    all_sorted = sorted(traj_spans, key=lambda t: (int(t["start_wall_ns"]), int(t["duration_ns"])))

    # Configure the initial row-ordering window and duration merge threshold
    # Requests in the first 120s are ordered by completion; merge durations within 20%
    base_start_thresh = 120_000_000_000  # 120 seconds
    all_durs = sorted(int(t["duration_ns"]) for t in traj_spans)
    base_dur_thresh = max(all_durs[len(all_durs) // 4] * 20 // 100, 10_000_000_000) if len(all_durs) >= 4 else 10_000_000_000

    def _try_merge(thresh_dur, max_per_bar=None):
        by_group: dict[Any, list] = defaultdict(list)
        for t in all_sorted:
            cycle = t.get("rl_cycle_id")
            status = t.get("status", "ok")
            by_group[(cycle, status)].append(t)
        merged = []
        for group_key in sorted(by_group.keys(), key=lambda c: (c[0] is None, c[0], c[1])):
            group = sorted(by_group[group_key], key=lambda t: int(t["duration_ns"]))
            current_bar = None
            base_dur = None
            members = []
            for t in group:
                t_start = int(t["start_wall_ns"])
                t_dur = int(t["duration_ns"])
                if current_bar is not None:
                    if (abs(t_dur - base_dur) <= thresh_dur
                            and (max_per_bar is None or current_bar["count"] < max_per_bar)):
                        current_bar["count"] += 1
                        members.append((t_start, t_start + t_dur))
                        continue
                if current_bar is not None:
                    # Use median start and end from members for bar position
                    members.sort()
                    med_idx = len(members) // 2
                    current_bar["start"] = members[med_idx][0]
                    current_bar["end"] = members[med_idx][1]
                    current_bar["dur"] = current_bar["end"] - current_bar["start"]
                    merged.append(current_bar)
                current_bar = {
                    "start": t_start,
                    "end": t_start + t_dur,
                    "dur": t_dur,
                    "cycle": group_key[0],
                    "status": group_key[1],
                    "count": 1,
                }
                base_dur = t_dur
                members = [(t_start, t_start + t_dur)]
            if current_bar is not None:
                members.sort()
                med_idx = len(members) // 2
                current_bar["start"] = members[med_idx][0]
                current_bar["end"] = members[med_idx][1]
                current_bar["dur"] = current_bar["end"] - current_bar["start"]
                merged.append(current_bar)
        merged.sort(key=lambda m: m["start"])
        return merged

    step_zero_starts = [
        int(event["start_wall_ns"])
        for event in display_spans
        if event.get("name") == "rl_step"
        and _safe_int(event.get("rl_cycle_id")) == 0
    ]

    def _count_non_overlap_rows(merged):
        if not merged:
            return 0, []

        initial_start = min(step_zero_starts) if step_zero_starts else min(m["start"] for m in merged)
        initial_cutoff = initial_start + base_start_thresh
        initial = [m for m in merged if m["start"] <= initial_cutoff]
        later = [m for m in merged if m["start"] > initial_cutoff]

        # Put requests launched around step 0 on rows by completion time. Requests
        # launched later stay in start-time order and reuse the earliest-ending available row.
        ordered = [
            *sorted(initial, key=lambda m: (m["end"], m["dur"], m["start"])),
            *sorted(later, key=lambda m: (m["start"], m["end"], m["dur"])),
        ]
        rows_end = []
        assignment_by_id = {}
        for m in ordered:
            best_ri = -1
            earliest_end = None
            for ri in range(len(rows_end)):
                if rows_end[ri] <= m["start"]:
                    if earliest_end is None or rows_end[ri] < earliest_end:
                        earliest_end = rows_end[ri]
                        best_ri = ri
            if best_ri >= 0:
                rows_end[best_ri] = m["end"]
                assignment_by_id[id(m)] = best_ri
            else:
                rows_end.append(m["end"])
                assignment_by_id[id(m)] = len(rows_end) - 1
        assignment = [assignment_by_id[id(m)] for m in merged]
        return len(rows_end), assignment

    # Initial merge: start with large max_per_bar, increase until fits
    max_per_bar = max(32, len(traj_spans) // ov_max_rows)
    merged_list = _try_merge(base_dur_thresh, max_per_bar)
    num_rows, row_assignment = _count_non_overlap_rows(merged_list)
    while num_rows > ov_max_rows:
        max_per_bar *= 2
        if max_per_bar > len(traj_spans):
            break
        merged_list = _try_merge(base_dur_thresh, max_per_bar)
        num_rows, row_assignment = _count_non_overlap_rows(merged_list)

    # Match requests without a cycle to a step using request completion time.
    if num_rows > ov_max_rows or any(t.get("rl_cycle_id") is None for t in all_sorted):
        ws_sorted = sorted(merged_ws, key=lambda ws: ws[0])
        def _match_step_by_complete(complete_ns):
            if not ws_sorted:
                return None
            count = sum(1 for _, ws_e in ws_sorted if ws_e <= complete_ns)
            return count

        for t in all_sorted:
            if t.get("rl_cycle_id") is None:
                t["rl_cycle_id"] = _match_step_by_complete(int(t["end_wall_ns"]))

        # Re-merge with updated cycles, increase max_per_bar until fits
        merged_list = _try_merge(base_dur_thresh, max_per_bar)
        num_rows, row_assignment = _count_non_overlap_rows(merged_list)
        while num_rows > ov_max_rows:
            max_per_bar *= 2
            if max_per_bar > len(traj_spans):
                break
            merged_list = _try_merge(base_dur_thresh, max_per_bar)
            num_rows, row_assignment = _count_non_overlap_rows(merged_list)

    # Sort merged_list by (start, dur) so row_assignment indices align
    merged_list.sort(key=lambda m: (m["start"], m["dur"]))
    num_rows, row_assignment = _count_non_overlap_rows(merged_list)

    ov_parts.append(f'<text x="4" y="{ov_y - 4}" font-size="11" fill="#e2e8f0" font-weight="bold">Requests</text>')

    # Render all bars on shared rows; style by status
    for m_idx, m in enumerate(merged_list):
        m_start = m["start"]
        m_end = m["end"]
        placed_ri = row_assignment[m_idx]
        cycle = m["cycle"]
        m_status = m.get("status", "ok")
        if m_status == "incomplete":
            color = "#fbbf24"
            opacity = 0.4
            dashed = True
        elif m_status == "error":
            color = "#ef4444"
            opacity = 0.5
            dashed = True
        else:
            color = step_color(cycle)
            opacity = 0.7
            dashed = False
        row_y = ov_y + placed_ri * ov_lane_h
        count = m["count"]
        avg_dur = m["dur"] / 1e9
        status_label = f" {m_status}" if m_status != "ok" else ""
        merge_label = f"{count}{status_label} requests, avg {avg_dur:.1f}s" if count > 1 else f"request{status_label}: {avg_dur:.1f}s"
        cut_points = _split_by_ws(m_start, m_end)
        for i in range(len(cut_points) - 1):
            seg_s = cut_points[i]
            seg_e = cut_points[i + 1]
            seg_dur = (seg_e - seg_s) / 1e9
            if seg_dur <= 0:
                continue
            seg_mid = (seg_s + seg_e) / 2
            in_ws = any(ws_s <= seg_mid < ws_e for ws_s, ws_e in merged_ws)
            if in_ws:
                continue
            ov_parts.append(_ov_bar(seg_s, seg_e - seg_s, row_y, color, merge_label, opacity, dashed=dashed))
        if count > 1 and (m_end - m_start) * ov_x_scale > 20:
            x_mid = _ov_x((m_start + m_end) / 2)
            ov_parts.append(f'<text x="{x_mid:.0f}" y="{row_y + 5}" font-size="8" fill="#fff" text-anchor="middle">x{count}</text>')
    ov_y += num_rows * ov_lane_h + 2

    ov_y += 8
    # GPU stream header
    ov_parts.append(f'<text x="4" y="{ov_y - 4}" font-size="11" fill="#3b82f6" font-weight="bold">GPU</text>')

    # GPU: rollout lane
    rollout_names = ["get_rollout_data"]
    for rn in rollout_names:
        rn_spans = [e for e in display_spans if e.get("name") == rn]
        if rn_spans:
            ov_parts.append(f'<text x="4" y="{ov_y + 8}" font-size="9" fill="#94a3b8">{rn}</text>')
            for e in rn_spans:
                s = int(e["start_wall_ns"])
                d = int(e["duration_ns"])
                cycle = e.get("rl_cycle_id")
                ov_parts.append(_ov_bar(s, d, ov_y, "#22c55e", f"{rn} step={cycle}: {d / 1e9:.1f}s"))
            ov_y += ov_lane_h

    # GPU: weight_sync lane
    ov_parts.append(f'<text x="4" y="{ov_y + 8}" font-size="9" fill="#94a3b8">weight sync</text>')
    for ws_s, ws_e in merged_ws:
        ov_parts.append(_ov_bar(ws_s, ws_e - ws_s, ov_y, "#7f1d1d", f"weight_sync: {(ws_e - ws_s) / 1e9:.1f}s"))
    ov_y += ov_lane_h

    # GPU: train_phase lane
    tp_spans = [e for e in display_spans if e.get("name") == "train_phase"]
    if tp_spans:
        ov_parts.append(f'<text x="4" y="{ov_y + 8}" font-size="9" fill="#94a3b8">train_phase</text>')
        for e in tp_spans:
            s = int(e["start_wall_ns"])
            d = int(e["duration_ns"])
            cycle = e.get("rl_cycle_id")
            ov_parts.append(_ov_bar(s, d, ov_y, "#60a5fa", f"train_phase step={cycle}: {d / 1e9:.1f}s"))
        ov_y += ov_lane_h

    # GPU: train sub-phases (one lane per sub-phase)
    train_sub_names = ["data_preprocess", "log_probs", "actor_compute_advantages", "critic_compute_advantages", "actor_train", "critic_train", "optimizer_step"]
    train_sub_colors = {"log_probs": "#fb923c", "actor_compute_advantages": "#a78bfa", "critic_compute_advantages": "#c084fc", "actor_train": "#3b82f6", "critic_train": "#ec4899", "optimizer_step": "#06b6d4"}
    for sub_name in train_sub_names:
        sub_spans = [e for e in display_spans if e.get("name") == sub_name or (sub_name == "log_probs" and e.get("name", "").endswith("log_probs"))]
        if not sub_spans:
            continue
        ov_parts.append(f'<text x="4" y="{ov_y + 8}" font-size="9" fill="#94a3b8">{sub_name}</text>')
        for e in sub_spans:
            s = int(e["start_wall_ns"])
            d = int(e["duration_ns"])
            cycle = e.get("rl_cycle_id")
            color = train_sub_colors.get(sub_name, "#64748b")
            ov_parts.append(_ov_bar(s, d, ov_y, color, f"{sub_name} step={cycle} rank={e.get('rank')}: {d / 1e9:.1f}s"))
        ov_y += ov_lane_h

    ov_y += 8
    # CPU stream header
    ov_parts.append(f'<text x="4" y="{ov_y - 4}" font-size="11" fill="#22c55e" font-weight="bold">CPU</text>')

    # CPU lanes
    cpu_names = ["_convert_samples_to_train_data", "_split_train_data_by_dp", "fill_routing_replay", "backup_weights"]
    cpu_colors = {"_convert_samples_to_train_data": "#14b8a6", "_split_train_data_by_dp": "#8b5cf6", "fill_routing_replay": "#eab308", "backup_weights": "#f97316"}
    for cpu_name in cpu_names:
        cpu_spans = [e for e in display_spans if e.get("name") == cpu_name]
        if not cpu_spans:
            continue
        ov_parts.append(f'<text x="4" y="{ov_y + 8}" font-size="9" fill="#94a3b8">{cpu_name}</text>')
        for e in cpu_spans:
            s = int(e["start_wall_ns"])
            d = int(e["duration_ns"])
            cycle = e.get("rl_cycle_id")
            color = cpu_colors.get(cpu_name, "#64748b")
            ov_parts.append(_ov_bar(s, d, ov_y, color, f"{cpu_name} step={cycle} rank={e.get('rank')}: {d / 1e9:.1f}s"))
        ov_y += ov_lane_h

    # Step separators
    for step_ns, step_cycle in step_boundaries:
        x = _ov_x(step_ns)
        ov_parts.append(f'<line x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="{ov_y}" stroke="#475569" stroke-dasharray="3,3" stroke-width="1" />')
        ov_parts.append(f'<text x="{x + 3:.1f}" y="18" font-size="9" fill="#64748b">step {step_cycle}</text>')

    ov_height = ov_y + 5
    overview_svg = f'<svg viewBox="0 0 {ov_label_w + ov_plot_w + 10} {ov_height}" style="background:#111827;border-radius:8px;display:block;margin:8px 0">{" ".join(ov_parts)}</svg>'

    counters = [
        event for event in events if event.get("type") == "counter"
    ]
    relations = [
        event for event in events
        if event.get("type") == "relation"
        and event.get("name") == "train_consumes_rollout"
    ]

    # Step aggregation table (convert to seconds)
    step_summary = _build_step_summary(events)
    step_rows: list[str] = []
    for s in step_summary:
        def ms2s(ms):
            return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"
        step_rows.append(
            f"<tr>"
            f"<td>{s['rl_cycle_id']}</td>"
            f"<td>{ms2s(s['step_time_ms'])}</td>"
            f"<td>{ms2s(s['rollout_busy_ms'])}</td>"
            f"<td>{ms2s(s['train_busy_ms'])}</td>"
            f"<td>{ms2s(s['sync_ms'])}</td>"
            f"<td>{ms2s(s['overlap_ms'])}</td>"
            f"<td>{s['overlap_ratio']:.1%}</td>"
            f"<td>{ms2s(s['unaccounted_ms'])}</td>"
            f"</tr>"
        )
    step_table_html = ""
    if step_rows:
        step_table_html = f"""
<h3>Step Summary</h3>
<table>
<thead><tr>
<th title="Training step (rl_cycle_id)">Step</th>
<th title="Total wall time from first span start to last span end">Step Time</th>
<th title="Union of request generation time (vllm_generate spans)">Rollout</th>
<th title="Union of train stream spans">Train</th>
<th title="Sum of weight_sync spans">Sync</th>
<th title="Time where rollout and train overlap">Overlap</th>
<th title="Overlap divided by Step Time">Overlap Ratio</th>
<th title="Step Time - Rollout - Train - Sync (overlap shown separately for async insight)">Unaccounted</th>
</tr></thead>
<tbody>
{''.join(step_rows)}
</tbody>
</table>
"""

    # Consume relations table — aggregate per step, compute staleness offline
    relation_rows: list[str] = []

    # Exclude requests known to be incomplete from consumption truncation rates.
    incomplete_sample_ids = {
        str(item.get("sample_id"))
        for item in traj_spans
        if item.get("status") == "incomplete" and item.get("sample_id") is not None
    }

    # Group relations by cycle
    from collections import defaultdict as _dd
    rels_by_cycle: dict[Any, list] = _dd(list)
    for rel in relations:
        cycle = rel.get("rl_cycle_id")
        if cycle is None:
            cycle = rel.get("target", {}).get("rl_cycle_id")
        rels_by_cycle[cycle].append(rel)

    for cycle in sorted(rels_by_cycle.keys()):
        rels = rels_by_cycle[cycle]
        all_samples = set()
        actor_v = None
        for rel in rels:
            source = rel.get("source", {})
            target = rel.get("target", {})
            for idx in source.get("sample_indices", []):
                sample_id = str(idx)
                if sample_id not in incomplete_sample_ids:
                    all_samples.add(sample_id)
            if actor_v is None:
                try:
                    actor_v = int(target.get("actor_policy_version")) if target.get("actor_policy_version") is not None else None
                except (TypeError, ValueError):
                    pass

        # The denominator is the consumed, completed request count.
        traj_count = len(all_samples)

        staleness_start_values = []
        staleness_end_values = []
        start_versions = set()
        end_versions = set()
        truncated_count = 0
        for sid in all_samples:
            info = traj_to_staleness.get(str(sid), {})
            s_start = info.get("staleness_from_start")
            s_end = info.get("staleness_from_end")
            sv = info.get("start_rollout_version")
            ev = info.get("end_rollout_version")
            if s_start is not None:
                staleness_start_values.append(s_start)
            if s_end is not None:
                staleness_end_values.append(s_end)
            if sv is not None:
                start_versions.add(sv)
            if ev is not None:
                end_versions.add(ev)
            fr = finish_reason_by_sample.get(str(sid))
            if fr is not None and fr != "stop":
                truncated_count += 1

        rollout_ver_min = min(start_versions | end_versions) if (start_versions | end_versions) else None
        rollout_ver_max = max(start_versions | end_versions) if (start_versions | end_versions) else None
        staleness_start_mean = sum(staleness_start_values) / len(staleness_start_values) if staleness_start_values else None
        staleness_end_mean = sum(staleness_end_values) / len(staleness_end_values) if staleness_end_values else None

        ver_range = (
            f"v{rollout_ver_min}" if rollout_ver_min == rollout_ver_max
            else f"v{rollout_ver_min}–v{rollout_ver_max}" if rollout_ver_min is not None
            else "—"
        )
        staleness_start_mean_str = (
            f"{staleness_start_mean:.1f}" if staleness_start_mean is not None else "—"
        )
        staleness_end_mean_str = (
            f"{staleness_end_mean:.1f}" if staleness_end_mean is not None else "—"
        )
        actor_ver_str = f"v{actor_v}" if actor_v is not None else "—"
        truncation_rate = f"{truncated_count / traj_count * 100:.1f}%" if traj_count > 0 else "—"

        relation_rows.append(
            f"<tr>"
            f"<td>step {cycle}</td>"
            f"<td>{traj_count}</td>"
            f"<td>{truncation_rate}</td>"
            f"<td>{ver_range}</td>"
            f"<td>{actor_ver_str}</td>"
            f"<td>{staleness_start_mean_str}</td>"
            f"<td>{staleness_end_mean_str}</td>"
            f"</tr>"
        )

    relation_table_html = ""
    if relation_rows:
        relation_table_html = f"""
<h3>Train Consumption</h3>
<table>
<thead><tr>
<th>Train Step</th>
<th>Requests</th>
<th>Truncation</th>
<th>Rollout Version</th><th>Actor Version</th>
<th>Staleness Start (mean)</th>
<th>Staleness End (mean)</th>
</tr></thead>
<tbody>
{''.join(relation_rows)}
</tbody>
</table>
"""

    # Async queue stats
    queue_counters = [e for e in counters if e.get("name") == "async_rollout_queue"]
    queue_stats_html = ""
    if queue_counters:
        inflight_vals = [e.get("values", {}).get("inflight", 0) for e in queue_counters]
        ready_vals = [e.get("values", {}).get("ready", 0) for e in queue_counters]
        max_concurrent_val = queue_counters[0].get("values", {}).get("max_concurrent", 0) if queue_counters else 0
        queue_stats_html = f"""
<h3>Async Queue Stats</h3>
<p>Max concurrent: {max_concurrent_val}</p>
<table>
<thead><tr><th>Metric</th><th>Min</th><th>Mean</th><th>Max</th><th>Samples</th></tr>
</thead>
<tbody>
<tr><td>Inflight (generating)</td><td>{min(inflight_vals)}</td><td>{sum(inflight_vals)/len(inflight_vals):.0f}</td><td>{max(inflight_vals)}</td><td>{len(inflight_vals)}</td></tr>
<tr><td>Ready (waiting for train)</td><td>{min(ready_vals)}</td><td>{sum(ready_vals)/len(ready_vals):.0f}</td><td>{max(ready_vals)}</td><td>{len(ready_vals)}</td></tr>
</tbody>
</table>
"""

    # Span type summary with nesting
    from collections import Counter as CCounter
    span_names = CCounter(e.get("name", "") for e in display_spans)
    _SPAN_NESTING = {
        "rl_step": ["get_rollout_data", "train_phase", "weight_sync"],
        "get_rollout_data": ["_convert_samples_to_train_data", "_split_train_data_by_dp"],
        "train_phase": ["data_preprocess", "log_probs", "actor_compute_advantages", "critic_compute_advantages", "optimizer_step", "backup_weights", "fill_routing_replay", "actor_train", "critic_train"],
    }
    _CHILD_TO_PARENT = {}
    for parent, children in _SPAN_NESTING.items():
        for child in children:
            _CHILD_TO_PARENT[child] = parent

    def _fmt_dur(sec):
        if sec >= 1:
            return f"{sec:.1f}s"
        return f"{sec*1000:.0f}ms"

    _TOP_ORDER = ["rl_step"]

    _step_zero_intervals = [
        (
            int(event["start_wall_ns"]),
            int(event["start_wall_ns"]) + int(event["duration_ns"]),
        )
        for event in display_spans
        if event.get("name") == "rl_step"
        and _safe_int(event.get("rl_cycle_id")) == 0
    ]

    def _is_step_zero_span(event: dict[str, Any]) -> bool:
        cycle = _safe_int(event.get("rl_cycle_id"))
        if cycle is not None:
            return cycle == 0
        start_ns = int(event["start_wall_ns"])
        return any(start <= start_ns < end for start, end in _step_zero_intervals)

    def _render_span_tree(name: str, depth: int) -> str:
        matching_spans = [e for e in display_spans if e.get("name") == name]
        durs = [int(e["duration_ns"]) for e in matching_spans]
        if not durs:
            return ""
        indent = "&nbsp;" * (depth * 3)
        if name == "get_rollout_data" and _step_zero_intervals:
            step_zero_durs = [int(e["duration_ns"]) for e in matching_spans if _is_step_zero_span(e)]
            later_durs = [int(e["duration_ns"]) for e in matching_spans if not _is_step_zero_span(e)]
            html_parts = []
            for label, group_durs in (("step 0", step_zero_durs), ("step 1+", later_durs)):
                if group_durs:
                    mean_dur = sum(group_durs) / len(group_durs) / 1e9
                    html_parts.append(f"<tr><td>{indent}{name} ({label})</td><td>{_fmt_dur(mean_dur)}</td></tr>")
        else:
            mean_dur = sum(durs) / len(durs) / 1e9
            html_parts = [f"<tr><td>{indent}{name}</td><td>{_fmt_dur(mean_dur)}</td></tr>"]
        for child in _SPAN_NESTING.get(name, []):
            html_parts.append(_render_span_tree(child, depth + 1))
        return "".join(html_parts)

    span_summary_html = "<h3>Span Summary</h3><table><thead><tr><th>Name</th><th>Mean Duration</th></tr></thead><tbody>"
    rendered_names = set()
    for top_name in _TOP_ORDER:
        if top_name in span_names:
            span_summary_html += _render_span_tree(top_name, 0)
            rendered_names.add(top_name)
            rendered_names.update(_SPAN_NESTING.get(top_name, []))
    for name in sorted(span_names, key=lambda n: -span_names[n]):
        if name in rendered_names or name in _CHILD_TO_PARENT:
            continue
        durs = [int(e["duration_ns"]) for e in display_spans if e.get("name") == name]
        mean_dur = sum(durs) / len(durs) / 1e9 if durs else 0
        span_summary_html += f"<tr><td>{name}</td><td>{_fmt_dur(mean_dur)}</td></tr>"
        rendered_names.add(name)
    span_summary_html += "</tbody></table>"

    # Request activity stats: split completed vs incomplete, compute truncation rate,
    # exclude weight_sync periods, add mean/std, also for response_length
    traj_stats_html = ""
    if traj_spans:
        import statistics as _stats

        # Build merged weight_sync intervals for exclusion
        _ws_intervals = sorted(merged_ws) if merged_ws else []

        def _exclude_ws(start_ns: int, end_ns: int) -> float:
            """Duration in seconds excluding weight_sync overlap."""
            total = end_ns - start_ns
            for ws_s, ws_e in _ws_intervals:
                ov_start = max(start_ns, ws_s)
                ov_end = min(end_ns, ws_e)
                if ov_end > ov_start:
                    total -= (ov_end - ov_start)
            return max(0, total) / 1e9

        completed = [t for t in traj_spans if t.get("status") != "incomplete"]
        incomplete = [t for t in traj_spans if t.get("status") == "incomplete"]

        def _get_response_length(t: dict) -> int | None:
            attrs = t.get("attrs", {})
            rl = attrs.get("response_length") or attrs.get("completion_tokens")
            return int(rl) if rl is not None else None

        def _stats_row(label: str, items: list) -> str:
            if not items:
                return f"<tr><td>{label}</td><td>0</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>"
            durs = sorted(_exclude_ws(int(t["start_wall_ns"]), int(t["end_wall_ns"])) for t in items)
            n = len(durs)
            mean_v = sum(durs) / n
            std_v = _stats.stdev(durs) if n > 1 else 0.0
            resp_lens = sorted(_get_response_length(t) for t in items if _get_response_length(t) is not None)
            if resp_lens:
                r_n = len(resp_lens)
                r_mean = sum(resp_lens) / r_n
                r_std = _stats.stdev(resp_lens) if r_n > 1 else 0.0
                r_min = resp_lens[0]
                r_p50 = resp_lens[r_n // 2]
                r_max = resp_lens[-1]
                resp_str = f"{r_min}</td><td>{r_mean:.0f}</td><td>±{r_std:.0f}</td><td>{r_p50}</td><td>{r_max}"
            else:
                resp_str = "—</td><td>—</td><td>—</td><td>—</td><td>—"
            return (
                f"<tr><td>{label}</td><td>{n}</td>"
                f"<td>{durs[0]:.1f}s</td><td>{mean_v:.1f}s</td>"
                f"<td>±{std_v:.1f}s</td>"
                f"<td>{durs[n//2]:.1f}s</td><td>{durs[-1]:.1f}s</td>"
                f"<td>{resp_str}</td></tr>"
            )

        # Truncation rate is defined over completed requests only.
        truncated = [
            t for t in completed
            if finish_reason_by_sample.get(str(t.get("sample_id", "")), "stop") != "stop"
        ]
        truncation_rate = (
            f"{len(truncated) / len(completed) * 100:.1f}%"
            if completed else "N/A"
        )

        wall_span = (max(int(t['end_wall_ns']) for t in traj_spans) - min(int(t['start_wall_ns']) for t in traj_spans)) / 1e9 if traj_spans else 0

        traj_stats_html = f"""
<h3>Request Activity</h3>
<p>Wall span: {wall_span:.1f}s &middot; Truncation rate: {truncation_rate}</p>
<table>
<thead><tr><th rowspan="2">Category</th><th rowspan="2">Count</th><th colspan="5">Duration (excl. weight_sync)</th><th colspan="5">Response Length</th></tr>
<tr><th>Min</th><th>Mean</th><th>Std</th><th>P50</th><th>Max</th><th>Min</th><th>Mean</th><th>Std</th><th>P50</th><th>Max</th></tr>
</thead>
<tbody>
{_stats_row("Completed", completed)}
{_stats_row("Incomplete", incomplete)}
</tbody>
</table>
"""

    # Token length histograms
    prompt_lens = []
    response_lens = []
    for e in request_spans:
        attrs = e.get("attrs", {})
        pl = attrs.get("prompt_length")
        rl = attrs.get("response_length")
        if pl is not None:
            prompt_lens.append(int(pl))
        if rl is not None:
            response_lens.append(int(rl))

    def _histogram_svg(values, title, color, bucket_count=20):
        if not values:
            return ""
        sorted_vals = sorted(values)
        vmin, vmax = sorted_vals[0], sorted_vals[-1]
        p50 = sorted_vals[len(sorted_vals) // 2]
        p95 = sorted_vals[int(len(sorted_vals) * 0.95)] if len(sorted_vals) > 1 else vmax
        # If integer values with small range, use one bar per value
        if all(isinstance(v, int) for v in values) and vmax - vmin <= 30:
            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            keys = sorted(counts.keys())
            max_count = max(counts.values())
            plot_w = 500
            bar_w = plot_w / len(keys)
            chart_h = 100
            bars = []
            for i, k in enumerate(keys):
                c = counts[k]
                x = 60 + i * bar_w
                h = int(c / max_count * 60) if max_count > 0 else 0
                y = chart_h - h - 18
                bars.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w - 2:.1f}" height="{h}" fill="{color}" rx="2"><title>staleness={k}: {c} requests</title></rect>')
                bars.append(f'<text x="{x + bar_w/2:.0f}" y="{chart_h - 4}" font-size="10" fill="#94a3b8" text-anchor="middle">{k}</text>')
                bars.append(f'<text x="{x + bar_w/2:.0f}" y="{y - 3}" font-size="9" fill="#e2e8f0" text-anchor="middle">{c}</text>')
            bars.append(f'<text x="60" y="10" font-size="10" fill="#94a3b8">{title}</text>')
            bars.append(f'<text x="560" y="10" font-size="9" fill="#64748b" text-anchor="end">n={len(values)} min={vmin} p50={p50} p95={p95} max={vmax}</text>')
            bars.append(f'<line x1="55" y1="12" x2="55" y2="{chart_h - 14}" stroke="#475569" stroke-width="1" />')
            return f'<svg viewBox="0 0 600 {chart_h}" style="background:#1e293b;border-radius:6px;margin:4px 0;display:block">{" ".join(bars)}</svg>'
        # Otherwise use bucketed histogram
        if vmax == vmin:
            vmax = vmin + 1
        bucket_size = (vmax - vmin) / bucket_count
        buckets = [0] * bucket_count
        for v in values:
            idx = min(int((v - vmin) / bucket_size), bucket_count - 1)
            buckets[idx] += 1
        max_count = max(buckets) if buckets else 1
        plot_w = 500
        bar_w = plot_w / bucket_count
        chart_h = 100
        bars = []
        for i, count in enumerate(buckets):
            x = 60 + i * bar_w
            h = int(count / max_count * 60) if max_count > 0 else 0
            y = chart_h - h - 18
            lo = int(vmin + i * bucket_size)
            hi = int(vmin + (i + 1) * bucket_size)
            bars.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w - 1:.1f}" height="{h}" fill="{color}" rx="1"><title>{lo}-{hi}: {count}</title></rect>')
            if i % 4 == 0:
                bars.append(f'<text x="{x + bar_w/2:.0f}" y="{chart_h - 4}" font-size="8" fill="#64748b" text-anchor="middle">{lo}</text>')
                if count > 0:
                    bars.append(f'<text x="{x + bar_w/2:.0f}" y="{y - 3}" font-size="8" fill="#e2e8f0" text-anchor="middle">{count}</text>')
        bars.append(f'<text x="60" y="10" font-size="10" fill="#94a3b8">{title}</text>')
        bars.append(f'<text x="560" y="10" font-size="9" fill="#64748b" text-anchor="end">n={len(values)} min={vmin} p50={p50} p95={p95} max={vmax}</text>')
        bars.append(f'<line x1="55" y1="12" x2="55" y2="{chart_h - 14}" stroke="#475569" stroke-width="1" />')
        return f'<svg viewBox="0 0 600 {chart_h}" style="background:#1e293b;border-radius:6px;margin:4px 0;display:block">{" ".join(bars)}</svg>'

    # Staleness histogram — horizontal bars, start vs end side by side
    staleness_start_values = []
    staleness_end_values = []
    for tid, info in traj_to_staleness.items():
        s_start = info.get("staleness_from_start")
        s_end = info.get("staleness_from_end")
        if s_start is not None:
            staleness_start_values.append(int(s_start))
        if s_end is not None:
            staleness_end_values.append(int(s_end))

    staleness_hist_html = ""
    if staleness_start_values or staleness_end_values:
        all_vals = staleness_start_values + staleness_end_values
        vmin = min(all_vals) if all_vals else 0
        vmax = max(all_vals) if all_vals else 1
        keys = list(range(vmin, vmax + 1))
        start_counts = {k: staleness_start_values.count(k) for k in keys}
        end_counts = {k: staleness_end_values.count(k) for k in keys}
        max_count = max(max(start_counts.values()), max(end_counts.values())) if keys else 1
        chart_w = 500
        label_w = 50
        row_h = 14
        chart_h = len(keys) * row_h + 18
        bar_h = 4
        bar_gap = 1
        plot_w = chart_w - label_w - 10
        bars = []
        for i, k in enumerate(keys):
            y = 16 + i * row_h
            sc = start_counts[k]
            ec = end_counts[k]
            sw = int(sc / max_count * plot_w) if max_count > 0 else 0
            ew = int(ec / max_count * plot_w) if max_count > 0 else 0
            bars.append(f'<text x="45" y="{y + bar_h}" font-size="5" fill="#94a3b8" text-anchor="end">{k}</text>')
            bars.append(f'<rect x="{label_w}" y="{y}" width="{sw}" height="{bar_h}" fill="#f59e0b" rx="0.5"><title>staleness={k} (start): {sc} requests</title></rect>')
            if sc > 0:
                bars.append(f'<text x="{label_w + sw + 2:.0f}" y="{y + bar_h - 0.5}" font-size="5" fill="#f59e0b">{sc}</text>')
            bars.append(f'<rect x="{label_w}" y="{y + bar_h + bar_gap}" width="{ew}" height="{bar_h}" fill="#3b82f6" rx="0.5"><title>staleness={k} (end): {ec} requests</title></rect>')
            if ec > 0:
                bars.append(f'<text x="{label_w + ew + 2:.0f}" y="{y + bar_h + bar_gap + bar_h - 0.5}" font-size="5" fill="#3b82f6">{ec}</text>')
        s_sorted = sorted(staleness_start_values)
        e_sorted = sorted(staleness_end_values)
        s_p50 = s_sorted[len(s_sorted)//2] if s_sorted else "—"
        e_p50 = e_sorted[len(e_sorted)//2] if e_sorted else "—"
        bars.append(f'<text x="50" y="8" font-size="5" fill="#f59e0b">\u25a0 Start</text>')
        bars.append(f'<text x="120" y="8" font-size="5" fill="#3b82f6">\u25a0 End</text>')
        bars.append(f'<text x="490" y="8" font-size="5" fill="#64748b" text-anchor="end">n={len(staleness_start_values)} p50={s_p50}/{e_p50}</text>')
        bars.append(f'<line x1="{label_w}" y1="11" x2="{label_w}" y2="{chart_h - 4}" stroke="#475569" stroke-width="0.5" />')
        staleness_hist_html = f'<div class="section"><h3>Staleness Distribution</h3><svg viewBox="0 0 {chart_w} {chart_h}" style="background:#1e293b;border-radius:6px;margin:4px 0;display:block">{" ".join(bars)}</svg></div>'

    summary = {
        "run_id": run_id,
        "total_events": len(events),
        "spans": len(display_spans),
        "request_activity_spans": len(traj_spans),
        "steps": len(step_summary),
        "relations": len(relations),
        "malformed_lines": malformed,
        "duplicate_events": duplicates,
        "wall_duration_s": (end - origin) / 1_000_000_000,
    }
    malformed_warning_html = ""
    if malformed > 0:
        malformed_warning_html = f'<div style="background:#78350f;padding:12px;border-radius:8px;margin:16px 0"><strong>{malformed} Malformed Lines</strong> — JSON parsing failed for these log lines. Affected events are skipped; other events render normally.</div>'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vime Performance Timeline</title>
<style>
body{{margin:0;background:#0f172a;color:#e2e8f0;font:14px system-ui}}
header{{position:sticky;top:0;padding:16px 24px;background:#111827;z-index:100}}
main{{padding:20px;overflow:auto}}
pre{{white-space:pre-wrap;background:#1e293b;padding:12px;border-radius:8px;font:12px ui-monospace,monospace}}
svg{{min-width:1000px;background:#111827;border-radius:8px;display:block;margin:16px 0}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font:12px ui-monospace,monospace}}
th,td{{border:1px solid #334155;padding:4px 8px;text-align:left}}
th{{background:#1e293b;color:#94a3b8}}
td{{color:#e2e8f0}}
tr:nth-child(even){{background:#1e293b}}
h3{{color:#94a3b8;margin-top:24px;margin-bottom:4px}}
.section{{margin-bottom:24px}}
.controls{{display:flex;gap:12px;flex-wrap:wrap;padding:12px;background:#1e293b;border-radius:8px;margin:12px 0}}
.controls button{{background:#334155;color:#e2e8f0;border:1px solid #475569;padding:6px 12px;border-radius:4px;cursor:pointer;font:12px ui-monospace,monospace}}
.controls button:hover{{background:#475569}}
.controls button.active{{background:#3b82f6;color:#fff}}
.zoom-hint{{color:#64748b;font:11px ui-monospace,monospace}}
.tick-label{{font:11px ui-monospace,monospace;fill:#64748b}}
.step-label{{font:11px ui-monospace,monospace;fill:#94a3b8}}
.lane-label{{font:12px ui-monospace,monospace;fill:#94a3b8}}
.bar-text{{font:10px ui-monospace,monospace;fill:#e2e8f0}}
.group-text{{font:10px ui-monospace,monospace;fill:#e2e8f0}}
.traj-bar{{opacity:0.7;pointer-events:all}}
.traj-bar:hover{{opacity:1;stroke:#fff;stroke-width:1}}
.lane-bg{{fill:#1e293b;stroke:#334155;stroke-width:1}}
.bar{{pointer-events:all}}
.bar:hover{{stroke:#fff;stroke-width:1.5}}
.arrow{{stroke-width:1.5;fill:none}}
.traj-arrow{{stroke-width:0.8;fill:none;opacity:0.4}}
.ws-band{{fill:#1e293b;opacity:1}}
.ws-label{{font:9px ui-monospace,monospace;fill:#64748b}}
#zoom-overlay{{position:absolute;background:rgba(59,130,246,0.2);border:1px solid #3b82f6;pointer-events:none;display:none}}
</style>
</head>
<body>
<header><strong>Vime Performance Timeline</strong> &middot; run {html.escape(run_id)}</header>
<main>
    {malformed_warning_html}
<div class="section">
<details><summary>Summary</summary>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
</details>
</div>
<div class="section" style="display:flex;flex-wrap:wrap;gap:20px">
<div style="flex:1;min-width:400px">
{step_table_html}
</div>
<div style="flex:1;min-width:300px">
{traj_stats_html}
{queue_stats_html}
</div>
</div>
<div class="section" style="display:flex;flex-wrap:wrap;gap:20px">
<div style="flex:1;min-width:400px">
{relation_table_html}
</div>
<div style="flex:1;min-width:300px">
{span_summary_html}
</div>
</div>
{staleness_hist_html}
<div class="section">
<h3>Overview</h3>
{overview_svg}
</div>
<div class="section">
<div class="controls">
<button id="btn-arrows" onclick="toggleArrows()">Show Arrows</button>
<button id="btn-ws" class="active" onclick="toggleWS()">Weight Sync</button>
<button onclick="resetZoom()">Reset Zoom</button>
<button id="btn-expand-all" onclick="toggleExpandAll()">Expand All Groups</button>
<span class="zoom-hint">Ctrl+Scroll to zoom &middot; Drag to pan &middot; Click group to expand</span>
</div>
<div style="position:relative;overflow:auto">
<div id="zoom-overlay"></div>
<svg id="timeline-svg" viewBox="0 0 {width} {svg_height}" xmlns="http://www.w3.org/2000/svg" style="user-select:none;cursor:grab">
<defs>
<marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#64748b"/></marker>
<marker id="arrowhead-sm" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0, 6 2, 0 4" fill="context-stroke"/></marker>
</defs>
{svg_content}
</svg>
</div>
</div>
</main>
<script>
var svg = document.getElementById('timeline-svg');
var origVB = [0, 0, {width}, {svg_height}];
var arrowsVisible = false;
var wsVisible = true;
var expandedGroups = new Set();
var currentShift = 0;
var EXPAND_H = {group_expand_height};

function toggleArrows() {{
    arrowsVisible = !arrowsVisible;
    svg.querySelectorAll('.group-arrow, .batch-arrow').forEach(function(el){{ el.style.display = arrowsVisible ? '' : 'none'; }});
    if (arrowsVisible) {{
        expandedGroups.forEach(function(gid) {{
            svg.querySelectorAll('.req-arrow[data-group="' + gid + '"]').forEach(function(el){{ el.style.display = ''; }});
        }});
    }} else {{
        svg.querySelectorAll('.req-arrow').forEach(function(el){{ el.style.display = 'none'; }});
    }}
    document.getElementById('btn-arrows').classList.toggle('active', arrowsVisible);
    document.getElementById('btn-arrows').textContent = arrowsVisible ? 'Hide Arrows' : 'Show Arrows';
    recalcLayout();
}}
function toggleWS() {{
    wsVisible = !wsVisible;
    svg.querySelectorAll('.ws-band, .ws-label').forEach(function(e){{e.style.display=wsVisible?'':'none'}});
    document.getElementById('btn-ws').classList.toggle('active', wsVisible);
}}
function resetZoom() {{
    svg.setAttribute('viewBox', origVB.join(' '));
}}
function recalcLayout() {{
    // Build row -> expand height mapping from expanded groups
    var rowExpandH = {{}};
    expandedGroups.forEach(function(egid) {{
        var el = svg.querySelector('.group-bar[data-group="' + egid + '"]');
        if (el) {{
            var ri = parseInt(el.getAttribute('data-row') || '0');
            var reqGroup = svg.querySelector('.req-group[data-group="' + egid + '"]');
            var eh = reqGroup ? parseInt(reqGroup.getAttribute('data-expand-height') || EXPAND_H) : EXPAND_H;
            rowExpandH[ri] = Math.max(rowExpandH[ri] || 0, eh);
        }}
    }});
    var expandedRows = Object.keys(rowExpandH).map(Number);
    expandedRows.sort(function(a,b){{return a-b}});
    currentShift = 0;
    for (var i = 0; i < expandedRows.length; i++) currentShift += rowExpandH[expandedRows[i]];
    svg.querySelectorAll('.group-bar, .group-text, .req-group').forEach(function(el) {{
        var elRow = parseInt(el.getAttribute('data-row') || el.parentNode.getAttribute('data-row') || '0');
        var shift = 0;
        for (var i = 0; i < expandedRows.length; i++) {{
            if (elRow > expandedRows[i]) shift += rowExpandH[expandedRows[i]];
        }}
        el.setAttribute('transform', 'translate(0,' + shift + ')');
    }});
    var groupBg = svg.querySelector('.group-lane-bg');
    if (groupBg) {{
        groupBg.setAttribute('height', {group_lane_height_collapsed} + currentShift);
    }}
    svg.querySelectorAll('.ws-band').forEach(function(el) {{
        el.setAttribute('height', {group_lane_height} + currentShift);
    }});
    svg.querySelectorAll('.below-group-elem').forEach(function(el) {{
        var origY = parseFloat(el.getAttribute('data-orig-y'));
        if (!isNaN(origY)) {{
            el.setAttribute('y', origY + currentShift);
        }}
    }});
    // Update arrow paths
    svg.querySelectorAll('.group-arrow, .req-arrow, .batch-arrow').forEach(function(el) {{
        var d = el.getAttribute('data-orig-d') || el.getAttribute('d');
        if (!el.getAttribute('data-orig-d')) el.setAttribute('data-orig-d', d);
        var arrowRow = parseInt(el.getAttribute('data-row') || el.parentNode.getAttribute('data-row') || '0');
        var arrowShift = 0;
        for (var i = 0; i < expandedRows.length; i++) {{
            if (arrowRow > expandedRows[i]) arrowShift += rowExpandH[expandedRows[i]];
        }}
        var newD = d.replace(/(-?[\\d.]+)\\s+(-?[\\d.]+)/g, function(match, xv, yv) {{
            var newY = parseFloat(yv);
            if (newY > {group_lane_height_collapsed + 24}) {{
                newY += currentShift;
            }} else if (newY > 24 + {header_height}) {{
                newY += arrowShift;
            }}
            return xv + ' ' + newY;
        }});
        el.setAttribute('d', newD);
    }});
    var vb = svg.getAttribute('viewBox').split(' ').map(Number);
    vb[3] = {svg_height} + currentShift;
    svg.setAttribute('viewBox', vb[0] + ' ' + vb[1] + ' ' + vb[2] + ' ' + vb[3]);
}}
function toggleExpandAll() {{
    var btn = document.getElementById('btn-expand-all');
    if (expandedGroups.size > 0) {{
        expandedGroups.clear();
        svg.querySelectorAll('.req-group').forEach(function(el){{ el.style.display = 'none'; }});
        btn.classList.remove('active');
        btn.textContent = 'Expand All Groups';
    }} else {{
        svg.querySelectorAll('.group-bar').forEach(function(el) {{
            var gid = el.getAttribute('data-group');
            expandedGroups.add(gid);
            var rg = svg.querySelector('.req-group[data-group="' + gid + '"]');
            if (rg) rg.style.display = '';
        }});
        btn.classList.add('active');
        btn.textContent = 'Collapse All Groups';
    }}
    recalcLayout();
}}

// Ctrl+wheel zoom; plain wheel scrolls page
svg.addEventListener('wheel', function(e) {{
    if (!e.ctrlKey) return;
    e.preventDefault();
    var rect = svg.getBoundingClientRect();
    var mx = (e.clientX - rect.left) / rect.width;
    var my = (e.clientY - rect.top) / rect.height;
    var vb = svg.getAttribute('viewBox').split(' ').map(Number);
    var factor = e.deltaY > 0 ? 1.2 : 0.83;
    var newW = Math.max(50, Math.min({width}, vb[2] * factor));
    var newH = vb[3] * (newW / vb[2]);
    var newX = vb[0] + (vb[2] - newW) * mx;
    var newY = vb[1] + (vb[3] - newH) * my;
    svg.setAttribute('viewBox', newX + ' ' + newY + ' ' + newW + ' ' + newH);
}}, {{passive: false}});

// Drag to pan
var isDragging = false, dragStartX = 0, dragStartY = 0, dragVB = null;
svg.addEventListener('mousedown', function(e) {{
    if (e.target.classList && (e.target.classList.contains('group-bar') || e.target.classList.contains('step-label') || e.target.classList.contains('req-bar'))) return;
    isDragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragVB = svg.getAttribute('viewBox').split(' ').map(Number);
    svg.style.cursor = 'grabbing';
}});
window.addEventListener('mousemove', function(e) {{
    if (!isDragging) return;
    var rect = svg.getBoundingClientRect();
    var dx = (e.clientX - dragStartX) / rect.width * dragVB[2];
    var dy = (e.clientY - dragStartY) / rect.height * dragVB[3];
    svg.setAttribute('viewBox', (dragVB[0] - dx) + ' ' + (dragVB[1] - dy) + ' ' + dragVB[2] + ' ' + dragVB[3]);
}});
window.addEventListener('mouseup', function() {{ isDragging = false; svg.style.cursor = 'grab'; }});

// Init: hide all arrows on page load
svg.querySelectorAll('.traj-arrow').forEach(function(el){{ el.style.display = 'none'; }});
// Group expand/collapse
svg.addEventListener('click', function(e) {{
    if (e.target.classList && e.target.classList.contains('group-bar')) {{
        var gid = e.target.getAttribute('data-group');
        if (!gid) return;
        var reqGroup = svg.querySelector('.req-group[data-group="' + gid + '"]');
        if (!reqGroup) return;
        if (expandedGroups.has(gid)) {{
            expandedGroups.delete(gid);
            reqGroup.style.display = 'none';
        }} else {{
            expandedGroups.add(gid);
            reqGroup.style.display = '';
        }}
        var allExpanded = svg.querySelectorAll('.group-bar').length === expandedGroups.size;
        var btn = document.getElementById('btn-expand-all');
        btn.classList.toggle('active', allExpanded);
        btn.textContent = allExpanded ? 'Collapse All Groups' : 'Expand All Groups';
        recalcLayout();
    }}
}});
</script>
</body>
</html>
"""


def write_json(path: Path, data: dict[str, Any]) -> None:
    payload = json.dumps(
        data, ensure_ascii=False, separators=(",", ":"), default=str
    )
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        path.write_text(payload, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build perf timeline HTML and Chrome Trace from Vime logs."
    )
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--step-start", type=int, default=None, help="Only render events from this step onwards")
    parser.add_argument("--step-end", type=int, default=None, help="Only render events up to this step")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    parsed = parse_logs(args.logs)
    try:
        run_id, events = select_run(parsed.events, args.run_id)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    events, relation_duplicates = dedupe_train_consume_relations(events)

    # Filter events by step range
    if args.step_start is not None or args.step_end is not None:
        step_cycles = set()
        for e in events:
            cycle = e.get("rl_cycle_id")
            if cycle is not None:
                step_cycles.add(int(cycle))
            cycle = e.get("attrs", {}).get("rl_cycle_id")
            if cycle is not None:
                c = _safe_int(cycle)
                if c is not None:
                    step_cycles.add(c)
            cycle = e.get("target", {}).get("rl_cycle_id")
            if cycle is not None:
                step_cycles.add(int(cycle))
        keep_cycles = set()
        for c in step_cycles:
            if args.step_start is not None and c < args.step_start:
                continue
            if args.step_end is not None and c > args.step_end:
                continue
            keep_cycles.add(c)
        # Build sample_to_batch from relation events to filter request spans by consumed step
        sample_to_cycle = {}
        for e in events:
            if e.get("type") == "relation" and e.get("name") == "train_consumes_rollout":
                tgt_cycle = e.get("target", {}).get("rl_cycle_id")
                if tgt_cycle is not None:
                    for idx in e.get("source", {}).get("sample_indices", []):
                        sample_to_cycle[str(idx)] = int(tgt_cycle)
        events = [
            e for e in events
            if _event_in_step_range(e, keep_cycles, sample_to_cycle)
        ]
        print(f"Filtered to steps {args.step_start}-{args.step_end}: {len(events)} events", file=sys.stderr)

    args.html.write_text(
        render_html(
            run_id,
            events,
            parsed.malformed_lines,
            parsed.duplicate_events + relation_duplicates,
        ),
        encoding="utf-8",
    )
    write_json(args.trace, chrome_trace(events))
    print(
        f"run={run_id} events={len(events)} "
        f"html={args.html} trace={args.trace}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
