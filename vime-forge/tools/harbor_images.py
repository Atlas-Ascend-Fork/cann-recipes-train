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

"""Inventory or stage immutable images from an existing prepared SWE bundle.

Run on a CPU Docker host. Configuration is external. The bundle is read-only;
no Harbor task conversion or qualification-record mutation is performed.
"""
from __future__ import annotations

import argparse
import codecs
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


class PullProgress:
    """Count byte advances, not Docker's repeated redraws of the same progress bar."""

    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    layer = re.compile(r"^([0-9a-f]{12}):\s*(.*)")
    amount = re.compile(r"([\d.]+)\s*([kMGT]?B)/")

    def __init__(self, now):
        self.last_progress = now
        self.downloaded = 0
        self.layers = {}
        self.buffer = ""

    def feed(self, text, now):
        # Keep incomplete lines (including split ANSI sequences) between reads.
        lines = re.split(r"[\r\n]", self.buffer + text)
        self.buffer = lines.pop()[-8192:]
        for line in lines:
            match = self.layer.match(self.ansi.sub("", line).strip())
            if not match:
                continue
            key, status = match.groups()
            phase = status.split(" [", 1)[0]
            old_phase, old_bytes = self.layers.get(key, (None, 0))
            amount = self.amount.search(status)
            current = (
                int(float(amount[1]) * 1000 ** {"B": 0, "kB": 1, "MB": 2, "GB": 3, "TB": 4}[amount[2]])
                if amount else 0
            )
            if phase != old_phase or current > old_bytes:
                self.last_progress = now
            if phase == "Downloading":
                self.downloaded += max(0, current - (old_bytes if old_phase == phase else 0))
            self.layers[key] = (phase, current)

    @property
    def downloading(self):
        return any(phase == "Downloading" for phase, _ in self.layers.values())


def pull_image(ref, timeout, idle_timeout=300, min_bytes_sec=0, slow_window=300):
    """Preserve Docker credentials/context, exposing byte progress through a PTY."""
    if not idle_timeout and not min_bytes_sec:
        process = subprocess.run(["docker", "pull", ref], capture_output=True, text=True, timeout=timeout)
        if process.returncode:
            raise RuntimeError(process.stderr[-1000:])
        return
    import pty
    master, slave = pty.openpty()
    process = None
    started = time.monotonic()
    progress = PullProgress(started)
    window_at, window_bytes, logged_at = started, 0, started
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    tail = ""
    try:
        process = subprocess.Popen(["docker", "pull", ref], stdin=subprocess.DEVNULL, stdout=slave, stderr=slave)
        os.close(slave)
        slave = None
        while True:
            if select.select([master], [], [], 0.5)[0]:
                try:
                    data = os.read(master, 65536)
                except OSError as exc:
                    # Linux PTYs return EIO when their last writer closes.
                    import errno
                    if exc.errno != errno.EIO:
                        raise
                    data = b""
                if data:
                    text = decoder.decode(data)
                    tail = (tail + text)[-8192:]
                    progress.feed(text, time.monotonic())
                elif process.poll() is not None:
                    break
            now = time.monotonic()
            if process.poll() is not None:
                break
            if now - started >= timeout:
                raise TimeoutError(f"Pull exceeded {timeout}s")
            if idle_timeout and now - progress.last_progress >= idle_timeout:
                raise TimeoutError(f"No pull progress for {idle_timeout}s")
            if now - window_at >= slow_window:
                rate = (progress.downloaded - window_bytes) / (now - window_at)
                if min_bytes_sec and progress.downloading and rate < min_bytes_sec:
                    raise TimeoutError(f"Pull averaged {rate:.0f} B/s over {slow_window}s (minimum {min_bytes_sec})")
                window_at, window_bytes = now, progress.downloaded
            if now - logged_at >= 30:
                logger.info(
                    "pull_progress %s bytes=%s idle_sec=%.0f",
                    ref, progress.downloaded, now - progress.last_progress,
                )
                logged_at = now
        if process.wait():
            raise RuntimeError(PullProgress.ansi.sub("", tail)[-1000:])
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        os.close(master)
        if slave is not None:
            os.close(slave)


def safe_error(exc):
    # Registry failures may contain temporary signed blob URLs.
    return re.sub(r"https?://[^\s\"']+", "<registry-url>", f"{type(exc).__name__}: {exc}")


def image_reference(source, registry):
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", source):
        raise ValueError(f"Image is not digest-pinned: {source}")
    if not registry:
        return source
    if not re.fullmatch(r"[\w.:-]+", registry):
        raise ValueError("Registry must be host[:port], without URL scheme or path")
    parts = source.split("/", 1)
    if len(parts) == 2 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        source = parts[1]
    return f"{registry}/{source}"


def inspect(row, registry, pull, min_free_gb, pull_timeout=3600, pull_options=None, pull_attempts=3):
    ref = image_reference(row["source_image"], registry)
    result = {"instance_id": row["instance_id"], "source_image": row["source_image"], "pull_image": ref,
              "base_commit": row["base_commit"], "workdir": row["workdir"], "resources": row.get("resources", {})}
    try:
        for attempt in range(3):
            try:
                process = subprocess.run(
                    ["docker", "manifest", "inspect", ref], capture_output=True, text=True, timeout=120
                )
            except subprocess.TimeoutExpired:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
                continue
            if process.returncode == 0:
                break
            time.sleep(2 ** attempt)
        if process.returncode:
            raise RuntimeError(process.stderr[:500])
        manifest = json.loads(process.stdout)
        if "manifests" in manifest:
            # Never substitute an arbitrary platform manifest for the bundle digest.
            raise ValueError("Multi-platform index: platform resolution needs explicit validation")
        layers = manifest.get("layers", [])
        if not layers:
            raise ValueError("Registry returned no image layers")
        result["layers"] = [{"digest": x["digest"], "size": x["size"]} for x in layers]
        result["compressed_bytes"] = sum(x["size"] for x in layers)
        result["status"] = "available"
        if pull:
            info = subprocess.check_output(["docker", "info", "--format", "{{.DockerRootDir}}"], text=True).strip()
            if shutil.disk_usage(info).free < min_free_gb * 1024**3:
                raise RuntimeError("Disk free space is below VIME_HARBOR_IMAGE_MIN_FREE_GB")
            for attempt in range(pull_attempts):
                if shutil.disk_usage(info).free < min_free_gb * 1024**3:
                    raise RuntimeError("Disk free space is below VIME_HARBOR_IMAGE_MIN_FREE_GB")
                try:
                    pull_image(ref, pull_timeout, **(pull_options or {}))
                    break
                except (TimeoutError, subprocess.TimeoutExpired, RuntimeError) as exc:
                    logger.warning(
                        "pull_retry %s attempt=%s/%s %s",
                        row["instance_id"], attempt + 1, pull_attempts, safe_error(exc),
                    )
                    if attempt + 1 == pull_attempts:
                        raise
                    time.sleep(min(30, 2 ** (attempt + 1)))
            detail = json.loads(subprocess.check_output(["docker", "image", "inspect", ref], text=True))[0]
            if detail["Architecture"] != "amd64" or detail["Os"] != "linux" or ref not in detail.get("RepoDigests", []):
                raise ValueError("Pulled image does not match the required linux/amd64 digest")
            result.update(
                status="pulled", image_id=detail["Id"], docker_reported_size_bytes=detail["Size"],
                pull_attempts=attempt + 1,
            )
    except Exception as exc:
        result.update(status="error", error=safe_error(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", action="store_true", help="Download images; default only inventories manifests")
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--retry-errors", action="store_true", help="Reuse successful entries in the existing report")
    args = parser.parse_args()
    bundle = Path(os.environ["VIME_SWEGYM_PREPARED_BUNDLE"]).resolve()
    report_path = Path(os.environ["VIME_HARBOR_IMAGE_REPORT"]).resolve()
    if report_path == bundle or bundle in report_path.parents:
        parser.error("Image report must be outside the immutable bundle")
    rows = [
        json.loads(line)
        for line in (bundle / "environment/agentenv_templates.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.instance_id:
        ids = set(args.instance_id)
        if ids - {r["instance_id"] for r in rows}:
            parser.error("Requested instance is absent from the bundle")
        rows = [row for row in rows if row["instance_id"] in ids]
    registry = os.environ.get("VIME_HARBOR_IMAGE_REGISTRY", "").rstrip("/")
    workers = int(os.environ.get("VIME_HARBOR_IMAGE_WORKERS", "2"))
    floor = int(os.environ.get("VIME_HARBOR_IMAGE_MIN_FREE_GB", "100"))
    pull_timeout = int(os.environ.get("VIME_HARBOR_IMAGE_PULL_TIMEOUT_SEC", "3600"))
    pull_attempts = int(os.environ.get("VIME_HARBOR_IMAGE_PULL_ATTEMPTS", "3"))
    pull_options = {
        "idle_timeout": int(os.environ.get("VIME_HARBOR_IMAGE_IDLE_TIMEOUT_SEC", "300")),
        "min_bytes_sec": int(os.environ.get("VIME_HARBOR_IMAGE_MIN_BYTES_SEC", "0")),
        "slow_window": int(os.environ.get("VIME_HARBOR_IMAGE_SLOW_WINDOW_SEC", "300")),
    }
    if (
        min(workers, floor, pull_timeout, pull_attempts, pull_options["slow_window"]) < 1
        or min(pull_options["idle_timeout"], pull_options["min_bytes_sec"]) < 0
    ):
        parser.error(
            "Workers, disk floor, timeouts/window and attempts must be positive; "
            "idle timeout and minimum rate may be zero"
        )
    results = []
    pending = rows
    if args.retry_errors:
        previous = json.loads(report_path.read_text())
        expected = {row["instance_id"]: image_reference(row["source_image"], registry) for row in rows}
        results = [
            item for item in previous["images"]
            if item["status"] in ({"pulled"} if args.pull else {"available", "pulled"})
            and expected.get(item["instance_id"]) == item["pull_image"]
        ]
        done = {item["instance_id"] for item in results}
        pending = [row for row in rows if row["instance_id"] not in done]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        unique = {layer["digest"]: layer["size"] for item in results for layer in item.get("layers", [])}
        report = {"bundle": str(bundle), "selected": len(rows), "completed": len(results),
                  "unique_compressed_layer_bytes": sum(unique.values()),
                  "note": "Compressed layer size is a lower bound, not an unpacked disk estimate.",
                  "images": sorted(results, key=lambda r: r["instance_id"])}
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        save()  # Remove stale errors immediately when resuming a report.
        futures = [
            pool.submit(inspect, row, registry, args.pull, floor, pull_timeout, pull_options, pull_attempts)
            for row in pending
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            save()
            logger.info("%s/%s %s %s", len(results), len(rows), result["instance_id"], result["status"])
    return int(any(result["status"] == "error" for result in results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
