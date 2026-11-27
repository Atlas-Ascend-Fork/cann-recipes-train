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

"""Exercise each configured CPU host without an agent, model or training job."""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import shlex
import tempfile

from vime.agent.harbor_sandbox import HarborSandbox

logger = logging.getLogger(__name__)


async def main(image, instance_id=None):
    entry = None
    if instance_id:
        mapping = Path(os.environ["VIME_SWEGYM_PREPARED_BUNDLE"]) / "environment/agentenv_templates.jsonl"
        entry = next(json.loads(line) for line in mapping.read_text().splitlines()
                     if line.strip() and json.loads(line)["instance_id"] == instance_id)
        if entry["source_image"] != image:
            raise ValueError("The requested image must match the bundle's source_image exactly")
    hosts = json.loads(os.environ["VIME_HARBOR_HOSTS"])
    original = os.environ["VIME_HARBOR_HOSTS"]
    try:
        for host in hosts:
            os.environ["VIME_HARBOR_HOSTS"] = json.dumps([host])
            async with HarborSandbox(image) as sandbox:
                logger.info("START %s", sandbox.sandbox_id)
                if entry:
                    workdir = shlex.quote(entry["workdir"])
                    head = await sandbox.exec(f"cd {workdir} && git rev-parse HEAD", check=True)
                    assert head[1].strip() == entry["base_commit"], head
                    clean = await sandbox.exec(
                        f"cd {workdir} && git status --porcelain --untracked-files=no", check=True
                    )
                    assert not clean[1].strip(), clean
                    prefix = os.environ.get("VIME_AGENT_TASK_CONDA_PREFIX")
                    if prefix:
                        python = shlex.quote(prefix + "/bin/python")
                        version = (await sandbox.exec(f"{python} --version", check=True))[1].strip()
                        pytest = (await sandbox.exec(f"{python} -m pytest --version", check=True))[1].strip()
                        validation = entry.get("resource_validation") or entry.get("validation", {})
                        assert version == validation.get("python_version", version), version
                        assert pytest == validation.get("pytest_version", pytest), pytest
                        logger.info("GUEST %s %s", version, pytest)
                    logger.info("BUNDLE IMAGE PASSED %s %s", instance_id, entry["base_commit"])
                assert (await sandbox.exec("printf '%s' \"$VIME_SMOKE\"", env={"VIME_SMOKE": "ok"}))[1] == "ok"
                assert (await sandbox.exec("exit 7"))[0] == 7
                try:
                    await sandbox.exec("exit 7", check=True)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("check=True did not raise")
                try:
                    result = await sandbox.exec("sleep 5", timeout=1)
                    assert result[0] != 0, "Command timeout was ignored"
                except RuntimeError:
                    pass  # Harbor reports command timeouts as operation errors.
                assert (await sandbox.exec("printf ready"))[1] == "ready"
                await sandbox.exec("id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent", check=True)
                await sandbox.write_file("/home/agent/space name.txt", "hello\n", user="agent")
                assert await sandbox.read_file("/home/agent/space name.txt", user="agent") == "hello\n"
                assert (await sandbox.exec("stat -c %U '/home/agent/space name.txt'"))[1].strip() == "agent"
                with tempfile.TemporaryDirectory() as tmp:
                    artifact = Path(tmp) / "artifact.bin"
                    artifact.write_bytes(bytes(range(256)) * 8193)
                    await sandbox.write_file("/tmp/artifact.bin", artifact)
                    assert int((await sandbox.exec("wc -c < /tmp/artifact.bin"))[1]) == artifact.stat().st_size
                await asyncio.gather(sandbox.write_file("/tmp/a", "a"), sandbox.write_file("/tmp/b", "b"))
                assert await sandbox.read_file("/tmp/a") == "a"
                assert await sandbox.read_file("/tmp/b") == "b"
                assert await sandbox.read_file("/no-such-smoke-file") == ""
                logger.info("CONTRACT PASSED %s", sandbox.sandbox_id)
            logger.info("CLOSED %s", sandbox.sandbox_id)
    finally:
        os.environ["VIME_HARBOR_HOSTS"] = original


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Digest-pinned Bash-capable fixture or task image")
    parser.add_argument("--instance-id", help="Also check the existing bundle's repository and guest-runtime identity")
    args = parser.parse_args()
    asyncio.run(main(args.image, args.instance_id))
