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

"""Select the sandbox used for SWE-Gym rollouts and fresh-image grading."""

from __future__ import annotations

from vime.agent.config import sandbox_backend
from vime.agent.sandbox import E2BSandbox, Sandbox


def create_sandbox(image: str, *, trace=None, observation_role: str = "agent", **kwargs) -> Sandbox:
    """Instantiate the provider this run selected; the image is provider-specific.

    Provider dependencies are imported lazily, so a run only loads the SDK of
    the backend it uses.
    """
    if sandbox_backend() == "agentenv":
        return E2BSandbox(image, trace=trace, observation_role=observation_role, **kwargs)
    from vime.agent.harbor_sandbox import HarborSandbox

    return HarborSandbox(image, **kwargs)
