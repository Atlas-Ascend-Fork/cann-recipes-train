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

"""vime-side torchtitan overrides, in the shape of ``torchtitan/overrides/``.

An override replaces one node of the ``Trainer.Config`` tree before anything is built. The recipes
list the ones vime needs in ``_VIME_OVERRIDES``; nothing here applies on its own.

Each module is a stopgap for something the vendor plugin does not provide yet.
"""
