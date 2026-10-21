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

"""Versioned AgentENV template mapping for SWE rollouts.

Dataset rows describe a source repository environment, while AgentENV creates
sandboxes from a prebuilt ``template_id``.  This module keeps that translation
explicit and validates the immutable provenance before a rollout starts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class TemplateMappingError(ValueError):
    """The mapping file or a dataset-to-template lookup is inconsistent."""


def _required_string(record: dict[str, Any], key: str, *, location: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TemplateMappingError(f"{location}: {key} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class TemplateMappingEntry:
    instance_id: str
    source_image: str
    source_image_tag: str | None
    image_digest: str
    template_id: str
    template_alias: str | None
    repo: str
    workdir: str
    base_commit: str
    raw: dict[str, Any]

    @classmethod
    def from_record(cls, record: dict[str, Any], *, location: str) -> TemplateMappingEntry:
        if record.get("schema_version") != 1:
            raise TemplateMappingError(f"{location}: schema_version must be 1")

        image_digest = _required_string(record, "image_digest", location=location)
        if not _DIGEST_RE.fullmatch(image_digest):
            raise TemplateMappingError(f"{location}: invalid image_digest {image_digest!r}")

        source_image = _required_string(record, "source_image", location=location)
        if not source_image.endswith(f"@{image_digest}"):
            raise TemplateMappingError(
                f"{location}: source_image must end with the immutable digest @{image_digest}"
            )

        template_id = _required_string(record, "template_id", location=location)
        try:
            UUID(template_id)
        except ValueError as exc:
            raise TemplateMappingError(f"{location}: invalid template_id {template_id!r}") from exc

        base_commit = _required_string(record, "base_commit", location=location).lower()
        if not _COMMIT_RE.fullmatch(base_commit):
            raise TemplateMappingError(f"{location}: invalid base_commit {base_commit!r}")

        source_image_tag = record.get("source_image_tag")
        if source_image_tag is not None and (not isinstance(source_image_tag, str) or not source_image_tag.strip()):
            raise TemplateMappingError(f"{location}: source_image_tag must be null or a non-empty string")

        template_alias = record.get("template_alias")
        if template_alias is not None and (not isinstance(template_alias, str) or not template_alias.strip()):
            raise TemplateMappingError(f"{location}: template_alias must be null or a non-empty string")

        return cls(
            instance_id=_required_string(record, "instance_id", location=location),
            source_image=source_image,
            source_image_tag=source_image_tag.strip() if isinstance(source_image_tag, str) else None,
            image_digest=image_digest,
            template_id=template_id,
            template_alias=template_alias.strip() if isinstance(template_alias, str) else None,
            repo=_required_string(record, "repo", location=location),
            workdir=_required_string(record, "workdir", location=location),
            base_commit=base_commit,
            raw=record,
        )


class TemplateMapping:
    def __init__(self, entries: list[TemplateMappingEntry]) -> None:
        self._by_instance: dict[str, TemplateMappingEntry] = {}
        for entry in entries:
            if entry.instance_id in self._by_instance:
                raise TemplateMappingError(f"duplicate instance_id in template mapping: {entry.instance_id}")
            self._by_instance[entry.instance_id] = entry

    def resolve(
        self,
        instance_id: str,
        *,
        source_image: str | None = None,
        base_commit: str | None = None,
        workdir: str | None = None,
        repo: str | None = None,
    ) -> TemplateMappingEntry:
        try:
            entry = self._by_instance[instance_id]
        except KeyError as exc:
            raise TemplateMappingError(f"no AgentENV template mapping for instance_id={instance_id!r}") from exc

        if source_image:
            accepted_images = {entry.source_image}
            if entry.source_image_tag:
                accepted_images.add(entry.source_image_tag)
            if source_image not in accepted_images:
                raise TemplateMappingError(
                    f"instance_id={instance_id!r}: dataset image {source_image!r} does not match "
                    f"mapped image {entry.source_image!r}"
                )
        if base_commit and base_commit.lower() != entry.base_commit:
            raise TemplateMappingError(
                f"instance_id={instance_id!r}: dataset base_commit {base_commit!r} does not match "
                f"mapped base_commit {entry.base_commit!r}"
            )
        if workdir and workdir != entry.workdir:
            raise TemplateMappingError(
                f"instance_id={instance_id!r}: dataset workdir {workdir!r} does not match "
                f"mapped workdir {entry.workdir!r}"
            )
        if repo and repo != entry.repo:
            raise TemplateMappingError(
                f"instance_id={instance_id!r}: dataset repo {repo!r} does not match mapped repo {entry.repo!r}"
            )
        return entry


@lru_cache(maxsize=None)
def load_template_mapping(path: str) -> TemplateMapping:
    mapping_path = Path(path).expanduser()
    if not mapping_path.is_file():
        raise TemplateMappingError(f"AgentENV template mapping does not exist: {mapping_path}")

    entries: list[TemplateMappingEntry] = []
    for line_number, line in enumerate(mapping_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        location = f"{mapping_path}:{line_number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TemplateMappingError(f"{location}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise TemplateMappingError(f"{location}: each JSONL row must be an object")
        entries.append(TemplateMappingEntry.from_record(record, location=location))

    if not entries:
        raise TemplateMappingError(f"AgentENV template mapping is empty: {mapping_path}")
    return TemplateMapping(entries)


def apply_template_mapping(md: dict[str, Any], mapping: TemplateMapping) -> dict[str, Any]:
    """Return SWE metadata whose sandbox selector is an AgentENV template ID."""
    instance_id = str(md.get("instance_id") or "")
    source_image = str(md.get("image") or "").strip() or None
    grading = md.get("grading") or {}
    sweb_instance = grading.get("sweb_instance") or {}
    base_commit = str(md.get("base_commit") or sweb_instance.get("base_commit") or "").strip() or None
    workdir = str(md.get("workdir") or "").strip() or None
    repo = str(md.get("repo") or sweb_instance.get("repo") or "").strip() or None

    entry = mapping.resolve(
        instance_id,
        source_image=source_image,
        base_commit=base_commit,
        workdir=workdir,
        repo=repo,
    )
    resolved = dict(md)
    resolved["source_image"] = source_image or entry.source_image
    resolved["image"] = entry.template_id
    resolved["template_id"] = entry.template_id
    resolved["template_alias"] = entry.template_alias
    resolved["image_digest"] = entry.image_digest
    resolved["base_commit"] = entry.base_commit
    resolved["repo"] = entry.repo
    return resolved
