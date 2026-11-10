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

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from examples.cann_bench_evaluator.submission import (
    get_submission_fields,
    parse_submission,
    serialize_submission,
    validate_source_fields,
    validate_submission,
)

MARKDOWN_FINAL_MARKER_RE = re.compile(r"(?m)^[ \t]*kernel_src[ \t]*\r?\n[ \t]*```cpp[ \t]*$")
LEGACY_ASSIGNMENT_MARKER_RE = re.compile(r"(?m)^kernel_src\s*=")
MARKDOWN_LANGUAGES = {
    "kernel_src": "cpp",
    "launch_h_src": "cpp",
    "plugin_src": "cpp",
    "cmake_src": "cmake",
}


@dataclass(frozen=True)
class ParsedResponse:
    reasoning_text: str
    submission: str
    answer_char_start: int | None
    format_valid: bool
    format_info: dict[str, Any]


def serialize_model_answer(sources: dict[str, str], format_name: str = "4string") -> str:
    """Serialize sources using the Markdown contract used by the SFT data."""

    fields = validate_source_fields(format_name, sources)

    blocks = []
    for field in fields:
        language = MARKDOWN_LANGUAGES[field]
        source = sources[field]
        separator = "" if source.endswith(("\n", "\r")) else "\n"
        blocks.append(f"{field}\n```{language}\n{source}{separator}```")
    return "\n\n".join(blocks) + "\n"


def _parse_markdown_submission(submission: str, format_name: str) -> tuple[str | None, dict[str, Any]]:
    fields = get_submission_fields(format_name)
    cursor = 0
    sources: dict[str, str] = {}
    info: dict[str, Any] = {
        "submission_format": format_name,
        "model_answer_format": "md_code_block",
        "format_recovered": False,
        "format_valid_for_submit": False,
    }

    for field in fields:
        language = MARKDOWN_LANGUAGES[field]
        header = re.compile(rf"\A\s*{re.escape(field)}[ \t]*\r?\n[ \t]*```{re.escape(language)}[ \t]*\r?\n")
        match = header.match(submission[cursor:])
        if match is None:
            info["format_parse_error"] = f"missing_or_invalid_{field}_code_block"
            return None, info
        content_start = cursor + match.end()
        closing = re.search(r"(?m)^[ \t]*```[ \t]*(?:\r?\n|\Z)", submission[content_start:])
        if closing is None:
            info["format_parse_error"] = f"unclosed_{field}_code_block"
            return None, info
        content_end = content_start + closing.start()
        sources[field] = submission[content_start:content_end]
        cursor = content_start + closing.end()

    trailing = submission[cursor:]
    # Recover transport layout only. Never remove prose, another code block,
    # or modify whitespace inside any of the four source files.
    if trailing.strip() and not re.fullmatch(r"(?:\s*```[ \t]*(?:\r?\n|\Z))+\s*", trailing):
        info["format_parse_error"] = "trailing_content_after_cmake_src"
        return None, info
    if trailing.strip() or re.search(r"(?m)^[ \t]+```", submission):
        info.update(format_recovered=True, format_recovery_method="markdown_layout")

    try:
        canonical = serialize_submission(format_name, sources).strip()
    except ValueError as exc:
        info["format_parse_error"] = f"invalid_source:{exc}"
        return None, info
    valid, validation_info = validate_submission(canonical, format_name)
    if not valid:
        info.update(validation_info)
        info["format_parse_error"] = "canonical_submission_failed_validation"
        return None, info
    validation_info.update(info)
    validation_info["format_valid_for_submit"] = True
    validation_info["format_parse_error"] = ""
    return canonical, validation_info


def _assignment_markers(text: str, fields: tuple[str, ...]) -> tuple[list[re.Match[str]], str | None]:
    markers: list[re.Match[str]] = []
    for field in fields:
        matches = list(re.finditer(rf"(?m)^\s*{re.escape(field)}\s*=", text))
        if len(matches) != 1:
            return [], f"{field}:{'missing' if not matches else 'duplicate'}"
        markers.append(matches[0])
    if [match.start() for match in markers] != sorted(match.start() for match in markers):
        return [], "wrong_assignment_order"
    return markers, None


def _recover_string_value(rhs: str) -> tuple[str | None, str | None]:
    value = rhs.strip()
    prefixes = ("", "r", "R", "u", "U")
    for prefix in prefixes:
        if prefix and not value.startswith(prefix):
            continue
        literal = value[len(prefix):]
        for delimiter in ("'''", '"""', "'", '"'):
            if not literal.startswith(delimiter):
                continue
            if len(literal) < 2 * len(delimiter) or not literal.endswith(delimiter):
                return None, f"unclosed_{len(delimiter)}_quote"
            # The field boundary identifies the outer closing delimiter. Keep
            # everything inside verbatim so embedded docstrings and C/C++
            # string literals cannot break the transport envelope.
            return literal[len(delimiter): -len(delimiter)], None
    return None, "missing_string_delimiter"


def _recover_submission_by_field_boundaries(
    submission: str,
    format_name: str,
) -> tuple[str | None, dict[str, Any]]:
    fields = get_submission_fields(format_name)
    markers, marker_error = _assignment_markers(submission, fields)
    recovery_info: dict[str, Any] = {
        "format_recovered": False,
        "format_recovery_method": "field_boundary_string_envelope",
    }
    if marker_error is not None:
        recovery_info["format_recovery_error"] = marker_error
        return None, recovery_info

    sources: dict[str, str] = {}
    for index, (field, marker) in enumerate(zip(fields, markers, strict=True)):
        segment_end = markers[index + 1].start() if index + 1 < len(markers) else len(submission)
        rhs = submission[marker.end(): segment_end]
        value, value_error = _recover_string_value(rhs)
        if value_error is not None:
            recovery_info["format_recovery_error"] = f"{field}:{value_error}"
            return None, recovery_info
        sources[field] = value or ""

    try:
        recovered = serialize_submission(format_name, sources).strip()
    except ValueError as exc:
        recovery_info["format_recovery_error"] = f"serialization_failed:{exc}"
        return None, recovery_info
    recovery_info["format_recovered"] = True
    return recovered, recovery_info


def _merge_recovery_diagnostics(recovered_info, recovery_info, original_info):
    recovered_info.update(recovery_info)
    recovered_info["format_original_parse_error"] = original_info.get("format_parse_error", "")
    if "format_syntax_error" in original_info:
        recovered_info["format_original_syntax_error"] = original_info["format_syntax_error"]


def _canonicalize_legacy_submission(submission, format_name):
    sources, info = parse_submission(submission, format_name)
    valid = sources is not None
    if not valid:
        original_info = info
        recovered, recovery_info = _recover_submission_by_field_boundaries(submission, format_name)
        if recovered is not None:
            recovered_valid, recovered_info = validate_submission(recovered, format_name)
            if recovered_valid:
                _merge_recovery_diagnostics(recovered_info, recovery_info, original_info)
                submission = recovered
                valid = True
                info = recovered_info
            else:
                info.update(recovery_info)
                info["format_recovery_error"] = "canonical_submission_failed_validation"
        else:
            info.update(recovery_info)
    else:
        submission = serialize_submission(format_name, sources).strip()
        info["format_recovered"] = True
        info["format_recovery_method"] = "legacy_python_assignment"
    info["model_answer_format"] = "legacy_python_assignment"
    return submission, valid, info


def parse_response(response: str, format_name: str = "4string") -> ParsedResponse:
    get_submission_fields(format_name)
    # Native Qwen decoding may retain EOS immediately after the final fence.
    # Strip only terminal control markers; preserve source and character offsets.
    text = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)[ \t\r\n]*\Z", "", response or "")
    thinking_end = text.rfind("</think>")
    answer_search_start = thinking_end + len("</think>") if thinking_end >= 0 else 0

    # Never interpret a code draft inside <think> as the submitted answer.
    # Once an explicit reasoning boundary exists, both the Markdown parser and
    # legacy recovery are restricted to text after that boundary.
    matches = list(MARKDOWN_FINAL_MARKER_RE.finditer(text, answer_search_start))
    match = matches[-1] if matches else None
    if match is not None:
        model_answer = text[match.start():].strip()
        submission, info = _parse_markdown_submission(model_answer, format_name)
        return ParsedResponse(
            reasoning_text=text[: match.start()].rstrip(),
            submission=submission or "",
            answer_char_start=match.start(),
            format_valid=submission is not None,
            format_info=info,
        )

    # Backward-compatible recovery for rollouts produced before the prompt was
    # aligned with the Markdown SFT contract. These answers remain scoreable,
    # but are marked recovered so the existing format penalty applies.
    matches = list(LEGACY_ASSIGNMENT_MARKER_RE.finditer(text, answer_search_start))
    match = matches[-1] if matches else None
    if match is None:
        _, info = validate_submission("", format_name)
        info.update(
            {
                "model_answer_format": "unknown",
                "format_recovered": False,
                "format_parse_error": "missing_kernel_src_code_block",
            }
        )
        return ParsedResponse(
            reasoning_text=text[:answer_search_start].strip() if thinking_end >= 0 else text.strip(),
            submission="",
            answer_char_start=None,
            format_valid=False,
            format_info=info,
        )

    submission = text[match.start():].strip()
    submission, valid, info = _canonicalize_legacy_submission(submission, format_name)
    return ParsedResponse(
        reasoning_text=text[: match.start()].rstrip(),
        submission=submission,
        answer_char_start=match.start(),
        format_valid=valid,
        format_info=info,
    )
