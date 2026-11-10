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

"""Safe four-file validation and readable evaluation TXT serialization."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

SUBMISSION_FORMAT_FIELDS = {"4string": ("kernel_src", "launch_h_src", "plugin_src", "cmake_src")}


def get_submission_fields(format_name: str) -> tuple[str, ...]:
    try:
        return SUBMISSION_FORMAT_FIELDS[format_name]
    except KeyError as exc:
        supported = ", ".join(SUBMISSION_FORMAT_FIELDS)
        raise ValueError(f"Unsupported submission format {format_name!r}; expected one of: {supported}") from exc


def _syntax_error_details(exc):
    source_line = (exc.text or "").rstrip("\r\n")
    column = int(exc.offset or 0)
    excerpt_start = max(0, column - 81)
    excerpt_end = min(len(source_line), max(column + 79, excerpt_start + 160))
    return {
        "message": exc.msg,
        "line": exc.lineno,
        "column": exc.offset,
        "end_line": exc.end_lineno,
        "end_column": exc.end_offset,
        "excerpt": source_line[excerpt_start:excerpt_end],
        "excerpt_start_column": excerpt_start + 1,
    }


def _collect_assignments(module, fields):
    assignments: dict[str, str] = {}
    ordered_names: list[str] = []
    unsupported_nodes: list[str] = []
    for node in module.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            unsupported_nodes.append(type(node).__name__)
            continue
        name = node.targets[0].id
        if name not in fields:
            unsupported_nodes.append(f"{name}:unexpected_assignment")
        elif name in assignments:
            unsupported_nodes.append(f"{name}:duplicate_assignment")
        elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            assignments[name] = node.value.value
            ordered_names.append(name)
        else:
            unsupported_nodes.append(f"{name}:non_string_value")
    return assignments, ordered_names, unsupported_nodes


def parse_submission(text: str, format_name: str = "4string") -> tuple[dict[str, str] | None, dict[str, Any]]:
    """Parse and validate once; never execute the source assignments."""
    fields = get_submission_fields(format_name)
    submission = (text or "").strip()
    forbidden = [marker for marker in ("<tool_call", "```") if marker in submission]
    info: dict[str, Any] = {
        "submission_format": format_name,
        "format_forbidden_markers": forbidden,
        "format_found_vars": [],
        "format_missing_vars": list(fields),
        "format_parse_error": "",
        "format_valid_for_submit": False,
    }
    if not submission:
        info["format_parse_error"] = "empty_submission"
        return None, info
    if forbidden:
        info["format_parse_error"] = "forbidden_marker"
        return None, info
    first_field = fields[0]
    if not submission.startswith(first_field):
        info["format_parse_error"] = f"does_not_start_with_{first_field}"
        return None, info

    try:
        module = ast.parse(submission, mode="exec")
    except SyntaxError as exc:
        info["format_parse_error"] = f"syntax_error:{exc.msg}"
        info["format_syntax_error"] = _syntax_error_details(exc)
        return None, info

    assignments, ordered_names, unsupported_nodes = _collect_assignments(module, fields)
    found_vars = [name for name in fields if name in assignments]
    missing_vars = [name for name in fields if name not in assignments]
    info.update(
        {
            "format_found_vars": found_vars,
            "format_missing_vars": missing_vars,
            "format_assignment_order": ordered_names,
            "format_unsupported_nodes": unsupported_nodes,
        }
    )
    if unsupported_nodes:
        info["format_parse_error"] = "unsupported_top_level_node"
        return None, info
    if found_vars != list(fields):
        info["format_parse_error"] = "missing_required_assignment"
        return None, info
    if ordered_names != list(fields):
        info["format_parse_error"] = "wrong_assignment_order"
        return None, info

    info["format_valid_for_submit"] = True
    return assignments, info


def validate_submission(text: str, format_name: str = "4string") -> tuple[bool, dict[str, Any]]:
    sources, info = parse_submission(text, format_name)
    return sources is not None, info


WORKSPACE_FILES = {
    "4string": {
        "kernel_src": "kernel.cpp",
        "launch_h_src": "launch.h",
        "plugin_src": "plugin.cpp",
        "cmake_src": "CMakeLists.txt",
    },
}


def get_workspace_files(format_name: str) -> dict[str, str]:
    get_submission_fields(format_name)
    return dict(WORKSPACE_FILES[format_name])


def _multiline_literal(source: str) -> str:
    # Keep real LF line breaks, but escape Python-literal syntax. In particular,
    # literal C++ \\n and backslash-newline continuations must not be interpreted.
    escaped = []
    for char in source:
        if char in ("\\", "'"):
            escaped.append("\\" + char)
        elif char == "\n" or char.isprintable():
            escaped.append(char)
        else:
            escaped.append(repr(char)[1:-1])
    return "'''" + "".join(escaped) + "'''"


def validate_source_fields(format_name: str, sources: Mapping[str, str]) -> tuple[str, ...]:
    """Check the exact source-field set shared by Markdown and assignment serializers."""
    fields = get_submission_fields(format_name)
    missing = [field for field in fields if field not in sources]
    extra = sorted(set(sources) - set(fields))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected fields: {', '.join(extra)}")
        raise ValueError("; ".join(details))
    return fields


def serialize_submission(format_name: str, sources: Mapping[str, str], *, multiline: bool = True) -> str:
    """Serialize source literals without interpreting C++ escape sequences."""
    fields = validate_source_fields(format_name, sources)
    for field in fields:
        if not isinstance(sources[field], str):
            raise ValueError(f"{field} must be a string")
        if any(marker in sources[field] for marker in ("<tool_call", "```")):
            raise ValueError(f"{field} contains a forbidden marker")

    literal = _multiline_literal if multiline else repr
    separator = "\n\n" if multiline else "\n"
    # The exact field set and escaped string literals guarantee the envelope;
    # untrusted TXT is still parsed at evaluate_submission's public boundary.
    return separator.join(f"{field} = {literal(sources[field])}" for field in fields) + "\n"
