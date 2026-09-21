"""Construct normalized Linux records from ReposVul data."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import asdict
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.c_function_extract import (
    FunctionBody,
    c_function_bodies,
    enclosing_function_body_from_code_before_patch,
)
from experiments.vulex.schemas import LINUX_PROJECT, ReposVulRecord

CWE_RE = re.compile(r"(?:CWE[-\s]*)?(\d+)", re.IGNORECASE)


def normalize_cwe(value: Any) -> str:
    """Normalize a ReposVul CWE value to `CWE-<number>`."""
    if isinstance(value, list):
        value = value[0]
    match = CWE_RE.search(str(value))
    if not match:
        raise ValueError(f"cannot normalize CWE value {value!r}")
    return f"CWE-{int(match.group(1))}"


def detail_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Return detail entries from one ReposVul record."""
    details = item.get("details")
    if isinstance(details, list):
        return [detail for detail in details if isinstance(detail, dict)]
    if isinstance(details, dict):
        return [details]
    return []


def function_before_text(detail: dict[str, Any]) -> str:
    """Extract candidate target-function text from function_before."""
    function_before = detail.get("function_before") or {}
    if isinstance(function_before, list):
        targets = [row for row in function_before if isinstance(row, dict) and row.get("target") == 1]
        rows = targets or [row for row in function_before if isinstance(row, dict)]
        return str(rows[0].get("function") or "").strip() if rows else ""
    if isinstance(function_before, dict):
        return str(function_before.get("function") or "").strip()
    return ""


def target_function(detail: dict[str, Any]) -> tuple[FunctionBody, str] | None:
    """Extract the complete target function from function_before or patch context."""
    function_text = function_before_text(detail)
    if function_text:
        bodies = c_function_bodies(function_text)
        if bodies:
            return bodies[0], "function_before"
    code_before = str(detail.get("code_before") or "")
    patch = str(detail.get("patch") or "")
    if code_before and patch:
        body = enclosing_function_body_from_code_before_patch(code_before, patch)
        if body:
            return body, "code_before_patch"
    return None


def normalize_reposvul_records(items: list[dict[str, Any]]) -> list[ReposVulRecord]:
    """Filter and expand ReposVul entries into Linux function-level records."""
    records: list[ReposVulRecord] = []
    for item in items:
        if item.get("project") != LINUX_PROJECT:
            continue
        for detail in detail_rows(item):
            if not all((item.get("cve_id"), item.get("cwe_id"), item.get("commit_id"))):
                continue
            file_path = str(detail.get("file_name") or detail.get("file_path") or "").strip()
            if not file_path:
                continue
            try:
                cwe = normalize_cwe(item["cwe_id"])
            except ValueError:
                continue
            result = target_function(detail)
            if result is None:
                continue
            function, source = result
            records.append(
                ReposVulRecord(
                    record_id=f"linux-{len(records)}",
                    project=LINUX_PROJECT,
                    cve_id=str(item["cve_id"]),
                    cwe=cwe,
                    commit_id=str(item["commit_id"]),
                    file_path=file_path,
                    function_name=function.name,
                    anchor=f"{file_path}::{function.name}",
                    commit_message=str(item.get("commit_message") or ""),
                    publish_date=str(item.get("publish_date") or ""),
                    target_function=function.body,
                    target_function_source=source,
                )
            )
    return records


def write_jsonl(path: pathlib.Path, records: list[Any]) -> None:
    """Write dataclass records as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")


def read_json_items(raw_path: pathlib.Path) -> list[dict[str, Any]]:
    """Read ReposVul JSON or JSONL files and return flattened entries."""
    paths = [raw_path] if raw_path.is_file() else sorted(raw_path.rglob("*.json")) + sorted(raw_path.rglob("*.jsonl"))
    items: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            items.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            items.extend(payload)
        elif isinstance(payload.get("data"), list):
            items.extend(payload["data"])
        else:
            items.append(payload)
    return items


def main(argv: list[str] | None = None) -> None:
    """Build Linux function-level records from raw ReposVul data."""
    parser = argparse.ArgumentParser(description="Build normalized Linux records from ReposVul data.")
    parser.add_argument("--raw", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    write_jsonl(args.out, normalize_reposvul_records(read_json_items(args.raw)))


if __name__ == "__main__":
    main()
