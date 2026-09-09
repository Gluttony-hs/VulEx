import argparse
import json
import pathlib
import re
from dataclasses import asdict
from typing import Any

from experiments.vulex.c_function_extract import (
    FunctionBody,
    c_function_bodies,
    enclosing_function_body_from_code_before_patch,
)
from experiments.vulex.schemas import LINUX_PROJECT, ReposVulRecord

CWE_RE = re.compile(r"(?:CWE[-\s]*)?(\d+)", re.IGNORECASE)


def normalize_cwe(value: Any) -> str:
    """Normalize CWE values from ReposVul to `CWE-<number>`."""
    if isinstance(value, list):
        if not value:
            raise ValueError("cannot normalize empty CWE list")
        value = value[0]
    match = CWE_RE.search(str(value or ""))
    if not match:
        raise ValueError(f"cannot normalize CWE value {value!r}")
    return f"CWE-{int(match.group(1))}"


def _details(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract usable detail entries from the `details` field."""
    details = item.get("details")
    if isinstance(details, list):
        return [detail for detail in details if isinstance(detail, dict)]
    if isinstance(details, dict):
        return [details]
    return []


def _function_text_from_function_before(detail: dict[str, Any]) -> str:
    """Extract candidate target-function text from ReposVul function_before."""
    function_before = detail.get("function_before") or {}
    if isinstance(function_before, list):
        targets = [item for item in function_before if isinstance(item, dict) and item.get("target") == 1]
        if targets:
            return str(targets[0].get("function") or "").strip()
        for item in function_before:
            if isinstance(item, dict):
                return str(item.get("function") or "").strip()
    if isinstance(function_before, dict):
        return str(function_before.get("function") or "").strip()
    return ""


def _target_function_from_function_before(detail: dict[str, Any]) -> FunctionBody | None:
    """Accept only function_before text parseable as a complete C function definition."""
    function_text = _function_text_from_function_before(detail)
    if function_text:
        bodies = c_function_bodies(function_text)
        if bodies:
            return bodies[0]
    return None


def _target_function(detail: dict[str, Any]) -> tuple[FunctionBody, str] | None:
    """Resolve the complete target function in function_before -> code_before+patch order."""
    from_function_before = _target_function_from_function_before(detail)
    if from_function_before:
        return from_function_before, "function_before"
    code_before = str(detail.get("code_before") or "")
    patch = str(detail.get("patch") or "")
    if not code_before or not patch:
        return None
    from_code = enclosing_function_body_from_code_before_patch(code_before, patch)
    if from_code:
        return from_code, "code_before_patch"
    return None


def _required_present(item: dict[str, Any], detail: dict[str, Any]) -> bool:
    """Check whether a detail has the minimum fields needed for a formal Linux record."""
    if not item.get("cve_id"):
        return False
    if not item.get("cwe_id"):
        return False
    if not item.get("commit_id"):
        return False
    if not item.get("project"):
        return False
    if not (detail.get("file_name") or detail.get("file_path")):
        return False
    return True


def _line_stats(line_counts: list[int]) -> dict[str, float | int]:
    """Run this module's helper logic."""
    if not line_counts:
        return {"count": 0, "min": 0, "max": 0, "avg": 0.0}
    return {
        "count": len(line_counts),
        "min": min(line_counts),
        "max": max(line_counts),
        "avg": sum(line_counts) / len(line_counts),
    }


def normalize_reposvul_records(items: list[dict[str, Any]]) -> tuple[list[ReposVulRecord], dict[str, Any]]:
    """Filter, expand, and normalize official ReposVul entries into Linux records."""
    records: list[ReposVulRecord] = []
    summary: dict[str, Any] = {
        "total": len(items),
        "project_matched": 0,
        "details_seen": 0,
        "missing_required": 0,
        "usable": 0,
        "cwe_distribution": {},
        "anchor_distribution": {},
        "target_function_sources": {
            "function_before": 0,
            "code_before_patch": 0,
        },
        "skipped_no_target_function": 0,
        "target_function_lines": {"count": 0, "min": 0, "max": 0, "avg": 0.0},
        "missing_code_before": 0,
    }
    target_function_line_counts: list[int] = []
    for item in items:
        if item.get("project") != LINUX_PROJECT:
            continue
        summary["project_matched"] += 1
        for detail in _details(item):
            summary["details_seen"] += 1
            if not detail.get("code_before"):
                summary["missing_code_before"] += 1
            if not _required_present(item, detail):
                summary["missing_required"] += 1
                continue
            try:
                cwe = normalize_cwe(item["cwe_id"])
            except ValueError:
                summary["missing_required"] += 1
                continue
            file_path = str(detail.get("file_name") or detail.get("file_path")).strip()
            target_result = _target_function(detail)
            if target_result is None:
                summary["skipped_no_target_function"] += 1
                continue
            target_function, target_function_source = target_result
            function_name = target_function.name
            anchor = f"{file_path}::{function_name}"
            record = ReposVulRecord(
                record_id=f"linux-{len(records)}",
                project=LINUX_PROJECT,
                cve_id=str(item["cve_id"]),
                cwe=cwe,
                commit_id=str(item["commit_id"]),
                file_path=file_path,
                function_name=function_name,
                anchor=anchor,
                commit_message=str(item.get("commit_message") or ""),
                publish_date=str(item.get("publish_date") or ""),
                target_function=target_function.body,
                target_function_source=target_function_source,
            )
            records.append(record)
            summary["cwe_distribution"][cwe] = summary["cwe_distribution"].get(cwe, 0) + 1
            summary["anchor_distribution"][anchor] = summary["anchor_distribution"].get(anchor, 0) + 1
            summary["target_function_sources"][target_function_source] += 1
            target_function_line_counts.append(len(target_function.body.splitlines()))
        if not _details(item):
            summary["missing_required"] += 1
    summary["usable"] = len(records)
    summary["target_function_lines"] = _line_stats(target_function_line_counts)
    return records, summary


def write_jsonl(path: pathlib.Path, records: list[Any]) -> None:
    """Write a dataclass list as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False) + "\n")


def _read_json_items(raw_path: pathlib.Path) -> list[dict[str, Any]]:
    """Recursively read raw ReposVul JSON/JSONL and return a flat list of dictionaries."""
    if raw_path.is_file():
        candidates = [raw_path]
    else:
        candidates = sorted(raw_path.rglob("*.json")) + sorted(raw_path.rglob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no json/jsonl ReposVul files found under {raw_path}")
    items: list[dict[str, Any]] = []
    for path in candidates:
        if path.suffix.lower() == ".jsonl":
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            items.extend(x for x in obj if isinstance(x, dict))
        elif isinstance(obj, dict):
            data = obj.get("data")
            if isinstance(data, list):
                items.extend(x for x in data if isinstance(x, dict))
            else:
                items.append(obj)
    return items


def main() -> None:
    """CLI entry point that normalizes raw official ReposVul data into Linux records."""
    parser = argparse.ArgumentParser(description="Normalize official ReposVul C data to torvalds/linux records.")
    parser.add_argument("--raw", required=True, type=pathlib.Path)
    parser.add_argument("--project", default=LINUX_PROJECT)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--summary", required=True, type=pathlib.Path)
    parser.add_argument("--min-records", type=int, default=200)
    args = parser.parse_args()
    if args.project != LINUX_PROJECT:
        raise ValueError(f"only {LINUX_PROJECT} is supported")
    records, summary = normalize_reposvul_records(_read_json_items(args.raw))
    if len(records) < args.min_records:
        raise RuntimeError(f"usable linux records {len(records)} < required {args.min_records}")
    write_jsonl(args.out, records)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
