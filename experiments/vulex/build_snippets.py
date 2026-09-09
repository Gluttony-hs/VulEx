"""Generate validated SSS suspect snippets from the generated SSS bank."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.llm_client import CachedLLMClient
from experiments.vulex.run_experiment import read_reposvul_jsonl
from experiments.vulex.sss import build_sss_prompt, sss_source_hash


def canonical(line: str) -> str:
    """Collapse whitespace in one line for conservative matching."""
    return " ".join(line.strip().split())


def validate_snippet(response: str, source: str) -> str:
    """Require one or two model-output lines and match each line back to the original function."""
    generated = [
        line.rstrip()
        for line in response.strip().splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    if not 1 <= len(generated) <= 2:
        raise ValueError("SSS snippet must contain one or two non-empty source lines")
    source_lines = source.splitlines()
    matched: list[str] = []
    start = 0
    for generated_line in generated:
        found = None
        for index in range(start, len(source_lines)):
            if generated_line == source_lines[index] or canonical(generated_line) == canonical(source_lines[index]):
                found = index
                break
        if found is None:
            raise ValueError(f"SSS snippet line is not present in source: {generated_line!r}")
        matched.append(source_lines[found].rstrip())
        start = found + 1
    return "\n".join(matched)


def read_json(path: pathlib.Path) -> Any:
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate SSS suspect snippets.")
    parser.add_argument("--records", required=True, type=pathlib.Path)
    parser.add_argument("--bank", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Generate, validate, and write snippets in bank order."""
    args = parse_args(argv)
    records = {record.record_id: record for record in read_reposvul_jsonl(args.records)}
    bank = read_json(args.bank)
    client = CachedLLMClient(args.cache_dir)
    rows: list[dict[str, Any]] = []
    for bank_row in bank["rows"]:
        cwe = str(bank_row["cwe"])
        for record_id in bank_row["payload_source_record_ids"]:
            record = records[str(record_id)]
            prompt = build_sss_prompt(record, cwe)
            source_hash = sss_source_hash(record, cwe)
            response = client.complete("sss", args.model, prompt, source_hash)
            rows.append(
                {
                    "record_id": record.record_id,
                    "anchor": record.anchor,
                    "cwe": cwe,
                    "source_hash": source_hash,
                    "snippet": validate_snippet(response, record.target_function),
                }
            )
    payload = {
        "model": args.model,
        "bank_selector": bank["selector"],
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
