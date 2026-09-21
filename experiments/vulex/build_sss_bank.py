"""Build the SSS bank from public records and the MSP/pretrain checkpoint."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.run_experiment import read_cwe_source, read_reposvul_jsonl, read_split_ids
from experiments.vulex.sss_embedding_fuzzy import fuzzy_pairwise_public_anchors, source_file
from experiments.vulex.stagedvulbert_encoder import (
    encode_records,
)


class PublicCandidate:
    """Expose the record ID and source file to the selector."""

    def __init__(self, record: Any) -> None:
        self.record = record
        self.anchor = record.anchor
        self.embedding_key = record.record_id
        self.source_file = source_file(record.anchor)


def body_lines(code: str) -> list[str]:
    """Return function-body lines after removing the signature, blank lines, and bare braces."""
    lines: list[str] = []
    in_body = False
    for raw in str(code).splitlines():
        if not in_body:
            if "{" in raw:
                in_body = True
                tail = raw.split("{", 1)[1].strip()
                if tail and tail != "}":
                    lines.append(tail)
            continue
        if raw.strip() and raw.strip() not in {"{", "}"}:
            lines.append(raw)
    return lines


def public_records_by_cwe(records: list[Any], public_ids: list[str], cwe_order: list[str]) -> dict[str, list[Any]]:
    """Group public records by CWE and deduplicate them by anchor."""
    records_by_id = {record.record_id: record for record in records}
    result: dict[str, list[Any]] = {}
    for cwe in cwe_order:
        by_anchor: dict[str, Any] = {}
        for record_id in public_ids:
            record = records_by_id[record_id]
            if record.cwe == cwe:
                by_anchor.setdefault(record.anchor, record)
        result[cwe] = [by_anchor[anchor] for anchor in sorted(by_anchor)]
    return result


def build_bank(
    *,
    cwe_order: list[str],
    public_records_by_cwe: dict[str, list[Any]],
    embeddings: dict[str, list[float]],
    repo_demand_source_files: list[str],
) -> dict[str, Any]:
    """Build a deterministic SSS bank with fixed filtering, prior weight, and top-k settings."""
    rows: list[dict[str, Any]] = []
    for probe_index, cwe in enumerate(cwe_order):
        source_records = public_records_by_cwe[cwe]
        candidates = [record for record in source_records if len(body_lines(record.target_function)) > 1]
        selected_anchors = fuzzy_pairwise_public_anchors(
            [PublicCandidate(record) for record in candidates],
            embeddings,
            5,
            prior_weight=0.5,
            repo_demand_source_files=repo_demand_source_files[:512],
        )
        records_by_anchor = {record.anchor: record for record in candidates}
        selected = [records_by_anchor[anchor] for anchor in selected_anchors]
        rows.append(
            {
                "cwe": cwe,
                "payload_anchors": [record.anchor for record in selected],
                "payload_source_record_ids": [record.record_id for record in selected],
            }
        )
    return {"rows": rows}


def read_json(path: pathlib.Path) -> Any:
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload: Any) -> None:
    """Write JSON in a stable format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build the VulEx SSS bank from public inputs.")
    parser.add_argument("--records", required=True, type=pathlib.Path)
    parser.add_argument("--visibility-split", required=True, type=pathlib.Path)
    parser.add_argument("--cwe-source", required=True, type=pathlib.Path)
    parser.add_argument("--repo-demand-source-files", required=True, type=pathlib.Path)
    parser.add_argument("--codebert-model", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--staged-source", required=True, type=pathlib.Path)
    parser.add_argument("--embedding-cache", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Generate MSP embeddings and the SSS bank."""
    args = parse_args(argv)
    records = read_reposvul_jsonl(args.records)
    cwe_order = read_cwe_source(args.cwe_source)[:30]
    public_by_cwe = public_records_by_cwe(
        records,
        read_split_ids(args.visibility_split)["public_record_ids"],
        cwe_order,
    )
    unique_records = {
        record.record_id: record
        for cwe in cwe_order
        for record in public_by_cwe[cwe]
        if len(body_lines(record.target_function)) > 1
    }
    embedding_records = [unique_records[record_id] for record_id in sorted(unique_records)]
    if args.embedding_cache.is_file():
        embeddings = read_json(args.embedding_cache)
    else:
        embeddings = encode_records(
            embedding_records,
            codebert_model=args.codebert_model,
            checkpoint=args.checkpoint,
            staged_source=args.staged_source,
        )
        write_json(args.embedding_cache, embeddings)
    repo_demand = read_json(args.repo_demand_source_files)
    bank = build_bank(
        cwe_order=cwe_order,
        public_records_by_cwe=public_by_cwe,
        embeddings=embeddings,
        repo_demand_source_files=repo_demand,
    )
    write_json(args.out, bank)
    return bank


if __name__ == "__main__":
    main()
