"""Build the SSS bank from public records and the MSP/pretrain checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.run_experiment import read_cwe_source, read_reposvul_jsonl, read_split_ids
from experiments.vulex.sss_embedding_fuzzy import fuzzy_pairwise_public_anchors, source_file
from experiments.vulex.stagedvulbert_encoder import (
    MSP_LINEAGE,
    encode_records,
    record_fingerprint,
)


SELECTOR = "stagedvulbert_cls_selector"
SELECTOR_POLICY = "pairwise_fuzzy_repository_coverage"
FILTER_POLICY = "exclude_short_public_functions"


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
    missing = [record_id for record_id in public_ids if record_id not in records_by_id]
    if missing:
        raise ValueError(f"public record ids missing from records: {missing[:5]}")
    result: dict[str, list[Any]] = {}
    for cwe in cwe_order:
        by_anchor: dict[str, Any] = {}
        for record_id in public_ids:
            record = records_by_id[record_id]
            if record.cwe == cwe:
                by_anchor.setdefault(record.anchor, record)
        if not by_anchor:
            raise ValueError(f"CWE has no public records: {cwe}")
        result[cwe] = [by_anchor[anchor] for anchor in sorted(by_anchor)]
    return result


def build_bank(
    *,
    cwe_order: list[str],
    public_records_by_cwe: dict[str, list[Any]],
    embeddings: dict[str, list[float]],
    embedding_metadata: dict[str, Any],
    repo_demand_source_files: list[str],
) -> dict[str, Any]:
    """Build a deterministic SSS bank with fixed filtering, prior weight, and top-k settings."""
    rows: list[dict[str, Any]] = []
    for probe_index, cwe in enumerate(cwe_order):
        source_records = public_records_by_cwe[cwe]
        candidates = [record for record in source_records if len(body_lines(record.target_function)) > 1]
        if not candidates:
            raise ValueError(f"no SSS candidates remain after filtering: {cwe}")
        missing = [record.record_id for record in candidates if record.record_id not in embeddings]
        if missing:
            raise ValueError(f"missing MSP embeddings for {cwe}: {missing[:5]}")
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
                "probe_index": probe_index,
                "cwe": cwe,
                "selector": SELECTOR,
                "selector_policy": SELECTOR_POLICY,
                "pooling": "fine_line_cls",
                "prior_weight": 0.5,
                "repo_demand_count": 512,
                "content_window": 512,
                "chunk_step": 512,
                "topk": 5,
                "payload_anchor_count": len(selected),
                "payload_anchors": [record.anchor for record in selected],
                "selected_anchors": [record.anchor for record in selected],
                "payload_source_record_ids": [record.record_id for record in selected],
                "payload_anchor_source_files": [record.file_path for record in selected],
                "public_record_count": len(source_records),
                "public_unique_anchor_count": len(source_records),
                "filtered_candidate_count": len(candidates),
                "filtered_out_count": len(source_records) - len(candidates),
            }
        )
    return {
        "selector": SELECTOR,
        "selector_policy": SELECTOR_POLICY,
        "lineage": MSP_LINEAGE,
        "topk": 5,
        "prior_weight": 0.5,
        "repo_demand_count": 512,
        "content_window": 512,
        "chunk_step": 512,
        "pooling": "fine_line_cls",
        "filter_policy": FILTER_POLICY,
        "embedding_metadata": embedding_metadata,
        "rows": rows,
    }


def read_json(path: pathlib.Path) -> Any:
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload: Any) -> None:
    """Write JSON in a stable format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def valid_embedding_cache(path: pathlib.Path, records: list[Any]) -> tuple[dict[str, list[float]], dict[str, Any]] | None:
    """Read an embedding cache matching the current records and MSP checkpoint."""
    if not path.is_file():
        return None
    data = read_json(path)
    metadata = data.get("metadata", {})
    expected_fingerprints = {record.record_id: record_fingerprint(record) for record in records}
    if (
        metadata.get("lineage") != MSP_LINEAGE
        or metadata.get("record_fingerprints") != expected_fingerprints
    ):
        return None
    embeddings = data.get("embeddings")
    dimension = metadata.get("embedding_dimension")
    if not isinstance(embeddings, dict) or not isinstance(dimension, int) or dimension <= 0:
        return None
    if set(embeddings) != set(expected_fingerprints):
        return None
    for vector in embeddings.values():
        if len(vector) != dimension or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
            return None
    return {key: [float(value) for value in vector] for key, vector in embeddings.items()}, metadata


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
    cached = valid_embedding_cache(args.embedding_cache, embedding_records)
    if cached is None:
        embeddings, checkpoint_load = encode_records(
            embedding_records,
            codebert_model=args.codebert_model,
            checkpoint=args.checkpoint,
            staged_source=args.staged_source,
        )
        dimensions = {len(vector) for vector in embeddings.values()}
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent embedding dimensions: {sorted(dimensions)}")
        metadata = {
            **checkpoint_load,
            "encoder": "msp-pretrain-fine-line-cls-extraction-v1",
            "record_count": len(embedding_records),
            "record_fingerprints": {
                record.record_id: record_fingerprint(record) for record in embedding_records
            },
            "embedding_dimension": dimensions.pop(),
            "batch_size": 4,
        }
        write_json(args.embedding_cache, {"metadata": metadata, "embeddings": embeddings})
    else:
        embeddings, metadata = cached
    repo_demand = read_json(args.repo_demand_source_files)
    if not isinstance(repo_demand, list) or not all(isinstance(item, str) for item in repo_demand):
        raise ValueError("repo demand source files must be a JSON string list")
    bank = build_bank(
        cwe_order=cwe_order,
        public_records_by_cwe=public_by_cwe,
        embeddings=embeddings,
        embedding_metadata=metadata,
        repo_demand_source_files=repo_demand,
    )
    write_json(args.out, bank)
    return bank


if __name__ == "__main__":
    main()
