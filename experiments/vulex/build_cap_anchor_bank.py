import argparse
import json
import pathlib
import sys
from dataclasses import dataclass

# Allow direct execution from the package root with `python experiments/vulex/build_cap_anchor_bank.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.cap_anchor_bank import (
    CAP_ANCHOR_BANK_PARAMS,
    cwe_prior_repository_coverage_cap_anchor_bank,
    repository_coverage_demand_source_files,
)
from experiments.vulex.c_function_extract import c_function_bodies
from experiments.vulex.run_experiment import cwe_anchor_targets, read_reposvul_jsonl, read_split_ids, records_from_ids


@dataclass(frozen=True)
class AnchorCandidate:
    """Represent one repository function considered by CAP."""

    anchor: str
    source_file: str
    source_function: str
    code: str


def collect_anchor_candidates(repo: pathlib.Path, excluded_anchors: set[str]) -> list[AnchorCandidate]:
    """Collect Linux C functions that are not public vulnerability anchors."""
    candidates: list[AnchorCandidate] = []
    for path in sorted(repo.rglob("*.c")):
        rel = path.relative_to(repo).as_posix()
        for function in c_function_bodies(path.read_text(encoding="utf-8")):
            anchor = f"{rel}::{function.name}"
            if anchor not in excluded_anchors:
                candidates.append(AnchorCandidate(anchor, rel, function.name, function.body))
    return candidates


def build_cwe_prior_repository_coverage_bank(args: argparse.Namespace) -> dict:
    """Build a per-CWE anchor bank with the current max joint-pool CAP algorithm."""
    records = read_reposvul_jsonl(args.records)
    records_by_id = {record.record_id: record for record in records}
    visibility_split = read_split_ids(args.visibility_split)
    public_records = records_from_ids(records_by_id, visibility_split["public_record_ids"])
    cwe_order, _ = cwe_anchor_targets(public_records)
    selected_cwes = cwe_order[: args.cwe_limit] if args.cwe_limit is not None else cwe_order
    public_anchors = {record.anchor for record in public_records}
    candidates = collect_anchor_candidates(args.repo_worktree, public_anchors)
    repo_demand_source_files = repository_coverage_demand_source_files(
        candidates,
        args.repo_demand_count,
        args.seed,
    )

    anchors_by_cwe_bank: dict[str, list[str]] = {}
    for index, cwe in enumerate(selected_cwes):
        demand_source_files = [
            record.anchor.split("::", 1)[0]
            for record in public_records
            if record.cwe == cwe
        ]
        anchors = cwe_prior_repository_coverage_cap_anchor_bank(
            candidates,
            demand_source_files,
            count=args.per_cwe_count,
            seed=args.seed + index,
            prior_weight=args.prior_weight,
            repo_demand_count=args.repo_demand_count,
            candidate_pool_size=args.candidate_pool_size,
            repository_demand_source_files=repo_demand_source_files,
        )
        anchors_by_cwe_bank[cwe] = anchors
    return {"anchors_by_cwe": anchors_by_cwe_bank}


def main(argv: list[str] | None = None) -> None:
    """Preprocess the final CAP anchor bank from all public repository functions."""
    parser = argparse.ArgumentParser(description="Build the CAP anchor bank.")
    parser.add_argument("--repo-worktree", required=True, type=pathlib.Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--records", required=True, type=pathlib.Path)
    parser.add_argument("--visibility-split", required=True, type=pathlib.Path)
    parser.add_argument("--cwe-limit", type=int)
    parser.add_argument("--per-cwe-count", required=True, type=int)
    parser.add_argument("--prior-weight", default=CAP_ANCHOR_BANK_PARAMS["prior_weight"], type=float)
    parser.add_argument("--repo-demand-count", default=CAP_ANCHOR_BANK_PARAMS["repo_demand_count"], type=int)
    parser.add_argument("--candidate-pool-size", default=CAP_ANCHOR_BANK_PARAMS["candidate_pool_size"], type=int)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    payload = build_cwe_prior_repository_coverage_bank(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
