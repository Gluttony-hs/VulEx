import argparse
import json
import pathlib
import sys

# Allow direct execution from the package root with `python experiments/vulex/build_cap_anchor_bank.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.cap_anchor_bank import (
    CAP_ANCHOR_BANK_PARAMS,
    CAP_ANCHOR_BANK_STRATEGY,
    cwe_prior_repository_coverage_cap_anchor_bank,
    repository_coverage_demand_source_files,
)
from experiments.vulex.ordinary_memory_build import collect_ordinary_functions
from experiments.vulex.run_experiment import cwe_anchor_targets, read_reposvul_jsonl, read_split_ids, records_from_ids


def build_cwe_prior_repository_coverage_bank(args: argparse.Namespace) -> dict:
    """Build a per-CWE anchor bank with the current max joint-pool CAP algorithm."""
    if args.records is None or args.visibility_split is None:
        raise ValueError(f"records and visibility-split are required for {CAP_ANCHOR_BANK_STRATEGY}")
    if args.per_cwe_count is None or args.per_cwe_count <= 0:
        raise ValueError(f"per-cwe-count must be positive for {CAP_ANCHOR_BANK_STRATEGY}")
    if args.cwe_limit is not None and args.cwe_limit <= 0:
        raise ValueError("cwe-limit must be positive when provided")

    records = read_reposvul_jsonl(args.records)
    records_by_id = {record.record_id: record for record in records}
    visibility_split = read_split_ids(args.visibility_split)
    public_records = records_from_ids(records_by_id, visibility_split["public_record_ids"], "public visibility")
    cwe_order, anchors_by_cwe, _, cwe_counts = cwe_anchor_targets(public_records)
    selected_cwes = cwe_order[: args.cwe_limit] if args.cwe_limit is not None else cwe_order
    public_anchors = {record.anchor for record in public_records}
    candidates = collect_ordinary_functions(args.repo_worktree, excluded_anchors=public_anchors)
    repo_demand_source_files = repository_coverage_demand_source_files(
        candidates,
        args.repo_demand_count,
        args.seed,
    )

    anchors_by_cwe_bank: dict[str, list[str]] = {}
    per_cwe_summary: dict[str, dict] = {}
    params = {
        "anti_concentration_weight": 0.0,
        "candidate_pool_size": args.candidate_pool_size,
        "prior_weight": args.prior_weight,
        "repo_demand_count": args.repo_demand_count,
    }
    candidate_pool_size = min(args.candidate_pool_size, len(candidates))
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
        per_cwe_summary[cwe] = {
            "anchor_count": len(anchors),
            "candidate_pool_size": candidate_pool_size,
            "demand_record_count": len(demand_source_files),
            "demand_unique_file_count": len(set(demand_source_files)),
            "public_cwe_count": cwe_counts[cwe],
            "public_anchor_count": len(anchors_by_cwe[cwe]),
            "repo_demand_count": len(repo_demand_source_files),
        }

    return {
        "strategy": CAP_ANCHOR_BANK_STRATEGY,
        "seed": args.seed,
        "cwe_limit": args.cwe_limit,
        "per_cwe_count": args.per_cwe_count,
        "params": params,
        "cwe_count": len(selected_cwes),
        "cwe_order": selected_cwes,
        "anchor_count": sum(len(anchors) for anchors in anchors_by_cwe_bank.values()),
        "anchor_count_by_cwe": {
            cwe: len(anchors) for cwe, anchors in sorted(anchors_by_cwe_bank.items())
        },
        "candidate_count": len(candidates),
        "excluded_public_anchor_count": len(public_anchors),
        "public_record_count": len(public_records),
        "anchors_by_cwe": anchors_by_cwe_bank,
        "per_cwe_summary": per_cwe_summary,
    }


def main(argv: list[str] | None = None) -> None:
    """Preprocess the final CAP anchor bank from all public repository functions."""
    parser = argparse.ArgumentParser(description=f"Build {CAP_ANCHOR_BANK_STRATEGY} CAP anchor bank.")
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
