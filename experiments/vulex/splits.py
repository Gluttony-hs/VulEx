import argparse
import json
import pathlib
import random
from typing import Any

from experiments.vulex.schemas import ReposVulRecord


def candidate_records(records: list[ReposVulRecord], exclude_file_scope: bool) -> list[ReposVulRecord]:
    """Build the dataset-level candidate pool; the current schema has no file_scope records."""
    return list(records)


def build_visibility_split(
    records: list[ReposVulRecord],
    seed: int,
    public_ratio: float = 0.3,
) -> dict[str, Any]:
    """Split candidate vulnerability records into an attacker-visible public pool and an evaluation protected pool."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    public_count = int(len(shuffled) * public_ratio)
    public = [record.record_id for record in shuffled[:public_count]]
    protected = [record.record_id for record in shuffled[public_count:]]
    return {
        "seed": seed,
        "public_ratio": public_ratio,
        "candidate_count": len(shuffled),
        "public_count": len(public),
        "protected_count": len(protected),
        "public_record_ids": public,
        "protected_record_ids": protected,
    }


def build_memory_split(
    visibility: dict[str, Any],
    memory_size: int,
    seed: int,
    public_ratio: float = 0.3,
) -> dict[str, Any]:
    """Sample a fixed-size target memory from the two pools in the visibility split."""
    public_pool = list(visibility["public_record_ids"])
    protected_pool = list(visibility["protected_record_ids"])
    public_count = int(memory_size * public_ratio)
    protected_count = memory_size - public_count
    if len(public_pool) < public_count:
        raise ValueError(f"not enough public records: {len(public_pool)} < requested {public_count}")
    if len(protected_pool) < protected_count:
        raise ValueError(f"not enough protected records: {len(protected_pool)} < requested {protected_count}")
    rng = random.Random(seed)
    rng.shuffle(public_pool)
    rng.shuffle(protected_pool)
    public_memory = public_pool[:public_count]
    protected_memory = protected_pool[:protected_count]
    protected_pools = {
        record_id: "embargoed" if index % 2 == 0 else "silent"
        for index, record_id in enumerate(protected_memory)
    }
    return {
        "seed": seed,
        "visibility_seed": visibility["seed"],
        "memory_size": memory_size,
        "public_count": len(public_memory),
        "protected_count": len(protected_memory),
        "public_record_ids": public_memory,
        "protected_record_ids": protected_memory,
        "protected_pools": protected_pools,
    }


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write structured split JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def read_json(path: pathlib.Path) -> dict[str, Any]:
    """Read structured split JSON."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """CLI entry point that constructs only the visibility and memory splits."""
    parser = argparse.ArgumentParser(description="Build formal linux VulEx visibility and memory split artifacts.")
    parser.add_argument("--records", required=True, type=pathlib.Path)
    parser.add_argument("--memory-size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--visibility-out", required=True, type=pathlib.Path)
    parser.add_argument("--memory-split-out", required=True, type=pathlib.Path)
    parser.add_argument("--visibility-split", type=pathlib.Path)
    parser.add_argument("--exclude-file-scope", action="store_true")
    args = parser.parse_args()

    from experiments.vulex.memory_build import read_reposvul_jsonl

    records = candidate_records(read_reposvul_jsonl(args.records), args.exclude_file_scope)
    visibility = read_json(args.visibility_split) if args.visibility_split else build_visibility_split(records, args.seed)
    memory_split = build_memory_split(visibility, args.memory_size, args.seed)
    write_json(args.visibility_out, visibility)
    write_json(args.memory_split_out, memory_split)
    print(
        json.dumps(
            {
                "records": str(args.records),
                "candidate_count": visibility["candidate_count"],
                "visibility_out": str(args.visibility_out),
                "memory_split_out": str(args.memory_split_out),
                "memory_size": args.memory_size,
                "seed": args.seed,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
