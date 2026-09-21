"""Construct visibility and memory splits for the release dataset."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.schemas import ReposVulRecord


def build_visibility_split(
    records: list[ReposVulRecord],
    seed: int,
    public_ratio: float = 0.3,
) -> dict[str, Any]:
    """Randomly split candidate records into public and protected sets."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    public_count = int(len(shuffled) * public_ratio)
    return {
        "public_record_ids": [record.record_id for record in shuffled[:public_count]],
        "protected_record_ids": [record.record_id for record in shuffled[public_count:]],
    }


def build_memory_split(
    visibility: dict[str, Any],
    memory_size: int,
    seed: int,
    public_ratio: float = 0.3,
) -> dict[str, Any]:
    """Sample target memory from the visibility split and assign protected pools."""
    public_pool = list(visibility["public_record_ids"])
    protected_pool = list(visibility["protected_record_ids"])
    rng = random.Random(seed)
    rng.shuffle(public_pool)
    rng.shuffle(protected_pool)
    public_count = int(memory_size * public_ratio)
    public = public_pool[:public_count]
    protected = protected_pool[: memory_size - public_count]
    return {
        "public_record_ids": public,
        "protected_record_ids": protected,
        "protected_pools": {
            record_id: "embargoed" if index % 2 == 0 else "silent"
            for index, record_id in enumerate(protected)
        },
    }


def read_json(path: pathlib.Path) -> dict[str, Any]:
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Generate visibility and memory splits."""
    parser = argparse.ArgumentParser(description="Build VulEx visibility and memory splits.")
    parser.add_argument("--records", required=True, type=pathlib.Path)
    parser.add_argument("--memory-size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--public-ratio", default=0.3, type=float)
    parser.add_argument("--visibility-out", required=True, type=pathlib.Path)
    parser.add_argument("--memory-split-out", required=True, type=pathlib.Path)
    parser.add_argument("--visibility-split", type=pathlib.Path)
    args = parser.parse_args(argv)

    from experiments.vulex.memory_build import read_reposvul_jsonl

    records = read_reposvul_jsonl(args.records)
    visibility = (
        read_json(args.visibility_split)
        if args.visibility_split
        else build_visibility_split(records, args.seed, args.public_ratio)
    )
    memory = build_memory_split(visibility, args.memory_size, args.seed, args.public_ratio)
    write_json(args.visibility_out, visibility)
    write_json(args.memory_split_out, memory)


if __name__ == "__main__":
    main()
