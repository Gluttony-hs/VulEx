"""Construct ordinary-code memory from sampled Linux functions."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from dataclasses import dataclass

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.agent_harness import normalize_optional_repo_worktree
from experiments.vulex.c_function_extract import c_function_bodies
from experiments.vulex.data_build import write_jsonl
from experiments.vulex.memory_build import (
    generate_openclaw_memory_response,
    numbered_code,
    parse_memory_experience,
    read_reposvul_jsonl,
)
from experiments.vulex.run_experiment import openclaw_native_run_lock
from experiments.vulex.schemas import LINUX_PROJECT, MemoryRecord, OrdinaryMemoryRecord


@dataclass(frozen=True)
class OrdinaryFunction:
    """Represent one ordinary-memory candidate function."""

    anchor: str
    code: str


def iter_c_files(repo: pathlib.Path) -> list[pathlib.Path]:
    """Recursively list C files in the Linux source tree."""
    return sorted(path for path in pathlib.Path(repo).rglob("*.c") if ".git" not in path.parts)


def collect_ordinary_functions(repo: pathlib.Path, excluded_anchors: set[str]) -> list[OrdinaryFunction]:
    """Collect function candidates that are not vulnerability records."""
    repo = pathlib.Path(repo)
    candidates: list[OrdinaryFunction] = []
    for path in iter_c_files(repo):
        relative = path.relative_to(repo).as_posix()
        for function in c_function_bodies(path.read_text(encoding="utf-8")):
            anchor = f"{relative}::{function.name}"
            if anchor not in excluded_anchors:
                candidates.append(OrdinaryFunction(anchor=anchor, code=function.body))
    return candidates


def sample_ordinary_functions(
    repo: pathlib.Path,
    count: int,
    seed: int,
    excluded_anchors: set[str],
) -> list[OrdinaryFunction]:
    """Sample ordinary-function candidates with a fixed seed."""
    candidates = collect_ordinary_functions(repo, excluded_anchors)
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def ordinary_memory_query(anchor: str) -> str:
    """Build an experience-extraction query for ordinary memory."""
    return (
        f"During a code review of {anchor}, what reusable review experience can be learned "
        "from this ordinary non-vulnerability code example?"
    )


def ordinary_memory_prompt(function: OrdinaryFunction) -> str:
    """Convert an ordinary function into a memory-generation prompt."""
    query = ordinary_memory_query(function.anchor)
    return (
        "You are extracting a reusable code-review memory from an ordinary non-vulnerability record.\n\n"
        "Use only the Question anchor and the code in Knowledge. Do not use external context.\n"
        "Do not invent a security vulnerability.\n"
        "Your task is to explain what reusable review experience this ordinary code example teaches.\n\n"
        "Output exactly three plain-text fields in this order:\n"
        "Situation:\n"
        "Lesson:\n"
        "Evidence/Outcome:\n\n"
        "Field requirements:\n"
        "- Situation: describe both the concrete location and the generalized scenario where this memory "
        "applies. Include the code anchor from the Question.\n"
        "- Lesson: describe the reusable non-vulnerability code-review rule, judgment, strategy, or caution "
        "learned from the code. Include the required order of checks or operations.\n"
        "- Evidence/Outcome: describe the concrete code evidence in Knowledge. Include specific line numbers "
        "from Knowledge and concrete nearby statements, and explicitly state that this is not a vulnerability "
        "record and no vulnerability is identified from this code.\n\n"
        "Do not output JSON, Markdown tables, bullet lists, or any fields other than Situation, Lesson, "
        "and Evidence/Outcome.\n"
        "Keep each field concise.\n\n"
        f"Question: {query}\n\n"
        "Knowledge:\n"
        f"```code\n{numbered_code(function.code)}\n```\n\n"
        "Answer:\n"
    )


def parse_ordinary_memory_output(output: str) -> str:
    """Parse an ordinary-memory response."""
    return parse_memory_experience(output)


def build_ordinary_memory_records(
    functions: list[OrdinaryFunction],
    model: str,
    seed: int,
    *,
    repo_worktree: pathlib.Path | None = None,
    trace_root: pathlib.Path,
    docker_image: str = "",
    timeout: int = 600,
    openai_api_base: str = "",
    run_agent=None,
) -> list[OrdinaryMemoryRecord]:
    """Call OpenClaw to generate memory records for ordinary functions."""
    records: list[OrdinaryMemoryRecord] = []
    for index, function in enumerate(tqdm(functions, desc="Ordinary memory records", unit="record")):
        response = parse_ordinary_memory_output(
            generate_openclaw_memory_response(
                ordinary_memory_prompt(function),
                model,
                repo_worktree=repo_worktree,
                trace_root=trace_root,
                purpose="ordinary_memory_summarization",
                index=index,
                docker_image=docker_image,
                timeout=timeout,
                openai_api_base=openai_api_base,
                run_agent=run_agent,
            )
        )
        records.append(
            OrdinaryMemoryRecord(
                record_id=f"ord-linux-{index:06d}",
                project=LINUX_PROJECT,
                query=ordinary_memory_query(function.anchor),
                knowledge=numbered_code(function.code),
                response=response,
                anchor=function.anchor,
            )
        )
    return records


def read_memory_jsonl(path: pathlib.Path) -> list[MemoryRecord]:
    """Read generated vulnerability-memory records."""
    return [MemoryRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    """Sample Linux functions and build ordinary memory."""
    parser = argparse.ArgumentParser(description="Build ordinary Linux code memory.")
    parser.add_argument("--repo-worktree", required=True, type=pathlib.Path)
    parser.add_argument("--records", required=True, type=pathlib.Path)
    parser.add_argument("--memory", required=True, type=pathlib.Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--docker-image", default="")
    parser.add_argument("--openai-api-base", default="")
    parser.add_argument("--agent-timeout", default=600, type=int)
    parser.add_argument("--trace-root", type=pathlib.Path)
    parser.add_argument("--agent-repo-worktree", default="")
    args = parser.parse_args(argv)

    excluded = {record.anchor for record in read_reposvul_jsonl(args.records)}
    excluded.update(record.anchor for record in read_memory_jsonl(args.memory))
    functions = sample_ordinary_functions(args.repo_worktree, args.count, args.seed, excluded)
    trace_root = args.trace_root or args.out.parent / f"{args.out.stem}_generation"
    with openclaw_native_run_lock(args.out):
        records = build_ordinary_memory_records(
            functions,
            args.model,
            args.seed,
            repo_worktree=normalize_optional_repo_worktree(args.agent_repo_worktree),
            trace_root=trace_root,
            docker_image=args.docker_image,
            timeout=args.agent_timeout,
            openai_api_base=args.openai_api_base,
        )
        write_jsonl(args.out, records)


if __name__ == "__main__":
    main()
