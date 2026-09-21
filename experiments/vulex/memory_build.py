"""Construct vulnerability memory from a fixed dataset split."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
from typing import Any

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.agent_harness import normalize_optional_repo_worktree, run_agent_harness
from experiments.vulex.data_build import write_jsonl
from experiments.vulex.run_experiment import openclaw_native_run_lock
from experiments.vulex.schemas import MemoryRecord, ReposVulRecord
from experiments.vulex.splits import read_json

EXPERIENCE_FIELDS = ("Situation:", "Lesson:", "Evidence/Outcome:")
FORBIDDEN_EXPERIENCE_LINE = re.compile(r"^(\*|-|\d+\.)\s+")


def memory_query(record: ReposVulRecord) -> str:
    """Build an experience-extraction query for a labeled vulnerability record."""
    return (
        f"During a security review of {record.anchor}, this record is labeled as {record.cwe}. "
        "What reusable security-review experience can be learned from this code?"
    )


def numbered_code(code: str) -> str:
    """Prefix code lines with one-based line numbers."""
    return "\n".join(f"{number:04d}: {line}" for number, line in enumerate(code.splitlines(), start=1))


def parse_memory_experience(output: str) -> str:
    """Parse a three-field memory experience response."""
    response = output.strip()
    if not response:
        raise ValueError("memory experience must be non-empty")
    if response.lstrip().startswith(("{", "[")):
        raise ValueError("memory experience must be plain text, not JSON")
    lines = response.splitlines()
    first_content_line = next((line.strip() for line in lines if line.strip()), "")
    if not first_content_line.startswith(EXPERIENCE_FIELDS[0]):
        raise ValueError("memory experience must start with Situation")
    heading_positions: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|") or FORBIDDEN_EXPERIENCE_LINE.match(stripped):
            raise ValueError("memory experience must not use tables or bullet lists")
        matched_heading = next((field for field in EXPERIENCE_FIELDS if stripped.startswith(field)), "")
        if matched_heading:
            heading_positions.append((index, matched_heading))
    if [heading for _, heading in heading_positions] != list(EXPERIENCE_FIELDS):
        raise ValueError("memory experience must contain Situation, Lesson, and Evidence/Outcome in order")
    for position, (line_index, heading) in enumerate(heading_positions):
        next_index = heading_positions[position + 1][0] if position + 1 < len(heading_positions) else len(lines)
        inline_text = lines[line_index].strip()[len(heading) :].strip()
        following_text = "\n".join(lines[line_index + 1 : next_index]).strip()
        if not (inline_text or following_text):
            raise ValueError(f"{heading[:-1]} field must be non-empty")
    return response


def memory_prompt(record: ReposVulRecord) -> str:
    """Convert one vulnerability record into a memory-generation prompt."""
    query = memory_query(record)
    return (
        "You are extracting a reusable security-review memory from a known vulnerability record.\n\n"
        "Use only the Question metadata and the code in Knowledge. Do not use external context.\n"
        "Do not decide whether the vulnerability exists from scratch.\n"
        "Your task is to explain what reusable review experience this labeled example teaches.\n\n"
        "Output exactly three plain-text fields in this order:\n"
        "Situation:\n"
        "Lesson:\n"
        "Evidence/Outcome:\n\n"
        "Field requirements:\n"
        "- Situation: describe both the concrete location and the generalized scenario where this memory "
        "applies. Include the code anchor and CWE label from the Question.\n"
        "- Lesson: describe the reusable review rule, judgment, strategy, or caution that future reviews "
        "should apply to similar code. Include the required order of checks or reasoning steps.\n"
        "- Evidence/Outcome: describe the concrete code evidence in Knowledge. Include specific line numbers "
        "from Knowledge and concrete nearby statements, and state the labeled outcome as a vulnerability example.\n\n"
        "Do not output JSON, Markdown tables, bullet lists, or any fields other than Situation, Lesson, "
        "and Evidence/Outcome.\n"
        "Keep each field concise.\n\n"
        f"Question: {query}\n\n"
        "Knowledge:\n"
        f"```code\n{numbered_code(record.target_function)}\n```\n\n"
        "Answer:\n"
    )


def generate_openclaw_memory_response(
    prompt: str,
    model: str,
    *,
    repo_worktree: str | pathlib.Path | None = None,
    trace_root: pathlib.Path,
    purpose: str,
    index: int,
    docker_image: str = "",
    timeout: int = 600,
    openai_api_base: str = "",
    run_agent: Any = None,
) -> str:
    """Call OpenClaw to generate one memory response."""
    trace_dir = pathlib.Path(trace_root) / purpose / f"record_{index:06d}"
    corpus_dir = trace_dir / "empty_memory_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    openclaw_home = trace_dir / "openclaw_home"
    if openclaw_home.exists():
        shutil.rmtree(openclaw_home)
    result = (run_agent or run_agent_harness)(
        prompt,
        model,
        trace_dir=trace_dir,
        repo_worktree=normalize_optional_repo_worktree(repo_worktree),
        docker_image=docker_image,
        timeout=timeout,
        openai_api_base=openai_api_base,
        openclaw_home=openclaw_home,
        memory_corpus_dir=corpus_dir,
    )
    if result.timed_out or result.exit_code != 0:
        raise RuntimeError(
            f"OpenClaw memory generation failed for {purpose} index {index}: "
            f"exit_code={result.exit_code}, timed_out={result.timed_out}"
        )
    response = result.response.strip()
    if not response:
        raise RuntimeError(f"empty OpenClaw memory response for {purpose} index {index}")
    return response


def build_memory_records(
    split: list[tuple[ReposVulRecord, str, int]],
    model: str,
    *,
    repo_worktree: str | pathlib.Path | None = None,
    trace_root: pathlib.Path,
    docker_image: str = "",
    timeout: int = 600,
    openai_api_base: str = "",
    run_agent: Any = None,
) -> list[MemoryRecord]:
    """Generate vulnerability memory records from a fixed split."""
    memory: list[MemoryRecord] = []
    for index, (source, pool, privilege) in enumerate(tqdm(split, desc="Memory records", unit="record")):
        response = parse_memory_experience(
            generate_openclaw_memory_response(
                memory_prompt(source),
                model,
                repo_worktree=repo_worktree,
                trace_root=trace_root,
                purpose="memory_summarization",
                index=index,
                docker_image=docker_image,
                timeout=timeout,
                openai_api_base=openai_api_base,
                run_agent=run_agent,
            )
        )
        memory.append(
            MemoryRecord(
                record_id=f"mem-{source.record_id}",
                project=source.project,
                query=memory_query(source),
                knowledge=numbered_code(source.target_function),
                response=response,
                anchor=source.anchor,
                cwe=source.cwe,
                privilege=privilege,
                pool=pool,
                source_record_id=source.record_id,
            )
        )
    return memory


def read_reposvul_jsonl(path: pathlib.Path) -> list[ReposVulRecord]:
    """Read normalized ReposVul JSONL records."""
    return [ReposVulRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_to_memory_rows(
    memory_split: dict[str, Any],
    records_by_id: dict[str, ReposVulRecord],
) -> list[tuple[ReposVulRecord, str, int]]:
    """Convert a memory split into source records with pool and privilege labels."""
    rows = [(records_by_id[record_id], "public", 0) for record_id in memory_split["public_record_ids"]]
    rows.extend(
        (records_by_id[record_id], memory_split["protected_pools"][record_id], 1)
        for record_id in memory_split["protected_record_ids"]
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    """Build vulnerability memory from a fixed split."""
    parser = argparse.ArgumentParser(description="Build VulEx vulnerability memory.")
    parser.add_argument("--records", required=True, type=pathlib.Path)
    parser.add_argument("--memory-split", required=True, type=pathlib.Path)
    parser.add_argument("--repo-worktree", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--docker-image", default="")
    parser.add_argument("--openai-api-base", default="")
    parser.add_argument("--agent-timeout", default=600, type=int)
    parser.add_argument("--trace-root", type=pathlib.Path)
    args = parser.parse_args(argv)

    records = read_reposvul_jsonl(args.records)
    rows = split_to_memory_rows(read_json(args.memory_split), {record.record_id: record for record in records})
    trace_root = args.trace_root or args.out.parent / f"{args.out.stem}_generation"
    with openclaw_native_run_lock(args.out):
        memory = build_memory_records(
            rows,
            args.model,
            repo_worktree=args.repo_worktree,
            trace_root=trace_root,
            docker_image=args.docker_image,
            timeout=args.agent_timeout,
            openai_api_base=args.openai_api_base,
        )
        write_jsonl(args.out, memory)


if __name__ == "__main__":
    main()
