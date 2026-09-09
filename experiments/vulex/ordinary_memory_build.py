import argparse
import hashlib
import json
import pathlib
import random
from dataclasses import dataclass

from tqdm import tqdm

from experiments.vulex.c_function_extract import c_function_bodies
from experiments.vulex.data_build import write_jsonl
from experiments.vulex.agent_harness import normalize_optional_repo_worktree
from experiments.vulex.memory_build import (
    generate_openclaw_memory_response,
    numbered_code,
    parse_memory_experience,
    read_reposvul_jsonl,
)
from experiments.vulex.run_experiment import openclaw_native_run_lock
from experiments.vulex.schemas import (
    LINUX_PROJECT,
    MemoryRecord,
    OrdinaryMemoryRecord,
)

@dataclass(frozen=True)
class OrdinaryFunction:
    anchor: str
    source_file: str
    source_function: str
    code: str
    source_hash: str


def iter_c_files(repo: pathlib.Path) -> list[pathlib.Path]:
    """Recursively list C files in the Linux worktree while excluding .git."""
    return sorted(path for path in pathlib.Path(repo).rglob("*.c") if ".git" not in path.parts)


def ordinary_function_hash(anchor: str, code: str) -> str:
    """Compute an ordinary-memory source hash from the anchor and function code."""
    payload = json.dumps({"anchor": anchor, "code": code}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_ordinary_functions(
    repo: pathlib.Path,
    count: int,
    seed: int,
    excluded_anchors: set[str],
) -> list[OrdinaryFunction]:
    """Sample functions deterministically from all C functions in the repository."""
    if count <= 0:
        raise ValueError("count must be positive")
    candidates = collect_ordinary_functions(repo, excluded_anchors)
    if len(candidates) < count:
        raise ValueError(f"not enough ordinary function candidates: {len(candidates)} < {count}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def collect_ordinary_functions(repo: pathlib.Path, excluded_anchors: set[str]) -> list[OrdinaryFunction]:
    """Collect repository-wide C-function candidates for CAP bank construction."""
    repo = pathlib.Path(repo)
    files = iter_c_files(repo)
    candidates: list[OrdinaryFunction] = []
    for path in files:
        rel = path.relative_to(repo).as_posix()
        text = path.read_text(encoding="utf-8")
        for function in c_function_bodies(text):
            anchor = f"{rel}::{function.name}"
            if anchor in excluded_anchors:
                continue
            candidates.append(
                OrdinaryFunction(
                    anchor=anchor,
                    source_file=rel,
                    source_function=function.name,
                    code=function.body,
                    source_hash=ordinary_function_hash(anchor, function.body),
                )
            )
    return candidates


def ordinary_memory_prompt(function: OrdinaryFunction) -> str:
    """Rewrite an ordinary-code function into an OpenClaw ordinary-memory generation prompt."""
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


def ordinary_memory_query(anchor: str) -> str:
    """Return the experience-extraction question for ordinary non-vulnerability code memory."""
    return (
        f"During a code review of {anchor}, what reusable review experience can be learned "
        "from this ordinary non-vulnerability code example?"
    )


def parse_ordinary_memory_output(output: str) -> str:
    """Parse input and return a normalized result."""
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
    """Call OpenClaw to generate ordinary-memory records for ordinary-code functions."""
    records: list[OrdinaryMemoryRecord] = []
    for index, function in enumerate(
        tqdm(functions, desc="Ordinary memory records", total=len(functions), unit="record")
    ):
        output = generate_openclaw_memory_response(
            ordinary_memory_prompt(function),
            model,
            function.source_hash,
            repo_worktree=repo_worktree,
            trace_root=trace_root,
            purpose="ordinary_memory_summarization",
            index=index,
            docker_image=docker_image,
            timeout=timeout,
            openai_api_base=openai_api_base,
            run_agent=run_agent,
        )
        query = ordinary_memory_query(function.anchor)
        response = parse_ordinary_memory_output(output)
        knowledge = numbered_code(function.code)
        records.append(
            OrdinaryMemoryRecord(
                record_id=f"ord-linux-{index:06d}",
                project=LINUX_PROJECT,
                query=query,
                knowledge=knowledge,
                response=response,
                anchor=function.anchor,
                source_file=function.source_file,
                source_function=function.source_function,
                source_hash=function.source_hash,
                sampling_seed=seed,
            )
        )
    return records


def read_memory_jsonl(path: pathlib.Path) -> list[MemoryRecord]:
    """Read the relevant input data."""
    records: list[MemoryRecord] = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(MemoryRecord(**json.loads(line)))
    return records


def reposvul_anchors(path: pathlib.Path) -> set[str]:
    """Read the vulnerability-function anchor set from ReposVul input."""
    return {record.anchor for record in read_reposvul_jsonl(path)}


def target_memory_anchors(path: pathlib.Path) -> set[str]:
    """Read the anchor set from the constructed target memory."""
    return {record.anchor for record in read_memory_jsonl(path)}


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and execute the selected flow."""
    parser = argparse.ArgumentParser(description="Build ordinary Linux code memory records from random C functions.")
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

    excluded = reposvul_anchors(args.records) | target_memory_anchors(args.memory)
    functions = sample_ordinary_functions(args.repo_worktree, args.count, args.seed, excluded)
    trace_root = args.trace_root or args.out.parent / f"{args.out.stem}_openclaw_memory_generation"
    with openclaw_native_run_lock(args.out):
        agent_repo_worktree = normalize_optional_repo_worktree(args.agent_repo_worktree)
        records = build_ordinary_memory_records(
            functions,
            args.model,
            args.seed,
            repo_worktree=agent_repo_worktree,
            trace_root=trace_root,
            docker_image=args.docker_image,
            timeout=args.agent_timeout,
            openai_api_base=args.openai_api_base,
        )
        write_jsonl(args.out, records)


if __name__ == "__main__":
    main()
