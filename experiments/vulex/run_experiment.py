import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any

from tqdm import tqdm

    # Allow direct runner execution from the package root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.agent_harness import (
    AgentRunResult,
    MEMORY_MAX_RESULTS_INSTRUCTION,
    MEMORY_ONLY_TOOLS,
    MEMORY_PRIVACY_RESTRICTION,
    OPENCLAW_DEFAULT_OPENAI_EMBEDDING_MODEL,
    OPENCLAW_MEMORY_CORPUS_CONTAINER_PATH,
    TAIL_SCHEMA,
    build_harness_prompt,
    harness_source_hash,
    normalize_optional_repo_worktree,
    run_agent_harness,
    run_openclaw_memory_index,
    write_openclaw_config,
)
from experiments.vulex.cap_anchor_bank import read_cap_anchor_bank_spec
from experiments.vulex.llm_client import CachedLLMClient, cache_key
from experiments.vulex.metrics import compute_native_agent_metrics
from experiments.vulex.openclaw_memory import write_openclaw_memory_corpus
from experiments.vulex.parse import parse_response
from experiments.vulex.probes import build_cap_probe, build_sss_probe
from experiments.vulex.schemas import (
    LINUX_PROJECT,
    OPENCLAW_NATIVE_BACKEND,
    OPENCLAW_NATIVE_RETRIEVER,
    MemoryRecord,
    OrdinaryMemoryRecord,
    ParsedTriple,
    ReposVulRecord,
    RunConfig,
)
from experiments.vulex.splits import read_json
from experiments.vulex.sss import build_sss_prompt, format_multi_anchor_sss_payload, generate_sss_payload, sss_source_hash

DEFAULT_SSS_PAYLOAD_ANCHOR_LIMIT = 1
OPENCLAW_AGENT_ATTEMPTS = 8
OPENCLAW_AGENT_RETRY_SLEEP_SECONDS = 5
OPENCLAW_TERMINAL_FAILURE_MARKERS = (
    "Agent couldn't generate a response",
    "The model did not produce a response before the model idle timeout",
    "Concurrency limit exceeded for account",
)
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_SEMANTIC_VERSION = 2
CHECKPOINT_DIRNAME = "checkpoint"
FINAL_OUTPUT_NAMES = {
    "config.json",
    "anchor_set.json",
    "cwe_set.json",
    "anchor_cwe_pairs.json",
    "cap_anchor_bank.json",
    "selected_sss_payload_anchors.json",
    "probes.jsonl",
    "responses.jsonl",
    "parsed.jsonl",
    "llm_prompts.jsonl",
    "probe_targets.json",
    "summary.json",
    "summary_scored.json",
    "versions.json",
    "manifest.json",
    "preflight.json",
    "tool_policy_audit.json",
}
PARTIAL_OUTPUT_NAMES = {"summary_partial.json", "failed_probes.json"}


class IncompleteExperimentError(RuntimeError):
    """Represent a partial probe failure after writing partial results that cannot produce formal metrics."""


class ProbeExecutionError(RuntimeError):
    """Represent a retryable execution failure for one OpenClaw probe."""

    def __init__(self, message: str, agent_result: AgentRunResult | None = None):
        """Store the underlying agent result so checkpoints can record the trace and exit status."""
        super().__init__(message)
        self.agent_result = agent_result


@contextmanager
def openclaw_native_run_lock(out_dir: pathlib.Path):
    """Limit OpenClaw native experiments to one instance to prevent concurrent Docker agents from contaminating results."""
    lock_path = pathlib.Path(".openclaw_native_memory.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(" ")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError as exc:
                holder = ""
                try:
                    handle.seek(0)
                    holder = handle.read().strip()
                except OSError:
                    pass
                raise RuntimeError(f"another OpenClaw native run is active: {holder}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError as exc:
                handle.seek(0)
                holder = handle.read().strip()
                raise RuntimeError(f"another OpenClaw native run is active: {holder}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {"pid": os.getpid(), "out_dir": str(out_dir), "started_at": datetime.now(timezone.utc).isoformat()},
                ensure_ascii=False,
            )
        )
        handle.flush()
        yield
    finally:
        if acquired:
            handle.seek(0)
            handle.truncate()
            handle.write(" ")
            handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.seek(0)
            handle.truncate()
            handle.flush()
        handle.close()


def numbered_memory_knowledge(code: str) -> str:
    """Return one-based numbered code stored in formal memory for downstream validation and generation alignment."""
    return "\n".join(f"{line_number:04d}: {line}" for line_number, line in enumerate(code.splitlines(), start=1))


def read_memory_jsonl(path: pathlib.Path) -> list[MemoryRecord]:
    """Read formal memory JSONL and construct MemoryRecord objects line by line."""
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[MemoryRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(MemoryRecord(**json.loads(line)))
    if not records:
        raise ValueError(f"memory file is empty: {path}")
    return records


def read_ordinary_memory_jsonl(path: pathlib.Path) -> list[OrdinaryMemoryRecord]:
    """Read ordinary-code memory JSONL and construct OrdinaryMemoryRecord objects line by line."""
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[OrdinaryMemoryRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(OrdinaryMemoryRecord(**json.loads(line)))
    if not records:
        raise ValueError(f"ordinary memory file is empty: {path}")
    return records


def read_cwe_source(path: pathlib.Path) -> list[str]:
    """Read the explicit CWE order from probe_targets or cwe_set JSON."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"cwe_source must be a JSON list: {path}")
    cwes: list[str] = []
    for row in rows:
        if isinstance(row, str):
            cwe = row
        elif isinstance(row, dict):
            cwe = row.get("cwe", "")
        else:
            cwe = ""
        if cwe:
            cwes.append(str(cwe))
    if not cwes:
        raise ValueError(f"cwe_source has no CWE entries: {path}")
    return list(dict.fromkeys(cwes))


def read_reposvul_jsonl(path: pathlib.Path) -> list[ReposVulRecord]:
    """Read formal ReposVul JSONL and construct ReposVulRecord objects line by line."""
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[ReposVulRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(ReposVulRecord(**json.loads(line)))
    if not records:
        raise ValueError(f"records file is empty: {path}")
    return records


def duplicate_record_ids(record_ids: list[str]) -> list[str]:
    """Return record IDs that first repeat in the input list, preserving discovery order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for record_id in record_ids:
        if record_id in seen and record_id not in duplicates:
            duplicates.append(record_id)
        seen.add(record_id)
    return duplicates


def read_split_ids(path: pathlib.Path) -> dict:
    """Read visibility and memory splits and validate their minimum fields."""
    if not path:
        raise ValueError("split path is required")
    if not path.exists():
        raise FileNotFoundError(path)
    data = read_json(path)
    for field in ("public_record_ids", "protected_record_ids"):
        if field not in data or not isinstance(data[field], list):
            raise ValueError(f"split {path} missing list field {field}")
        duplicates = duplicate_record_ids(data[field])
        if duplicates:
            raise ValueError(f"duplicate split record_id in {field}: {duplicates[:5]}")
    return data


def records_from_ids(records_by_id: dict[str, ReposVulRecord], record_ids: list[str], label: str) -> list[ReposVulRecord]:
    """Look up ReposVulRecords by split record IDs and fail immediately when any are missing."""
    missing = [record_id for record_id in record_ids if record_id not in records_by_id]
    if missing:
        raise ValueError(f"{label} record_id missing from records: {missing[:5]}")
    return [records_by_id[record_id] for record_id in record_ids]


def validate_memory_source_alignment(
    memory: list[MemoryRecord],
    records_by_id: dict[str, ReposVulRecord],
    memory_split: dict[str, Any],
) -> None:
    """Validate that each memory item faithfully corresponds to its ReposVul source in memory_split."""
    source_ids = [record.source_record_id for record in memory]
    seen: set[str] = set()
    duplicates: list[str] = []
    for source_id in source_ids:
        if source_id in seen and source_id not in duplicates:
            duplicates.append(source_id)
        seen.add(source_id)
    if duplicates:
        raise ValueError(f"duplicate memory source_record_id: {duplicates[:5]}")

    missing_sources = [source_id for source_id in source_ids if source_id not in records_by_id]
    if missing_sources:
        raise ValueError(f"memory source_record_id missing from records: {missing_sources[:5]}")

    public_ids = list(memory_split["public_record_ids"])
    protected_ids = list(memory_split["protected_record_ids"])
    expected_memory_sources = set(public_ids) | set(protected_ids)
    actual_memory_sources = set(source_ids)
    if expected_memory_sources != actual_memory_sources:
        raise ValueError("memory source_record_id set does not match memory split")

    protected_pools = memory_split.get("protected_pools", {})
    missing_protected_pools = [source_id for source_id in protected_ids if source_id not in protected_pools]
    if missing_protected_pools:
        raise ValueError(f"memory split missing protected_pools for protected source ids: {missing_protected_pools[:5]}")

    memory_by_source = {record.source_record_id: record for record in memory}
    for record in memory:
        source = records_by_id[record.source_record_id]
        if record.anchor != source.anchor:
            raise ValueError(f"memory anchor does not match source {record.source_record_id}")
        if record.cwe != source.cwe:
            raise ValueError(f"memory cwe does not match source {record.source_record_id}")
        if record.knowledge != numbered_memory_knowledge(source.target_function):
            raise ValueError(f"memory knowledge does not match source {record.source_record_id}")

    for source_id in public_ids:
        record = memory_by_source[source_id]
        if record.privilege != 0 or record.pool != "public":
            raise ValueError(f"public memory source {source_id} must have privilege=0 and pool='public'")

    for source_id in protected_ids:
        record = memory_by_source[source_id]
        expected_pool = protected_pools[source_id]
        if record.privilege != 1 or record.pool != expected_pool:
            raise ValueError(
                f"protected memory pool mismatch for source {source_id}: "
                f"got privilege={record.privilege}, pool={record.pool!r}; expected privilege=1, pool={expected_pool!r}"
            )


def ensure_linux_only(memory: list[MemoryRecord], records: list[ReposVulRecord], project: str) -> None:
    """Lock project, memory, and records to Linux before execution."""
    if project != LINUX_PROJECT:
        raise ValueError(f"only {LINUX_PROJECT} is supported in this experiment, got {project!r}")
    bad_memory = [record.record_id for record in memory if record.project != LINUX_PROJECT]
    bad_records = [record.record_id for record in records if record.project != LINUX_PROJECT]
    if bad_memory or bad_records:
        raise ValueError(
            f"all records must be {LINUX_PROJECT}; bad memory IDs={bad_memory[:5]}, bad record IDs={bad_records[:5]}"
        )


def write_json(path: pathlib.Path, payload: Any) -> None:
    """Write a structured JSON result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    """Write line-delimited JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def append_jsonl(path: pathlib.Path, row: dict[str, Any]) -> None:
    """Append one line to checkpoint JSONL and flush immediately to reduce interruption loss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact; return an empty list when absent for incremental checkpoint recovery."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def agent_attempt_trace_dir(out_dir: pathlib.Path, probe_index: int, attempt_index: int) -> pathlib.Path:
    """Assign a trace directory to one OpenClaw probe attempt so retries do not overwrite failure evidence."""
    if attempt_index == 0:
        return out_dir / "agent_traces" / f"probe_{probe_index:04d}"
    return out_dir / "agent_traces" / f"probe_{probe_index:04d}_attempt{attempt_index + 1:02d}"


def openclaw_terminal_response_failed(response: str) -> bool:
    """Identify terminal failure text where OpenClaw exits with code 0 but produces no valid answer."""
    return any(marker in response for marker in OPENCLAW_TERMINAL_FAILURE_MARKERS)


def openclaw_agent_succeeded(exit_code: int | None, timed_out: bool, response: str) -> bool:
    """Centralize whether the OpenClaw agent produced an acceptable terminal answer."""
    return exit_code == 0 and not timed_out and not openclaw_terminal_response_failed(response)


def run_configured_agent(
    config,
    out_dir: pathlib.Path,
    probe_index: int,
    message: str,
    openclaw_home: pathlib.Path,
    memory_corpus_dir: pathlib.Path,
) -> AgentRunResult:
    """Run one OpenClaw native-memory probe with unified configuration and retry transient empty-response failures."""
    repo_worktree = normalize_optional_repo_worktree(config.repo_worktree)
    memory_search_mode = getattr(config, "openclaw_memory_search", "bm25_fts")
    embedding_api_base = getattr(config, "openclaw_embedding_api_base", "")
    embedding_model = getattr(config, "openclaw_embedding_model", "")
    last_result = None
    for attempt_index in range(OPENCLAW_AGENT_ATTEMPTS):
        result = run_agent_harness(
            message,
            config.agent_model,
            trace_dir=agent_attempt_trace_dir(out_dir, probe_index, attempt_index),
            repo_worktree=repo_worktree,
            docker_image=config.docker_image,
            timeout=config.agent_timeout,
            openai_api_base=config.openai_api_base,
            memory_search_mode=memory_search_mode,
            embedding_api_base=embedding_api_base,
            embedding_model=embedding_model,
            openclaw_home=openclaw_home,
            memory_corpus_dir=memory_corpus_dir,
        )
        if openclaw_agent_succeeded(result.exit_code, result.timed_out, result.response):
            return result
        last_result = result
        if attempt_index < OPENCLAW_AGENT_ATTEMPTS - 1:
            time.sleep(OPENCLAW_AGENT_RETRY_SLEEP_SECONDS)
    if last_result is None:
        raise RuntimeError("OpenClaw agent attempt loop did not run")
    return last_result


def require_agent_success(result: AgentRunResult, probe_index: int) -> None:
    """Validate one agent result and raise failures for probe checkpoint capture."""
    if not openclaw_agent_succeeded(result.exit_code, result.timed_out, result.response):
        raise ProbeExecutionError(
            f"{OPENCLAW_NATIVE_BACKEND} probe {probe_index} failed: "
            f"exit_code={result.exit_code}, timed_out={result.timed_out}, trace_dir={result.trace_dir}",
            result,
        )


def sha256_file(path: pathlib.Path) -> str:
    """Compute a SHA256 digest for one input file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: pathlib.Path) -> str:
    """Compute a directory-tree content hash to record the local CodeBERT snapshot."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    """Capture the current worktree git state and write it to the experiment manifest."""
    root = pathlib.Path.cwd()
    data: dict[str, Any] = {"workdir": str(root)}
    try:
        data["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        data["branch"] = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
        data["status_short"] = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        data["error"] = str(exc)
    return data


def package_version(name: str) -> str | None:
    """Read a dependency version and return None when it is missing."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions(config: RunConfig) -> dict[str, Any]:
    """Record the runtime environment and key dependency versions."""
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "memory_delivery": OPENCLAW_NATIVE_RETRIEVER,
        "agent_model": config.agent_model,
        "sss_model": config.sss_model,
        "summarization_model": config.summarization_model,
        "agent_backend": OPENCLAW_NATIVE_BACKEND,
        "docker_image": config.docker_image,
        "packages": {
            "tree-sitter": package_version("tree-sitter"),
            "tree-sitter-c": package_version("tree-sitter-c"),
        },
    }
    if getattr(config, "method", "") == "vulex":
        semantics = current_prompt_semantics()
        payload["prompt_variant"] = semantics["prompt_variant"]
        payload["harness"] = harness_audit_fields(semantics)
        payload["prompting"] = semantics
        payload["banks"] = bank_audit_fields(config)
    return payload


def current_prompt_semantics() -> dict[str, Any]:
    """Return the current VulEx prompt and harness semantics for manifest and checkpoint auditing."""
    cap_hash = harness_source_hash(build_cap_probe("<anchor>", "<cwe>"))
    sss_hash = harness_source_hash(build_sss_probe("<cwe>", "<payload>"))
    return {
        "prompt_variant": "seeded_review",
        "cap_prompt_variant": "seeded_review",
        "sss_prompt_variant": "seeded_review",
        "seed_policy": "anchors_or_snippets_as_search_seeds_not_output_scope",
        "cap_seed_surface": "listed_locations_file_function_names",
        "sss_seed_surface": "provided_suspect_snippets",
        "cap_seed_wording": "Use <anchors> as search seeds for a seeded <CWE> review, not as the output scope.",
        "sss_seed_wording": "Use the provided suspect snippets as style seeds for a seeded <CWE> review, not as the output scope.",
        "max_results_instruction": MEMORY_MAX_RESULTS_INSTRUCTION,
        "memory_restriction": MEMORY_PRIVACY_RESTRICTION,
        "memory_privacy_restriction_included": True,
        "tools_policy": MEMORY_ONLY_TOOLS,
        "cap_schema_tail": TAIL_SCHEMA,
        "sss_schema_tail": TAIL_SCHEMA,
        "harness_schema_variant": "memory_only_tail",
        "sss_selector_policy": "pairwise_fuzzy_repository_coverage",
        "harness_source_hashes": {
            "cap": cap_hash,
            "sss": sss_hash,
        },
    }


def harness_audit_fields(semantics: dict[str, Any]) -> dict[str, Any]:
    """Flatten current harness hashes into field names used by historical run documents."""
    hashes = semantics["harness_source_hashes"]
    return {
        "prompt_variant": semantics["prompt_variant"],
        "schema_variant": semantics["harness_schema_variant"],
        "cap_harness_hash": hashes["cap"],
        "sss_harness_hash": hashes["sss"],
        "tools_policy": semantics["tools_policy"],
    }


def bank_audit_fields(config: Any) -> dict[str, dict[str, str]]:
    """Record formal frozen-bank input paths and content hashes; omit unconfigured banks."""
    banks: dict[str, dict[str, str]] = {}
    cap_anchor_bank = getattr(config, "cap_anchor_bank", "")
    if cap_anchor_bank:
        path = pathlib.Path(cap_anchor_bank)
        banks["cap_anchor_bank"] = {"path": str(path), "sha256": sha256_file(path)}
        expected_sha = getattr(config, "cap_anchor_bank_sha256", "")
        if expected_sha:
            banks["cap_anchor_bank"]["expected_sha256"] = expected_sha
    sss_payload_bank = getattr(config, "sss_payload_bank", "")
    if sss_payload_bank:
        path = pathlib.Path(sss_payload_bank)
        banks["sss_payload_bank"] = {"path": str(path), "sha256": sha256_file(path)}
        expected_sha = getattr(config, "sss_payload_bank_sha256", "")
        if expected_sha:
            banks["sss_payload_bank"]["expected_sha256"] = expected_sha
    return banks


def write_manifest(
    out_dir: pathlib.Path,
    memory_path: pathlib.Path,
    records_path: pathlib.Path,
    prompt_rows: list[dict[str, Any]],
    config: Any,
    ordinary_memory_path: pathlib.Path | None = None,
    memory_composition: dict[str, Any] | None = None,
    cap_anchor_bank_path: pathlib.Path | None = None,
    tool_policy_audit: dict[str, Any] | None = None,
) -> None:
    """Write the complete experiment manifest with input hashes, git state, and LLM cache keys."""
    cache_keys = sorted({row["cache_key"] for row in prompt_rows})
    inputs = {
        "memory": {"path": str(memory_path), "sha256": sha256_file(memory_path)},
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "visibility_split": {
            "path": config.visibility_split,
            "sha256": sha256_file(pathlib.Path(config.visibility_split)),
        },
        "memory_split": {
            "path": config.memory_split,
            "sha256": sha256_file(pathlib.Path(config.memory_split)),
        },
    }
    if ordinary_memory_path is not None:
        inputs["ordinary_memory"] = {"path": str(ordinary_memory_path), "sha256": sha256_file(ordinary_memory_path)}
    if cap_anchor_bank_path is not None:
        inputs["cap_anchor_bank"] = {
            "path": str(cap_anchor_bank_path),
            "sha256": sha256_file(cap_anchor_bank_path),
        }
        expected_cap_sha = getattr(config, "cap_anchor_bank_sha256", "")
        if expected_cap_sha:
            inputs["cap_anchor_bank"]["expected_sha256"] = expected_cap_sha
    sss_payload_bank = getattr(config, "sss_payload_bank", "")
    if sss_payload_bank:
        sss_payload_bank_path = pathlib.Path(sss_payload_bank)
        inputs["sss_payload_bank"] = {
            "path": str(sss_payload_bank_path),
            "sha256": sha256_file(sss_payload_bank_path),
        }
        expected_bank_sha = getattr(config, "sss_payload_bank_sha256", "")
        if expected_bank_sha:
            inputs["sss_payload_bank"]["expected_sha256"] = expected_bank_sha
    payload = {
        "inputs": inputs,
        "git": git_state(),
        "llm_cache": {
            "directory": str(out_dir / "llm_cache"),
            "cache_keys": cache_keys,
            "count": len(cache_keys),
        },
        "config": asdict(config),
    }
    if getattr(config, "method", "") == "vulex":
        semantics = current_prompt_semantics()
        payload["prompt_variant"] = semantics["prompt_variant"]
        payload["harness"] = harness_audit_fields(semantics)
        payload["prompting"] = semantics
    if memory_composition is not None:
        payload["memory_composition"] = memory_composition
    if tool_policy_audit is not None:
        payload["tool_policy_audit"] = tool_policy_audit
    write_json(out_dir / "manifest.json", payload)


def input_fingerprint(
    config: RunConfig,
    memory_path: pathlib.Path,
    records_path: pathlib.Path,
    ordinary_memory_path: pathlib.Path | None,
) -> dict[str, Any]:
    """Generate the resume validation fingerprint covering inputs and output-affecting run configuration."""
    semantics = current_prompt_semantics()
    probe_semantics = {
        **semantics,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "semantic_version": CHECKPOINT_SEMANTIC_VERSION,
        "cap_harness_hash": semantics["harness_source_hashes"]["cap"],
        "sss_harness_hash": semantics["harness_source_hashes"]["sss"],
        "sss_payload_anchor_limit": config.sss_payload_anchor_limit,
        "sss_probe_anchor_mode": "none",
        "sss_payload_mode": "public_cwe_anchor_snippets",
        "openclaw_served_memory_policy": config.openclaw_served_memory_policy,
        "cap_probe_hash": semantics["harness_source_hashes"]["cap"],
        "sss_probe_hash": semantics["harness_source_hashes"]["sss"],
    }
    inputs = {
        "memory": {"path": str(memory_path), "sha256": sha256_file(memory_path)},
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "visibility_split": {
            "path": config.visibility_split,
            "sha256": sha256_file(pathlib.Path(config.visibility_split)),
        },
        "memory_split": {
            "path": config.memory_split,
            "sha256": sha256_file(pathlib.Path(config.memory_split)),
        },
    }
    if ordinary_memory_path is not None:
        inputs["ordinary_memory"] = {
            "path": str(ordinary_memory_path),
            "sha256": sha256_file(ordinary_memory_path),
        }
    if config.cwe_source:
        cwe_source_path = pathlib.Path(config.cwe_source)
        inputs["cwe_source"] = {
            "path": str(cwe_source_path),
            "sha256": sha256_file(cwe_source_path),
        }
    if config.cap_anchor_bank:
        cap_anchor_bank_path = pathlib.Path(config.cap_anchor_bank)
        inputs["cap_anchor_bank"] = {
            "path": str(cap_anchor_bank_path),
            "sha256": sha256_file(cap_anchor_bank_path),
        }
        if config.cap_anchor_bank_sha256:
            inputs["cap_anchor_bank"]["expected_sha256"] = config.cap_anchor_bank_sha256
    if config.sss_payload_bank:
        sss_payload_bank_path = pathlib.Path(config.sss_payload_bank)
        inputs["sss_payload_bank"] = {
            "path": str(sss_payload_bank_path),
            "sha256": sha256_file(sss_payload_bank_path),
        }
        if config.sss_payload_bank_sha256:
            inputs["sss_payload_bank"]["expected_sha256"] = config.sss_payload_bank_sha256
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "semantics": probe_semantics,
        "config": asdict(config),
        "inputs": inputs,
    }


def write_preflight_audit(out_dir: pathlib.Path, config: RunConfig, fingerprint: dict[str, Any]) -> None:
    """Write prompt, bank, and input fingerprints validated before the formal run."""
    write_json(
        out_dir / "preflight.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_variant": current_prompt_semantics()["prompt_variant"],
            "prompting": current_prompt_semantics(),
            "harness": harness_audit_fields(current_prompt_semantics()),
            "banks": bank_audit_fields(config),
            "fingerprint": fingerprint,
        },
    )


def checkpoint_path(out_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Return the path of the requested checkpoint file."""
    return out_dir / CHECKPOINT_DIRNAME / name


def write_checkpoint_meta(out_dir: pathlib.Path, fingerprint: dict[str, Any], resume: bool) -> None:
    """Write or validate checkpoint metadata; resume requires identical inputs."""
    path = checkpoint_path(out_dir, "meta.json")
    if resume:
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != fingerprint:
                raise ValueError(f"checkpoint fingerprint mismatch: {path}")
            return
        checkpoint_root = out_dir / CHECKPOINT_DIRNAME
        existing_files = list(checkpoint_root.glob("*")) if checkpoint_root.exists() else []
        if existing_files:
            raise ValueError(f"checkpoint meta missing but checkpoint files exist: {checkpoint_root}")
    write_json(
        path,
        {
            "fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def latest_probe_statuses(out_dir: pathlib.Path) -> dict[int, dict[str, Any]]:
    """Read the latest checkpoint status for each probe."""
    statuses: dict[int, dict[str, Any]] = {}
    for row in read_jsonl_rows(checkpoint_path(out_dir, "probe_status.jsonl")):
        statuses[int(row["probe_index"])] = row
    return statuses


def clear_final_outputs(out_dir: pathlib.Path) -> None:
    """Clear old formal results before a non-resume run to avoid reading stale summaries after failure."""
    for name in FINAL_OUTPUT_NAMES | PARTIAL_OUTPUT_NAMES:
        path = out_dir / name
        if path.exists():
            path.unlink()


def clear_checkpoint_outputs(out_dir: pathlib.Path) -> None:
    """Clear old checkpoint files before a non-resume run to prevent stale state from contaminating the experiment."""
    directory = out_dir / CHECKPOINT_DIRNAME
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
        else:
            raise ValueError(f"unexpected checkpoint subdirectory: {path}")


def load_successful_checkpoint(
    out_dir: pathlib.Path,
    probe_index: int,
    prompt_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    probe_target_rows: list[dict[str, Any]],
) -> list:
    """Load successful probe checkpoint artifacts into memory and return parsed triples."""
    path = out_dir / CHECKPOINT_DIRNAME / f"probe_{probe_index:04d}.json"
    if not path.exists():
        raise ValueError(f"checkpoint success probe {probe_index} missing artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompt_rows.extend(payload["prompts"])
    response_rows.append(payload["response"])
    parsed_row = payload["parsed"]
    parsed_rows.append(parsed_row)
    probe_rows.append(payload["probe"])
    probe_target_rows.append(payload["target"])
    return [ParsedTriple(**triple) for triple in parsed_row["triples"]]


def record_probe_success(
    out_dir: pathlib.Path,
    probe_index: int,
    phase: str,
    prompt_delta: list[dict[str, Any]],
    response_delta: dict[str, Any],
    parsed_delta: dict[str, Any],
    probe_delta: dict[str, Any],
    target_delta: dict[str, Any],
) -> None:
    """Write a successful probe as a single-file checkpoint with the status line persisted last."""
    write_json(
        out_dir / CHECKPOINT_DIRNAME / f"probe_{probe_index:04d}.json",
        {
            "probe_index": probe_index,
            "phase": phase,
            "prompts": prompt_delta,
            "response": response_delta,
            "parsed": parsed_delta,
            "probe": probe_delta,
            "target": target_delta,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    append_jsonl(
        checkpoint_path(out_dir, "probe_status.jsonl"),
        {
            "probe_index": probe_index,
            "phase": phase,
            "status": "success",
            "triple_count": len(parsed_delta["triples"]),
            "trace_dir": response_delta.get("trace_dir", ""),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def record_probe_failure(
    out_dir: pathlib.Path,
    probe_index: int,
    phase: str,
    error: str,
    trace_dir: str | None = None,
) -> None:
    """Record a failed probe status; failures do not enter successful checkpoint artifacts."""
    append_jsonl(
        checkpoint_path(out_dir, "probe_status.jsonl"),
        {
            "probe_index": probe_index,
            "phase": phase,
            "status": "failed",
            "error": error,
            "trace_dir": trace_dir or "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def write_partial_outputs(out_dir: pathlib.Path, statuses: dict[int, dict[str, Any]], total_probes: int) -> None:
    """Write partial status for an incomplete run and explicitly prohibit its use as formal metrics."""
    missing = [
        probe_index
        for probe_index in range(total_probes)
        if probe_index not in statuses or statuses[probe_index].get("status") != "success"
    ]
    status_rows = []
    for probe_index in missing:
        status_rows.append(
            statuses.get(
                probe_index,
                {"probe_index": probe_index, "status": "missing", "updated_at": datetime.now(timezone.utc).isoformat()},
            )
        )
    payload = {
        "complete": False,
        "total_probes": total_probes,
        "success_count": total_probes - len(missing),
        "failed_count": len(status_rows),
        "missing_or_failed_probe_indices": missing,
        "failed_probes": status_rows,
    }
    write_json(out_dir / "summary_partial.json", payload)
    write_json(out_dir / "failed_probes.json", status_rows)


def memory_composition(memory: list[MemoryRecord], ordinary_memory: list[OrdinaryMemoryRecord]) -> dict[str, Any]:
    """Summarize vulnerability and ordinary-code record counts in the mixed OpenClaw memory corpus."""
    return {
        "target_vulnerability": len(memory),
        "ordinary_code": len(ordinary_memory),
        "mixed_total": len(memory) + len(ordinary_memory),
        "ordinary_sampling": "random_linux_functions",
    }


def memory_search_manifest(config: RunConfig) -> dict[str, Any]:
    """Record OpenClaw memory-search configuration; an empty embedding model selects the OpenClaw default."""
    memory_search_mode = getattr(config, "openclaw_memory_search", "bm25_fts")
    embedding_api_base = getattr(config, "openclaw_embedding_api_base", "")
    embedding_model = getattr(config, "openclaw_embedding_model", "")
    uses_embedding = memory_search_mode in {"vector", "hybrid"}
    omitted_model = embedding_model == ""
    return {
        "mode": memory_search_mode,
        "embedding_base_url": embedding_api_base,
        "embedding_model_config_field": "omitted" if omitted_model else embedding_model,
        "expected_openclaw_default_embedding_model": (
            OPENCLAW_DEFAULT_OPENAI_EMBEDDING_MODEL if uses_embedding and omitted_model else ""
        ),
        "hybrid_uses_openclaw_default_weights": memory_search_mode == "hybrid",
    }


def audit_openclaw_tool_policy(openclaw_home: pathlib.Path) -> dict[str, Any]:
    """Scan OpenClaw session JSONL to verify that runtime calls only memory-only tools."""
    sessions_dir = openclaw_home / "agents" / "vulex" / "sessions"
    audit: dict[str, Any] = {
        "declared_policy": MEMORY_ONLY_TOOLS,
        "verified": False,
        "session_count": 0,
        "tool_call_count": 0,
        "tool_counts": {},
        "forbidden_tools": [],
        "memory_search_max_results_max": None,
        "memory_search_missing_max_results": 0,
    }
    if not sessions_dir.exists():
        audit["reason"] = "sessions_dir_missing"
        return audit
    tool_counts: Counter[str] = Counter()
    max_results_values: list[int] = []
    missing_max_results = 0
    session_files = []
    for path in sorted(sessions_dir.glob("probe_*.jsonl")):
        suffix = path.stem.removeprefix("probe_")
        if len(suffix) == 4 and suffix.isdigit():
            session_files.append(path)
            continue
        if "_attempt" in suffix:
            probe_id, attempt_id = suffix.split("_attempt", 1)
            if len(probe_id) == 4 and probe_id.isdigit() and len(attempt_id) == 2 and attempt_id.isdigit():
                session_files.append(path)
    for session_file in session_files:
        for line in session_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            message = row.get("message") or {}
            if message.get("role") != "assistant":
                continue
            for item in message.get("content") or []:
                if item.get("type") != "toolCall":
                    continue
                name = item.get("name")
                if not isinstance(name, str):
                    continue
                tool_counts[name] += 1
                if name != "memory_search":
                    continue
                arguments = item.get("arguments") or {}
                max_results = arguments.get("maxResults")
                if isinstance(max_results, int):
                    max_results_values.append(max_results)
                else:
                    missing_max_results += 1
    forbidden = sorted(set(tool_counts) - set(MEMORY_ONLY_TOOLS["allow"]))
    max_results_max = max(max_results_values) if max_results_values else None
    audit.update(
        {
            "session_count": len(session_files),
            "tool_call_count": sum(tool_counts.values()),
            "tool_counts": dict(sorted(tool_counts.items())),
            "forbidden_tools": forbidden,
            "memory_search_max_results_max": max_results_max,
            "memory_search_missing_max_results": missing_max_results,
            "verified": bool(session_files)
            and not forbidden
            and missing_max_results == 0
            and (max_results_max is None or max_results_max <= 4),
        }
    )
    if not audit["verified"]:
        audit["reason"] = "runtime_tool_policy_not_verified"
    return audit


def cwe_anchor_targets(
    public_records: list[ReposVulRecord],
) -> tuple[list[str], dict[str, list[str]], dict[tuple[str, str], ReposVulRecord], Counter]:
    """Build CWE-level probe targets by public CWE frequency while retaining every public anchor per CWE."""
    cwe_counts = Counter(record.cwe for record in public_records)
    anchor_counts_by_cwe: dict[str, Counter] = {}
    public_records_by_cwe_anchor: dict[tuple[str, str], ReposVulRecord] = {}
    for record in public_records:
        anchor_counts_by_cwe.setdefault(record.cwe, Counter())[record.anchor] += 1
        public_records_by_cwe_anchor.setdefault((record.cwe, record.anchor), record)

    cwe_order = sorted(cwe_counts, key=lambda cwe: (-cwe_counts[cwe], cwe))
    anchors_by_cwe = {
        cwe: sorted(anchor_counts_by_cwe[cwe], key=lambda anchor: (-anchor_counts_by_cwe[cwe][anchor], anchor))
        for cwe in cwe_order
    }
    return cwe_order, anchors_by_cwe, public_records_by_cwe_anchor, cwe_counts


def records_by_cwe_anchor_order(public_records: list[ReposVulRecord]) -> dict[str, list[ReposVulRecord]]:
    """Return public records for each CWE ordered by the default SSS anchor ordering."""
    cwe_counts = Counter(record.cwe for record in public_records)
    anchor_counts_by_cwe: dict[str, Counter] = {}
    public_records_by_cwe_anchor: dict[tuple[str, str], ReposVulRecord] = {}
    for record in public_records:
        anchor_counts_by_cwe.setdefault(record.cwe, Counter())[record.anchor] += 1
        public_records_by_cwe_anchor.setdefault((record.cwe, record.anchor), record)
    cwe_order = sorted(cwe_counts, key=lambda cwe: (-cwe_counts[cwe], cwe))
    return {
        cwe: [
            public_records_by_cwe_anchor[(cwe, anchor)]
            for anchor in sorted(anchor_counts_by_cwe[cwe], key=lambda anchor: (-anchor_counts_by_cwe[cwe][anchor], anchor))
        ]
        for cwe in cwe_order
    }


def load_sss_payload_bank(path: pathlib.Path) -> dict[str, Any]:
    """Read the frozen CodeBERT SSS payload bank and perform minimal schema validation."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"sss_payload_bank rows must be a list: {path}")
    by_cwe: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("cwe"):
            raise ValueError(f"invalid sss_payload_bank row in {path}")
        cwe = str(row["cwe"])
        if cwe in by_cwe:
            raise ValueError(f"duplicate CWE in sss_payload_bank: {cwe}")
        anchors = row.get("payload_anchors", [])
        record_ids = row.get("payload_source_record_ids", [])
        if not isinstance(anchors, list) or not all(isinstance(anchor, str) for anchor in anchors):
            raise ValueError(f"invalid payload_anchors for {cwe} in {path}")
        if record_ids and (not isinstance(record_ids, list) or not all(isinstance(record_id, str) for record_id in record_ids)):
            raise ValueError(f"invalid payload_source_record_ids for {cwe} in {path}")
        by_cwe[cwe] = row
    return {"payload": data, "rows_by_cwe": by_cwe}


def require_sss_payload_bank_sha256(path: pathlib.Path, expected_sha256: str) -> None:
    """Validate a frozen bank against an externally fixed SHA to prevent replacement of a valid bank."""
    if not expected_sha256:
        return
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"sss_payload_bank sha256 mismatch: got {actual_sha256}, expected {expected_sha256}"
        )


def require_cap_anchor_bank_sha256(path: pathlib.Path, expected_sha256: str) -> None:
    """Validate the CAP bank against an externally fixed SHA to prevent anchor-input replacement."""
    if not expected_sha256:
        return
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"cap_anchor_bank sha256 mismatch: got {actual_sha256}, expected {expected_sha256}")


FINECLS_SSS_SELECTOR_POLICY = "pairwise_fuzzy_repository_coverage"
FINECLS_SSS_POOLING = "fine_line_cls"


def expected_finecls_bank_selector(config: RunConfig) -> str:
    """Return the neutral selector name used by the released SSS bank."""
    return "stagedvulbert_cls_selector"


def require_sss_bank_matches_config(bank: dict[str, Any], config: RunConfig) -> None:
    """Validate that the frozen SSS bank matches the current fine-line CLS selector parameters."""
    payload = bank["payload"]
    if config.sss_codebert_pooling != "cls":
        raise ValueError("this SSS bank requires sss_codebert_pooling='cls'")
    expected = {
        "selector": expected_finecls_bank_selector(config),
        "selector_policy": FINECLS_SSS_SELECTOR_POLICY,
        "topk": config.sss_payload_anchor_limit,
        "prior_weight": float(config.sss_codebert_prior_weight),
        "repo_demand_count": config.sss_codebert_repo_demand_count,
        "content_window": config.sss_codebert_content_window,
        "chunk_step": config.sss_codebert_chunk_step,
        "pooling": FINECLS_SSS_POOLING,
    }
    for key in ("selector", "selector_policy"):
        if key not in payload:
            raise ValueError(f"sss_payload_bank missing top-level {key}")
        if payload[key] != expected[key]:
            raise ValueError(f"sss_payload_bank {key} mismatch: got {payload[key]!r}, expected {expected[key]!r}")
    required_row_keys = set(expected)
    for row in bank["rows_by_cwe"].values():
        missing = sorted(required_row_keys - set(row))
        if missing:
            raise ValueError(f"sss_payload_bank row for {row['cwe']} missing selector metadata: {missing}")
        for key, expected_value in expected.items():
            actual = float(row[key]) if key == "prior_weight" else row[key]
            if actual != expected_value:
                raise ValueError(
                    f"sss_payload_bank {key} mismatch for {row['cwe']}: "
                    f"got {row[key]!r}, expected {expected_value!r}"
                )


def replay_sss_payload_bank_row(
    cwe: str,
    records_for_cwe: list[ReposVulRecord],
    bank_rows_by_cwe: dict[str, dict[str, Any]],
) -> tuple[list[ReposVulRecord], dict[str, Any]]:
    """Replay SSS payload records by frozen-bank row without reusing the repository selector."""
    if cwe not in bank_rows_by_cwe:
        raise ValueError(f"sss_payload_bank missing CWE {cwe}")
    row = bank_rows_by_cwe[cwe]
    records_by_id = {record.record_id: record for record in records_for_cwe}
    records_by_anchor: dict[str, list[ReposVulRecord]] = {}
    for record in records_for_cwe:
        records_by_anchor.setdefault(record.anchor, []).append(record)
    for required_key in ("payload_anchor_count", "payload_anchors", "payload_source_record_ids"):
        if required_key not in row:
            raise ValueError(f"sss_payload_bank row for {cwe} missing {required_key}")
    payload_anchors = [str(value) for value in row.get("payload_anchors", [])]
    source_ids = [str(record_id) for record_id in row.get("payload_source_record_ids", [])]
    expected_count = int(row["payload_anchor_count"])
    effective_topk = min(int(row["topk"]), len({record.anchor for record in records_for_cwe}))
    # A frozen selector may filter short functions before choosing payloads, so
    # its count can be below the public-pool top-k while remaining valid.
    if expected_count > effective_topk:
        raise ValueError(f"sss_payload_bank payload_anchor_count exceeds top-k for {cwe}")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"sss_payload_bank duplicate payload_source_record_ids for {cwe}")
    if len(payload_anchors) != len(set(payload_anchors)):
        raise ValueError(f"sss_payload_bank duplicate payload_anchors for {cwe}")
    if len(source_ids) != len(payload_anchors):
        raise ValueError(f"sss_payload_bank payload_source_record_ids length does not match payload_anchors for {cwe}")
    if source_ids and len(source_ids) != expected_count:
        raise ValueError(f"sss_payload_bank payload_anchor_count mismatch for {cwe}")
    if not source_ids and payload_anchors and len(payload_anchors) != expected_count:
        raise ValueError(f"sss_payload_bank payload_anchor_count mismatch for {cwe}")
    selected: list[ReposVulRecord] = []
    used_ids: set[str] = set()
    if source_ids:
        for index, record_id in enumerate(source_ids):
            if record_id not in records_by_id:
                raise ValueError(f"sss_payload_bank record {record_id} is not public for CWE {cwe}")
            if payload_anchors and records_by_id[record_id].anchor != payload_anchors[index]:
                raise ValueError(f"sss_payload_bank payload_anchors do not match payload_source_record_ids for {cwe}")
            selected.append(records_by_id[record_id])
            used_ids.add(record_id)
    else:
        for anchor in payload_anchors:
            matches = [record for record in records_by_anchor.get(anchor, []) if record.record_id not in used_ids]
            if not matches:
                raise ValueError(f"sss_payload_bank anchor {anchor!r} is not public for CWE {cwe}")
            selected.append(matches[0])
            used_ids.add(matches[0].record_id)
    if not selected:
        raise ValueError(f"sss_payload_bank row for {cwe} selects no payload records")
    selector_meta = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "probe_index",
            "phase",
            "cwe",
            "payload_anchors",
            "payload_source_record_ids",
            "payload_anchor_count",
            "payload_anchor_source_files",
            "public_record_count",
            "public_unique_anchor_count",
        }
    }
    selector_meta["selector_policy"] = row["selector_policy"]
    selector_meta["frozen_payload_bank"] = True
    return selected, selector_meta


def multi_anchor_text(anchors: list[str]) -> str:
    """Join all public anchors for one CWE into the existing prompt template's `<a>` placeholder."""
    return "; ".join(anchors)


def probe_target_row(
    probe_index: int,
    phase: str,
    cwe: str,
    anchors: list[str],
    public_records_by_cwe_anchor: dict[tuple[str, str], ReposVulRecord],
    cwe_counts: Counter,
    anchor_source: str,
    sss_payload_anchors: list[str] | None = None,
    sss_payload_anchor_limit: int = DEFAULT_SSS_PAYLOAD_ANCHOR_LIMIT,
) -> dict[str, Any]:
    """Generate a CWE-level probe-target audit row recording all anchors exposed by this substitution."""
    source_record_ids = [
        record.record_id
        for anchor in anchors
        if (record := public_records_by_cwe_anchor.get((cwe, anchor))) is not None
    ]
    anchor_text = multi_anchor_text(anchors)
    row = {
        "probe_index": probe_index,
        "phase": phase,
        "cwe": cwe,
        "anchors": anchors,
        "anchor_count": len(anchors),
        "source_record_ids": source_record_ids,
        "public_cwe_count": cwe_counts[cwe],
        "anchor_text": anchor_text,
        "anchor_text_chars": len(anchor_text),
        "anchor_source": anchor_source,
    }
    if sss_payload_anchors is not None:
        row["sss_payload_anchor_limit"] = sss_payload_anchor_limit
        row["sss_payload_anchors"] = sss_payload_anchors
        row["sss_payload_anchor_count"] = len(sss_payload_anchors)
        row["sss_payload_source_record_ids"] = [
            public_records_by_cwe_anchor[(cwe, anchor)].record_id for anchor in sss_payload_anchors
        ]
    return row


def prepare_openclaw_native_memory(
    out_dir: pathlib.Path,
    memory: list[MemoryRecord],
    config: RunConfig,
    ordinary_memory: list[OrdinaryMemoryRecord] | None = None,
    *,
    memory_path: pathlib.Path,
    ordinary_memory_path: pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the OpenClaw corpus according to served policy and retain scoring-memory input audit data."""
    ordinary_memory = ordinary_memory or []
    if ordinary_memory and ordinary_memory_path is None:
        raise ValueError("ordinary_memory_path is required when ordinary_memory is provided")
    served_policy = getattr(config, "openclaw_served_memory_policy", "configured")
    if served_policy not in {"configured", "empty"}:
        raise ValueError("openclaw_served_memory_policy must be one of: configured, empty")
    served_memory = [] if served_policy == "empty" else memory
    served_ordinary_memory = [] if served_policy == "empty" else ordinary_memory
    memory_root = out_dir / "openclaw_memory"
    corpus_dir = memory_root / "vulex_records"
    openclaw_home = memory_root / "openclaw_home"
    files = write_openclaw_memory_corpus(
        served_memory,
        corpus_dir,
        ordinary_memory=served_ordinary_memory,
    )
    memory_search_mode = getattr(config, "openclaw_memory_search", "bm25_fts")
    embedding_api_base = getattr(config, "openclaw_embedding_api_base", "")
    embedding_model = getattr(config, "openclaw_embedding_model", "")
    write_openclaw_config(
        openclaw_home,
        config.agent_model,
        config.openai_api_base or os.environ.get("OPENAI_BASE_URL", ""),
        OPENCLAW_MEMORY_CORPUS_CONTAINER_PATH,
        temperature=getattr(config, "agent_temperature", None),
        memory_search_mode=memory_search_mode,
        embedding_api_base=embedding_api_base,
        embedding_model=embedding_model,
    )
    index_result = run_openclaw_memory_index(
        openclaw_home,
        corpus_dir,
        config.docker_image,
        config.openai_api_base or os.environ.get("OPENAI_BASE_URL", ""),
        memory_search_mode=memory_search_mode,
        embedding_api_base=embedding_api_base,
        timeout=config.agent_timeout,
    )
    inputs = {"memory": {"path": str(memory_path), "sha256": sha256_file(memory_path)}}
    if ordinary_memory_path is not None:
        inputs["ordinary_memory"] = {
            "path": str(ordinary_memory_path),
            "sha256": sha256_file(ordinary_memory_path),
        }
    manifest = {
        "inputs": inputs,
        "served_memory_policy": served_policy,
        "corpus_dir": str(corpus_dir),
        "container_corpus_dir": OPENCLAW_MEMORY_CORPUS_CONTAINER_PATH,
        "file_count": len(files),
        "target_vulnerability_record_ids": [record.record_id for record in served_memory],
        "ordinary_memory_record_ids": [record.record_id for record in served_ordinary_memory],
        "memory_composition": memory_composition(served_memory, served_ordinary_memory),
        "memory_search": memory_search_manifest(config),
        "index": index_result,
    }
    if served_policy == "empty":
        manifest["scoring_memory_composition"] = memory_composition(memory, ordinary_memory)
    write_json(memory_root / "manifest.json", manifest)
    return openclaw_home, corpus_dir


def append_agent_artifacts(
    config: RunConfig,
    probe,
    probe_index: int,
    message: str,
    source_hash: str,
    agent_result: AgentRunResult,
    prompt_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
) -> list:
    """Write one probe's OpenClaw native artifact row and return parsed triples."""
    response = agent_result.response
    triples = parse_response(response)
    prompt_rows.append(
        {
            "purpose": "agent_harness",
            "model": config.agent_model,
            "cache_key": cache_key("agent_harness", config.agent_model, message, source_hash),
            "source_hash": source_hash,
            "probe_index": probe_index,
            "anchor": probe.anchor,
            "cwe": probe.cwe,
            "prompt": message,
        }
    )
    response_row = {
        "probe_index": probe_index,
        "model": config.agent_model,
        "raw_response": response,
        "backend": agent_result.backend,
        "trace_dir": agent_result.trace_dir,
        "exit_code": agent_result.exit_code,
        "timed_out": agent_result.timed_out,
    }
    if agent_result.final_prompt:
        response_row["openclaw_final_prompt"] = agent_result.final_prompt
        response_row["openclaw_final_prompt_matches_input"] = agent_result.final_prompt == message
    if agent_result.tool_summary:
        response_row["tool_summary"] = agent_result.tool_summary
    response_rows.append(response_row)
    parsed_rows.append(
        {
            "probe_index": probe_index,
            "triples": [asdict(triple) for triple in triples],
        }
    )
    return triples


def write_static_outputs(
    out_dir: pathlib.Path,
    config: RunConfig,
    anchor_set: list[str],
    cwe_set: list[str],
    pairs: list[tuple[str, str]],
    public_records_by_pair: dict[tuple[str, str], ReposVulRecord],
    selected_sss_payload_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Write static audit artifacts independent of probe success or failure."""
    write_json(out_dir / "config.json", asdict(config))
    write_json(out_dir / "anchor_set.json", anchor_set)
    write_json(out_dir / "cwe_set.json", cwe_set)
    if selected_sss_payload_rows is not None:
        write_json(out_dir / "selected_sss_payload_anchors.json", selected_sss_payload_rows)
    write_json(
        out_dir / "anchor_cwe_pairs.json",
        [
            {
                "pair_index": pair_index,
                "anchor": anchor,
                "cwe": cwe,
                "source_record_id": public_records_by_pair[(anchor, cwe)].record_id,
            }
            for pair_index, (anchor, cwe) in enumerate(pairs)
        ],
    )


def materialize_successful_outputs(
    out_dir: pathlib.Path,
    memory: list[MemoryRecord],
    memory_path: pathlib.Path,
    records_path: pathlib.Path,
    prompt_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    probe_target_rows: list[dict[str, Any]],
    parsed_lists: list[list],
    config: RunConfig,
    ordinary_memory: list[OrdinaryMemoryRecord],
    ordinary_memory_path: pathlib.Path | None,
    tool_policy_audit: dict[str, Any],
) -> dict[str, float]:
    """Write formal artifacts and scoring results after all probes succeed."""
    for name in PARTIAL_OUTPUT_NAMES:
        path = out_dir / name
        if path.exists():
            path.unlink()
    phases = [row["phase"] for row in probe_rows]
    summary = compute_native_agent_metrics(memory, parsed_lists, phases)
    write_jsonl(out_dir / "probes.jsonl", probe_rows)
    write_json(out_dir / "probe_targets.json", probe_target_rows)
    write_jsonl(out_dir / "llm_prompts.jsonl", prompt_rows)
    write_jsonl(out_dir / "responses.jsonl", response_rows)
    write_jsonl(out_dir / "parsed.jsonl", parsed_rows)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "versions.json", versions(config))
    write_json(out_dir / "tool_policy_audit.json", tool_policy_audit)
    write_manifest(
        out_dir,
        memory_path,
        records_path,
        prompt_rows,
        config,
        ordinary_memory_path=ordinary_memory_path,
        memory_composition=memory_composition(memory, ordinary_memory),
        cap_anchor_bank_path=pathlib.Path(config.cap_anchor_bank) if config.cap_anchor_bank else None,
        tool_policy_audit=tool_policy_audit,
    )
    from experiments.vulex.score_run import score_payload

    write_json(out_dir / "summary_scored.json", score_payload(out_dir, memory_path=memory_path))
    return summary


def total_expected_probe_count(config: RunConfig, cwe_set: list[str]) -> int:
    """Compute the total probes this run should execute from the deterministic CAP/SSS budget."""
    cap_limit = 0 if config.skip_cap_probes else config.cap_budget or config.budget
    cap_count = min(config.budget, cap_limit, len(cwe_set))
    remaining = config.budget - cap_count
    sss_count = min(remaining, len(cwe_set)) if remaining > 0 else 0
    return cap_count + sss_count


def cap_bank_anchors_for_cwe(cap_anchor_bank, cwe: str) -> tuple[list[str], str]:
    """Return the final CAP-bank anchors for the current CWE."""
    if cap_anchor_bank is None:
        raise ValueError("CAP anchor bank is required for CAP probes")
    if cwe not in cap_anchor_bank:
        raise ValueError(f"CAP anchor bank missing CWE {cwe}")
    return cap_anchor_bank[cwe], "cwe_prior_repository_coverage_cap_anchor_bank"


def validate_cap_anchor_bank_for_cwes(cap_anchor_bank, cwes: list[str]) -> None:
    """Before preparing OpenClaw, validate that the per-CWE CAP bank covers this run's CAP queries."""
    if cap_anchor_bank is None:
        raise ValueError("CAP anchor bank is required for CAP probes")
    missing = [cwe for cwe in cwes if cwe not in cap_anchor_bank]
    if missing:
        raise ValueError(f"CAP anchor bank missing CWE {missing[0]}")


def run_probe_with_checkpoint(
    config: RunConfig,
    out_dir: pathlib.Path,
    probe,
    probe_index: int,
    phase: str,
    message: str,
    source_hash: str,
    target_row: dict[str, Any],
    probe_row: dict[str, Any],
    openclaw_home: pathlib.Path,
    openclaw_corpus_dir: pathlib.Path,
    prompt_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    probe_target_rows: list[dict[str, Any]],
    checkpoint_prompt_start: int | None = None,
) -> list | None:
    """Run one probe and write its checkpoint; return None on failure or parsed triples on success."""
    agent_result = None
    local_prompt_start = len(prompt_rows)
    local_response_start = len(response_rows)
    local_parsed_start = len(parsed_rows)
    try:
        agent_result = run_configured_agent(config, out_dir, probe_index, message, openclaw_home, openclaw_corpus_dir)
        require_agent_success(agent_result, probe_index)
    except ProbeExecutionError as exc:
        prompt_rows[local_prompt_start:] = []
        response_rows[local_response_start:] = []
        parsed_rows[local_parsed_start:] = []
        trace_dir = exc.agent_result.trace_dir if exc.agent_result is not None else ""
        record_probe_failure(out_dir, probe_index, phase, str(exc), trace_dir=trace_dir)
        return None
    except (RuntimeError, TimeoutError, ConnectionError) as exc:
        prompt_rows[local_prompt_start:] = []
        response_rows[local_response_start:] = []
        parsed_rows[local_parsed_start:] = []
        record_probe_failure(out_dir, probe_index, phase, str(exc))
        return None
    else:
        triples = append_agent_artifacts(
            config,
            probe,
            probe_index,
            message,
            source_hash,
            agent_result,
            prompt_rows,
            response_rows,
            parsed_rows,
        )
    probe_target_rows.append(target_row)
    probe_rows.append(probe_row)
    record_probe_success(
        out_dir,
        probe_index,
        phase,
        prompt_rows[checkpoint_prompt_start if checkpoint_prompt_start is not None else local_prompt_start :],
        response_rows[-1],
        parsed_rows[-1],
        probe_row,
        target_row,
    )
    return triples


def run_single_experiment(
    config: RunConfig,
    memory_path: pathlib.Path,
    records_path: pathlib.Path,
    out_dir: pathlib.Path,
    llm=None,
    *,
    resume: bool = False,
) -> dict[str, float]:
    """Execute one formal experiment; continue after probe failures and rely on checkpoints for resumption."""
    previous_openai_base_url = os.environ.get("OPENAI_BASE_URL")
    if config.openai_api_base:
        os.environ["OPENAI_BASE_URL"] = config.openai_api_base
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not resume:
            clear_final_outputs(out_dir)
            clear_checkpoint_outputs(out_dir)
        memory = read_memory_jsonl(memory_path)
        ordinary_memory_path = pathlib.Path(config.ordinary_memory) if config.ordinary_memory else None
        ordinary_memory = read_ordinary_memory_jsonl(ordinary_memory_path) if ordinary_memory_path is not None else []
        records = read_reposvul_jsonl(records_path)
        ensure_linux_only(memory, records, config.project)
        if len(memory) != config.memory_size:
            raise ValueError(f"memory_size config {config.memory_size} does not match loaded records {len(memory)}")
        duplicate_records = duplicate_record_ids([record.record_id for record in records])
        if duplicate_records:
            raise ValueError(f"duplicate records record_id: {duplicate_records[:5]}")
        records_by_id = {record.record_id: record for record in records}
        visibility_split = read_split_ids(pathlib.Path(config.visibility_split))
        memory_split = read_split_ids(pathlib.Path(config.memory_split))
        validate_memory_source_alignment(memory, records_by_id, memory_split)

        public_records = records_from_ids(records_by_id, visibility_split["public_record_ids"], "public visibility")
        if not public_records:
            raise ValueError("at least one public visibility record is required to build attacker-visible anchors")
        bad_pairs = [
            record.record_id
            for record in public_records
            if not record.anchor or not record.anchor.strip() or not record.cwe or not record.cwe.strip()
        ]
        if bad_pairs:
            raise ValueError(f"public anchor-CWE pair has empty anchor or CWE in records: {bad_pairs[:5]}")
        anchor_counts = Counter(record.anchor for record in public_records)
        pair_counts = Counter((record.anchor, record.cwe) for record in public_records)
        pairs = sorted(
            pair_counts,
            key=lambda pair: (-anchor_counts[pair[0]], pair[0], -pair_counts[pair], pair[1]),
        )
        anchor_set = sorted(anchor_counts, key=lambda anchor: (-anchor_counts[anchor], anchor))
        cwe_set, anchors_by_cwe, public_records_by_cwe_anchor, cwe_counts = cwe_anchor_targets(public_records)
        if config.cwe_source:
            requested_cwes = read_cwe_source(pathlib.Path(config.cwe_source))
            missing_cwes = [cwe for cwe in requested_cwes if cwe not in anchors_by_cwe]
            if missing_cwes:
                raise ValueError(f"cwe_source contains CWE without public anchors: {missing_cwes[:5]}")
            cwe_set = requested_cwes
        # Frozen SSS banks identify exact public record_ids.  Preserve duplicate
        # anchors in that path so replay can recover the bank-selected record,
        # while retaining the historical one-record-per-anchor order otherwise.
        if config.sss_payload_bank:
            public_records_by_cwe = {
                cwe: sorted(
                    [record for record in public_records if record.cwe == cwe],
                    key=lambda record: (record.anchor, record.record_id),
                )
                for cwe in cwe_set
            }
        else:
            public_records_by_cwe = records_by_cwe_anchor_order(public_records)
        sss_payload_bank_rows_by_cwe: dict[str, dict[str, Any]] = {}
        if config.sss_payload_bank:
            sss_payload_bank_path = pathlib.Path(config.sss_payload_bank)
            require_sss_payload_bank_sha256(sss_payload_bank_path, config.sss_payload_bank_sha256)
            sss_payload_bank = load_sss_payload_bank(sss_payload_bank_path)
            require_sss_bank_matches_config(sss_payload_bank, config)
            sss_payload_bank_rows_by_cwe = sss_payload_bank["rows_by_cwe"]
        cap_limit = 0 if config.skip_cap_probes else config.cap_budget or config.budget
        cap_cwes = cwe_set[: min(config.budget, cap_limit, len(cwe_set))]
        if cap_cwes:
            cap_anchor_bank_path = pathlib.Path(config.cap_anchor_bank) if config.cap_anchor_bank else None
            if cap_anchor_bank_path is not None:
                require_cap_anchor_bank_sha256(cap_anchor_bank_path, config.cap_anchor_bank_sha256)
            cap_anchor_bank = read_cap_anchor_bank_spec(cap_anchor_bank_path) if cap_anchor_bank_path else None
            validate_cap_anchor_bank_for_cwes(cap_anchor_bank, cap_cwes)
        else:
            cap_anchor_bank = None
            config = replace(config, cap_anchor_bank="")
        remaining_after_cap = config.budget - len(cap_cwes)
        if remaining_after_cap > 0 and not config.sss_payload_bank:
            raise ValueError("sss_payload_bank is required when the run enters the SSS phase")
        public_records_by_pair: dict[tuple[str, str], ReposVulRecord] = {}
        for record in public_records:
            public_records_by_pair.setdefault((record.anchor, record.cwe), record)

        fingerprint = input_fingerprint(
            config,
            memory_path,
            records_path,
            ordinary_memory_path,
        )
        if not resume:
            write_preflight_audit(out_dir, config, fingerprint)
        write_checkpoint_meta(out_dir, fingerprint, resume)
        llm = llm or CachedLLMClient(out_dir / "llm_cache")
        openclaw_home, openclaw_corpus_dir = prepare_openclaw_native_memory(
            out_dir,
            memory,
            config,
            ordinary_memory=ordinary_memory,
            memory_path=memory_path,
            ordinary_memory_path=ordinary_memory_path,
        )
        if cap_anchor_bank is not None:
            cap_anchor_payload = {
                "path": config.cap_anchor_bank,
                "cwe_count": len(cap_anchor_bank),
                "anchor_count": sum(len(anchors) for anchors in cap_anchor_bank.values()),
                "anchors_by_cwe": cap_anchor_bank,
                "per_cwe_anchor_counts": {
                    cwe: len(anchors) for cwe, anchors in sorted(cap_anchor_bank.items())
                },
            }
            write_json(
                out_dir / "cap_anchor_bank.json",
                cap_anchor_payload,
            )

        response_rows: list[dict[str, Any]] = []
        parsed_rows: list[dict[str, Any]] = []
        prompt_rows: list[dict[str, Any]] = []
        probe_rows: list[dict[str, Any]] = []
        probe_target_rows: list[dict[str, Any]] = []
        selected_sss_payload_rows: list[dict[str, Any]] = []
        parsed_lists = []
        statuses = latest_probe_statuses(out_dir)
        probe_index = 0
        for cwe in tqdm(cap_cwes, desc="CAP probes", total=len(cap_cwes), unit="probe"):
            bank_anchors, anchor_source = cap_bank_anchors_for_cwe(cap_anchor_bank, cwe)
            anchors = bank_anchors
            anchor_text = multi_anchor_text(anchors)
            probe = build_cap_probe(anchor_text, cwe)
            message = build_harness_prompt(probe)
            source_hash = harness_source_hash(probe)
            target_row = probe_target_row(
                probe_index,
                "cap",
                cwe,
                anchors,
                public_records_by_cwe_anchor,
                cwe_counts,
                anchor_source=anchor_source,
            )
            probe_row = asdict(probe) | {"probe_index": probe_index}
            checkpoint_file = out_dir / CHECKPOINT_DIRNAME / f"probe_{probe_index:04d}.json"
            if resume and statuses.get(probe_index, {}).get("status") == "success" and checkpoint_file.exists():
                triples = load_successful_checkpoint(
                    out_dir,
                    probe_index,
                    prompt_rows,
                    response_rows,
                    parsed_rows,
                    probe_rows,
                    probe_target_rows,
                )
                parsed_lists.append(triples)
                probe_index += 1
                continue
            triples = run_probe_with_checkpoint(
                config,
                out_dir,
                probe,
                probe_index,
                "cap",
                message,
                source_hash,
                target_row,
                probe_row,
                openclaw_home,
                openclaw_corpus_dir,
                prompt_rows,
                response_rows,
                parsed_rows,
                probe_rows,
                probe_target_rows,
            )
            if triples is not None:
                parsed_lists.append(triples)
            probe_index += 1

        remaining = config.budget - probe_index
        if remaining > 0:
            sss_cwes = cwe_set[: min(remaining, len(cwe_set))]
            for cwe in tqdm(sss_cwes, desc="SSS probes", total=len(sss_cwes), unit="probe"):
                if cwe == "CWE-20" and config.sss_payload_bank:
                    print(
                        f"SSS_REPLAY_DEBUG cwe={cwe} public_pool={len(public_records_by_cwe[cwe])} "
                        f"linux-668={any(record.record_id == 'linux-668' for record in public_records_by_cwe[cwe])}",
                        flush=True,
                    )
                payload_records, selector_meta = replay_sss_payload_bank_row(
                    cwe,
                    public_records_by_cwe[cwe],
                    sss_payload_bank_rows_by_cwe,
                )
                payload_anchors = [record.anchor for record in payload_records]
                selected_sss_payload_rows.append(
                    {
                        "probe_index": probe_index,
                        "phase": "sss",
                        "cwe": cwe,
                        "sss_anchor_selection": config.sss_anchor_selection,
                        "payload_anchors": payload_anchors,
                        "payload_source_record_ids": [record.record_id for record in payload_records],
                        "payload_anchor_count": len(payload_records),
                        "payload_anchor_source_files": [record.file_path for record in payload_records],
                        "public_record_count": len(public_records_by_cwe[cwe]),
                        "public_unique_anchor_count": len({record.anchor for record in public_records_by_cwe[cwe]}),
                        **selector_meta,
                    }
                )
                checkpoint_file = out_dir / CHECKPOINT_DIRNAME / f"probe_{probe_index:04d}.json"
                if resume and statuses.get(probe_index, {}).get("status") == "success" and checkpoint_file.exists():
                    triples = load_successful_checkpoint(
                        out_dir,
                        probe_index,
                        prompt_rows,
                        response_rows,
                        parsed_rows,
                        probe_rows,
                        probe_target_rows,
                    )
                    parsed_lists.append(triples)
                    probe_index += 1
                    continue
                anchor_payloads: list[tuple[str, str]] = []
                sss_prompt_start = len(prompt_rows)
                sss_failed = False
                for source in payload_records:
                    sss_prompt = build_sss_prompt(source, cwe)
                    sss_hash = sss_source_hash(source, cwe)
                    try:
                        single_payload = generate_sss_payload(source, cwe, config.sss_model, llm)
                    except (RuntimeError, TimeoutError, ConnectionError, OSError) as exc:
                        prompt_rows[sss_prompt_start:] = []
                        record_probe_failure(out_dir, probe_index, "sss", str(exc))
                        probe_index += 1
                        sss_failed = True
                        break
                    prompt_rows.append(
                        {
                            "purpose": "sss",
                            "model": config.sss_model,
                            "cache_key": cache_key("sss", config.sss_model, sss_prompt, sss_hash),
                            "source_hash": sss_hash,
                            "probe_index": probe_index,
                            "anchor": source.anchor,
                            "source_record_id": source.record_id,
                            "cwe": cwe,
                            "prompt": sss_prompt,
                        }
                    )
                    anchor_payloads.append((source.anchor, single_payload))
                if sss_failed:
                    continue
                payload = format_multi_anchor_sss_payload(anchor_payloads)
                sss_probe_anchors: list[str] = []
                probe = build_sss_probe(cwe, payload)
                message = build_harness_prompt(probe)
                source_hash = harness_source_hash(probe)
                target_row = probe_target_row(
                    probe_index,
                    "sss",
                    cwe,
                    sss_probe_anchors,
                    public_records_by_cwe_anchor,
                    cwe_counts,
                    sss_payload_anchors=payload_anchors,
                    sss_payload_anchor_limit=config.sss_payload_anchor_limit,
                    anchor_source="none",
                )
                target_row["sss_anchor_selection"] = config.sss_anchor_selection
                target_row.update(selector_meta)
                probe_row = asdict(probe) | {"probe_index": probe_index}
                triples = run_probe_with_checkpoint(
                    config,
                    out_dir,
                    probe,
                    probe_index,
                    "sss",
                    message,
                    source_hash,
                    target_row,
                    probe_row,
                    openclaw_home,
                    openclaw_corpus_dir,
                    prompt_rows,
                    response_rows,
                    parsed_rows,
                    probe_rows,
                    probe_target_rows,
                    checkpoint_prompt_start=sss_prompt_start,
                )
                if triples is not None:
                    parsed_lists.append(triples)
                else:
                    del prompt_rows[sss_prompt_start : sss_prompt_start + len(anchor_payloads)]
                probe_index += 1

        expected_probe_count = total_expected_probe_count(config, cwe_set)
        statuses = latest_probe_statuses(out_dir)
        incomplete = [
            probe_id
            for probe_id in range(expected_probe_count)
            if statuses.get(probe_id, {}).get("status") != "success"
        ]
        if incomplete:
            for name in (FINAL_OUTPUT_NAMES - {"preflight.json"}) | PARTIAL_OUTPUT_NAMES:
                path = out_dir / name
                if path.exists():
                    path.unlink()
            write_partial_outputs(out_dir, statuses, expected_probe_count)
            raise IncompleteExperimentError(
                f"incomplete VulEx experiment: {len(incomplete)} probe(s) failed or missing; "
                f"see {out_dir / 'failed_probes.json'}"
            )
        write_static_outputs(
            out_dir,
            config,
            anchor_set,
            cwe_set,
            pairs,
            public_records_by_pair,
            selected_sss_payload_rows=selected_sss_payload_rows,
        )
        tool_policy_audit = audit_openclaw_tool_policy(openclaw_home)
        return materialize_successful_outputs(
            out_dir,
            memory,
            memory_path,
            records_path,
            prompt_rows,
            response_rows,
            parsed_rows,
            probe_rows,
            probe_target_rows,
            parsed_lists,
            config,
            ordinary_memory,
            ordinary_memory_path,
            tool_policy_audit,
        )
    finally:
        if config.openai_api_base:
            if previous_openai_base_url is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = previous_openai_base_url


def build_config(args: argparse.Namespace) -> RunConfig:
    """Convert CLI arguments into a validated RunConfig while preserving an optional empty repo_worktree."""
    repo_worktree = normalize_optional_repo_worktree(args.repo_worktree)
    if args.sss_payload_anchor_limit is not None:
        sss_payload_anchor_limit = args.sss_payload_anchor_limit
    elif args.sss_anchor_selection == "codebert_repo_mean":
        sss_payload_anchor_limit = 5
    else:
        sss_payload_anchor_limit = DEFAULT_SSS_PAYLOAD_ANCHOR_LIMIT
    return RunConfig(
        project=args.project,
        method=args.method,
        memory_size=args.memory_size,
        budget=args.budget,
        seed=args.seed,
        agent_model=args.agent_model,
        sss_model=args.sss_model,
        summarization_model=args.summarization_model,
        visibility_split=str(args.visibility_split),
        memory_split=str(args.memory_split),
        cwe_source=str(args.cwe_source) if args.cwe_source else "",
        cap_budget=args.cap_budget,
        skip_cap_probes=args.skip_cap_probes,
        sss_payload_anchor_limit=sss_payload_anchor_limit,
        repo_worktree=str(repo_worktree) if repo_worktree else "",
        ordinary_memory=str(args.ordinary_memory) if args.ordinary_memory else "",
        cap_anchor_bank=str(args.cap_anchor_bank) if args.cap_anchor_bank else "",
        cap_anchor_bank_sha256=args.cap_anchor_bank_sha256,
        sss_anchor_selection=args.sss_anchor_selection,
        sss_codebert_model=str(args.sss_codebert_model) if args.sss_codebert_model else "",
        sss_codebert_cache_dir=str(args.sss_codebert_cache_dir) if args.sss_codebert_cache_dir else "",
        sss_codebert_repo_worktree=str(args.sss_codebert_repo_worktree) if args.sss_codebert_repo_worktree else "",
        sss_codebert_prior_weight=args.sss_codebert_prior_weight,
        sss_codebert_repo_demand_count=args.sss_codebert_repo_demand_count,
        sss_codebert_content_window=args.sss_codebert_content_window,
        sss_codebert_chunk_step=args.sss_codebert_chunk_step,
        sss_codebert_pooling=args.sss_codebert_pooling,
        sss_payload_bank=str(args.sss_payload_bank) if args.sss_payload_bank else "",
        sss_payload_bank_sha256=args.sss_payload_bank_sha256,
        openai_api_base=args.openai_api_base,
        openclaw_served_memory_policy=args.openclaw_served_memory_policy,
        openclaw_memory_search=args.openclaw_memory_search,
        openclaw_embedding_api_base=args.openclaw_embedding_api_base,
        openclaw_embedding_model=args.openclaw_embedding_model,
        docker_image=args.docker_image,
        agent_timeout=args.agent_timeout,
    )


def main(argv: list[str] | None = None, llm=None) -> dict[str, float]:
    """CLI entry point for one OpenClaw native-memory-only VulEx Linux configuration."""
    parser = argparse.ArgumentParser(description="Run one OpenClaw native-memory VulEx linux configuration.")
    parser.add_argument("--memory", required=True, type=pathlib.Path)
    parser.add_argument("--ordinary-memory", default=None, type=pathlib.Path)
    parser.add_argument("--records", required=True, type=pathlib.Path)
    parser.add_argument("--project", default=LINUX_PROJECT)
    parser.add_argument("--method", default="vulex")
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--sss-model", required=True)
    parser.add_argument("--summarization-model", default="")
    parser.add_argument("--memory-size", type=int)
    parser.add_argument("--visibility-split", required=True, type=pathlib.Path)
    parser.add_argument("--memory-split", required=True, type=pathlib.Path)
    parser.add_argument("--cwe-source", default=None, type=pathlib.Path)
    parser.add_argument("--cap-budget", type=int, default=0)
    parser.add_argument("--skip-cap-probes", action="store_true")
    parser.add_argument("--sss-payload-anchor-limit", type=int, default=None)
    parser.add_argument("--sss-anchor-selection", choices=["codebert_repo_mean"], default="codebert_repo_mean")
    parser.add_argument("--sss-codebert-model", default=None, type=pathlib.Path)
    parser.add_argument("--sss-codebert-cache-dir", default=None, type=pathlib.Path)
    parser.add_argument("--sss-codebert-repo-worktree", default=None, type=pathlib.Path)
    parser.add_argument("--sss-codebert-prior-weight", type=float, default=0.50)
    parser.add_argument("--sss-codebert-repo-demand-count", type=int, default=512)
    parser.add_argument("--sss-codebert-content-window", type=int, default=512)
    parser.add_argument("--sss-codebert-chunk-step", type=int, default=512)
    parser.add_argument("--sss-codebert-pooling", choices=["mean", "cls"], default="cls")
    parser.add_argument("--sss-payload-bank", default=None, type=pathlib.Path)
    parser.add_argument("--sss-payload-bank-sha256", default="")
    parser.add_argument("--cap-anchor-bank", default=None, type=pathlib.Path)
    parser.add_argument("--cap-anchor-bank-sha256", default="")
    parser.add_argument("--repo-worktree", default="")
    parser.add_argument("--openai-api-base", default="")
    parser.add_argument("--openclaw-served-memory-policy", choices=["configured", "empty"], default="configured")
    parser.add_argument("--openclaw-memory-search", choices=["bm25_fts", "vector", "hybrid"], default="bm25_fts")
    parser.add_argument("--openclaw-embedding-api-base", default="")
    parser.add_argument("--openclaw-embedding-model", default="")
    parser.add_argument("--docker-image", default="")
    parser.add_argument("--agent-timeout", type=int, default=600)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.method != "vulex":
        raise ValueError(f"unsupported method {args.method!r}; this package contains only vulex")
    if args.memory_size is None:
        args.memory_size = sum(1 for line in args.memory.read_text(encoding="utf-8").splitlines() if line.strip())
    config = build_config(args)
    with openclaw_native_run_lock(args.out):
        return run_single_experiment(config, args.memory, args.records, args.out, llm, resume=args.resume)


if __name__ == "__main__":
    main()
