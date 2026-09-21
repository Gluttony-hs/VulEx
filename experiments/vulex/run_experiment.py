import argparse
import json
import os
import pathlib
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from tqdm import tqdm

    # Allow direct runner execution from the package root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.agent_harness import (
    AgentRunResult,
    OPENCLAW_MEMORY_CORPUS_CONTAINER_PATH,
    build_harness_prompt,
    normalize_optional_repo_worktree,
    run_agent_harness,
    run_openclaw_memory_index,
    write_openclaw_config,
)
from experiments.vulex.cap_anchor_bank import read_cap_anchor_bank_spec
from experiments.vulex.llm_client import CachedLLMClient
from experiments.vulex.metrics import compute_native_agent_metrics
from experiments.vulex.openclaw_memory import write_openclaw_memory_corpus
from experiments.vulex.parse import parse_response
from experiments.vulex.probes import build_cap_probe, build_sss_probe
from experiments.vulex.schemas import (
    MemoryRecord,
    OrdinaryMemoryRecord,
    ParsedTriple,
    ReposVulRecord,
    RunConfig,
)
from experiments.vulex.sss import format_multi_anchor_sss_payload, generate_sss_payload

OPENCLAW_AGENT_ATTEMPTS = 8
OPENCLAW_AGENT_RETRY_SLEEP_SECONDS = 5
OPENCLAW_TERMINAL_FAILURE_MARKERS = (
    "Agent couldn't generate a response",
    "The model did not produce a response before the model idle timeout",
    "Concurrency limit exceeded for account",
)
CHECKPOINT_DIRNAME = "checkpoint"
FINAL_OUTPUT_NAMES = {
    "probes.jsonl",
    "responses.jsonl",
    "parsed.jsonl",
    "summary.json",
    "summary_scored.json",
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


def read_memory_jsonl(path: pathlib.Path) -> list[MemoryRecord]:
    """Read formal memory JSONL and construct MemoryRecord objects line by line."""
    records: list[MemoryRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(MemoryRecord(**json.loads(line)))
    return records


def read_ordinary_memory_jsonl(path: pathlib.Path) -> list[OrdinaryMemoryRecord]:
    """Read ordinary-code memory JSONL and construct OrdinaryMemoryRecord objects line by line."""
    records: list[OrdinaryMemoryRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(OrdinaryMemoryRecord(**json.loads(line)))
    return records


def read_cwe_source(path: pathlib.Path) -> list[str]:
    """Read the explicit CWE order."""
    return list(dict.fromkeys(json.loads(path.read_text(encoding="utf-8"))))


def read_reposvul_jsonl(path: pathlib.Path) -> list[ReposVulRecord]:
    """Read formal ReposVul JSONL and construct ReposVulRecord objects line by line."""
    records: list[ReposVulRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(ReposVulRecord(**json.loads(line)))
    return records


def read_split_ids(path: pathlib.Path) -> dict:
    """Read visibility and memory splits."""
    return json.loads(path.read_text(encoding="utf-8"))


def records_from_ids(records_by_id: dict[str, ReposVulRecord], record_ids: list[str]) -> list[ReposVulRecord]:
    """Look up ReposVulRecords by split record IDs."""
    return [records_by_id[record_id] for record_id in record_ids]


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
            f"OpenClaw probe {probe_index} failed: "
            f"exit_code={result.exit_code}, timed_out={result.timed_out}, trace_dir={result.trace_dir}",
            result,
        )


def checkpoint_path(out_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Return the path of the requested checkpoint file."""
    return out_dir / CHECKPOINT_DIRNAME / name


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


def load_successful_checkpoint(
    out_dir: pathlib.Path,
    probe_index: int,
    response_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
) -> list:
    """Load successful probe checkpoint artifacts into memory and return parsed triples."""
    path = out_dir / CHECKPOINT_DIRNAME / f"probe_{probe_index:04d}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    response_rows.append(payload["response"])
    parsed_row = payload["parsed"]
    parsed_rows.append(parsed_row)
    probe_rows.append(payload["probe"])
    return [ParsedTriple(**triple) for triple in parsed_row["triples"]]


def record_probe_success(
    out_dir: pathlib.Path,
    probe_index: int,
    response_delta: dict[str, Any],
    parsed_delta: dict[str, Any],
    probe_delta: dict[str, Any],
) -> None:
    """Write a successful probe as a single-file checkpoint with the status line persisted last."""
    write_json(
        out_dir / CHECKPOINT_DIRNAME / f"probe_{probe_index:04d}.json",
        {
            "response": response_delta,
            "parsed": parsed_delta,
            "probe": probe_delta,
        },
    )
    append_jsonl(
        checkpoint_path(out_dir, "probe_status.jsonl"),
        {
            "probe_index": probe_index,
            "status": "success",
        },
    )


def record_probe_failure(
    out_dir: pathlib.Path,
    probe_index: int,
    error: str,
) -> None:
    """Record a failed probe status; failures do not enter successful checkpoint artifacts."""
    append_jsonl(
        checkpoint_path(out_dir, "probe_status.jsonl"),
        {
            "probe_index": probe_index,
            "status": "failed",
            "error": error,
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
                {"probe_index": probe_index, "status": "missing"},
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


def cwe_anchor_targets(
    public_records: list[ReposVulRecord],
) -> tuple[list[str], dict[str, list[str]]]:
    """Build CWE-level probe targets by public CWE frequency while retaining every public anchor per CWE."""
    cwe_counts = Counter(record.cwe for record in public_records)
    anchor_counts_by_cwe: dict[str, Counter] = {}
    for record in public_records:
        anchor_counts_by_cwe.setdefault(record.cwe, Counter())[record.anchor] += 1

    cwe_order = sorted(cwe_counts, key=lambda cwe: (-cwe_counts[cwe], cwe))
    anchors_by_cwe = {
        cwe: sorted(anchor_counts_by_cwe[cwe], key=lambda anchor: (-anchor_counts_by_cwe[cwe][anchor], anchor))
        for cwe in cwe_order
    }
    return cwe_order, anchors_by_cwe


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
    """Read SSS payload rows indexed by CWE."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["cwe"]): row for row in data["rows"]}


def replay_sss_payload_bank_row(
    cwe: str,
    records_for_cwe: list[ReposVulRecord],
    bank_rows_by_cwe: dict[str, dict[str, Any]],
) -> list[ReposVulRecord]:
    """Replay SSS payload records by bank row."""
    row = bank_rows_by_cwe[cwe]
    records_by_id = {record.record_id: record for record in records_for_cwe}
    return [records_by_id[str(record_id)] for record_id in row["payload_source_record_ids"]]


def multi_anchor_text(anchors: list[str]) -> str:
    """Join all public anchors for one CWE into the existing prompt template's `<a>` placeholder."""
    return "; ".join(anchors)


def prepare_openclaw_native_memory(
    out_dir: pathlib.Path,
    memory: list[MemoryRecord],
    config: RunConfig,
    ordinary_memory: list[OrdinaryMemoryRecord] | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write and index the OpenClaw corpus selected by the served-memory policy."""
    ordinary_memory = ordinary_memory or []
    served_policy = getattr(config, "openclaw_served_memory_policy", "configured")
    served_memory = [] if served_policy == "empty" else memory
    served_ordinary_memory = [] if served_policy == "empty" else ordinary_memory
    memory_root = out_dir / "openclaw_memory"
    corpus_dir = memory_root / "vulex_records"
    openclaw_home = memory_root / "openclaw_home"
    write_openclaw_memory_corpus(
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
    run_openclaw_memory_index(
        openclaw_home,
        corpus_dir,
        config.docker_image,
        config.openai_api_base or os.environ.get("OPENAI_BASE_URL", ""),
        memory_search_mode=memory_search_mode,
        embedding_api_base=embedding_api_base,
        timeout=config.agent_timeout,
    )
    return openclaw_home, corpus_dir


def append_agent_artifacts(
    probe_index: int,
    agent_result: AgentRunResult,
    response_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
) -> list:
    """Write one probe's OpenClaw native artifact row and return parsed triples."""
    response = agent_result.response
    triples = parse_response(response)
    response_row = {
        "probe_index": probe_index,
        "raw_response": response,
        "exit_code": agent_result.exit_code,
        "timed_out": agent_result.timed_out,
    }
    response_rows.append(response_row)
    parsed_rows.append(
        {
            "probe_index": probe_index,
            "triples": [asdict(triple) for triple in triples],
        }
    )
    return triples


def materialize_successful_outputs(
    out_dir: pathlib.Path,
    memory: list[MemoryRecord],
    memory_path: pathlib.Path,
    response_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    parsed_lists: list[list],
) -> dict[str, float]:
    """Write formal artifacts and scoring results after all probes succeed."""
    for name in PARTIAL_OUTPUT_NAMES:
        path = out_dir / name
        if path.exists():
            path.unlink()
    phases = [row["phase"] for row in probe_rows]
    summary = compute_native_agent_metrics(memory, parsed_lists, phases)
    write_jsonl(out_dir / "probes.jsonl", probe_rows)
    write_jsonl(out_dir / "responses.jsonl", response_rows)
    write_jsonl(out_dir / "parsed.jsonl", parsed_rows)
    write_json(out_dir / "summary.json", summary)
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


def cap_bank_anchors_for_cwe(cap_anchor_bank, cwe: str) -> list[str]:
    """Return the final CAP-bank anchors for the current CWE."""
    return cap_anchor_bank[cwe]


def run_probe_with_checkpoint(
    config: RunConfig,
    out_dir: pathlib.Path,
    probe,
    probe_index: int,
    message: str,
    probe_row: dict[str, Any],
    openclaw_home: pathlib.Path,
    openclaw_corpus_dir: pathlib.Path,
    response_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
) -> list | None:
    """Run one probe and write its checkpoint; return None on failure or parsed triples on success."""
    agent_result = None
    local_response_start = len(response_rows)
    local_parsed_start = len(parsed_rows)
    try:
        agent_result = run_configured_agent(config, out_dir, probe_index, message, openclaw_home, openclaw_corpus_dir)
        require_agent_success(agent_result, probe_index)
    except ProbeExecutionError as exc:
        response_rows[local_response_start:] = []
        parsed_rows[local_parsed_start:] = []
        record_probe_failure(out_dir, probe_index, str(exc))
        return None
    except (RuntimeError, TimeoutError, ConnectionError) as exc:
        response_rows[local_response_start:] = []
        parsed_rows[local_parsed_start:] = []
        record_probe_failure(out_dir, probe_index, str(exc))
        return None
    else:
        triples = append_agent_artifacts(
            probe_index,
            agent_result,
            response_rows,
            parsed_rows,
        )
    probe_rows.append(probe_row)
    record_probe_success(
        out_dir,
        probe_index,
        response_rows[-1],
        parsed_rows[-1],
        probe_row,
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
        records_by_id = {record.record_id: record for record in records}
        visibility_split = read_split_ids(pathlib.Path(config.visibility_split))

        public_records = records_from_ids(records_by_id, visibility_split["public_record_ids"])
        cwe_set, _ = cwe_anchor_targets(public_records)
        if config.cwe_source:
            cwe_set = read_cwe_source(pathlib.Path(config.cwe_source))
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
            sss_payload_bank_rows_by_cwe = load_sss_payload_bank(sss_payload_bank_path)
        cap_limit = 0 if config.skip_cap_probes else config.cap_budget or config.budget
        cap_cwes = cwe_set[: min(config.budget, cap_limit, len(cwe_set))]
        if cap_cwes:
            cap_anchor_bank_path = pathlib.Path(config.cap_anchor_bank) if config.cap_anchor_bank else None
            cap_anchor_bank = read_cap_anchor_bank_spec(cap_anchor_bank_path) if cap_anchor_bank_path else None
        else:
            cap_anchor_bank = None

        llm = llm or CachedLLMClient(out_dir / "llm_cache")
        openclaw_home, openclaw_corpus_dir = prepare_openclaw_native_memory(
            out_dir,
            memory,
            config,
            ordinary_memory=ordinary_memory,
        )
        response_rows: list[dict[str, Any]] = []
        parsed_rows: list[dict[str, Any]] = []
        probe_rows: list[dict[str, Any]] = []
        parsed_lists = []
        statuses = latest_probe_statuses(out_dir)
        probe_index = 0
        for cwe in tqdm(cap_cwes, desc="CAP probes", total=len(cap_cwes), unit="probe"):
            anchors = cap_bank_anchors_for_cwe(cap_anchor_bank, cwe)
            anchor_text = multi_anchor_text(anchors)
            probe = build_cap_probe(anchor_text, cwe)
            message = build_harness_prompt(probe)
            probe_row = {"probe_index": probe_index, "phase": probe.phase}
            checkpoint_file = out_dir / CHECKPOINT_DIRNAME / f"probe_{probe_index:04d}.json"
            if resume and statuses.get(probe_index, {}).get("status") == "success" and checkpoint_file.exists():
                triples = load_successful_checkpoint(
                    out_dir,
                    probe_index,
                    response_rows,
                    parsed_rows,
                    probe_rows,
                )
                parsed_lists.append(triples)
                probe_index += 1
                continue
            triples = run_probe_with_checkpoint(
                config,
                out_dir,
                probe,
                probe_index,
                message,
                probe_row,
                openclaw_home,
                openclaw_corpus_dir,
                response_rows,
                parsed_rows,
                probe_rows,
            )
            if triples is not None:
                parsed_lists.append(triples)
            probe_index += 1

        remaining = config.budget - probe_index
        if remaining > 0:
            sss_cwes = cwe_set[: min(remaining, len(cwe_set))]
            for cwe in tqdm(sss_cwes, desc="SSS probes", total=len(sss_cwes), unit="probe"):
                payload_records = replay_sss_payload_bank_row(
                    cwe,
                    public_records_by_cwe[cwe],
                    sss_payload_bank_rows_by_cwe,
                )
                checkpoint_file = out_dir / CHECKPOINT_DIRNAME / f"probe_{probe_index:04d}.json"
                if resume and statuses.get(probe_index, {}).get("status") == "success" and checkpoint_file.exists():
                    triples = load_successful_checkpoint(
                        out_dir,
                        probe_index,
                        response_rows,
                        parsed_rows,
                        probe_rows,
                    )
                    parsed_lists.append(triples)
                    probe_index += 1
                    continue
                anchor_payloads: list[tuple[str, str]] = []
                sss_failed = False
                for source in payload_records:
                    try:
                        single_payload = generate_sss_payload(source, cwe, config.sss_model, llm)
                    except (RuntimeError, TimeoutError, ConnectionError, OSError) as exc:
                        record_probe_failure(out_dir, probe_index, str(exc))
                        probe_index += 1
                        sss_failed = True
                        break
                    anchor_payloads.append((source.anchor, single_payload))
                if sss_failed:
                    continue
                payload = format_multi_anchor_sss_payload(anchor_payloads)
                probe = build_sss_probe(cwe, payload)
                message = build_harness_prompt(probe)
                probe_row = {"probe_index": probe_index, "phase": probe.phase}
                triples = run_probe_with_checkpoint(
                    config,
                    out_dir,
                    probe,
                    probe_index,
                    message,
                    probe_row,
                    openclaw_home,
                    openclaw_corpus_dir,
                    response_rows,
                    parsed_rows,
                    probe_rows,
                )
                if triples is not None:
                    parsed_lists.append(triples)
                probe_index += 1

        expected_probe_count = total_expected_probe_count(config, cwe_set)
        statuses = latest_probe_statuses(out_dir)
        incomplete = [
            probe_id
            for probe_id in range(expected_probe_count)
            if statuses.get(probe_id, {}).get("status") != "success"
        ]
        if incomplete:
            for name in FINAL_OUTPUT_NAMES | PARTIAL_OUTPUT_NAMES:
                path = out_dir / name
                if path.exists():
                    path.unlink()
            write_partial_outputs(out_dir, statuses, expected_probe_count)
            raise IncompleteExperimentError(
                f"incomplete VulEx experiment: {len(incomplete)} probe(s) failed or missing; "
                f"see {out_dir / 'failed_probes.json'}"
            )
        return materialize_successful_outputs(
            out_dir,
            memory,
            memory_path,
            response_rows,
            parsed_rows,
            probe_rows,
            parsed_lists,
        )
    finally:
        if config.openai_api_base:
            if previous_openai_base_url is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = previous_openai_base_url


def build_config(args: argparse.Namespace) -> RunConfig:
    """Convert CLI arguments into RunConfig while preserving an optional empty repo_worktree."""
    repo_worktree = normalize_optional_repo_worktree(args.repo_worktree)
    return RunConfig(
        budget=args.budget,
        agent_model=args.agent_model,
        sss_model=args.sss_model,
        visibility_split=str(args.visibility_split),
        cwe_source=str(args.cwe_source) if args.cwe_source else "",
        cap_budget=args.cap_budget,
        skip_cap_probes=args.skip_cap_probes,
        repo_worktree=str(repo_worktree) if repo_worktree else "",
        ordinary_memory=str(args.ordinary_memory) if args.ordinary_memory else "",
        cap_anchor_bank=str(args.cap_anchor_bank) if args.cap_anchor_bank else "",
        sss_payload_bank=str(args.sss_payload_bank) if args.sss_payload_bank else "",
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
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--sss-model", required=True)
    parser.add_argument("--visibility-split", required=True, type=pathlib.Path)
    parser.add_argument("--cwe-source", default=None, type=pathlib.Path)
    parser.add_argument("--cap-budget", type=int, default=0)
    parser.add_argument("--skip-cap-probes", action="store_true")
    parser.add_argument("--sss-payload-bank", default=None, type=pathlib.Path)
    parser.add_argument("--cap-anchor-bank", default=None, type=pathlib.Path)
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
    config = build_config(args)
    with openclaw_native_run_lock(args.out):
        return run_single_experiment(config, args.memory, args.records, args.out, llm, resume=args.resume)


if __name__ == "__main__":
    main()
