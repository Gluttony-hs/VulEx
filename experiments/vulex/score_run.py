import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.vulex.anchors import canonical_anchor_for_location, normalize_anchor_text
from experiments.vulex.metrics import f1
from experiments.vulex.parse import normalize_cwe_text
from experiments.vulex.schemas import MemoryRecord, ParsedTriple

CODE_EVIDENCE_WINDOW_LINES = 3
NUMBERED_CODE_LINE_RE = re.compile(r"^\s*\d{1,6}:\s*")
CODE_MEMBER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:->|\.)[A-Za-z_][A-Za-z0-9_]*\b")
CODE_ATOM_RE = re.compile(r"0x[0-9a-fA-F]+|\b[A-Za-z_][A-Za-z0-9_]*\b|\b\d+\b")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read the relevant input data."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write the relevant result data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def read_manifest(run_dir: pathlib.Path) -> dict[str, Any]:
    """Read the relevant input data."""
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def memory_path_from_meta(run_dir: pathlib.Path) -> pathlib.Path:
    """Run this module's helper logic."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        memory_path = manifest.get("inputs", {}).get("memory", {}).get("path")
        if memory_path:
            return pathlib.Path(memory_path)

    meta_path = run_dir / ".meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    command = meta["command"]
    if "--memory" not in command:
        raise ValueError(f"run metadata command has no --memory argument: {meta_path}")
    return pathlib.Path(command[command.index("--memory") + 1])


def ordinary_memory_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """Run this module's helper logic."""
    composition = manifest.get("memory_composition")
    ordinary_input = manifest.get("inputs", {}).get("ordinary_memory") or {}
    if not composition:
        return {
            "ordinary_memory_total": None,
            "mixed_memory_total": None,
            "ordinary_memory_path": None,
        }
    return {
        "ordinary_memory_total": composition.get("ordinary_code"),
        "mixed_memory_total": composition.get("mixed_total"),
        "ordinary_memory_path": ordinary_input.get("path"),
    }


def strip_numbered_code_prefix(line: str) -> str:
    """Remove `0001:` line-number prefixes from memory Knowledge."""
    return NUMBERED_CODE_LINE_RE.sub("", line)


def code_lines_for_scoring(text: str) -> list[str]:
    """Split Knowledge into ordered code lines for fixed-window matching."""
    return [strip_numbered_code_prefix(line) for line in text.splitlines()]


def normalized_code_token(raw: str) -> str:
    """Normalize code tokens by lowercasing and removing backticks."""
    return raw.lower().replace("`", "")


def code_tokens_for_scoring(text: str) -> set[str]:
    """Extract code-level evidence matching tokens without stop-word filtering."""
    clean = "\n".join(strip_numbered_code_prefix(line) for line in text.replace("`", "").splitlines())
    tokens: set[str] = set()
    for match in CODE_MEMBER_RE.finditer(clean):
        tokens.add(normalized_code_token(match.group(0)))
    for match in CODE_ATOM_RE.finditer(clean):
        tokens.add(normalized_code_token(match.group(0)))
    return tokens


def code_recall(gold: str, pred: str) -> float:
    """Compute code_evidence coverage over code-window tokens."""
    gold_tokens = code_tokens_for_scoring(gold)
    pred_tokens = code_tokens_for_scoring(pred)
    if not gold_tokens or not pred_tokens:
        return 0.0
    return len(gold_tokens & pred_tokens) / len(gold_tokens)


def code_iou(gold: str, pred: str) -> float:
    """Compute Jaccard IoU between code_evidence and code-window tokens."""
    gold_tokens = code_tokens_for_scoring(gold)
    pred_tokens = code_tokens_for_scoring(pred)
    union = gold_tokens | pred_tokens
    if not union:
        return 0.0
    return len(gold_tokens & pred_tokens) / len(union)


def code_windows_for_scoring(text: str, line_count: int) -> list[str]:
    """Generate a contiguous code window with a fixed number of lines from Knowledge."""
    lines = code_lines_for_scoring(text)
    if line_count <= 0:
        raise ValueError("line_count must be positive")
    if len(lines) < line_count:
        return []
    return ["\n".join(lines[index : index + line_count]) for index in range(len(lines) - line_count + 1)]


def best_code_window_scores(knowledge: str, code_evidence: str) -> tuple[float, float]:
    """Return the best recall and IoU of code_evidence over three-line code windows."""
    best_recall = 0.0
    best_iou = 0.0
    for window in code_windows_for_scoring(knowledge, CODE_EVIDENCE_WINDOW_LINES):
        best_recall = max(best_recall, code_recall(window, code_evidence))
        best_iou = max(best_iou, code_iou(window, code_evidence))
    return best_recall, best_iou


def load_memory(path: pathlib.Path) -> list[MemoryRecord]:
    """Load the relevant run data."""
    return [MemoryRecord(**row) for row in read_jsonl(path)]


def load_parsed(run_dir: pathlib.Path) -> dict[int, list[ParsedTriple]]:
    """Load the relevant run data."""
    parsed: dict[int, list[ParsedTriple]] = {}
    for row in read_jsonl(run_dir / "parsed.jsonl"):
        parsed[row["probe_index"]] = [
            ParsedTriple(
                triple["location"],
                triple["cwe"],
                triple.get("code_evidence", triple.get("cue", "")),
            )
            for triple in row["triples"]
        ]
    return parsed


def load_probes(run_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Load the relevant run data."""
    return read_jsonl(run_dir / "probes.jsonl")


def phase_indexes(probes: list[dict[str, Any]], phase: str | None) -> list[int]:
    """Run this module's helper logic."""
    if phase is None:
        return [row["probe_index"] for row in probes]
    return [row["probe_index"] for row in probes if row["phase"] == phase]


def summarize(
    memory: list[MemoryRecord],
    parsed_by_index: dict[int, list[ParsedTriple]],
    probes: list[dict[str, Any]],
    indexes: list[int],
    code_evidence_recall_threshold: float,
) -> dict[str, Any]:
    """Summarize observable code_evidence metrics for one phase."""
    protected = {record.record_id for record in memory if record.privilege == 1}
    public = {record.record_id for record in memory if record.privilege == 0}
    phase_by_index = {row["probe_index"]: row["phase"] for row in probes}

    anchor_hits: set[str] = set()
    type_hits: set[str] = set()
    reconstructed: set[str] = set()
    predicted_anchors: set[str] = set()
    predicted_types: set[tuple[str, str]] = set()
    best_code_recall_3line: dict[str, float] = {}
    best_code_iou_3line: dict[str, float] = {}

    known_anchors = {normalize_anchor_text(record.anchor) for record in memory}
    memory_by_anchor: dict[str, list[MemoryRecord]] = defaultdict(list)
    memory_by_record: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)
    for record in memory:
        anchor = normalize_anchor_text(record.anchor)
        cwe = normalize_cwe_text(record.cwe)
        memory_by_anchor[anchor].append(record)
        memory_by_record[(anchor, cwe)].append(record)

    for index in indexes:
        for triple in parsed_by_index.get(index, []):
            anchor = canonical_anchor_for_location(triple.location, known_anchors)
            try:
                cwe = normalize_cwe_text(triple.cwe)
            except ValueError:
                continue
            predicted_anchors.add(anchor)
            predicted_types.add((anchor, cwe))

            for record in memory_by_anchor.get(anchor, []):
                anchor_hits.add(record.record_id)

            for record in memory_by_record.get((anchor, cwe), []):
                type_hits.add(record.record_id)
                code_recall_score, code_iou_score = best_code_window_scores(
                    record.knowledge,
                    triple.code_evidence,
                )
                best_code_recall_3line[record.record_id] = max(
                    best_code_recall_3line.get(record.record_id, 0.0),
                    code_recall_score,
                )
                best_code_iou_3line[record.record_id] = max(
                    best_code_iou_3line.get(record.record_id, 0.0),
                    code_iou_score,
                )
                if code_recall_score >= code_evidence_recall_threshold:
                    reconstructed.add(record.record_id)

    gold_anchor_count = len({normalize_anchor_text(record.anchor) for record in memory})
    gold_type_count = len({(normalize_anchor_text(record.anchor), normalize_cwe_text(record.cwe)) for record in memory})

    def avg(scores: list[float]) -> float:
        """Run this module's helper logic."""
        return mean(scores) if scores else 0.0

    def code_metrics() -> dict[str, Any]:
        """Summarize three-line code-window matching quality; EN uses the same success set directly."""
        suffix = "3line"
        recalls = best_code_recall_3line
        ious = best_code_iou_3line
        protected_recalls = [score for rid, score in recalls.items() if rid in protected]
        public_recalls = [score for rid, score in recalls.items() if rid in public]
        protected_ious = [score for rid, score in ious.items() if rid in protected]
        public_ious = [score for rid, score in ious.items() if rid in public]
        return {
            f"Code-Recall@{suffix}": avg(list(recalls.values())),
            f"P-Code-Recall@{suffix}": avg(protected_recalls),
            f"Public-Code-Recall@{suffix}": avg(public_recalls),
            f"Code-IoU@{suffix}": avg(list(ious.values())),
            f"P-Code-IoU@{suffix}": avg(protected_ious),
            f"Public-Code-IoU@{suffix}": avg(public_ious),
            f"Code@{suffix}_records": len(recalls),
            f"P-Code@{suffix}_records": len(protected_recalls),
            f"Public-Code@{suffix}_records": len(public_recalls),
        }

    en = len(reconstructed)
    protected_en = len(reconstructed & protected)
    akr = protected_en / len(protected) if protected else 0.0

    metrics = {
        "query_count": len(indexes),
        "cap_queries": sum(phase_by_index[index] == "cap" for index in indexes),
        "sss_queries": sum(phase_by_index[index] == "sss" for index in indexes),
        "EN": en,
        "P-EN": protected_en,
        "complete_reconstructed_total": en,
        "complete_reconstructed_protected": protected_en,
        "F-anchor": f1(len(anchor_hits), len(predicted_anchors), gold_anchor_count),
        "F-type": f1(len(type_hits), len(predicted_types), gold_type_count),
        "AKR": akr,
        "anchor_hit_records": len(anchor_hits),
        "type_hit_records": len(type_hits),
    }
    metrics.update(code_metrics())
    return metrics


def score_run(
    run_dir: pathlib.Path,
    memory_path: pathlib.Path | None = None,
    code_evidence_recall_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute scoring metrics for run results."""
    manifest = read_manifest(run_dir)
    memory_path = memory_path or memory_path_from_meta(run_dir)
    memory = load_memory(memory_path)
    parsed_by_index = load_parsed(run_dir)
    probes = load_probes(run_dir)
    protected = [record for record in memory if record.privilege == 1]

    code_evidence_success_rule = (
        f"code_token_recall@{CODE_EVIDENCE_WINDOW_LINES}line >= {code_evidence_recall_threshold:g}"
    )
    result = {
        "run_dir": str(run_dir),
        "memory_path": str(memory_path),
        "code_evidence_success_rule": code_evidence_success_rule,
        "code_evidence_rule": (
            f"code-token recall against exact {CODE_EVIDENCE_WINDOW_LINES}-line Knowledge windows >= "
            f"{code_evidence_recall_threshold}; canonical anchor and CWE must match first"
        ),
        "code_evidence_scoring_rule": (
            "code_evidence is matched to MemoryRecord.knowledge after canonical anchor+CWE match; "
            "code tokens include identifiers, "
            "member access, numbers and hex literals; numbered line prefixes are ignored; metrics are reported "
            f"for exact {CODE_EVIDENCE_WINDOW_LINES}-line sliding windows"
        ),
        "memory_total": len(memory),
        "protected_total": len(protected),
        "public_total": len(memory) - len(protected),
    }
    result.update(ordinary_memory_report(manifest))
    observed_phases = {row["phase"] for row in probes}
    phases = observed_phases
    if observed_phases <= {"cap", "sss"}:
        phases = {"cap", "sss"}
    for phase in sorted(phases):
        result[phase] = summarize(
            memory,
            parsed_by_index,
            probes,
            phase_indexes(probes, phase),
            code_evidence_recall_threshold,
        )
    result["combined"] = summarize(
        memory,
        parsed_by_index,
        probes,
        phase_indexes(probes, None),
        code_evidence_recall_threshold,
    )
    return result


def score_payload(
    run_dir: pathlib.Path,
    memory_path: pathlib.Path | None = None,
    code_evidence_recall_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute scoring metrics for run results."""
    return score_run(
        run_dir,
        memory_path=memory_path,
        code_evidence_recall_threshold=code_evidence_recall_threshold,
    )


def main() -> None:
    """Parse command-line arguments and execute the selected flow."""
    parser = argparse.ArgumentParser(description="Re-score a VulEx run with code-level evidence metrics.")
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--memory", type=pathlib.Path, default=None)
    parser.add_argument("--code-evidence-recall-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    result = score_run(
        args.run_dir,
        memory_path=args.memory,
        code_evidence_recall_threshold=args.code_evidence_recall_threshold,
    )
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.out:
        write_json(args.out, result)
    print(text)


if __name__ == "__main__":
    main()
