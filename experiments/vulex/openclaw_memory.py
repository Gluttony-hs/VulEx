import os
import pathlib
import re

from experiments.vulex.schemas import MemoryRecord, OrdinaryMemoryRecord

SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
HAS_CONTENT_CHAR = re.compile(r"[A-Za-z0-9]")
EXPERIENCE_FIELDS = ("Situation:", "Lesson:", "Evidence/Outcome:")


def safe_memory_filename(record_id: str) -> str:
    """Generate a normalized value safe to write to disk."""
    value = record_id.strip()
    if not value:
        raise ValueError("record_id must be non-empty")
    safe = SAFE_FILENAME_CHARS.sub("_", value)
    if not HAS_CONTENT_CHAR.search(safe):
        raise ValueError(f"record_id has no safe filename content: {record_id!r}")
    return f"{safe}.md"


def openclaw_memory_text(record: MemoryRecord) -> str:
    """Render a vulnerability memory record as a three-part experience visible to OpenClaw."""
    return experience_memory_text(record.response)


def openclaw_ordinary_memory_text(record: OrdinaryMemoryRecord) -> str:
    """Render an ordinary-code memory record with the same three-part experience surface."""
    return experience_memory_text(record.response)


def experience_memory_text(response: str) -> str:
    """Render OpenClaw memory; the experiment side path may retain verified natural-language source text."""
    text = response.strip()
    if os.environ.get("OPENCLAW_ALLOW_NATURAL_MEMORY") == "1":
        if not text:
            raise ValueError("OpenClaw memory response must be non-empty")
        return f"{text}\n"
    validate_experience_memory_text(text)
    return f"{text}\n"


def validate_experience_memory_text(text: str) -> None:
    """Validate that OpenClaw-visible memory is a three-part experience text."""
    if not text:
        raise ValueError("OpenClaw memory response must be non-empty")
    first_content = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_content.startswith(EXPERIENCE_FIELDS[0]):
        raise ValueError("OpenClaw memory response must start with Situation")
    headings: list[tuple[int, str]] = []
    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        matched = next((field for field in EXPERIENCE_FIELDS if stripped.startswith(field)), "")
        if matched:
            headings.append((index, matched))
    if [heading for _, heading in headings] != list(EXPERIENCE_FIELDS):
        raise ValueError("OpenClaw memory response must contain Situation, Lesson, and Evidence/Outcome in order")
    lines = text.splitlines()
    for position, (line_index, heading) in enumerate(headings):
        next_index = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        inline_text = lines[line_index].strip()[len(heading) :].strip()
        following_text = "\n".join(lines[line_index + 1 : next_index]).strip()
        if not (inline_text or following_text):
            raise ValueError(f"OpenClaw memory {heading[:-1]} field must be non-empty")


def write_openclaw_memory_corpus(
    memory: list[MemoryRecord],
    corpus_dir: pathlib.Path,
    ordinary_memory: list[OrdinaryMemoryRecord] | None = None,
) -> list[pathlib.Path]:
    """Write vulnerability and ordinary memory as Markdown files in the OpenClaw corpus."""
    ordinary_memory = ordinary_memory or []
    seen: set[str] = set()
    for record in [*memory, *ordinary_memory]:
        if record.record_id in seen:
            raise ValueError(f"duplicate record_id: {record.record_id}")
        seen.add(record.record_id)

    pending: list[tuple[pathlib.Path, str]] = []
    seen_filenames: set[str] = set()
    for record in memory:
        filename = safe_memory_filename(record.record_id)
        if filename in seen_filenames:
            raise ValueError(f"duplicate memory filename: {filename}")
        seen_filenames.add(filename)
        path = corpus_dir / filename
        pending.append((path, openclaw_memory_text(record)))
    for record in ordinary_memory:
        filename = safe_memory_filename(record.record_id)
        if filename in seen_filenames:
            raise ValueError(f"duplicate memory filename: {filename}")
        seen_filenames.add(filename)
        pending.append((corpus_dir / filename, openclaw_ordinary_memory_text(record)))

    corpus_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in corpus_dir.glob("*.md"):
        stale_path.unlink()
    paths: list[pathlib.Path] = []
    for path, text in pending:
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
