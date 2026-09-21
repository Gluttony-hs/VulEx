import pathlib
import re

from experiments.vulex.schemas import MemoryRecord, OrdinaryMemoryRecord

SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def safe_memory_filename(record_id: str) -> str:
    """Generate a normalized value safe to write to disk."""
    safe = SAFE_FILENAME_CHARS.sub("_", record_id.strip())
    return f"{safe}.md"


def openclaw_memory_text(record: MemoryRecord) -> str:
    """Render a vulnerability memory record as a three-part experience visible to OpenClaw."""
    return experience_memory_text(record.response)


def openclaw_ordinary_memory_text(record: OrdinaryMemoryRecord) -> str:
    """Render an ordinary-code memory record with the same three-part experience surface."""
    return experience_memory_text(record.response)


def experience_memory_text(response: str) -> str:
    """Render one OpenClaw memory entry."""
    return f"{response.strip()}\n"


def write_openclaw_memory_corpus(
    memory: list[MemoryRecord],
    corpus_dir: pathlib.Path,
    ordinary_memory: list[OrdinaryMemoryRecord] | None = None,
) -> list[pathlib.Path]:
    """Write vulnerability and ordinary memory as Markdown files in the OpenClaw corpus."""
    ordinary_memory = ordinary_memory or []
    pending: list[tuple[pathlib.Path, str]] = []
    for record in memory:
        filename = safe_memory_filename(record.record_id)
        path = corpus_dir / filename
        pending.append((path, openclaw_memory_text(record)))
    for record in ordinary_memory:
        filename = safe_memory_filename(record.record_id)
        pending.append((corpus_dir / filename, openclaw_ordinary_memory_text(record)))

    corpus_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in corpus_dir.glob("*.md"):
        stale_path.unlink()
    paths: list[pathlib.Path] = []
    for path, text in pending:
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
