from collections import Counter
import re

from experiments.vulex.schemas import MemoryRecord, ReposVulRecord

ANCHOR_SUFFIX_BOUNDARY_RE = re.compile(r"^[`'\"\s:;,()\[\]{}#@]")


def build_anchor_set(records: list[MemoryRecord]) -> list[str]:
    """Extract public anchors with stable frequency and lexicographic ordering."""
    counts = Counter(record.anchor for record in records)
    return sorted(counts, key=lambda anchor: (-counts[anchor], anchor))


def build_anchor_set_from_records(records: list[ReposVulRecord]) -> list[str]:
    """Build the attacker-visible anchor set from dataset-level public records."""
    counts = Counter(record.anchor for record in records)
    return sorted(counts, key=lambda anchor: (-counts[anchor], anchor))


def normalize_anchor_text(text: str) -> str:
    """Normalize an anchor or model location while preserving its path::function core."""
    return " ".join(str(text).strip().strip("`'\"").split())


def location_matches_anchor(location: str, anchor: str) -> bool:
    """Check whether a model location preserves the complete canonical anchor with only a boundary suffix."""
    normalized_location = normalize_anchor_text(location).lstrip("`'\"")
    normalized_anchor = normalize_anchor_text(anchor)
    if normalized_location == normalized_anchor:
        return True
    if not normalized_location.startswith(normalized_anchor):
        return False
    suffix = normalized_location[len(normalized_anchor) :]
    return bool(ANCHOR_SUFFIX_BOUNDARY_RE.match(suffix))


def canonical_anchor_for_location(location: str, anchors: list[str] | set[str] | tuple[str, ...]) -> str:
    """Map a model location to a known anchor, or return its normalized form when unmatched."""
    for anchor in sorted(anchors, key=len, reverse=True):
        if location_matches_anchor(location, anchor):
            return normalize_anchor_text(anchor)
    return normalize_anchor_text(location)
