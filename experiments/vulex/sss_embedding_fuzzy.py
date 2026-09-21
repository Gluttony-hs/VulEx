"""Official CodeBERT repo_mean pairwise fuzzy selector for public SSS samples."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from experiments.vulex.cap_anchor_bank import coverage_gain, path_parts, path_similarity


def source_file(anchor: str) -> str:
    """Extract the file path from a `file::function` anchor."""
    return anchor.split("::", 1)[0]


def normalize_vector(values: list[float]) -> list[float]:
    """Return an L2-normalized vector; keep zero vectors unchanged."""
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for two vectors and clip floating-point error to [-1, 1]."""
    left_norm = normalize_vector(left)
    right_norm = normalize_vector(right)
    value = sum(a * b for a, b in zip(left_norm, right_norm))
    return max(-1.0, min(1.0, value))


def fuzzy_scale_from_distances(distances: list[float]) -> float:
    """Use the median positive distance as the RBF fuzzy scale, with a stable fallback for degenerate cases."""
    positive = [distance for distance in distances if distance > 1e-12]
    if not positive:
        return 0.1
    scale = median(positive)
    return scale if scale > 1e-12 else 0.1


def fuzzy_membership(distance: float, scale: float) -> float:
    """Convert cosine distance into RBF fuzzy membership."""
    return math.exp(-distance / scale)


def _anchor(candidate: Any) -> str:
    """Read the anchor field from a candidate object."""
    return str(candidate.anchor)


def _embedding_key(candidate: Any) -> str:
    """Read a candidate embedding key; default to the anchor for simple-test compatibility."""
    return str(getattr(candidate, "embedding_key", candidate.anchor))


def _candidate_source_file(candidate: Any) -> str:
    """Read a candidate source_file field, deriving it from the anchor when absent."""
    return str(getattr(candidate, "source_file", source_file(_anchor(candidate))))


def _embedding_for(candidate: Any, embeddings_by_anchor: dict[str, list[float]]) -> list[float]:
    """Read the embedding for a candidate anchor and fail fast when it is missing."""
    key = _embedding_key(candidate)
    if key not in embeddings_by_anchor:
        raise ValueError(f"missing embedding for public anchor {_anchor(candidate)} key={key}")
    return embeddings_by_anchor[key]


def pairwise_memberships(candidates: list[Any], embeddings_by_anchor: dict[str, list[float]]) -> dict[str, list[float]]:
    """Compute pairwise fuzzy membership among public candidates for one CWE."""
    normalized = {
        _anchor(candidate): normalize_vector(_embedding_for(candidate, embeddings_by_anchor))
        for candidate in candidates
    }
    anchors = [_anchor(candidate) for candidate in candidates]
    distances: list[float] = []
    for index, left_anchor in enumerate(anchors):
        for right_anchor in anchors[index + 1 :]:
            distances.append(1.0 - cosine_similarity(normalized[left_anchor], normalized[right_anchor]))
    scale = fuzzy_scale_from_distances(distances)
    memberships: dict[str, list[float]] = {}
    for left_anchor in anchors:
        row = []
        for right_anchor in anchors:
            distance = 1.0 - cosine_similarity(normalized[left_anchor], normalized[right_anchor])
            row.append(fuzzy_membership(distance, scale))
        memberships[left_anchor] = row
    return memberships


def repo_similarities(candidate: Any, repo_demand_source_files: list[str]) -> list[float]:
    """Compute path similarity between a candidate source_file and repository demand files."""
    candidate_parts = path_parts(_candidate_source_file(candidate))
    return [path_similarity(candidate_parts, path_parts(source_file)) for source_file in repo_demand_source_files]


def fuzzy_pairwise_public_anchors(
    candidates: list[Any],
    embeddings_by_anchor: dict[str, list[float]],
    count: int,
    *,
    prior_weight: float = 1.0,
    repo_demand_source_files: list[str] | None = None,
) -> list[str]:
    """Select SSS public payload anchors with pairwise fuzzy coverage."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not candidates:
        raise ValueError("at least one public SSS candidate is required")
    if not 0.0 <= prior_weight <= 1.0:
        raise ValueError("prior_weight must be in [0, 1]")
    target_count = min(count, len(candidates))
    ordered_candidates = sorted(candidates, key=_anchor)
    memberships = pairwise_memberships(ordered_candidates, embeddings_by_anchor)
    repo_demand_source_files = repo_demand_source_files or []
    repo_sims_by_anchor = {
        _anchor(candidate): repo_similarities(candidate, repo_demand_source_files)
        for candidate in ordered_candidates
    }

    selected: list[Any] = []
    remaining = ordered_candidates[:]
    semantic_coverage = [0.0 for _ in ordered_candidates]
    repo_coverage = [0.0 for _ in repo_demand_source_files]
    while len(selected) < target_count:
        ranked = []
        for candidate in remaining:
            anchor = _anchor(candidate)
            semantic_similarities = memberships[anchor]
            semantic_gain = coverage_gain(semantic_coverage, semantic_similarities)
            repo_gain = coverage_gain(repo_coverage, repo_sims_by_anchor[anchor]) if repo_demand_source_files else 0.0
            score = (
                prior_weight * semantic_gain + (1.0 - prior_weight) * repo_gain
                if repo_demand_source_files
                else semantic_gain
            )
            semantic_affinity = sum(semantic_similarities) / len(semantic_similarities)
            ranked.append((-score, -semantic_gain, -repo_gain, -semantic_affinity, anchor, candidate))
        ranked.sort(key=lambda item: item[:5])
        best_candidate = ranked[0][5]
        remaining.remove(best_candidate)
        selected.append(best_candidate)
        best_anchor = _anchor(best_candidate)
        semantic_coverage = [
            max(current, similarity)
            for current, similarity in zip(semantic_coverage, memberships[best_anchor])
        ]
        if repo_demand_source_files:
            repo_coverage = [
                max(current, similarity)
                for current, similarity in zip(repo_coverage, repo_sims_by_anchor[best_anchor])
            ]
    return [_anchor(candidate) for candidate in selected]
