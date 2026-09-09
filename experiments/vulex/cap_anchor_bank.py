import json
import math
import pathlib
import random
from collections import defaultdict
from typing import Any

CAP_ANCHOR_BANK_STRATEGY = "cwe_public_prior_repository_coverage"
CAP_ANCHOR_BANK_PARAMS = {
    "anti_concentration_weight": 0.0,
    "candidate_pool_size": 1024,
    "prior_weight": 0.30,
    "repo_demand_count": 512,
}
CapAnchorBankSpec = dict[str, list[str]]


def validate_current_cap_anchor_bank_params(params: Any, path: pathlib.Path) -> dict[str, Any]:
    """Validate current max joint-pool CAP parameters and retain only reproducibility fields."""
    if not isinstance(params, dict):
        raise ValueError(f"expected current max joint-pool CAP anchor bank params: {path}")
    required = {"anti_concentration_weight", "candidate_pool_size", "prior_weight", "repo_demand_count"}
    if set(params) != required:
        raise ValueError(f"expected current max joint-pool CAP anchor bank params: {path}")
    if params.get("anti_concentration_weight") != 0.0:
        raise ValueError(f"expected current max joint-pool CAP anchor bank params: {path}")
    if not isinstance(params.get("candidate_pool_size"), int) or params["candidate_pool_size"] <= 0:
        raise ValueError(f"expected current max joint-pool CAP anchor bank params: {path}")
    if not isinstance(params.get("repo_demand_count"), int) or params["repo_demand_count"] <= 0:
        raise ValueError(f"expected current max joint-pool CAP anchor bank params: {path}")
    prior_weight = params.get("prior_weight")
    if not isinstance(prior_weight, (int, float)) or not 0.0 <= float(prior_weight) <= 1.0:
        raise ValueError(f"expected current max joint-pool CAP anchor bank params: {path}")
    return params


def path_parts(source_file: str) -> tuple[str, ...]:
    """Split source_file into path components for path-similarity scoring."""
    parts = tuple(part for part in source_file.split("/") if part)
    return parts or (".",)


def path_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Compute distance from the longest common path prefix; longer prefixes give smaller distances."""
    if not left:
        left = (".",)
    if not right:
        right = (".",)
    lcp = 0
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        lcp += 1
    return 1.0 - (lcp / max(len(left), len(right)))


def path_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Convert path distance into the similarity used for coverage gain."""
    return 1.0 - path_distance(left, right)


def file_unique_candidates(candidates: list[Any], count: int) -> list[Any]:
    """Deduplicate by source_file and retain the lexicographically smallest anchor per file."""
    by_file: dict[str, Any] = {}
    for candidate in sorted(candidates, key=lambda item: (item.source_file, item.anchor)):
        by_file.setdefault(candidate.source_file, candidate)
    unique_candidates = list(by_file.values())
    if len(unique_candidates) < count:
        raise ValueError(f"not enough file-unique CAP anchor candidates: {len(unique_candidates)} < {count}")
    return unique_candidates


def _path_prefix_regions(source_file: str) -> tuple[tuple[str, ...], ...]:
    """Return directory-prefix regions for constructing repository coverage demand."""
    parts = path_parts(source_file)
    if len(parts) == 1:
        return (parts,)
    return tuple(parts[:depth] for depth in range(1, len(parts)))


def repository_coverage_demand_source_files(candidates: list[Any], count: int, seed: int) -> list[str]:
    """Select repository-coverage demand files from candidate functions as global targets."""
    if count <= 0:
        raise ValueError("count must be positive")
    unique_file_count = len({candidate.source_file for candidate in candidates})
    if unique_file_count == 0:
        raise ValueError("at least one CAP anchor candidate is required")

    selected: list[Any] = []
    remaining = sorted(file_unique_candidates(candidates, min(count, unique_file_count)), key=lambda item: item.anchor)
    candidate_parts = {candidate.anchor: path_parts(candidate.source_file) for candidate in remaining}
    candidate_regions = {candidate.anchor: _path_prefix_regions(candidate.source_file) for candidate in remaining}
    candidate_jitter = {candidate.anchor: random.Random(f"{seed}:{candidate.anchor}").random() for candidate in remaining}
    min_distances = {candidate.anchor: 1.0 for candidate in remaining}
    region_counts: dict[tuple[str, ...], int] = defaultdict(int)
    target_count = min(count, unique_file_count)
    while len(selected) < target_count:
        best_index = 0
        best_key: tuple[float, float, float, str] | None = None
        for index, candidate in enumerate(remaining):
            regions = candidate_regions[candidate.anchor]
            region_gain = sum(1.0 / math.sqrt(1.0 + region_counts[region]) for region in regions) / len(regions)
            novelty = min_distances[candidate.anchor] if selected else 0.0
            jitter = candidate_jitter[candidate.anchor]
            key = (region_gain, novelty, jitter, candidate.anchor)
            if best_key is None or key[:3] > best_key[:3] or (key[:3] == best_key[:3] and candidate.anchor < best_key[3]):
                best_index = index
                best_key = key
        best_candidate = remaining.pop(best_index)
        selected.append(best_candidate)
        for region in candidate_regions[best_candidate.anchor]:
            region_counts[region] += 1
        best_parts = candidate_parts[best_candidate.anchor]
        for candidate in remaining:
            distance = path_distance(candidate_parts[candidate.anchor], best_parts)
            min_distances[candidate.anchor] = min(min_distances[candidate.anchor], distance)
    return [candidate.source_file for candidate in selected]


def coverage_gain(coverage: list[float], similarities: list[float]) -> float:
    """Compute the average marginal gain when adding a candidate to coverage targets."""
    if not coverage:
        return 0.0
    return sum(max(current, similarity) - current for current, similarity in zip(coverage, similarities)) / len(coverage)


def cwe_prior_repository_coverage_cap_anchor_bank(
    candidates: list[Any],
    demand_source_files: list[str],
    count: int,
    seed: int,
    *,
    prior_weight: float = 0.30,
    repo_demand_count: int = 512,
    candidate_pool_size: int = 1024,
    repository_demand_source_files: list[str] | None = None,
) -> list[str]:
    """Generate the current CAP anchor bank with a max joint pool over CWE priors and repository coverage."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not demand_source_files:
        raise ValueError("at least one demand source file is required")
    if not 0.0 <= prior_weight <= 1.0:
        raise ValueError("prior_weight must be in [0, 1]")
    if repo_demand_count <= 0:
        raise ValueError("repo_demand_count must be positive")
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")

    unique_candidates = sorted(file_unique_candidates(candidates, count), key=lambda item: item.anchor)
    prior_parts = [path_parts(source_file) for source_file in demand_source_files]
    if repository_demand_source_files is None:
        repo_demand_files = repository_coverage_demand_source_files(unique_candidates, repo_demand_count, seed)
    else:
        repo_demand_files = list(dict.fromkeys(repository_demand_source_files))[:repo_demand_count]
    if not repo_demand_files:
        raise ValueError("at least one repository demand source file is required")
    repo_parts = [path_parts(source_file) for source_file in repo_demand_files]

    scored_candidates = []
    for candidate in unique_candidates:
        candidate_parts = path_parts(candidate.source_file)
        prior_similarities = [path_similarity(candidate_parts, demand) for demand in prior_parts]
        repo_similarities = [path_similarity(candidate_parts, demand) for demand in repo_parts]
        prior_score = max(prior_similarities)
        repo_score = max(repo_similarities)
        joint_score = prior_weight * prior_score + (1.0 - prior_weight) * repo_score
        scored_candidates.append(
            (
                joint_score,
                prior_score,
                repo_score,
                sum(prior_similarities) / len(prior_similarities),
                sum(repo_similarities) / len(repo_similarities),
                candidate.anchor,
                candidate,
            )
        )
    scored_candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4], item[5]))

    pool_size = min(len(unique_candidates), max(count, candidate_pool_size))
    pool = sorted((candidate for *_, candidate in scored_candidates[:pool_size]), key=lambda item: item.anchor)
    if len(pool) < count:
        raise ValueError(f"not enough CWE prior repository joint-pool candidates: {len(pool)} < {count}")

    candidate_parts_by_anchor = {candidate.anchor: path_parts(candidate.source_file) for candidate in pool}
    prior_sims_by_anchor = {
        candidate.anchor: [path_similarity(candidate_parts_by_anchor[candidate.anchor], demand) for demand in prior_parts]
        for candidate in pool
    }
    repo_sims_by_anchor = {
        candidate.anchor: [path_similarity(candidate_parts_by_anchor[candidate.anchor], demand) for demand in repo_parts]
        for candidate in pool
    }

    selected: list[Any] = []
    prior_coverage = [0.0 for _ in prior_parts]
    repo_coverage = [0.0 for _ in repo_parts]
    remaining = pool[:]
    while len(selected) < count:
        best_index = 0
        best_key: tuple[float, float, float, float, str] | None = None
        for index, candidate in enumerate(remaining):
            prior_similarities = prior_sims_by_anchor[candidate.anchor]
            repo_similarities = repo_sims_by_anchor[candidate.anchor]
            prior_gain = coverage_gain(prior_coverage, prior_similarities)
            repo_gain = coverage_gain(repo_coverage, repo_similarities)
            prior_affinity = sum(prior_similarities) / len(prior_similarities)
            diversity = (
                min(
                    path_distance(candidate_parts_by_anchor[candidate.anchor], path_parts(selected_candidate.source_file))
                    for selected_candidate in selected
                )
                if selected
                else 0.0
            )
            score = prior_weight * prior_gain + (1.0 - prior_weight) * repo_gain
            key = (score, diversity, repo_gain, prior_weight * prior_affinity, candidate.anchor)
            if best_key is None or key[:4] > best_key[:4] or (key[:4] == best_key[:4] and candidate.anchor < best_key[4]):
                best_index = index
                best_key = key

        best_candidate = remaining.pop(best_index)
        selected.append(best_candidate)
        prior_coverage = [
            max(current, similarity)
            for current, similarity in zip(prior_coverage, prior_sims_by_anchor[best_candidate.anchor])
        ]
        repo_coverage = [
            max(current, similarity)
            for current, similarity in zip(repo_coverage, repo_sims_by_anchor[best_candidate.anchor])
        ]
    return [candidate.anchor for candidate in selected]


def validate_anchor_list(anchors: Any, path: pathlib.Path, label: str) -> list[str]:
    """Validate a CAP anchor list and return it unchanged."""
    if not isinstance(anchors, list) or not all(isinstance(anchor, str) and anchor.strip() for anchor in anchors):
        raise ValueError(f"invalid CAP anchor bank {label}: {path}")
    if not anchors:
        raise ValueError(f"empty CAP anchor bank {label}: {path}")
    if len(set(anchors)) != len(anchors):
        raise ValueError(f"duplicate CAP anchor in bank {label}: {path}")
    return anchors


def read_cap_anchor_bank_spec(path: pathlib.Path) -> CapAnchorBankSpec:
    """Read the per-CWE CAP anchor bank produced by the final method."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("strategy") != CAP_ANCHOR_BANK_STRATEGY:
        raise ValueError(f"expected {CAP_ANCHOR_BANK_STRATEGY} CAP anchor bank: {path}")
    validate_current_cap_anchor_bank_params(data.get("params"), path)
    anchors_by_cwe = data.get("anchors_by_cwe")
    if not isinstance(anchors_by_cwe, dict):
        raise ValueError(f"invalid anchors_by_cwe in CAP anchor bank: {path}")

    result: dict[str, list[str]] = {}
    for cwe, anchors in anchors_by_cwe.items():
        if not isinstance(cwe, str) or not cwe.strip():
            raise ValueError(f"invalid CWE key in CAP anchor bank: {path}")
        result[cwe] = validate_anchor_list(anchors, path, cwe)
    if not result:
        raise ValueError(f"empty anchors_by_cwe in CAP anchor bank: {path}")
    return result
