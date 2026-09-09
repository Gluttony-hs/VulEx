"""Official CodeBERT repo_mean pairwise fuzzy selector for public SSS samples."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
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


def target_function_sha256(record: Any) -> str:
    """Compute a stable hash of a public record target_function."""
    return hashlib.sha256(str(record.target_function).encode("utf-8")).hexdigest()


def path_content_sha256(path: pathlib.Path) -> str:
    """Compute a file or directory-tree content hash to prevent reuse of an embedding cache from an old model."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def load_embedding_cache(
    cache_path: pathlib.Path,
    records: list[Any],
    *,
    model_path: pathlib.Path,
    pooling: str,
    content_window: int,
    chunk_step: int,
) -> dict[str, list[float]] | None:
    """Read a cache matching the current inputs and embedding parameters; return None on mismatch."""
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    expected = {
        "model_path": str(model_path),
        "model_sha256_tree": path_content_sha256(model_path),
        "pooling": pooling,
        "content_window": content_window,
        "chunk_step": chunk_step,
        "record_fingerprints": {
            record.record_id: target_function_sha256(record)
            for record in records
        },
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            return None
    embeddings = data.get("embeddings", {})
    if not isinstance(embeddings, dict):
        return None
    return {str(anchor): [float(value) for value in vector] for anchor, vector in embeddings.items()}


def write_embedding_cache(
    cache_path: pathlib.Path,
    records: list[Any],
    embeddings_by_anchor: dict[str, list[float]],
    *,
    model_path: pathlib.Path,
    pooling: str,
    content_window: int,
    chunk_step: int,
) -> None:
    """Write the project-local embedding cache and minimal reproduction metadata."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_path": str(model_path),
        "model_sha256_tree": path_content_sha256(model_path),
        "pooling": pooling,
        "content_window": content_window,
        "chunk_step": chunk_step,
        "input_field": "target_function",
        "record_count": len(records),
        "record_fingerprints": {
            record.record_id: target_function_sha256(record)
            for record in records
        },
    }
    payload = {
        "metadata": metadata,
        "embeddings": embeddings_by_anchor,
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def pool_batch_hidden_states(hidden: Any, attention_mask: Any, pooling: str) -> Any:
    """Aggregate CodeBERT token hidden states into a chunk embedding according to the configuration."""
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    if pooling == "cls":
        return hidden[:, 0, :]
    raise ValueError("pooling must be one of: mean, cls")


def load_or_build_codebert_embeddings(
    records: list[Any],
    *,
    model_path: pathlib.Path,
    cache_path: pathlib.Path,
    pooling: str,
    content_window: int = 510,
    chunk_step: int = 255,
    batch_size: int = 8,
) -> dict[str, list[float]]:
    """Generate or read public target_function embeddings from the project-local CodeBERT path."""
    if pooling not in {"mean", "cls"}:
        raise ValueError("pooling must be one of: mean, cls")
    if content_window <= 0 or chunk_step <= 0:
        raise ValueError("content_window and chunk_step must be positive")
    if not model_path.exists():
        raise FileNotFoundError(f"CodeBERT local model path does not exist: {model_path}")
    cached = load_embedding_cache(
        cache_path,
        records,
        model_path=model_path,
        pooling=pooling,
        content_window=content_window,
        chunk_step=chunk_step,
    )
    if cached is not None:
        return cached

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model.eval()

    embeddings_by_anchor: dict[str, list[float]] = {}
    with torch.no_grad():
        for record in records:
            token_ids = tokenizer.encode(str(record.target_function), add_special_tokens=False)
            if not token_ids:
                token_ids = [tokenizer.unk_token_id]
            chunks = []
            for start in range(0, len(token_ids), chunk_step):
                chunk_ids = token_ids[start : start + content_window]
                if not chunk_ids:
                    continue
                chunks.append(tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=False))
                if start + content_window >= len(token_ids):
                    break
            chunk_vectors = []
            for batch_start in range(0, len(chunks), batch_size):
                batch = tokenizer(
                    chunks[batch_start : batch_start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                outputs = model(**batch)
                vectors = pool_batch_hidden_states(outputs.last_hidden_state, batch["attention_mask"], pooling)
                chunk_vectors.extend(vectors.detach().cpu())
            function_vector = torch.stack(chunk_vectors, dim=0).mean(dim=0)
            vector = normalize_vector([float(value) for value in function_vector.tolist()])
            embeddings_by_anchor[record.record_id] = vector

    write_embedding_cache(
        cache_path,
        records,
        embeddings_by_anchor,
        model_path=model_path,
        pooling=pooling,
        content_window=content_window,
        chunk_step=chunk_step,
    )
    return embeddings_by_anchor
