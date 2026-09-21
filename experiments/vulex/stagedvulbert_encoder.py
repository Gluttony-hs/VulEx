"""Use the StagedVulBERT MSP/pretrain checkpoint as the SSS encoder."""

from __future__ import annotations

import pathlib
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StagedTokenFeatures:
    """Store tokens and source-line indices for one StagedVulBERT segment."""

    input_ids: list[int]
    row_idx: list[int]


def l2_normalize(values: list[float]) -> list[float]:
    """Return an L2-normalized vector."""
    norm = sum(value * value for value in values) ** 0.5
    if norm <= 0.0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def staged_code_segments(
    func: str,
    tokenizer: Any,
    block_size: int,
    segment_count: int,
) -> list[StagedTokenFeatures | None]:
    """Split a function at source-line boundaries into a fixed number of StagedVulBERT segments."""
    if block_size <= 2:
        raise ValueError("block_size must be greater than 2")
    code_tokens_by_row: list[list[str]] = []
    for row in str(func).split("\n"):
        tokens = tokenizer.tokenize("\n" if row == "" else row)
        if tokens:
            code_tokens_by_row.append(tokens)
    if not code_tokens_by_row:
        code_tokens_by_row = [[tokenizer.unk_token]]

    code_tokens: list[str] = []
    row_idx: list[int] = []
    for row_number, tokens in enumerate(code_tokens_by_row, start=1):
        code_tokens.extend(tokens)
        row_idx.extend([row_number] * len(tokens))

    segments: list[StagedTokenFeatures | None] = []
    start_idx = 0
    max_segments = min((len(code_tokens) + block_size - 1) // block_size + 1, segment_count)
    for _ in range(max_segments):
        if start_idx >= len(code_tokens):
            break
        end_token_idx = start_idx + block_size - 2
        if end_token_idx < len(code_tokens) - 1:
            last_row = row_idx[end_token_idx]
            end_idx = row_idx.index(last_row)
            if end_idx <= start_idx:
                end_idx = min(len(code_tokens), start_idx + block_size - 2)
        else:
            end_idx = len(code_tokens)

        sequence = [tokenizer.cls_token] + code_tokens[start_idx:end_idx]
        base_row = row_idx[start_idx]
        sequence_rows = [0] + [index - base_row + 1 for index in row_idx[start_idx:end_idx]]
        input_ids = tokenizer.convert_tokens_to_ids(sequence)
        input_ids += [tokenizer.pad_token_id] * (block_size - len(input_ids))
        segments.append(StagedTokenFeatures(input_ids, sequence_rows))
        start_idx = end_idx

    segments += [None] * (segment_count - len(segments))
    return segments[:segment_count]


def staged_segment_tensors(feature: StagedTokenFeatures | None, block_size: int) -> tuple[Any, ...]:
    """Build tensors required by the model for one segment."""
    import torch

    if feature is None:
        return (
            torch.ones(block_size).long(),
            torch.zeros(block_size, block_size).bool(),
            torch.zeros(block_size, block_size),
            torch.zeros(block_size).bool(),
            torch.zeros(1).bool(),
        )

    row_token_counts = Counter(feature.row_idx)
    row_starts = [feature.row_idx.index(row_id) for row_id in row_token_counts] + [len(feature.row_idx)]
    attention_mask = torch.zeros(block_size, block_size)
    attention_mask[: len(feature.row_idx), : len(feature.row_idx)] = 1
    row_to_row_mask = torch.zeros(block_size, block_size)
    for index in range(len(row_token_counts)):
        row_to_row_mask[row_starts[index] : row_starts[index + 1], row_starts[index] : row_starts[index + 1]] = 1
    row_to_row_mask[0, : len(feature.row_idx)] = 1
    first_token_mask = torch.zeros(block_size)
    first_token_mask[row_starts[:-1]] = 1
    return (
        torch.tensor(feature.input_ids),
        attention_mask.bool(),
        row_to_row_mask,
        first_token_mask.bool(),
        torch.ones(1).bool(),
    )


def staged_collate(
    segments_by_record: list[list[StagedTokenFeatures | None]],
    block_size: int,
    device: Any,
) -> list[tuple[Any, ...]]:
    """Stack same-position segments from multiple records into a batch."""
    import torch

    segment_count = len(segments_by_record[0])
    result = []
    for segment_index in range(segment_count):
        tensors = [
            staged_segment_tensors(segments[segment_index], block_size)
            for segments in segments_by_record
        ]
        result.append(
            tuple(torch.stack([item[field] for item in tensors]).to(device) for field in range(5))
        )
    return result


def roberta_attention_mask(mask: Any) -> Any:
    """Convert the legacy 3D token mask to the additive mask accepted by current Transformers."""
    import torch

    if mask.dim() != 3:
        return mask
    keep = mask[:, None, :, :].bool()
    additive = torch.zeros(keep.shape, dtype=torch.float32, device=mask.device)
    return additive.masked_fill(~keep, torch.finfo(additive.dtype).min)


def fine_get_row_rep(model: Any, token_features: tuple[Any, ...]) -> tuple[list[Any], list[Any]]:
    """Reproduce line-level representation extraction from the fine/line-style TE transformer."""
    import torch
    import torch.nn.functional as functional

    input_ids, attention_mask, row_to_row_mask, first_token_mask, is_split = token_features
    batch_size = input_ids.shape[0]
    hidden_size = model.config.hidden_size
    active = is_split.squeeze(-1).bool().view(-1)
    empty_rows = [torch.zeros(0, hidden_size, device=input_ids.device) for _ in range(batch_size)]
    empty_cls = [torch.zeros(0, hidden_size, device=input_ids.device) for _ in range(batch_size)]
    if not bool(active.any()):
        return empty_rows, empty_cls

    selected_ids = input_ids[active]
    selected_attention = attention_mask[active]
    selected_row_to_row = row_to_row_mask[active]
    selected_first = first_token_mask[active]
    token_outputs = model.TEtransformer.roberta(
        selected_ids,
        attention_mask=roberta_attention_mask(selected_attention),
    )[0]
    weights = selected_row_to_row.clone()
    weights[~selected_first] = 0
    weights = weights / (weights.sum(-1) + 1e-10)[:, :, None]
    weights[weights == 0] = float("-1e9")
    weights = functional.softmax(weights, dim=2).clone()
    weights[~selected_first] = 0
    averaged = torch.einsum("abc,acd->abd", weights, token_outputs)

    row_representations = empty_rows[:]
    cls_representations = empty_cls[:]
    for selected_index, original_index in enumerate(torch.nonzero(active, as_tuple=False).view(-1).tolist()):
        selected_rows = averaged[selected_index, selected_first[selected_index]]
        cls_representations[original_index] = selected_rows[0].unsqueeze(0)
        row_representations[original_index] = selected_rows[1:]
    return row_representations, cls_representations


def fine_encode_cls_batch(model: Any, batch_segments: list[tuple[Any, ...]]) -> Any:
    """Extract function-level CLS vectors from the fine/line-style SE transformer."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    rows: list[Any] | None = None
    cls_rows: list[Any] | None = None
    for segment in batch_segments:
        segment_rows, segment_cls = fine_get_row_rep(model, segment)
        if rows is None:
            rows, cls_rows = segment_rows, segment_cls
            continue
        rows = [torch.cat((left, right), dim=0) for left, right in zip(rows, segment_rows)]
        cls_rows = [torch.cat((left, right), dim=0) for left, right in zip(cls_rows, segment_cls)]
    if rows is None or cls_rows is None:
        raise ValueError("empty StagedVulBERT batch")

    cls_rows = [
        torch.mean(item, dim=0).unsqueeze(0)
        if item.shape[0]
        else torch.zeros(1, model.config.hidden_size, device=model.args.device)
        for item in cls_rows
    ]
    rows = [torch.cat((cls_item, row_item), dim=0) for cls_item, row_item in zip(cls_rows, rows)]
    padded = pad_sequence(rows, batch_first=True, padding_value=0)
    attention_mask = torch.tensor(
        [[1] * item.shape[0] + [0] * (padded.shape[1] - item.shape[0]) for item in rows],
        device=model.args.device,
    )
    return model.SETransformer.roberta(inputs_embeds=padded, attention_mask=attention_mask)[0][:, 0, :]


def load_fine_line_model(
    codebert_model: pathlib.Path,
    checkpoint: pathlib.Path,
    staged_source: pathlib.Path,
    device: Any,
) -> tuple[Any, Any]:
    """Construct the line-style model and load only shape-matching TE/SE tensors from MSP."""
    import torch
    from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

    if str(staged_source) not in sys.path:
        sys.path.insert(0, str(staged_source))
    from Models.StagedModel_line_vul import Model

    config = RobertaConfig.from_pretrained(str(codebert_model), local_files_only=True)
    config.num_labels = 1
    config.num_attention_heads = 12
    config.num_hidden_layers = 6
    config._attn_implementation = "eager"
    tokenizer = RobertaTokenizer.from_pretrained(str(codebert_model), local_files_only=True)
    args = type("StagedArgs", (), {"device": device})()
    model = Model(
        RobertaForSequenceClassification(config=config),
        RobertaForSequenceClassification(config=config),
        config,
        tokenizer,
        args,
    ).to(device)
    for transformer in (model.TEtransformer, model.SETransformer):
        transformer.config._attn_implementation = "eager"

    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    own_state = model.state_dict()
    for key, value in state_dict.items():
        if key not in own_state:
            continue
        if tuple(own_state[key].shape) != tuple(value.shape):
            continue
        own_state[key].copy_(value)
    model.eval()
    return model, tokenizer


def encode_records(
    records: list[Any],
    *,
    codebert_model: pathlib.Path,
    checkpoint: pathlib.Path,
    staged_source: pathlib.Path,
    batch_size: int = 4,
) -> dict[str, list[float]]:
    """Generate normalized fine-line CLS embeddings for records on CPU."""
    import torch

    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model, tokenizer = load_fine_line_model(
        codebert_model,
        checkpoint,
        staged_source,
        device,
    )
    embeddings: dict[str, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            segments = [staged_code_segments(record.target_function, tokenizer, 512, 4) for record in batch]
            tensors = staged_collate(segments, 512, device)
            vectors = fine_encode_cls_batch(model, tensors)
            for record, vector in zip(batch, vectors.detach().cpu(), strict=True):
                embeddings[record.record_id] = l2_normalize([float(value) for value in vector.tolist()])
            print(f"embedded {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    return embeddings
