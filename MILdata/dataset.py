"""Utility helpers for loading MIL text datasets with soft labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .collator import MILDataCollator

_SENTIMENT_MAP: Dict[str, int] = {"+": 1, "-": -1, "0": 0}


@dataclass(frozen=True)
class Segment:
    """Represents a single EDU or sentence with its token-level sentiment tag."""

    label: int
    positive_prob: float
    text: str


@dataclass(frozen=True)
class DocumentSample:
    """Container for a full document and its derived soft label."""

    doc_id: str
    rating: int
    positive_prob: float
    segments: List[Segment]
    granularity: Literal["edu", "sent"]
    source: Literal["imdb", "yelp"]


_DATASETS = {
    "imdb_edus": {"filename": "imdb_edus.txt", "scale": 9, "source": "imdb", "granularity": "edu"},
    "imdb_sent": {"filename": "imdb_sent.txt", "scale": 9, "source": "imdb", "granularity": "sent"},
    "yelp_edus": {"filename": "yelp_edus.txt", "scale": 4, "source": "yelp", "granularity": "edu"},
    "yelp_sent": {"filename": "yelp_sent.txt", "scale": 4, "source": "yelp", "granularity": "sent"},
}


def load_dataset(name: str, data_dir: Optional[Path] = None) -> List[DocumentSample]:
    """Load a dataset by name and convert raw ratings into soft binary labels."""

    if name not in _DATASETS:
        valid = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Expected one of: {valid}.")

    config = _DATASETS[name]
    base_dir = data_dir or Path(__file__).resolve().parent
    file_path = base_dir / config["filename"]
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")

    return list(
        _parse_file(
            file_path=file_path,
            scale=config["scale"],
            source=config["source"],
            granularity=config["granularity"],
        )
    )


def _parse_file(
    *,
    file_path: Path,
    scale: int,
    source: Literal["imdb", "yelp"],
    granularity: Literal["edu", "sent"],
) -> Iterable[DocumentSample]:
    current_id: Optional[str] = None
    current_rating: Optional[int] = None
    current_segments: List[Segment] = []

    def finalize() -> Optional[DocumentSample]:
        if current_id is None or current_rating is None:
            return None
        sample = DocumentSample(
            doc_id=current_id,
            rating=current_rating,
            positive_prob=_soft_label(current_rating, scale),
            segments=list(current_segments),
            granularity=granularity,
            source=source,
        )
        return sample

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                sample = finalize()
                if sample:
                    yield sample
                current_id = None
                current_rating = None
                current_segments = []
                continue

            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                sample = finalize()
                if sample:
                    yield sample
                current_rating = int(parts[0])
                current_id = parts[1]
                current_segments = []
                continue

            if current_id is None or current_rating is None:
                raise ValueError(f"Encountered content line before header in {file_path}.")

            sentiment_token = parts[0]
            if sentiment_token not in _SENTIMENT_MAP:
                raise ValueError(f"Unknown sentiment token '{sentiment_token}' in {file_path}.")
            raw_text = parts[1] if len(parts) > 1 else ""
            current_segments.append(
                Segment(
                    label=_SENTIMENT_MAP[sentiment_token],
                    positive_prob=_segment_positive_prob(_SENTIMENT_MAP[sentiment_token]),
                    text=_clean_segment_text(raw_text),
                )
            )

    sample = finalize()
    if sample:
        yield sample


def _soft_label(rating: int, scale: int) -> float:
    if scale <= 0:
        raise ValueError("Scale must be positive.")
    return round(rating / scale, 4)


def _segment_positive_prob(label: int) -> float:
    if label == 1:
        return 1.0
    if label == -1:
        return 0.0
    if label == 0:
        return 0.5
    raise ValueError("Segment label must be one of {-1, 0, 1}.")


def _clean_segment_text(text: str) -> str:
    """Strip trailing sentence tokens so segments can be concatenated cleanly."""

    cleaned = text.rstrip()
    if cleaned.endswith("<s>"):
        cleaned = cleaned[:-3].rstrip()
    return cleaned


class TokenizedDocumentDataset(Dataset):
    """Torch dataset that tokenizes documents and stores sentence boundary offsets."""

    def __init__(
        self,
        samples: Sequence[DocumentSample],
        tokenizer: Any,
        *,
        max_length: Optional[int] = None,
    ) -> None:
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("Tokenizer must define pad_token_id for batching.")
        self._samples = list(samples)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._pad_token_id = tokenizer.pad_token_id

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        tokenized = _encode_document(sample, self._tokenizer, self._max_length)
        tokenized.update(
            {
                "doc_id": sample.doc_id,
                "positive_prob": sample.positive_prob,
                "rating": sample.rating,
                "source": sample.source,
                "granularity": sample.granularity,
                "document_text": " ".join(segment.text for segment in sample.segments).strip(),
                "segment_texts": [segment.text for segment in sample.segments],
            }
        )
        return tokenized


def build_document_dataloader(
    samples: Sequence[DocumentSample],
    tokenizer: Any,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    max_length: Optional[int] = None,
    collate_fn: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
) -> DataLoader:
    """Create a DataLoader that yields padded batches with sentence-boundary offsets."""

    dataset = TokenizedDocumentDataset(samples, tokenizer, max_length=max_length)
    if collate_fn is None:
        collate_fn = MILDataCollator(pad_token_id=tokenizer.pad_token_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def _encode_document(
    sample: DocumentSample,
    tokenizer: Any,
    max_length: Optional[int],
) -> Dict[str, Any]:
    flat_ids: List[int] = []
    segment_lengths: List[int] = []
    segment_token_ids: List[List[int]] = []
    for segment in sample.segments:
        encoded = tokenizer(segment.text, add_special_tokens=False)
        ids = _ensure_1d_ids(encoded)
        segment_token_ids.append(ids)
        segment_lengths.append(len(ids))
        flat_ids.extend(ids)

    input_ids = list(flat_ids)

    segment_ends: List[int] = []
    cursor = 0
    for length in segment_lengths:
        if length == 0:
            segment_ends.append(-1 if cursor == 0 else cursor - 1)
            continue
        cursor += length
        segment_ends.append(cursor - 1)

    if max_length is not None and max_length > 0 and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    attention_mask = [1] * len(input_ids)
    segment_ends = [pos for pos in segment_ends if 0 <= pos < len(input_ids)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ends": segment_ends,
        "segment_token_ids": segment_token_ids,
    }


def _ensure_1d_ids(tokenizer_output: Any) -> List[int]:
    """Normalize tokenizer outputs (list or dict) into a flat list of token ids."""

    if hasattr(tokenizer_output, "input_ids"):
        ids = tokenizer_output.input_ids
    else:
        ids = tokenizer_output

    if ids is None:
        raise ValueError("Tokenizer output does not contain 'input_ids'.")

    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Batch encoding is not supported for segment-level tokenization.")
        ids = ids[0]

    return list(ids)


def create_mil_data_collator(tokenizer: Any) -> MILDataCollator:
    """Convenience helper to build the default MILDataCollator from a tokenizer."""

    if getattr(tokenizer, "pad_token_id", None) is None:
        raise ValueError("Tokenizer must define pad_token_id for batching.")
    return MILDataCollator(pad_token_id=tokenizer.pad_token_id)


__all__ = [
    "DocumentSample",
    "Segment",
    "TokenizedDocumentDataset",
    "build_document_dataloader",
    "create_mil_data_collator",
    "load_dataset",
]
