"""Shared data structures and helpers for MIL datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from torch.utils.data import DataLoader, Dataset
from MILdata.collator import MILDataCollator


@dataclass(frozen=True)
class Segment:
    """Represents a single reasoning step and its correctness label."""

    label: int
    positive_prob: float
    text: str


@dataclass(frozen=True)
class DocumentSample:
    """Container for a prompt with its step-level supervision."""

    doc_id: str
    rating: float
    positive_prob: float
    prompt: str
    segments: List[Segment]
    granularity: Literal["step"]
    source: str


def load_hf_split(*, split_name: str, dataset_id: str, config_name: Optional[str]):
    """Load a Hugging Face split or a local saved dataset."""

    try:
        from datasets import load_dataset as hf_load_dataset, load_from_disk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "datasets package is required for backend='hf'. Install via 'pip install datasets'."
        ) from exc

    dataset_path = Path(dataset_id).expanduser()
    if dataset_path.exists():
        dataset = load_from_disk(str(dataset_path))
        if split_name not in dataset:
            raise ValueError(f"Split '{split_name}' not found in dataset stored at '{dataset_path}'.")
        return dataset[split_name]

    return hf_load_dataset(dataset_id, name=config_name, split=split_name)


class TokenizedDocumentDataset(Dataset):
    """Torch dataset that tokenizes documents and stores sentence boundary offsets."""

    def __init__(
        self,
        samples: Sequence[DocumentSample],
        tokenizer: Any,
        *,
        separator: Optional[str] = '\n\n',
        apply_chat_template: bool = False,
        max_length: Optional[int] = None,
    ) -> None:
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("Tokenizer must define pad_token_id for batching.")
        self._samples = list(samples)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._pad_token_id = tokenizer.pad_token_id

        # separator used to join prompt and segments for the "document_text" field
        self.separator = separator if separator is not None else ''
        # whether to apply a chat-style template to the prompt and segments when constructing "document_text"
        self.apply_chat_template = apply_chat_template

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        tokenized = _encode_document(
            sample, 
            self._tokenizer, 
            separator=self.separator,
            apply_chat_template=self.apply_chat_template,
            max_length=self._max_length
        )
        tokenized.update(
            {
                "doc_id": sample.doc_id,
                "positive_prob": sample.positive_prob,
                "rating": sample.rating,
                "source": sample.source,
                "granularity": sample.granularity,
                "segment_positive_probs": [segment.positive_prob for segment in sample.segments],
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
    separator: Optional[str] = '\n\n',
    apply_chat_template: Optional[bool] = False,
    max_length: Optional[int] = None
) -> Dict[str, Any]:
    separator = separator if separator is not None else ''
    prompt = sample.prompt + separator
    if apply_chat_template:
        messages = [
            {
                'role': 'user',
                'content': prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # if 'qwen3' in tokenizer.name_or_path.lower():
        #     prompt += "<think>\n\n</think>\n\n"
    prompt_ids = _ensure_1d_ids(tokenizer(prompt, add_special_tokens=False))

    flat_ids = prompt_ids.copy()
    segment_lengths: List[int] = []
    segment_token_ids: List[List[int]] = []
    for segment in sample.segments:
        encoded = tokenizer(segment.text + separator, add_special_tokens=False)
        ids = _ensure_1d_ids(encoded)
        segment_token_ids.append(ids)
        segment_lengths.append(len(ids))
        flat_ids.extend(ids)

    input_ids = list(flat_ids)

    segment_ends: List[int] = []
    cursor = len(prompt_ids)
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
        "prompt_ids": prompt_ids,
        "prompt_text": prompt,
        "segment_texts": [segment.text + separator for segment in sample.segments],
        "document_text": prompt + "".join(segment.text + separator for segment in sample.segments).strip(),
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
    "Segment",
    "DocumentSample",
    "TokenizedDocumentDataset",
    "build_document_dataloader",
    "create_mil_data_collator",
    "load_hf_split",
]
