from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Literal
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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
    rating: int
    positive_prob: float
    prompt: str
    segments: List[Segment]
    granularity: Literal["step"]
    source: Literal["math-shepherd/gsm8k", "math-shepherd/math"]


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

    @staticmethod
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

    @staticmethod
    def _encode_document(
        sample: DocumentSample,
        tokenizer: Any,
        max_length: Optional[int],
    ) -> Dict[str, Any]:
        prompt_ids = TokenizedDocumentDataset._ensure_1d_ids(tokenizer(sample.prompt, add_special_tokens=False))
        flat_ids = prompt_ids.copy()
        segment_lengths: List[int] = []
        segment_token_ids: List[List[int]] = []
        for segment in sample.segments:
            encoded = tokenizer(segment.text, add_special_tokens=False)
            ids = TokenizedDocumentDataset._ensure_1d_ids(encoded)
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
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        tokenized = TokenizedDocumentDataset._encode_document(sample, self._tokenizer, self._max_length)
        tokenized.update(
            {
                "doc_id": sample.doc_id,
                "positive_prob": sample.positive_prob,
                "rating": sample.rating,
                "source": sample.source,
                "granularity": sample.granularity,
                "prompt_text": sample.prompt,
                "document_text": sample.prompt + "".join(segment.text for segment in sample.segments).strip(),
                "segment_texts": [segment.text for segment in sample.segments],
                "segment_positive_probs": [segment.positive_prob for segment in sample.segments],
            }
        )
        return tokenized