"""Custom data collator for MIL datasets that aligns documents and segments with left padding."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor


class MILDataCollator:
    """Pads document- and segment-level sequences for MIL training."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Collate batch is empty.")

        max_seq_len = max(len(item["input_ids"]) for item in batch)
        max_segments = max(len(item["segment_ends"]) for item in batch)
        batch_size = len(batch)

        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        segment_ends = (
            torch.full((batch_size, max_segments), -1, dtype=torch.long)
            if max_segments > 0
            else torch.empty((batch_size, 0), dtype=torch.long)
        )

        total_segments = sum(len(item["segment_token_ids"]) for item in batch)
        max_segment_len = (
            max((len(seg) for item in batch for seg in item["segment_token_ids"]), default=0)
            if total_segments > 0
            else 0
        )
        if total_segments > 0 and max_segment_len > 0:
            segment_input_ids = torch.full(
                (total_segments, max_segment_len), self.pad_token_id, dtype=torch.long
            )
            segment_attention_mask = torch.zeros((total_segments, max_segment_len), dtype=torch.long)
        else:
            segment_input_ids = torch.empty((total_segments, 0), dtype=torch.long)
            segment_attention_mask = torch.empty((total_segments, 0), dtype=torch.long)
        segment_to_doc = torch.empty(total_segments, dtype=torch.long)
        document_texts: List[str] = []
        segment_texts: List[str] = []

        segment_row = 0
        for row, item in enumerate(batch):
            length = len(item["input_ids"])
            start = max_seq_len - length
            input_ids[row, start:] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[row, start:] = 1

            ends = item["segment_ends"]
            if max_segments > 0 and ends:
                shifted = [pos + start if pos >= 0 else -1 for pos in ends]
                segment_ends[row, : len(ends)] = torch.tensor(shifted, dtype=torch.long)

            document_texts.append(item.get("document_text", ""))
            segment_text_entries = item.get("segment_texts", [])
            if len(segment_text_entries) != len(item["segment_token_ids"]):
                raise ValueError("segment_texts must align with segment_token_ids.")

            for segment_tokens, segment_text in zip(item["segment_token_ids"], segment_text_entries):
                if max_segment_len > 0:
                    seg_length = len(segment_tokens)
                    trim_tokens = segment_tokens[-max_segment_len:]
                    effective_len = len(trim_tokens)
                    seg_start = max_segment_len - effective_len
                    if effective_len > 0:
                        segment_input_ids[segment_row, seg_start:] = torch.tensor(
                            trim_tokens, dtype=torch.long
                        )
                        segment_attention_mask[segment_row, seg_start:] = 1
                segment_to_doc[segment_row] = row
                segment_texts.append(segment_text)
                segment_row += 1

        batch_tensor = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "segment_ends": segment_ends,
            "positive_prob": torch.tensor(
                [item["positive_prob"] for item in batch], dtype=torch.float32
            ),
            "rating": torch.tensor([item["rating"] for item in batch], dtype=torch.long),
            "segment_input_ids": segment_input_ids,
            "segment_attention_mask": segment_attention_mask,
            "segment_to_doc": segment_to_doc,
        }
        # keep metadata as lists for downstream logging/debugging
        batch_tensor.update(
            {
                "doc_ids": [item["doc_id"] for item in batch],
                "source": [item["source"] for item in batch],
                "granularity": [item["granularity"] for item in batch],
                "document_texts": document_texts,
                "segment_texts": segment_texts,
            }
        )
        return batch_tensor


__all__ = ["MILDataCollator"]
