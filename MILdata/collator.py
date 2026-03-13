"""Custom data collator for MIL datasets that aligns documents and segments with left padding."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
class MILDataCollator:
    """Pads document- and segment-level sequences for MIL training."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Collate batch is empty.")

        max_seq_len = max(len(item["input_ids"]) for item in batch)
        max_segments = max(len(item["segment_token_ids"]) for item in batch)
        batch_size = len(batch)

        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        segment_ends = (
            torch.full((batch_size, max_segments), -1, dtype=torch.long)
            if max_segments > 0
            else torch.empty((batch_size, 0), dtype=torch.long)
        )
        max_segment_len = (
            max((len(seg) for item in batch for seg in item["segment_token_ids"]), default=0)
            if max_segments > 0
            else 0
        )
        if max_segments == 0:
            segment_input_ids = torch.empty((batch_size, 0, 0), dtype=torch.long)
            segment_attention_mask = torch.empty((batch_size, 0, 0), dtype=torch.long)
            segment_positive_probs = torch.empty((batch_size, 0), dtype=torch.float32)
        elif max_segment_len == 0:
            segment_input_ids = torch.empty((batch_size, max_segments, 0), dtype=torch.long)
            segment_attention_mask = torch.empty((batch_size, max_segments, 0), dtype=torch.long)
            segment_positive_probs = torch.zeros((batch_size, max_segments), dtype=torch.float32)
        else:
            segment_input_ids = torch.full(
                (batch_size, max_segments, max_segment_len), self.pad_token_id, dtype=torch.long
            )
            segment_attention_mask = torch.zeros(
                (batch_size, max_segments, max_segment_len), dtype=torch.long
            )
            segment_positive_probs = torch.zeros((batch_size, max_segments), dtype=torch.float32)
        document_texts: List[str] = []
        prompt_texts: List[str] = []
        segment_texts: List[List[str]] = []

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
            prompt_texts.append(item.get("prompt_text", ""))
            segment_text_entries = item.get("segment_texts", [])
            if len(segment_text_entries) != len(item["segment_token_ids"]):
                raise ValueError("segment_texts must align with segment_token_ids.")
            segment_texts.append(list(segment_text_entries))

            segment_probs = item.get("segment_positive_probs")
            if segment_probs is None:
                raise ValueError("segment_positive_probs missing from batch item.")
            if len(segment_probs) != len(item["segment_token_ids"]):
                raise ValueError("segment_positive_probs must align with segment_token_ids.")
            if max_segments > 0 and len(segment_probs) > 0:
                segment_positive_probs[row, : len(segment_probs)] = torch.tensor(
                    segment_probs, dtype=torch.float32
                )

            for seg_idx, (segment_tokens, _) in enumerate(
                zip(item["segment_token_ids"], segment_text_entries)
            ):
                if max_segment_len == 0:
                    continue
                trim_tokens = segment_tokens[-max_segment_len:]
                effective_len = len(trim_tokens)
                seg_start = max_segment_len - effective_len
                if effective_len > 0:
                    segment_input_ids[row, seg_idx, seg_start:] = torch.tensor(
                        trim_tokens, dtype=torch.long
                    )
                    segment_attention_mask[row, seg_idx, seg_start:] = 1

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
            "segment_positive_probs": segment_positive_probs,
        }
        # keep metadata as lists for downstream logging/debugging
        batch_tensor.update(
            {
                "doc_ids": [item["doc_id"] for item in batch],
                "source": [item["source"] for item in batch],
                "granularity": [item["granularity"] for item in batch],
                "document_texts": document_texts,
                "prompt_texts": prompt_texts,
                "segment_texts": segment_texts,
            }
        )
        return batch_tensor


__all__ = ["MILDataCollator"]
