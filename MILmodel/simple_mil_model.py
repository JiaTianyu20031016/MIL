"""Minimal MIL model using a transformer backbone plus linear classification head."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoConfig, PretrainedConfig

from .mil_base import BaseMILModel, Batch

class SimpleMILModel(BaseMILModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = nn.Linear(hidden_size, 2).to(dtype=backbone_dtype)

    def _forward_impl(self, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]

        if segment_ids.dim() != 3 or segment_mask.dim() != 3:
            raise ValueError("Segment tensors must be 3D with shape [batch, segments, seq_len].")

        batch_size, max_segments, segment_length = segment_ids.shape
        dtype = self.classifier.weight.dtype
        device = segment_ids.device

        if max_segments == 0 or segment_length == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        # flatten segments to a single batch dimension for processing, then un-flatten results back to [batch, max_segments, classes]
        valid_segment_mask = segment_mask.any(dim=-1)
        flat_valid_mask = valid_segment_mask.reshape(-1)
        valid_count = int(flat_valid_mask.sum().item())

        if valid_count == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        flat_segment_ids = segment_ids.reshape(-1, segment_length)[flat_valid_mask]
        flat_segment_mask = segment_mask.reshape(-1, segment_length)[flat_valid_mask]

        # forward
        outputs = self.pretrained_model(
            input_ids=flat_segment_ids,
            attention_mask=flat_segment_mask,
            output_hidden_states=True,
        )
        segment_embeddings = outputs.hidden_states[-1][:, -1]
        segment_logits = self.classifier(segment_embeddings)
        segment_probs = torch.softmax(segment_logits, dim=-1)

        # un-flatten segment predictions back to [batch, max_segments, classes]
        num_classes = segment_probs.size(-1)
        segment_logits_grid = segment_logits.new_zeros((batch_size, max_segments, num_classes))
        segment_probs_grid = segment_probs.new_zeros((batch_size, max_segments, num_classes))

        mask_bool = valid_segment_mask.bool()
        segment_logits_grid[mask_bool] = segment_logits
        segment_probs_grid[mask_bool] = segment_probs

        # average segment predictions to document-level
        default_prob_row = torch.zeros((num_classes,), dtype=segment_probs.dtype, device=device)
        default_prob_row[0] = 1.0
        document_probs = self._average_segments_by_document(
            segment_probs_grid,
            valid_segment_mask,
            default_row=default_prob_row,
        )
        document_logits = self._average_segments_by_document(
            segment_logits_grid,
            valid_segment_mask,
            default_row=torch.zeros((num_classes,), dtype=segment_logits.dtype, device=device),
        )

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
        }
        return document_probs, segment_probs_grid, extras

    def _average_segments_by_document(
        self,
        values: Tensor,
        valid_mask: Tensor,
        *,
        default_row: Optional[Tensor] = None,
    ) -> Tensor:
        if values.dim() != 3:
            raise ValueError("values must be shaped [batch, segments, classes].")
        if valid_mask.dim() != 2:
            raise ValueError("valid_mask must be shaped [batch, segments].")
        if values.shape[:2] != valid_mask.shape:
            raise ValueError("values and valid_mask must share batch and segment dimensions.")

        num_docs, _, num_classes = values.shape
        mask = valid_mask.unsqueeze(-1).to(values.dtype)
        summed = (values * mask).sum(dim=1)
        counts = mask.sum(dim=1)

        doc_values = values.new_zeros((num_docs, num_classes))
        nonzero = counts.squeeze(-1) > 0
        if nonzero.any():
            doc_values[nonzero] = summed[nonzero] / counts[nonzero].clamp_min(1e-6)

        if default_row is not None:
            if default_row.dim() != 1 or default_row.size(0) != num_classes:
                raise ValueError("default_row must be a 1D tensor matching the class dimension.")
            if (~nonzero).any():
                doc_values[~nonzero] = default_row.to(values.device, values.dtype)

        return doc_values


__all__ = ["SimpleMILModel", "SimpleMILConfig"]
