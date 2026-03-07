"""Minimal MIL model using a transformer backbone plus linear classification head."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
        if segment_ids.size(0) == 0:
            num_docs = self._document_count(batch)
            dtype = self.classifier.weight.dtype
            device = segment_ids.device
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            doc_probs = torch.zeros((num_docs, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            return doc_probs, empty, None

        outputs = self.pretrained_model(input_ids=segment_ids, attention_mask=segment_mask, output_hidden_states=True)
        segment_embeddings = outputs.hidden_states[-1][:,-1]
        segment_logits = self.classifier(segment_embeddings)
        segment_probs = torch.softmax(segment_logits, dim=-1)

        document_probs = self._average_segments_by_document(segment_probs, batch)
        extras = {
            "segment_logits": segment_logits,
            "document_logits": self._average_segments_by_document(segment_logits, batch),
        }
        return document_probs, segment_probs, extras

    @staticmethod
    def _mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        mask = attention_mask.unsqueeze(-1)
        masked_hidden = hidden_states * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1e-6)
        return summed / counts

    def _average_segments_by_document(self, values: Tensor, batch: Batch) -> Tensor:
        segment_to_doc: Tensor = batch.get("segment_to_doc")
        num_docs = self._document_count(batch)
        if segment_to_doc is None or segment_to_doc.size(0) == 0:
            return values.new_zeros((num_docs, values.size(-1)))

        device = values.device
        num_classes = values.size(-1)
        document_values = torch.zeros((num_docs, num_classes), device=device, dtype=values.dtype)
        segment_to_doc_expanded = segment_to_doc.to(device).unsqueeze(-1).expand(-1, num_classes)
        document_values.scatter_add_(0, segment_to_doc_expanded, values)

        counts = torch.zeros(num_docs, device=device, dtype=values.dtype)
        counts.scatter_add_(0, segment_to_doc.to(device), torch.ones(values.size(0), device=device, dtype=values.dtype))
        counts = counts.clamp_min(1e-6)
        return document_values / counts.unsqueeze(-1)


__all__ = ["SimpleMILModel", "SimpleMILConfig"]
