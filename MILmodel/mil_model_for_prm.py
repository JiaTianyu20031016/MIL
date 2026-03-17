"""Minimal MIL model using a transformer backbone plus linear classification head."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoConfig, PretrainedConfig, AutoModelForCausalLM

from .mil_base import BaseMILModel, MLP, LinearAttention, Batch

class ProbAveragePoolMILModelforPRM(BaseMILModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)

    def _forward_impl(self, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]
        document_ids: Tensor = batch["input_ids"]
        document_mask: Tensor = batch["attention_mask"]

        if segment_ids.dim() != 3 or segment_mask.dim() != 3:
            raise ValueError("Segment tensors must be 3D with shape [batch, segments, seq_len].")

        batch_size, max_segments, segment_length = segment_ids.shape
        dtype = self.backbone_dtype
        device = segment_ids.device

        if max_segments == 0 or segment_length == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        # forward
        outputs = self.pretrained_model(
            input_ids=document_ids,
            attention_mask=document_mask,
            output_hidden_states=True,
        )

        # extract segment embeddings at each segment's end position
        end_positions = batch["segment_ends"]  # shape [batch, max_segments]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_segments)
        segment_embeddings_grid = outputs.hidden_states[-1][batch_indices, end_positions] # shape [batch, max_segments, hidden]
        segment_logits_grid = self.classifier(segment_embeddings_grid)    # shape [batch, max_segments, classes]
        segment_probs_grid = torch.softmax(segment_logits_grid, dim=-1)   # shape [batch, max_segments, classes]

        # average segment predictions to document-level
        num_classes = segment_logits_grid.size(-1)
        valid_segment_mask = segment_mask.any(dim=-1)  # shape [batch, max_segments]
        
        default_prob_row = torch.zeros((num_classes,), dtype=dtype, device=device)
        default_prob_row[0] = 1.0
        document_probs = self._average_segments_by_document(
            segment_probs_grid,
            valid_segment_mask,
            default_row=default_prob_row,
        )
        document_logits = self._average_segments_by_document(
            segment_logits_grid,
            valid_segment_mask,
            default_row=torch.zeros((num_classes,), dtype=dtype, device=device),
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

class InstanceAveragePoolMILModelforPRM(BaseMILModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)

    def _forward_impl(self, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]
        document_ids: Tensor = batch["input_ids"]
        document_mask: Tensor = batch["attention_mask"]

        if segment_ids.dim() != 3 or segment_mask.dim() != 3:
            raise ValueError("Segment tensors must be 3D with shape [batch, segments, seq_len].")

        batch_size, max_segments, segment_length = segment_ids.shape
        dtype = self.backbone_dtype
        device = segment_ids.device

        if max_segments == 0 or segment_length == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        # forward
        outputs = self.pretrained_model(
            input_ids=document_ids,
            attention_mask=document_mask,
            output_hidden_states=True,
        )

        # extract segment embeddings at each segment's end position
        end_positions = batch["segment_ends"]  # shape [batch, max_segments]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_segments)
        segment_embeddings_grid = outputs.hidden_states[-1][batch_indices, end_positions] # shape [batch, max_segments, hidden]
        segment_logits_grid = self.classifier(segment_embeddings_grid)    # shape [batch, max_segments, classes]
        segment_probs_grid = torch.softmax(segment_logits_grid, dim=-1)   # shape [batch, max_segments, classes]

        # average segment predictions to document-level
        hidden_dim = segment_embeddings_grid.size(-1)
        valid_segment_mask = segment_mask.any(dim=-1)  # shape [batch, max_segments]
        
        document_embeddings = self._average_segments_by_document(
            segment_embeddings_grid,
            valid_segment_mask,
            default_row=torch.zeros((hidden_dim,), dtype=dtype, device=device),
        )
        document_logits = self.classifier(document_embeddings)
        document_probs = torch.softmax(document_logits, dim=-1)


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

class AttentionPoolMILModelforPRM(BaseMILModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    supported_modules = ("classifier","attention")

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)
        self.attention = LinearAttention(input_dim=hidden_size).to(dtype=self.backbone_dtype)

    def _forward_impl(self, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]
        document_ids: Tensor = batch["input_ids"]
        document_mask: Tensor = batch["attention_mask"]

        if segment_ids.dim() != 3 or segment_mask.dim() != 3:
            raise ValueError("Segment tensors must be 3D with shape [batch, segments, seq_len].")

        batch_size, max_segments, segment_length = segment_ids.shape
        dtype = self.backbone_dtype
        device = segment_ids.device

        if max_segments == 0 or segment_length == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        # forward
        outputs = self.pretrained_model(
            input_ids=document_ids,
            attention_mask=document_mask,
            output_hidden_states=True,
        )

        # extract segment embeddings at each segment's end position
        end_positions = batch["segment_ends"]  # shape [batch, max_segments]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_segments)
        segment_embeddings_grid = outputs.hidden_states[-1][batch_indices, end_positions] # shape [batch, max_segments, hidden]
        segment_logits_grid = self.classifier(segment_embeddings_grid)    # shape [batch, max_segments, classes]
        segment_probs_grid = torch.softmax(segment_logits_grid, dim=-1)   # shape [batch, max_segments, classes]

        # weighted-average segment predictions to document-level
        hidden_dim = segment_embeddings_grid.size(-1)
        valid_segment_mask = segment_mask.any(dim=-1)  # shape [batch, max_segments]
        
        attention_weights = self.attention(segment_embeddings_grid, mask=valid_segment_mask)  # shape [batch, max_segments]
        document_embeddings = (segment_embeddings_grid * attention_weights).sum(dim=1)  # shape [batch, hidden]
        document_logits = self.classifier(document_embeddings)
        document_probs = torch.softmax(document_logits, dim=-1)

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
        }
        return document_probs, segment_probs_grid, extras

class ConjucturePoolMILModelforPRM(BaseMILModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    supported_modules = ("classifier","attention")

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)
        self.attention = LinearAttention(input_dim=hidden_size).to(dtype=self.backbone_dtype)

    def _forward_impl(self, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]
        document_ids: Tensor = batch["input_ids"]
        document_mask: Tensor = batch["attention_mask"]

        if segment_ids.dim() != 3 or segment_mask.dim() != 3:
            raise ValueError("Segment tensors must be 3D with shape [batch, segments, seq_len].")

        batch_size, max_segments, segment_length = segment_ids.shape
        dtype = self.backbone_dtype
        device = segment_ids.device

        if max_segments == 0 or segment_length == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        # forward
        outputs = self.pretrained_model(
            input_ids=document_ids,
            attention_mask=document_mask,
            output_hidden_states=True,
        )

        # extract segment embeddings at each segment's end position
        end_positions = batch["segment_ends"]  # shape [batch, max_segments]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_segments)
        segment_embeddings_grid = outputs.hidden_states[-1][batch_indices, end_positions] # shape [batch, max_segments, hidden]
        segment_logits_grid = self.classifier(segment_embeddings_grid)    # shape [batch, max_segments, classes]
        segment_probs_grid = torch.softmax(segment_logits_grid, dim=-1)   # shape [batch, max_segments, classes]

        # weighted-average segment predictions to document-level
        num_classes = segment_logits_grid.size(-1)
        valid_segment_mask = segment_mask.any(dim=-1)  # shape [batch, max_segments]
        
        attention_weights = self.attention(segment_embeddings_grid, mask=valid_segment_mask)  # shape [batch, max_segments]
        document_probs = (segment_probs_grid * attention_weights).sum(dim=1)  # shape [batch, classes]
        document_logits = (segment_logits_grid * attention_weights).sum(dim=1)  # shape [batch, classes]

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
        }
        return document_probs, segment_probs_grid, extras

class MinPoolMILModelforPRM(BaseMILModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, dropout=0.0, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)
        self.dropout = dropout

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)

    def _forward_impl(self, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]
        document_ids: Tensor = batch["input_ids"]
        document_mask: Tensor = batch["attention_mask"]

        if segment_ids.dim() != 3 or segment_mask.dim() != 3:
            raise ValueError("Segment tensors must be 3D with shape [batch, segments, seq_len].")

        batch_size, max_segments, segment_length = segment_ids.shape
        dtype = self.backbone_dtype
        device = segment_ids.device

        if max_segments == 0 or segment_length == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        # forward
        outputs = self.pretrained_model(
            input_ids=document_ids,
            attention_mask=document_mask,
            output_hidden_states=True,
        )

        # extract segment embeddings at each segment's end position
        end_positions = batch["segment_ends"]  # shape [batch, max_segments]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_segments)
        segment_embeddings_grid = outputs.hidden_states[-1][batch_indices, end_positions] # shape [batch, max_segments, hidden]
        segment_logits_grid = self.classifier(segment_embeddings_grid)    # shape [batch, max_segments, classes]
        segment_probs_grid = torch.softmax(segment_logits_grid, dim=-1)   # shape [batch, max_segments, classes]

        # document-level prediction by taking the minimum segment probability for the positive class
        positive_probs = segment_probs_grid[..., 1]  # shape [batch, max_segments]
        valid_segment_mask = segment_mask.any(dim=-1)  # shape [batch, max_segments]
        # random dropout of some valid segments during training for regularization (similar to DropBlock)
        dropout_mask = (torch.rand_like(positive_probs) < self.dropout) & valid_segment_mask  # shape [batch, max_segments]
        masked_positive_probs = positive_probs.masked_fill(~valid_segment_mask | dropout_mask, 1.0)  # treat invalid segments as having max positive probability
        min_positive_probs, _ = masked_positive_probs.min(dim=1)  # shape [batch]
        document_probs = torch.stack([1 - min_positive_probs, min_positive_probs], dim=-1)  # shape [batch, 2]
        document_logits = torch.log(document_probs + 1e-6)  # add small constant for numerical stability

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
        }
        return document_probs, segment_probs_grid, extras

class SoftMinPoolMILModelforPRM(BaseMILModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, dropout=0.0, temperature=0.5, **kwargs):
        self.temperature = temperature
        self.dropout = dropout
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)

    def _forward_impl(self, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]
        document_ids: Tensor = batch["input_ids"]
        document_mask: Tensor = batch["attention_mask"]

        if segment_ids.dim() != 3 or segment_mask.dim() != 3:
            raise ValueError("Segment tensors must be 3D with shape [batch, segments, seq_len].")

        batch_size, max_segments, segment_length = segment_ids.shape
        dtype = self.backbone_dtype
        device = segment_ids.device

        if max_segments == 0 or segment_length == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        # forward
        outputs = self.pretrained_model(
            input_ids=document_ids,
            attention_mask=document_mask,
            output_hidden_states=True,
        )

        # extract segment embeddings at each segment's end position
        end_positions = batch["segment_ends"]  # shape [batch, max_segments]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_segments)
        segment_embeddings_grid = outputs.hidden_states[-1][batch_indices, end_positions] # shape [batch, max_segments, hidden]
        segment_logits_grid = self.classifier(segment_embeddings_grid)    # shape [batch, max_segments, classes]
        segment_probs_grid = torch.softmax(segment_logits_grid, dim=-1)   # shape [batch, max_segments, classes]

        # document-level prediction by 'soft' minimum segment probability for the positive class
        # the softmin weights are computed as a softmax over the negative of the positive class probabilities, so that segments with lower positive probability get higher weight in the final document prediction
        positive_probs = segment_probs_grid[..., 1]  # shape [batch, max_segments]
        valid_segment_mask = segment_mask.any(dim=-1)  # shape [batch, max_segments]
        masked_positive_probs = positive_probs.masked_fill(~valid_segment_mask, torch.inf)  # treat invalid segments as having max positive probability
        # we apply stop_gradient to the softmin weights
        softmin_weights = torch.softmax((1 - masked_positive_probs) / self.temperature, dim=1).detach()  # shape [batch, max_segments]
        
        # random dropout of some valid segments during training for regularization (similar to DropBlock)
        # the dropout probability is scaled by the softmin weights 
        # so that segments with lower positive probability are more likely to be dropped, encouraging the model to consider multiple segments rather than just the most negative one
        doc_positive_prob = batch.get("positive_prob")
        if doc_positive_prob is not None:
            negative_doc_mask = (doc_positive_prob.to(device=device) == 0).unsqueeze(1)
        else:
            print("Warning: positive_prob not found in batch, using no negative doc masking for dropout.")
            negative_doc_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        dropout_mask = (
            torch.rand_like(positive_probs) < self.dropout * softmin_weights
        ) & valid_segment_mask & negative_doc_mask  # shape [batch, max_segments]
        masked_softmin_weights = softmin_weights.masked_fill(~valid_segment_mask | dropout_mask, 0.0)  # shape [batch, max_segments], zero out weights for invalid and dropped segments
        masked_softmin_weights = masked_softmin_weights / masked_softmin_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)  # renormalize weights to sum to 1

        document_probs = (segment_probs_grid * masked_softmin_weights.unsqueeze(-1)).sum(dim=1)  # shape [batch, classes]
        document_logits = torch.log(document_probs + 1e-6)  # add small constant for numerical stability

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
        }
        return document_probs, segment_probs_grid, extras

class NaiveMILModelforPRM(BaseMILModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)

    def _forward_impl(self, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]
        document_ids: Tensor = batch["input_ids"]
        document_mask: Tensor = batch["attention_mask"]

        if segment_ids.dim() != 3 or segment_mask.dim() != 3:
            raise ValueError("Segment tensors must be 3D with shape [batch, segments, seq_len].")

        batch_size, max_segments, segment_length = segment_ids.shape
        dtype = self.backbone_dtype
        device = segment_ids.device

        if max_segments == 0 or segment_length == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        # forward
        outputs = self.pretrained_model(
            input_ids=document_ids,
            attention_mask=document_mask,
            output_hidden_states=True,
        )

        # extract segment embeddings at each segment's end position
        end_positions = batch["segment_ends"]  # shape [batch, max_segments]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_segments)
        segment_embeddings_grid = outputs.hidden_states[-1][batch_indices, end_positions] # shape [batch, max_segments, hidden]
        segment_logits_grid = self.classifier(segment_embeddings_grid)    # shape [batch, max_segments, classes]
        segment_probs_grid = torch.softmax(segment_logits_grid, dim=-1)   # shape [batch, max_segments, classes]

        # document-level prediction by taking last valid segment's prediction
        last_valid_indices = segment_mask.any(dim=-1).sum(dim=-1) - 1  # shape [batch]
        batch_indices = torch.arange(batch_size, device=device)
        document_probs = segment_probs_grid[batch_indices, last_valid_indices]  # shape [batch, classes]
        document_logits = segment_logits_grid[batch_indices, last_valid_indices]  # shape [batch, classes]

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
        }
        return document_probs, segment_probs_grid, extras


class BufferBaselineModelforPRM(BaseMILModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)

    def _forward_impl(self, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]
        document_ids: Tensor = batch["input_ids"]
        document_mask: Tensor = batch["attention_mask"]

        if segment_ids.dim() != 3 or segment_mask.dim() != 3:
            raise ValueError("Segment tensors must be 3D with shape [batch, segments, seq_len].")

        batch_size, max_segments, segment_length = segment_ids.shape
        dtype = self.backbone_dtype
        device = segment_ids.device

        if max_segments == 0 or segment_length == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        # forward
        outputs = self.pretrained_model(
            input_ids=document_ids,
            attention_mask=document_mask,
            output_hidden_states=True,
        )

        # extract segment embeddings at each segment's end position
        end_positions = batch["segment_ends"]  # shape [batch, max_segments]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_segments)
        segment_embeddings_grid = outputs.hidden_states[-1][batch_indices, end_positions] # shape [batch, max_segments, hidden]
        segment_logits_grid = self.classifier(segment_embeddings_grid)    # shape [batch, max_segments, classes]
        segment_probs_grid = torch.softmax(segment_logits_grid, dim=-1)   # shape [batch, max_segments, classes]

        # document-level prediction by taking last valid segment's prediction
        last_valid_indices = segment_mask.any(dim=-1).sum(dim=-1) - 1  # shape [batch]
        batch_indices = torch.arange(batch_size, device=device)
        document_probs = segment_probs_grid[batch_indices, last_valid_indices]  # shape [batch, classes]
        document_logits = segment_logits_grid[batch_indices, last_valid_indices]  # shape [batch, classes]

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
        }
        return document_probs, segment_probs_grid, extras


class DPOBaselineModelforPRM(BaseMILModel):
    """Feeds segment batches through a pretrained backbone and averages predictions per document."""

    transformers_parent_class = AutoModelForCausalLM
    supported_modules = ("ref_model",)

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        self.ref_model = self.pretrained_model.clone()

    def _forward_impl(self, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]
        document_ids: Tensor = batch["input_ids"]
        document_mask: Tensor = batch["attention_mask"]

        if segment_ids.dim() != 3 or segment_mask.dim() != 3:
            raise ValueError("Segment tensors must be 3D with shape [batch, segments, seq_len].")

        batch_size, max_segments, segment_length = segment_ids.shape
        dtype = self.backbone_dtype
        device = segment_ids.device

        if max_segments == 0 or segment_length == 0:
            doc_probs = torch.zeros((batch_size, 2), dtype=dtype, device=device)
            doc_probs[:, 0] = 1.0
            empty = torch.zeros((0, 2), dtype=dtype, device=device)
            return doc_probs, empty, None

        # forward
        outputs = self.pretrained_model(
            input_ids=document_ids,
            attention_mask=document_mask,
            output_hidden_states=True,
        )

        # extract segment embeddings at each segment's end position
        end_positions = batch["segment_ends"]  # shape [batch, max_segments]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_segments)
        segment_embeddings_grid = outputs.hidden_states[-1][batch_indices, end_positions] # shape [batch, max_segments, hidden]
        segment_logits_grid = self.classifier(segment_embeddings_grid)    # shape [batch, max_segments, classes]
        segment_probs_grid = torch.softmax(segment_logits_grid, dim=-1)   # shape [batch, max_segments, classes]

        # document-level prediction by taking last valid segment's prediction
        last_valid_indices = segment_mask.any(dim=-1).sum(dim=-1) - 1  # shape [batch]
        batch_indices = torch.arange(batch_size, device=device)
        document_probs = segment_probs_grid[batch_indices, last_valid_indices]  # shape [batch, classes]
        document_logits = segment_logits_grid[batch_indices, last_valid_indices]  # shape [batch, classes]

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
        }
        return document_probs, segment_probs_grid, extras


__all__ = [
    "ProbAveragePoolMILModelforPRM", 
    "InstanceAveragePoolMILModelforPRM", 
    "AttentionPoolMILModelforPRM", 
    "ConjucturePoolMILModelforPRM", 
    "MinPoolMILModelforPRM", 
    "SoftMinPoolMILModelforPRM",
    "NaiveMILModelforPRM",
]
