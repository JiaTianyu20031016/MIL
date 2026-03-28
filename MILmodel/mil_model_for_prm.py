"""Minimal MIL model using a transformer backbone plus linear classification head."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoConfig, PretrainedConfig, AutoModelForCausalLM
from trl.trainer.utils import selective_log_softmax
from trl.models.utils import disable_gradient_checkpointing
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

    def _forward_impl(self, eval, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
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
    """
    Feeds segment batches through a pretrained backbone,
      averages the segment embeddings per document as document embeddings,
      and feeds the document embeddings through the classifier to get document predictions.
    """

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)

    def _forward_impl(self, eval, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
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
    """
    Feeds segment batches through a pretrained backbone and averages segment embeddings per document.
    The segment embeddings are weighted by an attention mechanism before being averaged to form the document embedding, 
    which is then fed through the classifier to get document predictions.
    """

    supported_modules = ("classifier","attention")

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)
        self.attention = LinearAttention(input_dim=hidden_size).to(dtype=self.backbone_dtype)

    def _forward_impl(self, eval, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
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

class ConjunctivePoolMILModelforPRM(BaseMILModel):
    """
    Feeds segment batches through a pretrained backbone and averages predictions per document.
    The segment predictions are weighted by an attention mechanism before being averaged to form the document prediction.
    """

    supported_modules = ("classifier","attention")

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)
        self.attention = LinearAttention(input_dim=hidden_size).to(dtype=self.backbone_dtype)

    def _forward_impl(self, eval, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
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
    """
    Feeds segment batches through a pretrained backbone, 
    and gets document-level predictions by taking the minimum segment probability for the positive class.
    Instance dropout is optional.
    """

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, dropout=0.0, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)
        self.dropout = dropout

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)

    def _forward_impl(self, eval, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
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
    """
    'Soft' version of MinPoolMILModelforPRM where segment probabilities are combined using a weighted average with softmin weights based on the positive class probabilities, rather than taking a hard minimum.
    Control the softness of the minimum with a temperature parameter, and optionally apply instance dropout for regularization as in MinPoolMILModelforPRM.
    """

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, dropout=0.0, temperature=0.5, **kwargs):
        self.temperature = temperature
        self.dropout = dropout
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)

    def _forward_impl(self, eval, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
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
        
        # debug metrics
        weight_entropy = -(softmin_weights * (softmin_weights + 1e-8).log()).sum(dim=1)  # shape [batch], measure of how many segments are contributing to the prediction (higher entropy means more segments are contributing)
        relative_positions = torch.arange(max_segments, device=device).unsqueeze(0).expand(batch_size, -1) / max_segments  # shape [batch, max_segments], relative position of each segment in the document
        weighted_relative_position = (relative_positions * softmin_weights).sum(dim=1)  # shape [batch], average relative position of the segments contributing to the prediction (higher means more later segments are contributing)

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
            "softmin_weights": softmin_weights,
            "masked_softmin_weights": masked_softmin_weights,
            "debug_weight_entropy": weight_entropy,
            "debug_weighted_relative_position": weighted_relative_position,
        }
        return document_probs, segment_probs_grid, extras

class NaiveMILModelforPRM(BaseMILModel):
    """
    Feeds segment batches through a pretrained backbone, and
    document-level predictions are made by taking the prediction of the last valid segment.
    """

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)
    

    def _forward_impl(self, eval, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
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

class NoisyORPoolMILModelforPRM(BaseMILModel):
    """
    Feeds segment batches through a pretrained backbone, and
    document-level predictions are made by treating the positive class probabilities of the segments as independent and computing the probability that all segments are positive (i.e. Noisy-OR).
    """

    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, temperature=0.5, **kwargs):
        self.temperature = temperature
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=2).to(dtype=self.backbone_dtype)

    def _forward_impl(self, eval, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
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

        # document-level prediction by multiplying the positive class probabilities across segments (i.e. Noisy-OR)
        segment_positive_probs = segment_probs_grid[..., 1]  # shape [batch, max_segments]
        valid_segment_mask = segment_mask.any(dim=-1)  # shape [batch, max_segments]
        masked_positive_probs = segment_positive_probs.masked_fill(~valid_segment_mask, 1.0) 
        document_positive_probs = torch.prod(masked_positive_probs, dim=1)  # shape [batch], probability that all segments are positive
        document_probs = torch.stack([1 - document_positive_probs, document_positive_probs], dim=-1)  # shape [batch, 2]
        document_logits = torch.log(document_probs + 1e-6)  # add small constant for numerical stability

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
        }
        return document_probs, segment_probs_grid, extras


class BufferBaselineModelforPRM(BaseMILModel):
    supported_modules = ("classifier",)

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)

    def _init_weights(self, **kwargs):
        hidden_size = self.pretrained_model.config.hidden_size
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier = MLP(input_dim=hidden_size, hidden_dim=hidden_size, output_dim=3).to(dtype=self.backbone_dtype)

    def _forward_impl(self, eval, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
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

        # random buffer dropout: beta ~ Bornulli(segment_buffer_prob)
        segment_buffer_prob = segment_probs_grid[..., 2]  # shape [batch, max_segments]
        if not eval:   
            random_buffer_mask = (torch.rand((batch_size, max_segments), device=device) > segment_buffer_prob)  # shape [batch, max_segments]
        else:
            random_buffer_mask = torch.zeros((batch_size, max_segments), dtype=torch.bool, device=device)
        # random_buffer_mask = torch.ones((batch_size, max_segments), dtype=torch.bool, device=device)
        negative_probs = segment_probs_grid[..., 0] + segment_buffer_prob * random_buffer_mask.float()
        positive_probs = segment_probs_grid[..., 1] + segment_buffer_prob * random_buffer_mask.float()
        segment_probs_grid = torch.stack([negative_probs, positive_probs], dim=-1)

        # document-level prediction by taking last valid segment's prediction
        last_valid_indices = segment_mask.any(dim=-1).sum(dim=-1) - 1  # shape [batch]
        batch_indices = torch.arange(batch_size, device=device)
        document_probs = segment_probs_grid[batch_indices, last_valid_indices]  # shape [batch, classes]
        document_logits = segment_logits_grid[batch_indices, last_valid_indices]  # shape [batch, classes]

        # debug metircs
        avg_buffer_prob = segment_buffer_prob[segment_mask.any(dim=-1)].mean()  # average buffer probability across all valid segments in the batch, for monitoring how much the model is relying on the buffer
        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
            "random_buffer_mask": random_buffer_mask,
            "debug_avg_buffer_prob": avg_buffer_prob,
        }
        return document_probs, segment_probs_grid, extras


class DPOBaselineModelforPRM(BaseMILModel):
    """
    Using a pretrained causal language model to compute token-level log-probabilities for the document and a reference model, then aggregating these log-probabilities at the segment level and feeding them through a classifier to get document-level predictions.
    """

    transformers_parent_class = AutoModelForCausalLM
    supported_modules = ()

    def __init__(self, 
                 pretrained_model, 
                 ref_model='/data2/jty/models/Qwen2.5-Math-7B-Instruct',
                 decision_threshold=0.5, 
                 beta=0.05, 
                 accumulate_mode=False,
                 **kwargs):
        self.beta = beta
        self.accumulate_mode = accumulate_mode
        self.ref_model = ref_model
        super().__init__(pretrained_model, decision_threshold=decision_threshold, **kwargs)


    def _init_weights(self, **kwargs):
        self.backbone_dtype = next(self.pretrained_model.parameters()).dtype
        if isinstance(self.ref_model, str):
            self.ref_model = self.transformers_parent_class.from_pretrained(
                self.ref_model,
            ).to(dtype=self.backbone_dtype)

    @staticmethod
    def compute_completion_mask(batch: Batch) -> Tensor:
        # the following is left padded
        document_mask = batch["attention_mask"]
        prompt_mask = batch["prompt_attention_mask"]
        seq_len = document_mask.size(1)
        device = document_mask.device

        prompt_mask = prompt_mask.to(device)
        doc_lengths = document_mask.sum(dim=1)
        prompt_lengths = prompt_mask.sum(dim=1)
        assert (prompt_lengths <= doc_lengths).all(), "Prompt length cannot exceed document length for any example in the batch."
        completion_lengths = (doc_lengths - prompt_lengths).clamp_min(0)
        completion_start = seq_len - completion_lengths

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        completion_mask = (positions >= completion_start.unsqueeze(1)) & document_mask.bool()   # shape: [batch, seq_len]
        return completion_mask.long()


    def _forward_impl(self, eval, batch: Batch) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        segment_ids: Tensor = batch["segment_input_ids"]
        segment_mask: Tensor = batch["segment_attention_mask"]
        document_ids: Tensor = batch["input_ids"]
        document_mask: Tensor = batch["attention_mask"]
        completion_mask = self.compute_completion_mask(batch)

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

        model_kwargs = {
            "input_ids": document_ids, 
            "attention_mask": document_mask, 
            "use_cache": False
        }

        outputs = self.pretrained_model(**model_kwargs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = document_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens

        # When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad():
            ref_outputs = self.ref_model(**model_kwargs)
            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens

        # compute per-segment logits from cumulative log-probability ratios
        end_positions = batch["segment_ends"]  # shape [batch, max_segments]
        valid_segment_mask = segment_mask.any(dim=-1)

        logp_ratio = self.beta * (per_token_logps - ref_per_token_logps)  # shape [batch, seq_len - 1]
        token_len = logp_ratio.size(-1)

        if token_len > 0:
            # aggregate all the previous tokens' log probability ratios up to each segment end position
            cumulative_ratio = logp_ratio.cumsum(dim=-1)
            segment_end_indices = end_positions - 1  # align with shifted log-probs
            clamped_indices = segment_end_indices.clamp(min=0, max=token_len - 1)
            expanded_cumsum = cumulative_ratio.unsqueeze(1).expand(-1, max_segments, -1)
            gathered = torch.gather(expanded_cumsum, dim=2, index=clamped_indices.unsqueeze(-1)).squeeze(-1)
            if not self.accumulate_mode:
                # take the difference between cumulative sums to get the sum within each segment
                gathered = torch.cat([
                    torch.zeros((batch_size, 1), dtype=dtype, device=device),
                    gathered
                ], dim=1)
                gathered = gathered.diff(dim=1)
            # mask out invalid segments (those with end positions outside the valid token range) by setting their scores to zero
            segment_scores = torch.where(
                segment_end_indices >= 0,
                gathered,
                torch.zeros_like(gathered),
            )

        else:
            segment_scores = torch.zeros((batch_size, max_segments), dtype=dtype, device=device)

        segment_pos_probs = torch.sigmoid(segment_scores)
        segment_probs_grid = torch.stack([1 - segment_pos_probs, segment_pos_probs], dim=-1)
        segment_logits_grid = torch.stack([-segment_scores, segment_scores], dim=-1)

        default_prob_row = torch.tensor([1.0, 0.0], dtype=dtype, device=device).view(1, 1, 2)
        segment_probs_grid = torch.where(valid_segment_mask.unsqueeze(-1), segment_probs_grid, default_prob_row)
        segment_logits_grid = torch.where(
            valid_segment_mask.unsqueeze(-1),
            segment_logits_grid,
            torch.zeros_like(segment_logits_grid),
        )

        doc_scores = logp_ratio.sum(dim=-1)
        doc_pos_probs = torch.sigmoid(doc_scores)
        document_probs = torch.stack([1 - doc_pos_probs, doc_pos_probs], dim=-1)
        document_logits = torch.stack([-doc_scores, doc_scores], dim=-1)

        extras = {
            "segment_logits": segment_logits_grid,
            "document_logits": document_logits,
        }
        return document_probs, segment_probs_grid, extras


__all__ = [
    "ProbAveragePoolMILModelforPRM", 
    "InstanceAveragePoolMILModelforPRM", 
    "AttentionPoolMILModelforPRM", 
    "ConjunctivePoolMILModelforPRM", 
    "MinPoolMILModelforPRM", 
    "SoftMinPoolMILModelforPRM",
    "NaiveMILModelforPRM",
    "NoisyORPoolMILModelforPRM",
    "BufferBaselineModelforPRM",
    "DPOBaselineModelforPRM",
]
