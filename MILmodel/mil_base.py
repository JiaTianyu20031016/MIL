"""Abstract base class for MIL models handling both document- and segment-level outputs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from transformers.utils import ModelOutput

from MILmodel.model_wrapper import PreTrainedModelWrapper

Batch = Mapping[str, Any]


@dataclass
class MILModelOutput(ModelOutput):
    """ModelOutput carrying probabilities and predictions"""

    document_probs: Optional[Tensor] = None
    segment_probs: Optional[Tensor] = None
    document_predictions: Optional[Tensor] = None
    segment_predictions: Optional[Tensor] = None


class BaseMILModel(PreTrainedModelWrapper):
    """Shared helpers for MIL classifiers working on document batches."""

    transformers_parent_class = AutoModelForSequenceClassification
    supported_args = ()
    supported_modules = ()
    supported_rm_modules = ()

    def __init__(self, pretrained_model, decision_threshold=0.5, **kwargs):
        """
        Initializes the model.

        Args:
            pretrained_model ([`~transformers.PreTrainedModel`]):
                The model to wrap. It should be a causal language model such as GPT2. or any model mapped inside the
                `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the [`ValueHead`] class.
        """
        super().__init__(pretrained_model, **kwargs)
        self.decision_threshold = decision_threshold
        supported_kwargs, _, _ = self._split_kwargs(kwargs)
        self._init_weights(**supported_kwargs)

    @staticmethod
    def _document_count(batch: Batch) -> int:
        input_ids = batch.get("input_ids")
        if not isinstance(input_ids, Tensor):  # type: ignore[str-bytes-safe]
            raise ValueError("Batch must contain tensor 'input_ids'.")
        return input_ids.size(0)

    @staticmethod
    def _segment_count(batch: Batch) -> int:
        segment_ids = batch.get("segment_input_ids")
        if segment_ids is None:
            return 0
        if not isinstance(segment_ids, Tensor):
            raise ValueError("'segment_input_ids' must be a tensor if present.")
        return segment_ids.size(0)

    @staticmethod
    def _ensure_prob_matrix(matrix: Tensor, batch_size: int, *, matrix_name: str) -> Tensor:
        if batch_size == 0:
            return matrix.view(0, 2)
        if matrix.dim() != 2 or matrix.size(1) != 2:
            raise ValueError(f"{matrix_name} must be a [batch, 2] tensor.")
        if matrix.size(0) != batch_size:
            raise ValueError(f"{matrix_name} length does not match batch size.")
        return matrix

    @staticmethod
    def _ensure_target_vector(vector: Tensor, batch_size: int, *, vector_name: str) -> Tensor:
        if batch_size == 0:
            return vector.view(0)
        vector = vector.view(-1)
        if vector.size(0) != batch_size:
            raise ValueError(f"{vector_name} length does not match batch size.")
        return vector

    # @staticmethod
    # def _compute_loss(
    #     *,
    #     document_probs: Tensor,
    #     document_positive_prob: Optional[Tensor],
    # ) -> Optional[Tensor]:
    #     if document_positive_prob is None or document_probs.size(0) == 0:
    #         return None
    #     target = document_positive_prob.to(document_probs.device, dtype=document_probs.dtype)
    #     target = target.clamp(0.0, 1.0)
    #     target_matrix = torch.stack([1.0 - target, target], dim=-1)
    #     log_probs = torch.log(document_probs.clamp_min(1e-8))
    #     loss = -(target_matrix * log_probs).sum(dim=-1).mean()
    #     return loss

    @staticmethod
    def _validate_batch(batch: Batch) -> None:
        required = ["input_ids", "attention_mask", "segment_input_ids", "segment_attention_mask"]
        for key in required:
            if key not in batch:
                raise ValueError(f"Batch is missing required key '{key}'.")


    def forward(
        self,
        **batch: Tensor,
    ) -> MILModelOutput:  # type: ignore[override]
        """Run inference on a batch from the MIL data loader."""

        working_batch: Dict[str, Any] = dict(batch)

        self._validate_batch(working_batch)
        document_probs, segment_probs, extras = self._forward_impl(working_batch)
        doc_count = self._document_count(working_batch)
        seg_count = self._segment_count(working_batch)
        document_probs = self._ensure_prob_matrix(
            document_probs,
            batch_size=doc_count,
            matrix_name="document_probs",
        )
        segment_probs = self._ensure_prob_matrix(
            segment_probs,
            batch_size=seg_count,
            matrix_name="segment_probs",
        )

        document_pos_probs = document_probs[:, 1]
        segment_pos_probs = segment_probs[:, 1]
        document_predictions = (document_pos_probs >= self.decision_threshold).long()
        segment_predictions = (segment_pos_probs >= self.decision_threshold).long()

        output = MILModelOutput(
            document_probs=document_probs,
            segment_probs=segment_probs,
            document_predictions=document_predictions,
            segment_predictions=segment_predictions,
        )
        if extras:
            output.update(extras)
        return output


    def _forward_impl(
        self, batch: Batch
    ) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        """Subclasses must return document probs, segment probs, optional extras."""
        raise NotImplementedError


    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the supprted modules.
        """
        raise NotImplementedError


    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the 
        supported modules to the state dictionary of the wrapped model by 
        prepending the key with the name of the supported modules.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        for module_name in self.supported_modules:
            module = getattr(self, module_name, None)
            if module is not None:
                module_state_dict = module.state_dict(*args, **kwargs)
                for k, v in module_state_dict.items():
                    pretrained_model_state_dict[f"{module_name}.{k}"] = v
        return pretrained_model_state_dict


    def push_to_hub(self, *args, **kwargs):
        for module_name in self.supported_modules:
            module = getattr(self, module_name, None)
            if module is not None:
                self.pretrained_model.__setattr__(module_name, module)

        return self.pretrained_model.push_to_hub(*args, **kwargs)


    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the supported modules to the state dictionary of the wrapped model by prepending the
        key with the name of the supported modules. This function removes the prefix from the keys of the supported modules' state
        dictionary.
        """
        for module_name in self.supported_modules:
            tmp_state_dict = {}
            for k in list(state_dict.keys()):
                if f"{module_name}." in k:
                    tmp_state_dict[k.replace(f"{module_name}.", "")] = state_dict.pop(k)
            module = getattr(self, module_name, None)
            if module is not None:
                module.load_state_dict(tmp_state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]
            if isinstance(first_device, int):
                if is_torch_npu_available():
                    first_device = f"npu:{first_device}"
                elif is_torch_xpu_available():
                    first_device = f"xpu:{first_device}"
                else:
                    first_device = f"cuda:{first_device}"
            self.v_head = self.v_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True



__all__ = ["BaseMILModel", "MILModelOutput"]
