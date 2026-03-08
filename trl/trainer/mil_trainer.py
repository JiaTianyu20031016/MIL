# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import json
import logging
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import transformers
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model
from datasets import Dataset, IterableDataset
from packaging.version import Version
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    set_seed,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.modeling_layers import GenericForSequenceClassification
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ..chat_template_utils import clone_chat_template
from ..data_utils import is_conversational
from ..models import get_act_offloading_ctx_manager
from .base_trainer import _BaseTrainer
from .reward_config import RewardConfig
from .utils import create_model_from_path, disable_dropout_in_model, get_config_model_id, pad, remove_none_values

from MILmodel.mil_base import MILModelOutput

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model


logger = get_logger(__name__)


# AutoModelForSequenceClassification adds a new classification head when loading a CausalLM. That head is randomly
# initialized and triggers a harmless warning about uninitialized weights. We suppress just that specific warning to
# avoid confusing users.


# Old approach using logging filter (for transformers < 4.57.0)
@contextmanager
def suppress_from_pretrained_warning(logger: logging.Logger):
    pattern = re.compile(
        r"^Some weights of \S+ were not initialized from the model checkpoint at \S+ and are newly initialized: "
        r"\[.*\]\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and "
        r"inference\.$"
    )

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not pattern.search(record.getMessage())

    f = _Filter()
    logger.addFilter(f)
    try:
        yield
    finally:
        logger.removeFilter(f)


# New approach using scoped override (for transformers >= 4.57.0)
@contextmanager
def ignore_seqcls_score_missing_key():
    # Scoped override: ignore only the expected seq-clf head key.
    old = getattr(GenericForSequenceClassification, "_keys_to_ignore_on_load_missing", None)
    merged = list(old) if old is not None else []
    pattern = r"^score\.weight$"
    if pattern not in merged:
        merged.append(pattern)
    GenericForSequenceClassification._keys_to_ignore_on_load_missing = merged
    try:
        yield
    finally:
        GenericForSequenceClassification._keys_to_ignore_on_load_missing = old


# Version-aware wrapper that chooses the appropriate approach
@contextmanager
def suppress_seqcls_warning():
    # Use the new approach for transformers >= 4.57.0, old approach for earlier versions
    # The old approach is needed for 4.56.2 to avoid meta tensor issues with device_map=None
    if Version(transformers.__version__) >= Version("4.57.0"):
        with ignore_seqcls_score_missing_key():
            yield
    else:
        # Get the transformers logger
        transformers_logger = logging.getLogger("transformers.modeling_utils")
        with suppress_from_pretrained_warning(transformers_logger):
            yield


def get_dataset_column_names(dataset: Dataset | IterableDataset) -> list[str]:
    return list(next(iter(dataset)).keys()) if dataset.column_names is None else dataset.column_names


class MILTrainer(_BaseTrainer):
    """
    Trainer for Multiple Instance Learning (MIL) models.

    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from trl import RewardTrainer
    from datasets import load_dataset

    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    trainer = RewardTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `AutoModelForSequenceClassification.from_pretrained` with the keyword arguments in
              `args.model_init_kwargs`.
            - A sequence classification [`~transformers.PreTrainedModel`] object.
            - A sequence classification [`~peft.PeftModel`] object.
        args ([`RewardConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. This trainer supports [preference](#preference) type (both implicit and
            explicit prompt). The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            The trainer also supports processed datasets (tokenized) as long as they contain `chosen_ids` and
            `rejected_ids` fields.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Tokenizer used to process the data. If `None`, the tokenizer is loaded from the model's name with
            [`~transformers.AutoTokenizer.from_pretrained`]. A padding token, `processing_class.pad_token`, must be
            set. If the processing class has not set a padding token, `processing_class.eos_token` will be used as the
            default.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.EvalPrediction`] and return a dictionary string to metric values. When passing
            [`RewardConfig`] with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a
            boolean `compute_result` argument. This will be triggered after the last eval batch to signal that the
            function needs to calculate and return the global summary statistics rather than accumulating the
            batch-level statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before
            initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped. Note that if the loaded
            model is a causal LM, it's highly recommended to set `modules_to_save=["score"]` in the PEFT configuration
            to ensure that the reward head is properly trained.
    """

    _tag_names = ["trl", "MIL-trainer"]
    _name = "MIL"
    _template_file = "rm_model_card.md"

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        args: RewardConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = RewardConfig(f"{model_name}-Reward")

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif isinstance(train_dataset, IterableDataset):
            # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
            # batches from multiple processes, leading to mismatch errors.
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `RewardConfig` or set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False

        # Model
        # As AutoModelForSequenceClassification.from_pretrained() will add a random head for the model, set_seed must
        # be done before loading the model to ensure reproducibility.
        set_seed(args.seed)
        if isinstance(model, str):
            raise NotImplementedError(
                "Passing a model name or path as a string is not currently supported in MILTrainer. Please load your "
                "model as a `PreTrainedModel` instance and pass it directly to the trainer."
            )
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model_init_kwargs["num_labels"] = 1  # the only output of the model is the reward score
            with suppress_seqcls_warning():
                model = create_model_from_path(model, AutoModelForSequenceClassification, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `RewardConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(get_config_model_id(model.config))

        # Handle pad token for processors or tokenizers
        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = processing_class.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            processing_class.eos_token_id = eos_token_id

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
                added_tokens = []
            else:
                model, processing_class, added_tokens = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )
        else:
            added_tokens = []

        # PEFT configuration and model wrapping
        if peft_config is not None:
            raise NotImplementedError(
                "PEFT is not currently supported in MILTrainer. Please set `peft_config` to `None`."
            )
            if added_tokens:
                # Ensure that the added tokens are trainable
                if peft_config.trainable_token_indices is None:
                    peft_config.trainable_token_indices = {"embed_tokens": added_tokens}
                elif "embed_tokens" not in peft_config.trainable_token_indices:
                    peft_config.trainable_token_indices["embed_tokens"] = added_tokens
                else:
                    peft_config.trainable_token_indices["embed_tokens"].extend(added_tokens)

                # Ensure that the lm_head is trainable
                if peft_config.modules_to_save is None or "lm_head" not in peft_config.modules_to_save:
                    logger.warning(
                        "Cloning chat template added new tokens to the tokenizer, but 'lm_head' is not in PEFT's "
                        "`modules_to_save`. As a result, the model may not learn to generate outputs with these new "
                        "tokens, leading to degraded generation quality. To fix this, add "
                        "`modules_to_save=['lm_head']` to your PEFT configuration."
                    )

                    if peft_config.modules_to_save is None:
                        peft_config.modules_to_save = ["lm_head"]
                    else:
                        peft_config.modules_to_save.append("lm_head")

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                "with the new `peft_config` to the trainer."
            )

        # Create PEFT model
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_available() and is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper (see https://huggingface.co/papers/2305.14314, paragraph 3). Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(model)

        # Pad token (needed for SequenceClassification models)
        # If not provided, use the one from the processing class or the eos token if the processing class does not have
        # a pad token.
        pad_token = args.pad_token or processing_class.pad_token or processing_class.eos_token
        pad_token_id = processing_class.convert_tokens_to_ids(pad_token)
        if pad_token_id is None:
            raise ValueError(
                f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                "in the vocabulary before using it as a padding token."
            )
        model.config.pad_token_id = pad_token_id
        processing_class.pad_token_id = pad_token_id

        # Data collator
        if data_collator is None:
            raise ValueError("`data_collator` is required. Please provide an instance of `MILDataCollator` or a custom data collator.")
        
        # Dataset
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        if eval_dataset is not None:
            eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # During evaluation, Trainer calls compute_loss() only if can_return_loss is True and label_names is empty.
        self.can_return_loss = True
        self.label_names = []

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        self.loss_type = args.loss_type


    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase,
        args: RewardConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        return dataset

    # NOTE: this function is of no use in MIL training
    # def _set_signature_columns_if_needed(self):
    #     # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
    #     # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
    #     # and "attention_mask").
    #     if self._signature_columns is None:
    #         self._signature_columns = ["chosen_ids", "rejected_ids", "margin"]

    @staticmethod
    def document_loss(outputs: MILModelOutput, document_target_prob: torch.Tensor) -> torch.Tensor:
        '''
        binary classification loss for document-level prediction.
        '''
        document_pred_probs = outputs.document_probs
        if document_target_prob is None or document_pred_probs.size(0) == 0:
            return None
        
        target = document_target_prob.to(document_pred_probs.device, dtype=document_pred_probs.dtype)
        target = target.clamp(0.0, 1.0)
        target_matrix = torch.stack([1.0 - target, target], dim=-1)
        log_probs = torch.log(document_pred_probs.clamp_min(1e-8))
        loss = -(target_matrix * log_probs).sum(dim=-1).mean()
        return loss
    
    @staticmethod
    def segment_loss(outputs: MILModelOutput, segment_target_prob: torch.Tensor, mask_ambiguous_labels: bool = True) -> torch.Tensor:
        '''
        Binary classification loss for segment-level prediction. 
        When trained with segment-level labels, this is not actually MIL training, but rather standard supervised learning. 
        '''
        
        # segment_valid_mask: (batch_size, max_segments)
        # segment_pred_probs: (batch_size, max_segments, 2)
        # segment_target_prob: (batch_size, max_segments)

        # If mask_ambiguous_labels is True, we will ignore the segments after the first errorous segment (i.e. the first segment with target_prob < 1.0)
        # since for math reasoning task, the segments after the first error are not labeled and can be either correct or incorrect, which can confuse the model during training. 
        # If mask_ambiguous_labels is False, we will keep all segments for training.
        # by default, we expect the segments after the first error are labeled as 0.
        segment_valid_mask = outputs.segment_attention_mask.any(dim=-1)
        if mask_ambiguous_labels:
            first_error_mask = (segment_target_prob < 1.0) & segment_valid_mask
            first_error_indices = torch.where(first_error_mask, torch.arange(segment_target_prob.size(1), device=segment_target_prob.device), segment_target_prob.size(1))
            first_error_position = first_error_indices.min(dim=-1).values
            batch_indices = torch.arange(segment_target_prob.size(0), device=segment_target_prob.device)
            segment_valid_mask[batch_indices[:, None], torch.arange(segment_target_prob.size(1), device=segment_target_prob.device)[None, :]] &= (torch.arange(segment_target_prob.size(1), device=segment_target_prob.device)[None, :] < first_error_position[:, None])

        segment_pred_probs = outputs.segment_probs[segment_valid_mask]
        segment_target_prob = segment_target_prob[segment_valid_mask]
        if segment_target_prob is None or segment_pred_probs.size(0) == 0:
            return None
        
        target = segment_target_prob.to(segment_pred_probs.device, dtype=segment_pred_probs.dtype)
        target = target.clamp(0.0, 1.0)
        target_matrix = torch.stack([1.0 - target, target], dim=-1)
        log_probs = torch.log(segment_pred_probs.clamp_min(1e-8))
        loss = -(target_matrix * log_probs).sum(dim=-1).mean()
        return loss
    
    @staticmethod
    def noisy_segment_loss(outputs: MILModelOutput, document_target_prob: torch.Tensor) -> torch.Tensor:
        '''
        propagate the document-level label to segments as noisy labels, and calculate the cross-entropy loss for all segments with valid labels. 
        '''
        segment_valid_mask = outputs.segment_attention_mask.any(dim=-1)
        segment_pred_probs = outputs.segment_probs[segment_valid_mask]
        segment_target_prob = document_target_prob.unsqueeze(1).expand_as(outputs.segment_probs)[segment_valid_mask]
        if segment_target_prob is None or segment_pred_probs.size(0) == 0:
            return None

        target = segment_target_prob.to(segment_pred_probs.device, dtype=segment_pred_probs.dtype)
        target = target.clamp(0.0, 1.0)
        target_matrix = torch.stack([1.0 - target, target], dim=-1)
        log_probs = torch.log(segment_pred_probs.clamp_min(1e-8))
        loss = -(target_matrix * log_probs).sum(dim=-1).mean()
        return loss


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"

        # If not set, defaults from model config and may warn since cache isn't compatible with gradient checkpointing
        inputs["use_cache"] = False
        outputs = model(**inputs)

        document_target_prob = inputs.get("positive_prob")
        segment_target_prob = inputs.get("segment_positive_prob")
        if self.loss_type == "noisy_segment":
            loss = self.noisy_segment_loss(outputs, document_target_prob)
        elif self.loss_type == "document":
            loss = self.document_loss(outputs, document_target_prob)
        elif self.loss_type == "segment":
            loss = self.segment_loss(outputs, segment_target_prob)
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}. Supported values are 'document', 'segment', and 'noisy_segment'.")
        
        # metrics
        with torch.no_grad():
            if document_target_prob is not None:
                document_loss = self.document_loss(outputs, document_target_prob)
                self._metrics[mode]["document_loss"].append(document_loss.item())
            if segment_target_prob is not None:
                segment_loss = self.segment_loss(outputs, segment_target_prob)
                self._metrics[mode]["segment_loss"].append(segment_loss.item())

            document_pred_labels = (outputs.document_probs[:, 1] >= 0.5).long()
            segment_pred_labels = (outputs.segment_probs[:, :, 1] >= 0.5).long()
            if document_target_prob is not None:
                document_target_labels = (document_target_prob >= 0.5).long()
                self._metrics[mode]["document_accuracy"].append((document_pred_labels == document_target_labels).float().mean().item())
                if torch.any(document_target_labels == 1):
                    self._metrics[mode]["document_positive_accuracy"].append(((document_pred_labels == 1) & (document_target_labels == 1)).float().sum().item() / (document_target_labels == 1).float().sum().item())
                if torch.any(document_target_labels == 0):
                    self._metrics[mode]["document_negative_accuracy"].append(((document_pred_labels == 0) & (document_target_labels == 0)).float().sum().item() / (document_target_labels == 0).float().sum().item())
            if segment_target_prob is not None:
                segment_target_labels = (segment_target_prob >= 0.5).long()
                valid_mask = outputs.segment_attention_mask.any(dim=-1)
                if valid_mask.sum() > 0:
                    self._metrics[mode]["segment_accuracy"].append((segment_pred_labels[valid_mask] == segment_target_labels[valid_mask]).float().mean().item())
                    if torch.any(segment_target_labels[valid_mask] == 1):
                        self._metrics[mode]["segment_positive_accuracy"].append(((segment_pred_labels[valid_mask] == 1) & (segment_target_labels[valid_mask] == 1)).float().sum().item() / (segment_target_labels[valid_mask] == 1).float().sum().item())
                    if torch.any(segment_target_labels[valid_mask] == 0):   
                        self._metrics[mode]["segment_negative_accuracy"].append(((segment_pred_labels[valid_mask] == 0) & (segment_target_labels[valid_mask] == 0)).float().sum().item() / (segment_target_labels[valid_mask] == 0).float().sum().item())

        return (loss, outputs) if return_outputs else loss

    # Override training step to add activation offloading context.
    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
