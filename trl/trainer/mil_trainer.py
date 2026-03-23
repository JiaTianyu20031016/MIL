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
        self._metrics_backup = {}
        self._total_train_tokens = 0

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        # loss type: "document" for document-level loss only, "segment" for segment-level loss only, "noisy_segment" for using document-level labels as noisy labels for segments, and "pgpu_document" for using PGPU relabeling for document-level loss.
        self.loss_type = args.loss_type
        # PU warmup steps for PGPU relabeling. During the warmup period, no relabeling will be applied to allow the model to learn from the original labels first. After the warmup period, PGPU relabeling will be applied to potentially noisy positive samples to reduce the noise in the labels and improve the model's robustness. The optimal value for this hyperparameter may depend on the dataset and task, and can be tuned based on validation performance.
        self.pu_warmup_steps = args.pu_warmup_steps
        # if annotation_output is True, will record model's prediction for each sample in the eval dataset and save to disk for analysis. This is useful for understanding the model's behavior and diagnosing potential issues with the training data or model architecture.
        self.annotation_output = args.annotation_output


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
    def segment_loss(outputs: MILModelOutput, segment_target_prob: torch.Tensor, segment_valid_mask: torch.Tensor, mask_ambiguous_labels: bool = False) -> torch.Tensor:
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
        effective_mask = segment_valid_mask
        if mask_ambiguous_labels:
            # Clone to avoid in-place edits on tensors that autograd may need for gradient scattering.
            effective_mask = segment_valid_mask.clone()
            seq_indices = torch.arange(segment_target_prob.size(1), device=segment_target_prob.device)
            first_error_mask = (segment_target_prob < 1.0) & segment_valid_mask
            first_error_indices = torch.where(first_error_mask, seq_indices, segment_target_prob.size(1))
            first_error_position = first_error_indices.min(dim=-1).values
            batch_indices = torch.arange(segment_target_prob.size(0), device=segment_target_prob.device)
            effective_mask[batch_indices[:, None], seq_indices[None, :]] &= (
                seq_indices[None, :] <= first_error_position[:, None]
            )

        segment_pred_probs = outputs.segment_probs[effective_mask]
        segment_target_prob = segment_target_prob[effective_mask]
        if segment_target_prob is None or segment_pred_probs.size(0) == 0:
            return None
        
        target = segment_target_prob.to(segment_pred_probs.device, dtype=segment_pred_probs.dtype)
        target = target.clamp(0.0, 1.0)
        target_matrix = torch.stack([1.0 - target, target], dim=-1)
        log_probs = torch.log(segment_pred_probs.clamp_min(1e-8))
        loss = -(target_matrix * log_probs).sum(dim=-1).mean()
        return loss
    
    @staticmethod
    def noisy_segment_loss(
        outputs: MILModelOutput, 
        document_target_prob: torch.Tensor, 
        segment_valid_mask: torch.Tensor,
        last_index_scale: float = 1.0
    ) -> torch.Tensor:
        '''
        propagate the document-level label to segments as noisy labels, and calculate the cross-entropy loss for all segments with valid labels. 
        '''
        segment_pred_probs = outputs.segment_probs[segment_valid_mask]
        segment_target_prob = document_target_prob.unsqueeze(1).expand_as(outputs.segment_probs[:,:,0])[segment_valid_mask]
        if segment_target_prob is None or segment_pred_probs.size(0) == 0:
            return None
        
        valid_segment_cnt = segment_valid_mask.sum(dim=-1)  # shape: (batch_size,)
        last_segment_index = valid_segment_cnt.cumsum(dim=0) - 1  # shape: (batch_size,). Cumulative sum to get the correct last segment index after flattening.
        last_segment_mask = torch.arange(segment_pred_probs.size(0), device=segment_pred_probs.device).unsqueeze(1) == last_segment_index.unsqueeze(0)  # shape: (num_valid_segments, batch_size)
        last_segment_mask = last_segment_mask.any(dim=-1)  # shape: (num_valid_segments,). True for the last valid segment of each sample.
        last_segment_factor = 1.0 + (last_index_scale - 1.0) * last_segment_mask.float()  # shape: (num_valid_segments,). Scale the loss for the last valid segment if last_index_scale > 1.0.

        target = segment_target_prob.to(segment_pred_probs.device, dtype=segment_pred_probs.dtype)
        target = target.clamp(0.0, 1.0)
        target_matrix = torch.stack([1.0 - target, target], dim=-1)
        log_probs = torch.log(segment_pred_probs.clamp_min(1e-8))
        loss = -(target_matrix * log_probs).sum(dim=-1) * last_segment_factor
        return loss.mean()

    def pgpu_document_loss(self, outputs: MILModelOutput, document_target_prob: torch.Tensor) -> torch.Tensor:
        '''
        binary classification loss for document-level prediction, using PGPU (Probability Gap Positive Unlabeled Learning) for relabeling.
        '''
        metrics = {}

        document_pred_probs = outputs.document_probs
        if document_target_prob is None or document_pred_probs.size(0) == 0:
            return None, None
        
        # PGPU relabeling: for (potentially noisy) positive documents with observed probability gap < 0, the true probability gap must be lower than observed
        # so we can relabel the positive document with low predicted probability as negative to reduce the noise in the labels.
        if self.state.global_step > self.pu_warmup_steps:
            pgpu_relabel_mask = (document_pred_probs[:,1] < 0.5) & (document_target_prob == 1)  # shape: (batch_size,)
        else:
            pgpu_relabel_mask = torch.zeros_like(document_target_prob, dtype=torch.bool)
        
        target = document_target_prob.to(document_pred_probs.device, dtype=document_pred_probs.dtype)
        target = target.clamp(0.0, 1.0)
        target_matrix = torch.stack([1.0 - target, target], dim=-1)
        log_probs = torch.log(document_pred_probs.clamp_min(1e-8))
        loss = -(target_matrix * log_probs).masked_fill(pgpu_relabel_mask.unsqueeze(-1), 0).sum(dim=-1).mean()
        
        if (document_target_prob == 1).any():
            metrics["pgpu_relabel_ratio"] = pgpu_relabel_mask.float().mean().item() / (document_target_prob == 1).float().mean().item()
        return loss, metrics
    
    def pad_and_gather_for_metrics(self, tensor: torch.Tensor, pad_dim: int = 0, pad_value: int = 0) -> torch.Tensor:
        # Pad the tensor to the same length across all processes before gathering for metrics calculation. This is needed for segment-level predictions since different processes may have different max_segments due to dynamic padding.
        
        # NOTE: never use self.accelerator.gather_for_metrics to gather batch-inconsistent tensor
        # because gather_for_metrics will try to truncate the additional data in the last batch.
        # if the tensor to be gathered is not in shape [batch, ...], the truncation may cause out-of-expectation errors.
        # should use gather instead.
        max_length = self.accelerator.gather(torch.tensor(tensor.shape[pad_dim], device=tensor.device)).max().item()
        if tensor.shape[pad_dim] < max_length:
            pad_sizes = [(0, 0)] * len(tensor.shape)
            pad_sizes[pad_dim] = (0, max_length - tensor.shape[pad_dim])
            tensor = torch.nn.functional.pad(tensor, [size for pair in reversed(pad_sizes) for size in pair], value=pad_value)
        return self.accelerator.gather_for_metrics(tensor)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"

        # If not set, defaults from model config and may warn since cache isn't compatible with gradient checkpointing
        inputs["use_cache"] = False
        outputs = model(eval=not self.model.training, **inputs)

        document_target_prob = inputs.get("positive_prob")
        segment_target_prob = inputs.get("segment_positive_probs")
        segment_valid_mask = inputs.get("segment_attention_mask").any(dim=-1)
        assert document_target_prob is not None and segment_target_prob is not None and segment_valid_mask is not None
        assert segment_valid_mask.any(dim=-1).all()

        if self.loss_type == "noisy_segment":
            loss = self.noisy_segment_loss(outputs, document_target_prob, segment_valid_mask)
        elif self.loss_type == "document":
            loss = self.document_loss(outputs, document_target_prob)
        elif self.loss_type == "segment":
            loss = self.segment_loss(outputs, segment_target_prob, segment_valid_mask)
        elif self.loss_type == "pgpu_document":
            loss, pgpu_metrics = self.pgpu_document_loss(outputs, document_target_prob)
            for key, value in pgpu_metrics.items():
                self._metrics[mode][key].append(value)
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}. Supported values are 'document', 'segment', and 'noisy_segment'.")
        
        # metrics
        with torch.no_grad():
            document_loss = self.document_loss(outputs, document_target_prob)
            self._metrics[mode]["document_loss"].append(
                self.accelerator.gather(document_loss).mean().item()
            )
            segment_loss = self.segment_loss(outputs, segment_target_prob, segment_valid_mask)
            self._metrics[mode]["segment_loss"].append(
                self.accelerator.gather(segment_loss).mean().item()
            )

            # gather results across ranks for calculating metrics
            document_pred_labels = self.accelerator.gather_for_metrics(outputs.document_predictions).long()  # (batch_size,)
            document_target_labels = self.accelerator.gather_for_metrics((document_target_prob >= 0.5).long())  # (batch_size,)
            segment_pred_labels = self.pad_and_gather_for_metrics(outputs.segment_predictions, pad_dim=1, pad_value=0).long()    # (batch_size, max_segments)
            segment_target_labels = self.pad_and_gather_for_metrics((segment_target_prob >= 0.5).long(), pad_dim=1, pad_value=0)  # (batch_size, max_segments)
            segment_valid_mask = self.pad_and_gather_for_metrics(segment_valid_mask, pad_dim=1, pad_value=0)  # (batch_size, max_segments)

            document_accuracies = (document_pred_labels == document_target_labels).float()
            document_positive_mask = document_target_labels == 1
            segment_accuracies = (segment_pred_labels[segment_valid_mask] == segment_target_labels[segment_valid_mask]).float()
            segment_positive_mask = segment_target_labels[segment_valid_mask] == 1

            # document level prediction accuracy
            self._metrics[mode]["document_accuracy"].append(
                document_accuracies.mean().item()
            )
            if torch.any(document_positive_mask):
                self._metrics[mode]["document_positive_accuracy"].append(
                    document_accuracies[document_positive_mask].mean().item()
                )
            if torch.any(~document_positive_mask):
                self._metrics[mode]["document_negative_accuracy"].append(
                    document_accuracies[~document_positive_mask].mean().item()
                )

            # segement level prediction accuracy
            self._metrics[mode]["segment_accuracy"].append(
                segment_accuracies.mean().item()
            )
            if torch.any(segment_positive_mask):
                self._metrics[mode]["segment_positive_accuracy"].append(
                    segment_accuracies[segment_positive_mask].mean().item()
                )
            if torch.any(~segment_positive_mask):
                self._metrics[mode]["segment_negative_accuracy"].append(
                    segment_accuracies[~segment_positive_mask].mean().item()
                )

            # first-error detection accuracy: whether the model can correctly identify the first error segment. using F1 score
            seq_indices = torch.arange(segment_pred_labels.size(1), device=segment_pred_labels.device)
            seq_indices = seq_indices.unsqueeze(0).expand(segment_pred_labels.size(0), -1)
            sentinel = segment_pred_labels.size(-1)
            first_error_pred = torch.where(
                ((segment_pred_labels == 0) & segment_valid_mask), seq_indices, sentinel
            ).min(dim=-1).values
            first_error_target = torch.where(
                ((segment_target_labels == 0) & segment_valid_mask), seq_indices, sentinel
            ).min(dim=-1).values
            first_error_accuracies = (first_error_pred == first_error_target).float()
            if torch.any(document_positive_mask):
                error_detection_positive_accuracy = first_error_accuracies[document_positive_mask].mean().item()
                self._metrics[mode]["error_detection_positive_accuracy"].append(
                    error_detection_positive_accuracy
                )
            if torch.any(~document_positive_mask):
                error_detection_negative_accuracy = first_error_accuracies[~document_positive_mask].mean().item()
                self._metrics[mode]["error_detection_negative_accuracy"].append(
                    error_detection_negative_accuracy
                )
            if torch.any(document_positive_mask) and not torch.all(document_positive_mask):
                self._metrics[mode]["error_detection_f1"].append(
                    200 * error_detection_positive_accuracy * error_detection_negative_accuracy / (error_detection_positive_accuracy + error_detection_negative_accuracy + 1e-8)
                )

            # other extra metrics
            for key in outputs.extras:
                if 'debug' in key:
                    # we may not assume that the tensors in outputs.extras are batch-consistent
                    # so use accelerator.gather instead of gather_for_metrics to gather the tensors
                    metric_value = self.accelerator.gather(outputs.extras[key]).mean().item()
                    self._metrics[mode][key].append(metric_value)

        # dump the model predictions for analysis if annotation_output is set
        if self.annotation_output is not None:
            from accelerate.utils import gather_object
            doc_ids = gather_object(inputs.get("doc_ids"))
            prompts = gather_object(inputs.get("prompt_texts"))
            completions = gather_object(inputs.get("segment_texts"))
            sources = gather_object(inputs.get("source"))
            document_annotations = document_pred_labels.tolist()
            document_labels = document_target_labels.tolist()
            segment_labels = segment_target_labels.tolist()
            segment_num = segment_valid_mask.sum(dim=-1).tolist()
            segment_labels = [labels[:num] for labels, num in zip(segment_labels, segment_num)]
            # note that the length of document_annotations and segment_labels may be smaller than doc_ids, prompts, and completions since the inputs may be padded for distributed training
            # the padding occurs in the end of the batch, so we can simply ignore the extra samples after the length of document_annotations and segment_labels
            annotation_data = [
                {
                    "id": doc_ids[i],
                    "prompt": (
                        prompts[i] 
                        if not prompts[i].endswith(self.args.step_separator) 
                        else prompts[i][:-len(self.args.step_separator)]
                    ),
                    "completions": [
                        (
                            c 
                            if not c.endswith(self.args.step_separator) 
                            else c[:-len(self.args.step_separator)] 
                        )
                        for c in  completions[i]
                    ],
                    "annotation": document_annotations[i] if document_labels[i] == 1 else 0,  # if the document is labeled as negative, we will label all segments as negative regardless of the model prediction
                    "labels": segment_labels[i],
                    "source": sources[i],
                }
                for i in range(len(document_annotations))
            ]
            if self.accelerator.is_main_process:
                annotation_output_path = os.path.join(self.annotation_output, f"{mode}_annotations.jsonl")
                os.makedirs(self.annotation_output, exist_ok=True)
                with open(annotation_output_path, "a", encoding="utf-8") as f:
                    for record in annotation_data:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

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

        self._metrics_backup = metrics  # backup the metrics in case we need to access them later (e.g., in callbacks) after they are cleared
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


    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        metrics.update(self._metrics_backup)  # add the metrics calculated in compute_loss to the final metrics returned by evaluate()
        return metrics