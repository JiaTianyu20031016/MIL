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

from dataclasses import dataclass, field
from typing import Any

from transformers import TrainingArguments

from .base_config import _BaseConfig


@dataclass
class MILConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`MILTrainer`].

    This class includes only the parameters that are specific to MIL training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`RewardTrainer`] is provided as a string. If you're training a MoE architecture and want
            to include the load balancing/auxiliary loss as a part of the final loss, remember to set
            `output_router_logits=True` in this dictionary.
        chat_template_path (`str`, *optional*):
            If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory
            or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must
            ensure that any special tokens referenced in the template are added to the tokenizer and that the model's
            embedding layer is resized accordingly.
        
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the sequences (prompt + completion) used for truncation.
        max_completion_length (`int`, *optional*):
            Maximum length of the completion used for truncation. The completion is the concatenation of the steps.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        eos_token (`str`, *optional*):
            Token used to indicate the end of a turn or sequence. If `None`, it defaults to
            `processing_class.eos_token`.
        pad_token (`str`, *optional*):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        step_separator (`str`, *optional*, defaults to `"\n\n"`):
            Separator used to separate each step of the reasoning process.
        train_on_last_step_only (`bool`, *optional*, defaults to `False`):
            Whether to train only on the last step.
        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        
        architecture (`str`, *optional*):
            The architecture of the MIL model to use. If not specified, defaults to `InstanceAveragePoolMILModelForPRM`.
        loss_type (`str`, *optional*, defaults to `"document"`):
            The type of loss to use for training. Supported values are `"document"`, `"segment"` and `"noisy_segment"`.

        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.
    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-5` instead of `5e-5`.
    """
    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]
    
    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    # Parameters that control the model
    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `RewardTrainer` is provided as a string. If you're training a MoE architecture and want to include "
            "the load balancing/auxiliary loss as a part of the final loss, remember to set "
            "`output_router_logits=True` in this dictionary."
        },
    )
    chat_template_path: str | None = field(
        default=None,
        metadata={
            "help": "If specified, sets the model's chat template. This can either be the path to a tokenizer (local "
            "directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, "
            "you must ensure that any special tokens referenced in the template are added to the tokenizer and "
            "that the model's embedding layer is resized accordingly."
        },
    )

    max_length: int | None = field(
        default=1024,
        metadata={"help": "Maximum length of the sequences (prompt + completion) used for truncation."},
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "Maximum length of the completion used for truncation. The completion is the concatenation of the "
            "steps."
        },
    )
    
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model and reference model."},
    )
    step_separator: str = field(
        default="\n\n",
        metadata={"help": "Separator used to separate each step of the reasoning process."},
    )
    train_on_last_step_only: bool = field(
        default=False,
        metadata={"help": "Whether to train only on the last step."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    eos_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`."
        },
    )
    pad_token: str | None = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that "
            "is also `None`, it falls back to `processing_class.eos_token`."
        },
    )

    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Whether to offload the activations to the CPU."},
    )

    architecture: str | None = field(
        default=None,
        metadata={"help": "The architecture of the MIL model to use. If not specified, defaults to `InstanceAveragePoolMILModelForPRM`."},
    )
    loss_type: str = field(
        default="document",
        metadata={"help": "The type of loss to use for training. Supported values are `document`, `segment` and `noisy_segment`."},
    )