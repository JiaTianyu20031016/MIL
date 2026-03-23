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

# /// script
# dependencies = [
#     "trl",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Full training:
CUDA_VISIBLE_DEVICES=4,5 python scripts/run_mil.py \
    --model_name_or_path /data2/Common_LLM_Base/Qwen/Qwen3-Embedding-0.6B/" \
    --dataset_name MILdata/PRM800K/data/data_balanced \
    --output_dir ckpts/debug \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 1.0e-5 \
    --eval_strategy steps \
    --eval_steps 50
"""

import os

import torch
from accelerate import logging
from transformers import AutoModelForTokenClassification, AutoTokenizer, HfArgumentParser


from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.mil_trainer import MILTrainer
from trl.trainer.mil_config import MILConfig

from MILdata.collator import MILDataCollator
from MILdata.dataset_common import (
    TokenizedDocumentDataset,
    create_mil_data_collator
)
from MILdata.ProcessBench.dataset import load_dataset as load_process_bench_dataset
from MILdata.PRM800K.dataset import load_dataset as load_prm800k_dataset
from MILdata.shepherd.dataset import load_dataset as load_math_shepherd_dataset
from MILdata.annotation.dataset import load_dataset as load_annotation_dataset
from MILmodel.mil_model_for_prm import *

logger = logging.get_logger(__name__)

ARCHITECTURE_TO_MODEL_CLASS = {
    "ProbAveragePoolMILModelforPRM": ProbAveragePoolMILModelforPRM,
    "InstanceAveragePoolMILModelforPRM": InstanceAveragePoolMILModelforPRM,
    "AttentionPoolMILModelforPRM": AttentionPoolMILModelforPRM,
    "ConjucturePoolMILModelforPRM": ConjucturePoolMILModelforPRM,
    "MinPoolMILModelforPRM": MinPoolMILModelforPRM,
    "SoftMinPoolMILModelforPRM": SoftMinPoolMILModelforPRM,
    "NaiveMILModelforPRM": NaiveMILModelforPRM,
    "DPOBaselineModelforPRM": DPOBaselineModelforPRM,
    "BufferBaselineModelforPRM": BufferBaselineModelforPRM,
}

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

def parse_model_kwargs(model_kwargs_list):
    """
    Parses a list of model keyword arguments in the format ["key1=value1", "key2=value2"] into a dictionary.

    Args:
        model_kwargs_list (list[str]): A list of strings, each in the format "key=value".

    Returns:
        dict: A dictionary containing the parsed key-value pairs.
    """
    model_kwargs = {}
    for item in model_kwargs_list:
        if "=" not in item:
            logger.warning(f"Skipping invalid model_kwargs item '{item}' as it does not contain '='.")
            continue
        key, value = item.split("=", 1)
        # Attempt to interpret the value as a Python literal (e.g., int, float, bool), otherwise keep it as a string
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            pass
        model_kwargs[key] = value
    return model_kwargs


if __name__ == "__main__":
    
    parser = HfArgumentParser((ScriptArguments, MILConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    model_class = ARCHITECTURE_TO_MODEL_CLASS.get(training_args.architecture, None)
    if model_class is None:
        raise ValueError(f"Unsupported architecture '{training_args.architecture}'. Supported architectures are: {list(ARCHITECTURE_TO_MODEL_CLASS.keys())}")
    model = model_class.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    if model_args.use_peft and model_args.lora_task_type != "TOKEN_CLS":
        logger.warning(
            "You are using a `task_type` that is different than `TOKEN_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type TOKEN_CLS when using this script with PEFT.",
        )

    ##############
    # Load dataset
    ##############
    collator = create_mil_data_collator(tokenizer)

    def load_dataset_fn(name, split):
        if name.endswith('jsonl'):
            return load_annotation_dataset(file_path=name)
        elif 'shepherd' in name.lower():
            return load_math_shepherd_dataset(hf_dataset=name, split=split)
        elif 'prm800k' in name.lower():
            return load_prm800k_dataset(hf_dataset=name, split=split)
        elif 'processbench' in name.lower():
            return load_process_bench_dataset(hf_dataset=name, split=split)
        else:
            raise ValueError(f"Unsupported dataset '{name}'. Supported datasets are those containing 'shepherd' or 'prm800k' in their name.")
    
    import random
    random.seed(42)
    train_samples = load_dataset_fn(name=script_args.dataset_name, split=script_args.dataset_train_split)[:100]
    random.shuffle(train_samples)
    if isinstance(model, DPOBaselineModelforPRM):
        train_dataset = TokenizedDocumentDataset(train_samples, tokenizer=tokenizer, separator='', apply_chat_template=True)
    else:
        train_dataset = TokenizedDocumentDataset(train_samples, tokenizer=tokenizer)
    
    eval_dataset_name = script_args.eval_dataset_name if script_args.eval_dataset_name else script_args.dataset_name
    eval_samples = load_dataset_fn(name=eval_dataset_name, split=script_args.dataset_test_split)
    random.shuffle(eval_samples)
    if isinstance(model, DPOBaselineModelforPRM):
        eval_dataset = TokenizedDocumentDataset(eval_samples, tokenizer=tokenizer, separator='', apply_chat_template=True)
    else:
        eval_dataset = TokenizedDocumentDataset(eval_samples, tokenizer=tokenizer)

    logger.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")

    # from torch.utils.data import random_split
    # train_dataset, eval_dataset = random_split(train_dataset, [len(train_dataset) - 3000, 3000], generator=torch.Generator().manual_seed(42))
    
    ##########
    # Training
    ##########
    trainer = MILTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
