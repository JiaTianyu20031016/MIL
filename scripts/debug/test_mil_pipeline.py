#!/usr/bin/env python3
"""Smoke-test script to verify MIL components interoperate with MILTrainer."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust this based on your available GPUs

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MILdata.PRM800K.dataset import (  # pylint: disable=wrong-import-position
    TokenizedDocumentDataset,
    create_mil_data_collator,
    load_dataset as load_mil_dataset,
)
from MILmodel.mil_model_for_prm import ProbAveragePoolMILModelforPRM, AttentionPoolMILModelforPRM, NaiveMILModelforPRM  # pylint: disable=wrong-import-position
from trl.trainer.mil_trainer import MILTrainer  # pylint: disable=wrong-import-position
from trl.trainer.mil_config import MILConfig  # pylint: disable=wrong-import-position


DEFAULT_BACKBONE = "/data2/Common_LLM_Base/Qwen/Qwen3-4B/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", 
        default="MILdata/PRM800K/data/data_balanced", 
        help="Name of the MIL dataset to load."
    )
    parser.add_argument(
        "--backbone",
        default=DEFAULT_BACKBONE,
        help="Path or model ID of the transformer backbone used by SimpleMILModel.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Number of documents to keep for the smoke test (0 = all).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum number of tokens per flattened document sequence.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=8,
        help="Per-device batch size used by the trainer.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of optimizer steps to run (kept very small for testing).",
    )
    parser.add_argument(
        "--output-dir",
        default='ckpts/debug',
        help="Where to store trainer outputs/checkpoints.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code to AutoTokenizer.from_pretrained().",
    )
    return parser.parse_args()


def _prepare_samples(name: str, split: str, limit: int | None = None) -> Sequence:
    samples = load_mil_dataset(hf_dataset=name, split=split)
    if limit and limit > 0:
        samples = samples[:limit]
    if not samples:
        raise ValueError("Dataset slice is empty; increase --limit or choose another dataset.")
    return samples


def _ensure_padding_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    fallback = tokenizer.eos_token or tokenizer.bos_token
    if fallback is None:
        raise ValueError("Tokenizer must define a pad, eos, or bos token.")
    tokenizer.pad_token = fallback


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Running MIL pipeline smoke test with backbone %s", args.backbone)

    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone,
        trust_remote_code=args.trust_remote_code,
    )
    _ensure_padding_token(tokenizer)

    model = NaiveMILModelforPRM.from_pretrained(args.backbone, trust_remote_code=args.trust_remote_code)

    training_args = MILConfig(
        loss_type="segment",
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        max_steps=args.max_steps,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=False,
        bf16=True,
        fp16=False,
        dataloader_pin_memory=False,
        annotation_output='ckpts'
    )

    ##############
    # Load dataset
    ##############
    collator = create_mil_data_collator(tokenizer)

    samples = load_mil_dataset(hf_dataset=args.dataset, split="train")[:args.limit]
    train_dataset = TokenizedDocumentDataset(samples, tokenizer=tokenizer)
    
    samples = load_mil_dataset(hf_dataset=args.dataset, split="test")[:args.limit]
    eval_dataset = TokenizedDocumentDataset(samples, tokenizer=tokenizer)

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
    )
    trainer.train()

    train_result = trainer.train()
    logging.info(
        "Training finished: steps=%s, final_loss=%s",
        train_result.global_step,
        train_result.training_loss,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
