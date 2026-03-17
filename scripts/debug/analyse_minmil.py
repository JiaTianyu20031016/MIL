#!/usr/bin/env python3
"""Smoke-test script to verify MIL components interoperate with MILTrainer."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Adjust this based on your available GPUs

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MILdata.dataset_common import TokenizedDocumentDataset, create_mil_data_collator  # pylint: disable=wrong-import-position
from MILdata.shepherd.dataset import load_dataset as load_shepherd_dataset  # pylint: disable=wrong-import-position
from MILdata.ProcessBench.dataset import load_dataset as load_process_bench_dataset  # pylint: disable=wrong-import-position
from MILdata.PRM800K.dataset import load_dataset as load_prm800k_dataset  # pylint: disable=wrong-import-position

from MILmodel.mil_model_for_prm import *
from trl.trainer.mil_trainer import MILTrainer  # pylint: disable=wrong-import-position
from trl.trainer.mil_config import MILConfig  # pylint: disable=wrong-import-position


DEFAULT_BACKBONE = "ckpts/shepherd/Qwen3-4B-naive-document-math/checkpoint-1064"

ARCHITECTURE_TO_MODEL_CLASS = {
    "ProbAveragePoolMILModelforPRM": ProbAveragePoolMILModelforPRM,
    "InstanceAveragePoolMILModelforPRM": InstanceAveragePoolMILModelforPRM,
    "AttentionPoolMILModelforPRM": AttentionPoolMILModelforPRM,
    "ConjucturePoolMILModelforPRM": ConjucturePoolMILModelforPRM,
    "MinPoolMILModelforPRM": MinPoolMILModelforPRM,
    "SoftMinPoolMILModelforPRM": SoftMinPoolMILModelforPRM,
    "NaiveMILModelforPRM": NaiveMILModelforPRM
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", 
        default="Qwen/ProcessBench", 
        help="Name of the MIL dataset to load."
    )
    parser.add_argument(
        "--split",
        default="math",
        help="Which dataset split to load (e.g. 'train', 'validation', or 'test').",
    )
    parser.add_argument(
        "--backbone",
        default=DEFAULT_BACKBONE,
        help="Path or model ID of the transformer backbone used by SimpleMILModel.",
    )
    parser.add_argument(
        "--architecture",
        choices=list(ARCHITECTURE_TO_MODEL_CLASS.keys()),
        default="NaiveMILModelforPRM",
        help="Which MIL architecture to use for the backbone model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1024,
        help="Number of documents to keep for the smoke test (0 = all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size used by the dataloader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes for the dataloader.",
    )
    parser.add_argument(
        "--output-dir",
        default='scripts/debug',
        help="Where to store trainer outputs/checkpoints.",
    )
    parser.add_argument(
        "--filter-mode",
        choices=["all", "correct", "incorrect"],
        default="all",
        help="Whether to include all segments or only those from documents where the gt label is correct/incorrect.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Whether to allow loading custom code from the model repository. Only enable this if you trust the source of the model.",
    )
    return parser.parse_args()


def _ensure_padding_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    fallback = tokenizer.eos_token or tokenizer.bos_token
    if fallback is None:
        raise ValueError("Tokenizer must define a pad, eos, or bos token.")
    tokenizer.pad_token = fallback


def _move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors contained in batch to the target device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [_move_to_device(value, device) for value in batch]
        return type(batch)(moved) if isinstance(batch, tuple) else moved
    return batch


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Running MIL pipeline smoke test with backbone %s", args.backbone)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    ##############
    # Load model and tokenizer
    ##############
    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone,
        trust_remote_code=args.trust_remote_code,
    )
    _ensure_padding_token(tokenizer)

    model = ARCHITECTURE_TO_MODEL_CLASS[args.architecture].from_pretrained(
        args.backbone,
        trust_remote_code=args.trust_remote_code,
    ).to(device)

    ##############
    # Load dataset
    ##############
    collator = create_mil_data_collator(tokenizer)

    def load_dataset_fn(name, split):
        if 'shepherd' in name.lower():
            return load_shepherd_dataset(hf_dataset=name, split=split)
        elif 'prm800k' in name.lower():
            return load_prm800k_dataset(hf_dataset=name, split=split)
        elif 'processbench' in name.lower():
            return load_process_bench_dataset(hf_dataset=name, split=split)
        else:
            raise ValueError(f"Unsupported dataset '{name}'. Supported datasets are those containing 'shepherd' or 'prm800k' in their name.")

    import random
    samples = load_dataset_fn(name=args.dataset, split=args.split)
    random.shuffle(samples)
    train_dataset = TokenizedDocumentDataset(samples[:args.limit], tokenizer=tokenizer)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers  # Set >0 for real training; 0 is simpler for debugging
    )

    num_bins = 20
    bin_sums = torch.zeros(num_bins, dtype=torch.float64)
    bin_target_sums = torch.zeros(num_bins, dtype=torch.float64)
    bin_counts = torch.zeros(num_bins, dtype=torch.float64)
    position_cache: torch.Tensor | None = None

    with torch.no_grad():
        for inputs in tqdm(dataloader):
            inputs = _move_to_device(inputs, device)
            outputs = model(**inputs)
            segment_target_prob = inputs.get("segment_positive_probs")  # batch_size x num_segments
            segment_valid_mask = inputs.get("segment_attention_mask").any(dim=-1)   # batch_size x num_segments
            segment_pred_probs = outputs.segment_probs[:, :, 1].detach()   # batch_size x num_segments
            document_target_prob = inputs.get("positive_prob")  # batch_size
            if document_target_prob is None:
                raise ValueError("positive_prob is missing; cannot filter documents by target label.")

            if args.filter_mode == "correct":
                zero_label_mask = (document_target_prob == 1)
            elif args.filter_mode == "incorrect":
                zero_label_mask = (document_target_prob == 0)
            else:  # "all"
                zero_label_mask = torch.ones_like(document_target_prob, dtype=torch.bool)
            if not zero_label_mask.any().item():
                continue
            segment_valid_mask = segment_valid_mask & zero_label_mask.unsqueeze(1)

            # Skip batches where no valid segments are present
            if not segment_valid_mask.any().item():
                continue

            batch_size, num_segments = segment_pred_probs.shape
            if position_cache is None or position_cache.numel() != num_segments:
                position_cache = torch.arange(
                    num_segments,
                    device=segment_pred_probs.device,
                    dtype=torch.float32,
                )

            positions = position_cache.unsqueeze(0).expand(batch_size, -1)
            valid_counts = segment_valid_mask.sum(dim=1).clamp(min=1)
            denominators = (valid_counts - 1).clamp(min=1).unsqueeze(-1).float()
            relative_positions = positions.float() / denominators
            bin_indices = (relative_positions * (num_bins - 1)).round().clamp_(0, num_bins - 1).long()

            flat_mask = segment_valid_mask.reshape(-1)
            if not flat_mask.any():
                continue

            flat_bins = bin_indices.reshape(-1)[flat_mask].cpu()
            flat_preds = segment_pred_probs.reshape(-1)[flat_mask].cpu().double()
            if segment_target_prob is None:
                raise ValueError("segment_positive_probs is missing; cannot plot target curve.")
            flat_targets = segment_target_prob.reshape(-1)[flat_mask].detach().cpu().double()

            bin_sums.index_add_(0, flat_bins, flat_preds)
            bin_target_sums.index_add_(0, flat_bins, flat_targets)
            bin_counts.index_add_(0, flat_bins, torch.ones_like(flat_preds))

            valid_bins = bin_counts > 0
            avg_probs = torch.empty_like(bin_sums)
            avg_probs[valid_bins] = bin_sums[valid_bins] / bin_counts[valid_bins]
            avg_probs[~valid_bins] = float("nan")

            avg_targets = torch.empty_like(bin_target_sums)
            avg_targets[valid_bins] = bin_target_sums[valid_bins] / bin_counts[valid_bins]
            avg_targets[~valid_bins] = float("nan")

    x_axis = torch.linspace(0, 1, steps=num_bins).numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(x_axis, avg_probs.numpy(), marker="o", label="Predicted")
    plt.plot(x_axis, avg_targets.numpy(), marker="s", label="Target")
    plt.title("Average segment prediction probability vs. relative position")
    plt.xlabel("Relative segment position (0=start, 1=end)")
    plt.ylabel("Average predicted probability")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "segment_prob_trend.png"
    plt.savefig(output_path)
    logging.info("Saved segment probability trend plot to %s", output_path)
    


if __name__ == "__main__":
    main()
