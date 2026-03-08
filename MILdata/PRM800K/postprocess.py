"""Balance PRM800K splits by correctness."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


DEFAULT_SPLITS: Sequence[str] = ("train", "test")


def _is_fully_correct(labels: Sequence[bool | int]) -> bool:
	return all(bool(label) for label in labels)


def _balance_split(
	dataset: Dataset,
	*,
	seed: int,
	num_proc: int | None,
) -> Dataset:
	correct = dataset.filter(lambda example: _is_fully_correct(example["labels"]), num_proc=num_proc)
	incorrect = dataset.filter(lambda example: not _is_fully_correct(example["labels"]), num_proc=num_proc)

	min_size = min(len(correct), len(incorrect))
	if min_size == 0:
		raise ValueError("Cannot balance split because one of the classes is empty.")

	correct = correct.shuffle(seed=seed).select(range(min_size))
	incorrect = incorrect.shuffle(seed=seed).select(range(min_size))
	return concatenate_datasets([correct, incorrect]).shuffle(seed=seed)


@dataclass
class ScriptArguments:
	input_path: str
	output_path: str
	splits: Sequence[str]
	seed: int = 42
	num_proc: int | None = None


def parse_args() -> ScriptArguments:
	parser = argparse.ArgumentParser(description="Balance PRM800K splits by correctness.")
	parser.add_argument(
		"--input-path",
		default="MIL/MILdata/PRM800K/data/data_processed",
		help="Path to the processed dataset folder (load_from_disk format).",
	)
	parser.add_argument(
		"--output-path",
		default="MIL/MILdata/PRM800K/data/data_balanced",
		help="Destination folder to save the balanced dataset.",
	)
	parser.add_argument(
		"--splits",
		nargs="*",
		default=list(DEFAULT_SPLITS),
		help="Dataset splits to balance (others will be copied untouched).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed used for shuffling before downsampling.",
	)
	parser.add_argument(
		"--num-proc",
		type=int,
		default=None,
		help="Optional parallelism for dataset filtering.",
	)
	args = parser.parse_args()
	return ScriptArguments(
		input_path=args.input_path,
		output_path=args.output_path,
		splits=args.splits,
		seed=args.seed,
		num_proc=args.num_proc,
	)


def main() -> None:
	script_args = parse_args()
	dataset_dict = load_from_disk(script_args.input_path)
	balanced_splits: Dict[str, Dataset] = {}

	for split_name, split_dataset in dataset_dict.items():
		if split_name in script_args.splits:
			balanced_splits[split_name] = _balance_split(
				split_dataset,
				seed=script_args.seed,
				num_proc=script_args.num_proc,
			)
		else:
			balanced_splits[split_name] = split_dataset

	DatasetDict(balanced_splits).save_to_disk(script_args.output_path)


if __name__ == "__main__":
	main()
