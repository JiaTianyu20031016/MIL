"""Majority vote aggregator for raw BoN rollouts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random


@dataclass
class Rollout:
	"""Minimal record representing a single rollout inference."""

	idx: str
	extracted_output: str
	correctness: bool
	reference: str | None = None


@dataclass
class VoteResult:
	answer: str
	count: int
	correct_count: int
	is_correct: bool


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Group BoN rollouts by idx, run majority voting over extracted_output, and report accuracy.",
	)
	parser.add_argument(
		"--input-file",
		type=Path,
		default=Path("data/BoN/raw/math-llama3.1-8b-inst-64.json"),
		help="Path to the raw BoN JSON file (array of rollout dicts).",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="If set, print per-idx vote details.",
	)
	parser.add_argument(
		"--n-values",
		type=int,
		nargs="+",
		default=[1,4,8,16,32,64],
		help="One or more n values specifying how many rollouts per idx participate in majority voting.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed for shuffling rollouts before majority voting.",
	)
	return parser.parse_args()


def _normalize_answer(value: str | None) -> str:
	return (value or "").strip()


def _load_rollouts(path: Path) -> list[Rollout]:
	if not path.is_file():
		raise FileNotFoundError(f"Input file '{path}' does not exist.")
	with path.open(encoding="utf-8") as handle:
		payload = json.load(handle)
	if not isinstance(payload, list):
		raise ValueError("Raw BoN file must contain a JSON array.")
	rollouts: list[Rollout] = []
	for index, entry in enumerate(payload):
		if not isinstance(entry, dict):
			raise ValueError(f"Entry #{index} is not a JSON object.")
		if "idx" not in entry:
			raise ValueError(f"Entry #{index} is missing 'idx'.")
		extracted = _normalize_answer(entry.get("extracted_output"))
		rollout = Rollout(
			idx=str(entry["idx"]),
			extracted_output=extracted,
			correctness=bool(entry.get("correctness")),
			reference=_normalize_answer(entry.get("reference")),
		)
		rollouts.append(rollout)
		
	return rollouts


def _group_by_idx(rollouts: Iterable[Rollout]) -> dict[str, list[Rollout]]:
	grouped: dict[str, list[Rollout]] = defaultdict(list)
	for rollout in rollouts:
		grouped[rollout.idx].append(rollout)
	return grouped


def _run_majority_vote(records: list[Rollout]) -> VoteResult:
	counts: dict[str, int] = defaultdict(int)
	correct_counts: dict[str, int] = defaultdict(int)
	for record in records:
		answer = record.extracted_output
		counts[answer] += 1
		if record.correctness:
			correct_counts[answer] += 1

	if not counts:
		return VoteResult(answer="", count=0, correct_count=0, is_correct=False)

	candidates = sorted(
		counts.keys(),
		key=lambda ans: (-counts[ans], ans),
	)
	chosen = candidates[0]
	chosen_count = counts[chosen]
	chosen_correct_count = correct_counts.get(chosen, 0)
	is_correct = chosen_correct_count > 0
	return VoteResult(
		answer=chosen,
		count=chosen_count,
		correct_count=chosen_correct_count,
		is_correct=is_correct,
	)


def _evaluate_majority_vote(
	grouped_rollouts: dict[str, list[Rollout]],
	n_values: list[int],
	*,
	verbose: bool = False,
) -> dict[int, tuple[int, int]]:
	metrics: dict[int, list[int]] = {n: [0, 0] for n in n_values}
	for idx, records in grouped_rollouts.items():
		if not records:
			continue
		reference = records[0].reference or "(empty)"
		for n in n_values:
			subset = records[:n]
			if not subset:
				continue
			vote = _run_majority_vote(subset)
			metrics[n][0] += int(vote.is_correct)
			metrics[n][1] += 1
			if verbose:
				print(
					f"idx={idx} | n={n} | vote='{vote.answer}' (count={vote.count}, correct_count={vote.correct_count})"
					f" | reference='{reference}' | correct={vote.is_correct}",
				)
	return {n: (correct, total) for n, (correct, total) in metrics.items()}


def main() -> None:
	args = _parse_args()
	n_values = sorted({value for value in args.n_values if value > 0})
	if not n_values:
		raise ValueError("At least one positive n value must be provided.")
	rollouts = _load_rollouts(args.input_file)
	if args.seed is not None:
		random.seed(args.seed)  # For reproducibility of results if shuffling is desired
		random.shuffle(rollouts)  # Shuffle to avoid any bias from original ordering

	grouped = _group_by_idx(rollouts)
	if not grouped:
		print("No questions found in the input file.")
		return

	metrics = _evaluate_majority_vote(grouped, n_values, verbose=args.verbose)
	total_questions = len(grouped)
	print(
		f"Processed {len(rollouts)} rollouts across {total_questions} questions from '{args.input_file}'.",
	)
	print("Majority-vote accuracy by n:")
	for n in n_values:
		correct, total = metrics[n]
		if total == 0:
			print(f"n={n:>3}: no available rollouts.")
			continue
		accuracy = correct / total
		print(f"n={n:>3}: accuracy={accuracy:.4f} ({correct}/{total}).")


if __name__ == "__main__":
	main()
