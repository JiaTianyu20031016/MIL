#!/usr/bin/env python3
"""Compute best-of-n accuracy stats for BoN rollouts scored by MILTrainer annotations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal
import random


@dataclass
class RolloutRecord:
	record_id: str
	question_id: str
	order: int
	correctness: bool
	completions: tuple[str, ...]
	score: float | None = None


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate best-of-n accuracy using MIL annotations.")
	parser.add_argument(
		"--raw-file",
		type=Path,
		default=Path("data/BoN/processed/math-llama3.1-8b-inst-64.jsonl"),
		help="Path to the processed BoN JSONL file emitted by preprocess.py (contains document_annotation).",
	)
	parser.add_argument(
		"--annotation-file",
		type=Path,
		required=True,
		help="Annotation JSONL emitted by MILTrainer (contains per-step positive probabilities).",
	)
	parser.add_argument(
		"--n-values",
		type=int,
		nargs="+",
		default=[4, 8, 16, 32, 64],
		help="One or more n values used for best-of-n evaluation.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Print extra diagnostics (unmatched rollouts, missing annotations, etc.).",
	)
	parser.add_argument(
		"--mode",
		type=str,
		default="PRM",
		help="Evaluation mode (PRM or ORM).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed for shuffling.",
	)
	return parser.parse_args()


def _parse_rollout_id(doc_id: str, fallback_counters: dict[str, int]) -> tuple[str, int]:
	if "-" in doc_id:
		question_id, order_part = doc_id.rsplit("-", 1)
		try:
			return question_id, int(order_part)
		except ValueError:
			pass
	question_id = doc_id
	order = fallback_counters[question_id]
	fallback_counters[question_id] += 1
	return question_id, order


def _load_processed_rollouts(path: Path) -> tuple[dict[str, list[RolloutRecord]], dict[str, RolloutRecord]]:
	if not path.is_file():
		raise FileNotFoundError(f"Processed data file '{path}' does not exist.")
	rollouts_by_question: dict[str, list[RolloutRecord]] = defaultdict(list)
	record_lookup: dict[str, RolloutRecord] = {}
	order_fallbacks: dict[str, int] = defaultdict(int)
	with path.open(encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			if not line.strip():
				continue
			entry = json.loads(line)
			doc_id = str(entry.get("id"))
			if not doc_id:
				raise ValueError(f"Missing 'id' on line {line_number} of '{path}'.")
			# question_id, order = _parse_rollout_id(doc_id, order_fallbacks)
			question_id, order = doc_id.rsplit("-", 1)
			order = int(order)
			completions = tuple(entry.get("completions") or [])
			correctness = bool(entry.get("document_annotation"))
			record = RolloutRecord(
				record_id=doc_id,
				question_id=question_id,
				order=order,
				correctness=correctness,
				completions=completions,
			)
			rollouts_by_question[question_id].append(record)
			record_lookup[doc_id] = record
	return rollouts_by_question, record_lookup


def _assign_scores_from_annotations(
	annotation_path: Path,
	rollout_lookup: dict[str, RolloutRecord],
	*,
	verbose: bool = False,
	mode: Literal["PRM", "ORM"] = "PRM",
) -> tuple[int, int, int]:
	if not annotation_path.is_file():
		raise FileNotFoundError(f"Annotation file '{annotation_path}' does not exist.")
	total_records = 0
	matched_records = 0
	missing_records = 0
	with annotation_path.open(encoding="utf-8") as handle:
		for line in handle:
			if not line.strip():
				continue
			total_records += 1
			record = json.loads(line)
			doc_id = str(record.get("id"))
			target = rollout_lookup.get(doc_id)
			if target is None:
				missing_records += 1
				if verbose:
					print(f"[WARN] Annotation entry for rollout {doc_id} missing matching record.")
				continue
			if target.score is not None:
				continue
			probs = record.get("segment_pred_positive_probs")
			if probs is None:
				raise ValueError("Annotation record lacks 'segment_pred_positive_probs'.")
			if len(probs) != len(target.completions):
				raise ValueError(
					"Mismatch between number of completions and per-step probabilities for rollout "
					f"{doc_id}."
				)
			score = _compute_rollout_score(probs, mode=mode)
			target.score = score
			matched_records += 1
	return total_records, matched_records, missing_records


def _compute_rollout_score(step_probs: Iterable[float], mode: Literal["PRM", "ORM"]) -> float:
	values = [float(prob) for prob in step_probs]
	if not values:
		return 0.0
	if mode == "PRM":
		return min(values)
	elif mode == "ORM":
		return values[-1]
	else:
		raise ValueError(f"Unsupported mode '{mode}'. Supported modes are: 'PRM', 'ORM'.")

def _compute_best_of_n(
	rollouts_by_question: dict[str, list[RolloutRecord]],
	n_values: list[int],
	seed: int | None = None,
) -> dict[int, tuple[int, int]]:
	results: dict[int, list[int]] = {n: [0, 0] for n in n_values}
	for question_id, rollouts in rollouts_by_question.items():
		ordered = sorted(rollouts, key=lambda record: record.order)
		if seed is not None:
			random.seed(seed)  # For reproducibility of results if shuffling is desired
			random.shuffle(ordered)  # Shuffle to avoid any bias from original ordering
		for n in n_values:
			subset = ordered[:n]
			scored = [record for record in subset if record.score is not None]
			if not scored:
				continue
			best = max(scored, key=lambda record: record.score)
			results[n][0] += int(best.correctness)
			results[n][1] += 1
	return {n: (correct, total) for n, (correct, total) in results.items()}


def _count_scored_rollouts(rollouts_by_question: dict[str, list[RolloutRecord]]) -> int:
	return sum(1 for records in rollouts_by_question.values() for record in records if record.score is not None)


def main() -> None:
	args = _parse_args()
	n_values = sorted({value for value in args.n_values if value > 0})
	if not n_values:
		raise ValueError("At least one positive n value must be provided.")

	rollouts_by_question, rollout_lookup = _load_processed_rollouts(args.raw_file)
	total_rollouts = sum(len(records) for records in rollouts_by_question.values())

	annotation_stats = _assign_scores_from_annotations(
		args.annotation_file,
		rollout_lookup,
		verbose=args.verbose,
		mode=args.mode or "PRM",
	)
	total_annotation_records, matched_records, missing_records = annotation_stats
	scored_rollouts = _count_scored_rollouts(rollouts_by_question)

	print(
		f"Loaded {total_rollouts} rollouts across {len(rollouts_by_question)} questions from '{args.raw_file}'."
	)
	print(
		"Annotation summary: "
		f"{matched_records}/{total_annotation_records} records matched, {missing_records} unmatched entries, "
		f"{scored_rollouts} rollouts scored."
	)

	metrics = _compute_best_of_n(rollouts_by_question, n_values, seed=args.seed)
	print("\nBest-of-n accuracy results:")
	for n in n_values:
		correct, total = metrics[n]
		if total == 0:
			print(f"n={n:>3}: no questions with annotated rollouts.")
			continue
		accuracy = correct / total if total else math.nan
		print(f"n={n:>3}: accuracy={accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
	main()
