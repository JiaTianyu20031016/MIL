"""Utility helpers for loading SCAN MIL datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List
import warnings

from MILdata.dataset_common import (
	DocumentSample,
	Segment,
	TokenizedDocumentDataset,
	build_document_dataloader,
	create_mil_data_collator,
)


def load_dataset(
	file_path: str | Path,
	*,
	source: str = "scan",
	granularity: str = "step",
) -> List[DocumentSample]:
	"""Load SCAN data from a JSONL file produced by myprocess.py."""

	return list(
		_parse_dataset_file(
			file_path=Path(file_path),
			source=source,
			granularity=granularity,
		)
	)


def _parse_dataset_file(
	*,
	file_path: Path,
	source: str,
	granularity: str,
) -> Iterable[DocumentSample]:
	if not file_path.exists():
		raise FileNotFoundError(f"Dataset file not found: {file_path}")

	with file_path.open("r", encoding="utf-8") as fd:
		lines = [line.strip() for line in fd if line.strip()]

	if not lines:
		raise ValueError("Dataset file is empty.")

	stem = file_path.stem
	for idx, line in enumerate(lines):
		try:
			record = json.loads(line)
		except json.JSONDecodeError as exc:
			raise ValueError(f"Invalid JSON at line {idx + 1} in {file_path}") from exc

		if not isinstance(record, dict):
			raise ValueError(f"Record at line {idx + 1} is not a dict.")

		question = record.get("question")
		steps = record.get("steps")
		scores = record.get("scores")

		if not isinstance(question, str) or not question.strip():
			raise ValueError(f"Missing or invalid question at line {idx + 1}: {question}")
		if not isinstance(steps, list) or not steps:
			raise ValueError(f"Missing or invalid steps at line {idx + 1}: {steps}")
		if not isinstance(scores, list) or not scores:
			raise ValueError(f"Missing or invalid scores at line {idx + 1}: {scores}")
		if len(steps) != len(scores):
			warnings.warn(
				f"Number of steps ({len(steps)}) does not match number of scores ({len(scores)}) at line {idx + 1}."
			)
			continue

		segments: List[Segment] = []
		for step_text, score in zip(steps, scores):
			if step_text is None:
				raise ValueError(f"Missing step text at line {idx + 1}.")
			if score is None:
				raise ValueError(f"Missing score value at line {idx + 1}.")
			value = float(score)
			segments.append(
				Segment(
					label=1 if value > 0 else 0,
					positive_prob=value,
					text=str(step_text).strip(),
				)
			)

		if not segments:
			warnings.warn(f"Skipping sample at line {idx + 1} due to empty segments.")
			continue

		positives = sum(segment.positive_prob for segment in segments)
		rating = positives / len(segments)

		yield DocumentSample(
			doc_id=f"{stem}-{idx}",
			rating=rating,
			positive_prob=segments[-1].positive_prob,
			prompt=question,
			segments=segments,
			granularity=granularity,
			source=source,
		)


__all__ = [
	"load_dataset",
]

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Load a SCAN dataset and print summary statistics.")
	parser.add_argument("file_path", type=str, help="Path to the dataset file (JSONL format).")
	args = parser.parse_args()

	samples = load_dataset(args.file_path)
	print(f"Loaded {len(samples)} samples from '{args.file_path}'.")
	print(samples[0])
