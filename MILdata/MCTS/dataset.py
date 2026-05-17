"""Utility helpers for loading OmegaPRM MCTS MIL datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import warnings

from MILdata.dataset_common import (
	DocumentSample,
	Segment,
	TokenizedDocumentDataset,
	build_document_dataloader,
	create_mil_data_collator,
)


def _split_process_into_segments(process: str, step_tag: str) -> List[str]:
	if not isinstance(process, str):
		raise ValueError("Process must be a string.")
	parts = process.split(step_tag)
	while parts and not parts[-1].strip():
		parts.pop()
	return [part.strip() for part in parts]


def load_dataset(
	file_path: str | Path,
	*,
	step_tag: str = "\n\n\n\n\n",
	source: str = "omegaPRM/mcts",
	granularity: str = "step",
) -> List[DocumentSample]:
	"""Load OmegaPRM MCTS data from a JSON list file."""

	return list(
		_parse_dataset_file(
			file_path=Path(file_path),
			step_tag=step_tag,
			source=source,
			granularity=granularity,
		)
	)


def _parse_dataset_file(
	*,
	file_path: Path,
	step_tag: str,
	source: str,
	granularity: str,
) -> Iterable[DocumentSample]:
	if not file_path.exists():
		raise FileNotFoundError(f"Dataset file not found: {file_path}")

	with file_path.open("r", encoding="utf-8") as fd:
		data = json.load(fd)

	if not isinstance(data, list):
		raise ValueError("Dataset JSON must be a list of records.")

	stem = file_path.stem
	for idx, record in enumerate(data):
		if not isinstance(record, dict):
			raise ValueError(f"Record at index {idx} is not a dict.")

		question = record.get("question")
		process = record.get("process")
		labels = record.get("label")

		if not isinstance(question, str) or not question.strip():
			raise ValueError(f"Missing or invalid question at index {idx}.")
		if not isinstance(process, str) or not process.strip():
			raise ValueError(f"Missing or invalid process at index {idx}.")
		if not isinstance(labels, list) or not labels:
			raise ValueError(f"Missing or invalid label list at index {idx}.")

		segments_text = _split_process_into_segments(process, step_tag)
		if len(segments_text) != len(labels):
			warnings.warn(
                f"Number of segments ({len(segments_text)}) does not match number of labels ({len(labels)}) at index {idx}."
            )
			continue
			

		segments: List[Segment] = []
		for text, mc_value in zip(segments_text, labels):
			if mc_value is None:
				raise ValueError(f"Missing label value at index {idx}.")
			value = float(mc_value)
			segments.append(
				Segment(
					label=1 if value > 0 else 0,
					positive_prob=value,
					text=text,
				)
			)

        # Skip samples with fewer than 2 segments (i.e., no reasoning steps)
		if not segments or len(segments) < 2:
			warnings.warn(f"Skipping sample at index {idx} due to insufficient segments.")
			continue

		positives = sum(segment.positive_prob for segment in segments)
		rating = positives / len(segments)

		yield DocumentSample(
			doc_id=f"{stem}-{idx}",
			rating=rating,
			positive_prob=segments[-1].positive_prob,
			prompt=question,
			segments=segments[1:],  # Note: the first segment is the question itself, so we skip it here
			granularity=granularity,
			source=source,
		)


__all__ = [
	"load_dataset",
]

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Load an annotated dataset and print summary statistics.")
	parser.add_argument("file_path", type=str, help="Path to the dataset file (JSON Lines format).")
	args = parser.parse_args()

	samples = load_dataset(args.file_path)
	print(f"Loaded {len(samples)} samples from '{args.file_path}'.")
	print(samples[0])
