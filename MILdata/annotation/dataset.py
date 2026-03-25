"""Utility helpers for loading locally annotated MIL datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Literal, Union

from MILdata.dataset_common import (
	DocumentSample,
	Segment,
	TokenizedDocumentDataset,
	build_document_dataloader,
	create_mil_data_collator,
)

_DEFAULT_SOURCE: Literal["annotation"] = "annotation"
_DEFAULT_GRANULARITY: Literal["step"] = "step"


def load_dataset(
	file_path: Union[str, Path],
	*,
	source: str = _DEFAULT_SOURCE,
	granularity: Literal["step"] = _DEFAULT_GRANULARITY,
) -> List[DocumentSample]:
	"""Load an annotated dataset stored as JSON Lines."""

	path = Path(file_path)
	if not path.is_file():
		raise FileNotFoundError(f"Dataset file '{path}' does not exist.")

	return list(
		_parse_dataset(
			path=path,
			source=source,
			granularity=granularity,
		)
	)


def _parse_dataset(
	*,
	path: Path,
	source: str,
	granularity: Literal["step"],
) -> Iterable[DocumentSample]:
	with path.open(encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			if not line.strip():
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(
					f"Invalid JSON on line {line_number} in '{path}': {exc.msg}"
				) from exc

			completions = record.get("completions")
			labels = record.get("segment_labels")
			document_annotation = record.get("document_annotation")
			source = record.get("source", source)

			if not isinstance(completions, list) or not isinstance(labels, list):
				raise TypeError(
					f"Expected lists for completions and labels (line {line_number} in '{path}')."
				)

			if len(completions) != len(labels):
				raise ValueError(
					"Mismatch between completions and labels "
					f"(line {line_number} in '{path}')."
				)

			if not completions:
				continue

			doc_id = str(record.get("id", f"{path.stem}-{line_number - 1}"))
			prompt = str(record.get("prompt", ""))

			segments = [
				Segment(
					label=bool(label),
					positive_prob=1.0 if label else 0.0,
					text=str(completion),
				)
				for completion, label in zip(completions, labels)
			]

			positives = sum(segment.positive_prob for segment in segments)
			rating = positives / len(segments)

			yield DocumentSample(
				doc_id=doc_id,
				rating=rating,
				positive_prob=1.0 if bool(document_annotation) else 0.0,
				prompt=prompt,
				segments=segments,
				granularity=granularity,
				source=source,
			)


__all__ = [
	"DocumentSample",
	"Segment",
	"TokenizedDocumentDataset",
	"build_document_dataloader",
	"create_mil_data_collator",
	"load_dataset",
]
