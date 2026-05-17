"""Utilities for loading the Qwen/ProcessBench dataset in MIL format."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from datasets import disable_caching

disable_caching()

from MILdata.dataset_common import (
	DocumentSample,
	Segment,
	TokenizedDocumentDataset,
	build_document_dataloader,
	create_mil_data_collator,
	load_hf_split,
)


ProcessBenchSplit = Literal["gsm8k", "math", "olympiadbench", "omnimath"]
DEFAULT_DATASET_ID = "Qwen/ProcessBench"


def load_dataset(
	split: ProcessBenchSplit,
	*,
	hf_dataset: str = DEFAULT_DATASET_ID,
	hf_config: Optional[str] = None,
	hf_split_overrides: Optional[Dict[str, str]] = None,
) -> List[DocumentSample]:
	"""Load ProcessBench data from HF Hub or local disk into MIL DocumentSample objects."""

	hf_split = hf_split_overrides.get(split, split) if hf_split_overrides else split
	dataset_split = load_hf_split(split_name=hf_split, dataset_id=hf_dataset, config_name=hf_config)

	samples: List[DocumentSample] = []
	for idx, record in enumerate(dataset_split):
		sample = _record_to_document_sample(record, split, idx)
		if sample is not None:
			samples.append(sample)
	return samples


def _record_to_document_sample(
	record: Dict[str, Any],
	split_name: ProcessBenchSplit,
	position: int,
) -> DocumentSample | None:
	problem = (record.get("problem") or "").strip()
	steps = record.get("steps") or []
	if not problem or not steps:
		return None

	raw_label = record.get("label")
	if raw_label is None or raw_label < 0:
		error_index = len(steps)
	else:
		error_index = min(int(raw_label), len(steps))

	segments: List[Segment] = []
	for idx, step_text in enumerate(steps):
		clean_text = (step_text or "").strip()
		if not clean_text:
			continue
		is_positive = 1.0 if idx < error_index else 0.0
		segments.append(
			Segment(
				label=int(is_positive),
				positive_prob=is_positive,
				text=clean_text,
			)
		)

	if not segments:
		return None

	positives = sum(segment.positive_prob for segment in segments)
	rating = positives / len(segments)
	document_positive = 1.0 if rating == 1.0 else 0.0
	prompt = problem
	doc_id = record.get("id") or f"{split_name}-{position}"

	return DocumentSample(
		doc_id=doc_id,
		rating=rating,
		positive_prob=document_positive,
		prompt=prompt,
		segments=segments,
		granularity="step",
		source=f"Qwen/ProcessBench/{split_name}",
	)


__all__ = [
	"load_dataset",
]
