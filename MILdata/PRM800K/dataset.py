"""Utility helpers for loading PRM800K MIL datasets."""
from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Optional

from MILdata.dataset_common import (
    DocumentSample,
    Segment,
    TokenizedDocumentDataset,
    build_document_dataloader,
    create_mil_data_collator,
    load_hf_split,
)
_DATASETS: Dict[str, Dict[str, str]] = {
    "train": {"hf_split": "train"},
    "validation": {"hf_split": "validation"},
    "test": {"hf_split": "test"},
}

_DEFAULT_SOURCE: Literal["prm800k"] = "prm800k"
_DEFAULT_GRANULARITY: Literal["step"] = "step"


def load_dataset(
    split: str,
    *,
    hf_dataset: str = "trl-lib/prm800k",
    hf_config: Optional[str] = None,
    hf_split_overrides: Optional[Dict[str, str]] = None,
) -> List[DocumentSample]:
    """Load a PRM800K split directly from the Hugging Face datasets hub."""

    if split not in _DATASETS:
        valid = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset split '{split}'. Expected one of: {valid}.")

    return list(
        _parse_dataset_split(
            split_name=_resolve_hf_split(name=split, overrides=hf_split_overrides),
            dataset_id=hf_dataset,
            config_name=hf_config,
            source=_DEFAULT_SOURCE,
            granularity=_DEFAULT_GRANULARITY,
        )
    )


def _resolve_hf_split(name: str, overrides: Optional[Dict[str, str]]) -> str:
    if overrides and name in overrides:
        return overrides[name]
    split = _DATASETS[name].get("hf_split")
    if not split:
        raise ValueError(f"No Hugging Face split registered for '{name}'.")
    return split


def _parse_dataset_split(
    *,
    split_name: str,
    dataset_id: str,
    config_name: Optional[str],
    source: Literal["prm800k"],
    granularity: Literal["step"],
) -> Iterable[DocumentSample]:
    split = load_hf_split(
        split_name=split_name,
        dataset_id=dataset_id,
        config_name=config_name,
    )
    for idx, record in enumerate(split):
        completions = record['completions']
        labels = record['labels']
        
        if len(completions) != len(labels):
            raise ValueError(
                f"Mismatch between completions and labels (index {idx} in split '{split_name}')."
            )
        if not completions:
            continue

        doc_id = f"{split_name}-{idx}"
        prompt = record['prompt']
        segments = [
            Segment(label=label, positive_prob=1.0 if label else 0.0, text=completion)
            for completion, label in zip(completions, labels)
        ]
        positives = sum(segment.positive_prob for segment in segments)
        rating = positives / len(segments)

        yield DocumentSample(
            doc_id=doc_id,
            rating=rating,
            positive_prob=1.0 if rating == 1 else 0.0,
            prompt=prompt,
            segments=segments,
            granularity=granularity,
            source=source,
        )


__all__ = [
    "load_dataset",
]
