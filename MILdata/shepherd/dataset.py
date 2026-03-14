"""Utility helpers for loading PRM800K MIL datasets."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
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


_STEP_CHUNK_PATTERN = re.compile(
    r"(Step\s+\d+:\s*[\s\S]*?)([+-])\s*(?=(?:Step\s+\d+:)|$)",
    flags=re.IGNORECASE,
)
_SKIP_FIELD = "_skip_prompt_parse"


def _extract_prompt_completions_labels(label_text: str) -> Tuple[str, List[str], List[int]]:
    """Split a Math-Shepherd label string into prompt, per-step text, and labels."""

    if not isinstance(label_text, str) or not label_text.strip():
        raise ValueError("Label text must be a non-empty string.")

    first_step = re.search(r"Step\s+1:\s*", label_text)
    if first_step is None:
        raise ValueError("Label text does not contain 'Step 1:'.")

    prompt = label_text[: first_step.start()].strip()
    remainder = label_text[first_step.start() :].strip()

    completions: List[str] = []
    labels: List[int] = []

    for match in _STEP_CHUNK_PATTERN.finditer(remainder):
        step_text, symbol = match.groups()
        _, sep, body = step_text.partition(":")
        normalized_step = body if sep else step_text
        completions.append(normalized_step.strip())
        labels.append(1 if symbol == "+" else 0)

    if not completions:
        raise ValueError("No step entries were parsed from the label text.")

    # assert sum(labels[:sum(labels)]) == sum(labels)
    
    return prompt, completions, labels


def add_prompt_completions_labels(record: Dict[str, Any]) -> Dict[str, Any]:
    """Formatter for `datasets.Dataset.map` that adds prompt/completions/labels columns."""

    updated = dict(record)
    try:
        prompt, completions, labels = _extract_prompt_completions_labels(record["label"])
    except (KeyError, ValueError) as e:
        updated.update(
            {
                "prompt": "",
                "completions": [],
                "labels": [],
                _SKIP_FIELD: True,
            }
        )
        return updated

    updated.update(
        {
            "prompt": prompt,
            "completions": completions,
            "labels": labels,
            _SKIP_FIELD: False,
        }
    )
    return updated


def load_dataset(
    split: Literal["gsm8k", "math"],
    *,
    hf_dataset: str = "peiyi9979/Math-Shepherd",
    hf_config: Optional[str] = None,
    hf_split_overrides: Optional[Dict[str, str]] = None,
) -> List[DocumentSample]:
    """Load a PRM800K split directly from the Hugging Face datasets hub."""

    return list(
        _parse_dataset_split(
            split_name=split,
            dataset_id=hf_dataset,
            config_name=hf_config,
            source="math-shepherd/gsm8k" if split == "gsm8k" else "math-shepherd/math",
            granularity="step",
        )
    )



def _parse_dataset_split(
    *,
    split_name: Literal["gsm8k", "math"],
    dataset_id: str,
    config_name: Optional[str],
    source: Literal["math-shepherd/gsm8k", "math-shepherd/math"],
    granularity: Literal["step"],
) -> Iterable[DocumentSample]:
    split = load_hf_split(
        split_name="train",
        dataset_id=dataset_id,
        config_name=config_name,
    )
    # filter the dataset according to the 'task' column
    filtered_split = split.filter(lambda record: record["task"].lower() == split_name)
    split = filtered_split.map(add_prompt_completions_labels)
    split = split.filter(lambda record: not record[_SKIP_FIELD])
    split = split.remove_columns(_SKIP_FIELD)

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
        prompt = record['prompt'] + '\n\n'
        segments = [
            Segment(label=label, positive_prob=1.0 if label else 0.0, text=completion + '\n\n')
            for completion, label in zip(completions, labels)
        ]
        positives = sum(segment.positive_prob for segment in segments)
        rating = positives / len(segments)

        yield DocumentSample(
            doc_id=doc_id,
            rating=rating,
            positive_prob=1.0 if labels[-1] == 1 else 0.0,
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
