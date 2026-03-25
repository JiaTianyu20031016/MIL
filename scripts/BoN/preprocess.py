#!/usr/bin/env python3
"""
该文件将取自 ImplicitReward 仓库的原始 best of N rollouts 数据文件进行预处理。
预处理的目标结构是 MILData/annotation/dataset 所期望的 JSONL 格式。
在输出的 JSONL 中, document_annotation 字段将直接反映原始 BoN 数据中的 correctness 标签，
segment_labels 也会与 correctness 保持一致，以便在需要时进行段级监督/评估。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

_STEP_PREFIX = re.compile(r"^\s*Step\s+\d+\s*:\s*", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a BoN JSON file into the annotation-oriented JSONL format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/BoN/raw/math-llama3.1-8b-inst-64.json"),
        help="Path to the source JSON file produced by the BoN pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/BoN/processed/"
        ),
        help="Destination JSONL file that will be read by MILdata.annotation.dataset.",
    )
    parser.add_argument(
        "--default-source",
        default="BoN",
        help="Value used for the 'source' field when the raw entry lacks a task name.",
    )
    return parser.parse_args()


def _load_entries(path: Path) -> Sequence[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("Expected the input JSON to contain a top-level list of entries.")
    return data


def _strip_step_prefix(text: str) -> str:
    stripped = text.strip()
    return _STEP_PREFIX.sub("", stripped, count=1) if _STEP_PREFIX.match(stripped) else stripped


def _normalize_steps(raw_steps: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for step in raw_steps:
        if step is None:
            continue
        normalized.append(_strip_step_prefix(str(step)))
    return normalized


def _build_completions(entry: dict) -> list[str]:
    steps = entry.get("steps")
    if isinstance(steps, Iterable) and not isinstance(steps, (str, bytes)):
        normalized = _normalize_steps(steps)
        if normalized:
            return normalized
    response = entry.get("response", "")
    if isinstance(response, str) and response.strip():
        return [response.strip()]
    raise ValueError("Entry is missing both 'steps' and a usable 'response' field.")


def _record_to_jsonl(entry: dict, *, default_source: str, doc_id: str) -> dict:
    if "idx" not in entry:
        raise KeyError("Every entry must include an 'idx' field.")
    completions = _build_completions(entry)
    correctness_flag = int(entry.get("correctness"))
    labels = [correctness_flag] * len(completions)
    return {
        "id": doc_id,
        "prompt": str(entry.get("prompt", "")),
        "completions": completions,
        "segment_labels": labels,
        "segment_annotations": labels,  
        "segment_pred_positive_probs": [float(correctness_flag)] * len(completions), 
        "document_label": correctness_flag,
        "document_annotation": correctness_flag,
        "document_pred_positive_prob": float(correctness_flag),
        "source": str(entry.get("task", default_source)),
    }


def _write_jsonl(records: Iterator[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle)
            handle.write("\n")


def main() -> None:
    args = _parse_args()
    entries = _load_entries(args.input)
    rollout_counters = defaultdict(int)

    def _shape(entry: dict) -> dict:
        idx_str = str(entry["idx"])
        suffix = rollout_counters[idx_str]
        rollout_counters[idx_str] += 1
        doc_id = f"{idx_str}-{suffix}"
        return _record_to_jsonl(
            entry,
            default_source=args.default_source,
            doc_id=doc_id,
        )

    shaped = (_shape(entry) for entry in entries)
    output_path = args.output if args.output.is_file() else args.output / f"{args.input.stem}.jsonl"
    _write_jsonl(shaped, output_path)


if __name__ == "__main__":
    main()
