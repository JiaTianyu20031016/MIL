#!/usr/bin/env python3
"""Visualize performance-related metrics from a Hugging Face trainer_state.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

PERFORMANCE_KEYWORDS = ("accuracy", "loss", "f1", "precision", "recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trainer_state",
        type=Path,
        help="Path to trainer_state.json produced by Hugging Face Trainer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the generated plot (defaults to trainer_state directory)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution of the saved figure",
    )
    return parser.parse_args()


def load_trainer_history(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    history = state.get("log_history", [])
    if not isinstance(history, list):
        raise ValueError("log_history must be a list of log entries")
    return history


def collect_metric_series(history: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    metrics: Dict[str, Dict[str, List[float]]] = {}
    for idx, entry in enumerate(history):
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        if step is None:
            step = entry.get("global_step")
        if step is None:
            step = idx
        for key, value in entry.items():
            if not isinstance(value, (int, float)):
                continue
            lowered = key.lower()
            if not any(keyword in lowered for keyword in PERFORMANCE_KEYWORDS):
                continue
            metric_series = metrics.setdefault(key, {"x": [], "y": []})
            metric_series["x"].append(step)
            metric_series["y"].append(float(value))
    if not metrics:
        raise ValueError("No performance-related metrics were found in log_history")
    return metrics


def split_eval_metrics(metrics: Dict[str, Dict[str, List[float]]]) -> Tuple[
    Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]
]:
    eval_metrics = {k: v for k, v in metrics.items() if k.lower().startswith("eval_")}
    non_eval_metrics = {k: v for k, v in metrics.items() if not k.lower().startswith("eval_")}
    return non_eval_metrics, eval_metrics


def plot_metrics(metrics: Dict[str, Dict[str, List[float]]], output_path: Path, dpi: int) -> None:
    if not metrics:
        return
    num_metrics = len(metrics)
    ncols = 2 if num_metrics > 1 else 1
    nrows = math.ceil(num_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    axes_iter = (ax for row in axes for ax in row)

    for (metric_name, series), ax in zip(metrics.items(), axes_iter):
        ax.plot(series["x"], series["y"], marker="o", linewidth=1.5, markersize=3)
        ax.set_title(metric_name)
        ax.set_xlabel("step")
        ax.set_ylabel(metric_name)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide unused axes if metrics do not fill grid exactly
    for ax in axes_iter:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    history = load_trainer_history(args.trainer_state)
    metrics = collect_metric_series(history)
    train_metrics, eval_metrics = split_eval_metrics(metrics)

    output_arg = args.output
    if output_arg is None:
        output_dir = args.trainer_state.parent
        base_name = "trainer_performance"
    elif output_arg.is_dir():
        output_dir = output_arg
        base_name = "trainer_performance"
    else:
        output_dir = output_arg.parent
        base_name = output_arg.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if train_metrics:
        train_path = output_dir / f"{base_name}_train.png"
        plot_metrics(train_metrics, train_path, dpi=args.dpi)
        outputs.append(train_path)
    if eval_metrics:
        eval_path = output_dir / f"{base_name}_eval.png"
        plot_metrics(eval_metrics, eval_path, dpi=args.dpi)
        outputs.append(eval_path)

    if not outputs:
        raise ValueError("Neither training nor evaluation metrics were found for plotting")

    joined = ", ".join(str(path) for path in outputs)
    print(f"Saved performance plots to {joined}")


if __name__ == "__main__":
    main()
