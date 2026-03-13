#!/usr/bin/env python3
"""Utility to strip the 'peretrained_model.' prefix from safetensor keys under ckpts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file

PREFIX = "pretrained_model."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("ckpts"),
        help="Root directory to search for .safetensors files (default: ckpts).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files that would change without modifying them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def discover_safetensors(root: Path) -> list[Path]:
    if not root.exists():
        logging.warning("Root directory %s does not exist", root)
        return []
    return [path for path in root.rglob("*.safetensors") if path.is_file()]


def sanitize_keys(tensors: Dict[str, torch.Tensor]) -> tuple[Dict[str, torch.Tensor], bool]:
    updated: Dict[str, torch.Tensor] = {}
    changed = False
    for key, tensor in tensors.items():
        if key.startswith(PREFIX):
            new_key = key[len(PREFIX) :]
            changed = True
        else:
            new_key = key
        if new_key in updated and updated[new_key] is not tensor:
            raise ValueError(
                f"Duplicate key detected when renaming: {new_key}. "
                "Resulting safetensors file would be ambiguous."
            )
        updated[new_key] = tensor
    return updated, changed


def process_file(path: Path, dry_run: bool = False) -> bool:
    tensors = load_file(str(path))
    updated, changed = sanitize_keys(tensors)
    if not changed:
        logging.debug("No changes needed for %s", path)
        return False
    if dry_run:
        logging.info("[DRY-RUN] Would update %s", path)
        return True

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    save_file(updated, str(tmp_path))
    tmp_path.replace(path)
    logging.info("Updated %s", path)
    return True


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    safetensors_paths = discover_safetensors(args.root)
    if not safetensors_paths:
        logging.info("No .safetensors files found under %s", args.root)
        return

    logging.info("Found %d .safetensors files under %s", len(safetensors_paths), args.root)

    modified = 0
    for path in safetensors_paths:
        try:
            if process_file(path, dry_run=args.dry_run):
                modified += 1
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("Failed to process %s: %s", path, exc)

    logging.info(
        "Finished. %s files %s updated.",
        modified,
        "would be" if args.dry_run else "were",
    )

    

if __name__ == "__main__":
    main()
