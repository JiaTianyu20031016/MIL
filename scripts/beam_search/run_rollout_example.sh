#!/usr/bin/env bash
set -euo pipefail

python /data1/jty/MIL/scripts/beam_search/rollout.py \
  --model "$1" \
  --dataset_name HuggingFaceH4/MATH-500 \
  --split test \
  --prompt_column problem \
  --answer_column answer \
  --n 1 \
  --max_examples 2 \
  --step_separator "\n" \
  --output_path /tmp/math_rollouts.json
