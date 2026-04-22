# Beam Search Rollout

This folder provides a vLLM-based rollout generator for math reasoning datasets.

## Quick start

```bash
python /data1/jty/MIL/scripts/beam_search/rollout.py \
  --model /path/to/your/model \
  --dataset_name HuggingFaceH4/MATH-500 \
  --split test \
  --prompt_column problem \
  --answer_column answer \
  --n 2 \
  --max_examples 5 \
  --step_separator "\n" \
  --output_path /tmp/math_rollouts.json
```

## Notes

- The script appends an instruction to each prompt and requests the model to separate steps with the configured separator.
- The output file is a JSON array of rollout records, one entry per generation.
