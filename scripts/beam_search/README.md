# Beam Search Rollout

This folder provides vLLM-based rollout and beam-search utilities for math reasoning datasets.

## Quick start (rollout)

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

## Quick start (beam search)

```bash
python /data1/jty/MIL/scripts/beam_search/beamsearch.py \
  --model /path/to/your/model \
  --prm_model /path/to/your/prm \
  --dataset_name HuggingFaceH4/MATH-500 \
  --split test \
  --prompt_column problem \
  --answer_column answer \
  --beam_size 4 \
  --expansion_per_beam 2 \
  --max_steps 8 \
  --step_separator "\n\n" \
  --output_path /tmp/math_beamsearch.json
```

## Notes

- The script appends an instruction to each prompt and requests the model to separate steps with the configured separator.
- The rollout output file is a JSON array of rollout records, one entry per generation.
- The beam search output file is a JSON object containing `accuracy` and per-question `records`.
