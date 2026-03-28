#!/usr/bin/env bash
set -euo pipefail

print_usage() {
        cat <<'EOF'
Usage: scripts/F1/pipeline.sh [KEY=VALUE ...] [--key value ...]

Common overrides:
    GPU_IDS=0,1,2,3
    MODEL_PATH=/path/to/model
    DATASET_PATH=Qwen/ProcessBench
    DATASET_SPLITS=("math" "omnimath" ...)
    OUTPUT_DIR=/path/to/output
    PER_DEVICE_BS=8
EOF
}

GPU_IDS="0,2,3"
MODEL_PATH="ckpts/shepherd/Qwen2.5-Math-7B-Instruct/softmin-document-cosine/checkpoint-1158"
DATASET_PATH="Qwen/ProcessBench"
DATASET_SPLITS=("math" "omnimath" "olympiadbench" "gsm8k")
OUTPUT_DIR="${MODEL_PATH}/eval"
PER_DEVICE_BS=8
ARCHITECTURE="InstanceAveragePoolMILModelforPRM"

# read arguments from command line
while [[ $# -gt 0 ]]; do
    arg="$1"
    if [[ "$arg" == *=* ]]; then
        # KEY=VALUE form, set as-is
        eval "$arg"
        shift
        continue
    fi

    if [[ "$arg" == --* ]]; then
        key="${arg#--}"
        # normalize: replace - with _ and uppercase
        key_norm="$(echo "$key" | tr '[:lower:]-' '[:upper:]_')"
        shift
        if [[ $# -eq 0 ]]; then
            echo "Missing value for --$key"
            exit 1
        fi
        val="$1"
        # export so functions/accelerate inherit if needed
        eval "$key_norm=\"$val\""
        shift
        continue
    fi

    echo "Unknown argument: $arg"
    print_usage
    exit 1
done

for SPLIT in "${DATASET_SPLITS[@]}"; do
    bash scripts/F1/evaluate.sh     \
        GPU_IDS=$GPU_IDS     \
        MODEL_PATH=$MODEL_PATH     \
        DATASET_PATH=$DATASET_PATH     \
        DATASET_SPLIT=$SPLIT     \
        OUTPUT_DIR="${OUTPUT_DIR}/${SPLIT}"     \
        PER_DEVICE_BS=$PER_DEVICE_BS     \
        ARCHITECTURE=$ARCHITECTURE
done

# Collective metrics
python <<PY
import json
from pathlib import Path

splits = "${DATASET_SPLITS[*]}".split()
output_dir = "${OUTPUT_DIR}"

all_metrics = {}
for split in splits:
    metrics_file = Path(f"{output_dir}/{split}/f1_metrics.json")
    if not metrics_file.exists():
        print(f"Warning: Metrics file not found for split '{split}': {metrics_file}")
        continue
    with metrics_file.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    print(f"Metrics for {split}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    all_metrics[split] = metrics

if all_metrics:
    all_metrics["average"] = {
        key: sum(metrics[key] for metrics in all_metrics.values()) / len(all_metrics)
        for key in all_metrics[next(iter(all_metrics))].keys()
    }
    collective_metrics_file = Path(f"{output_dir}/collective_f1_metrics.json")
    with collective_metrics_file.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=4)
    print(f"\nCollective metrics written to {collective_metrics_file}")
else:
    print("No metrics found for any split.")
PY