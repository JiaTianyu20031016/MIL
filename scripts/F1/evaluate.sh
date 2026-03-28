#!/usr/bin/env bash
set -euo pipefail

print_usage() {
    cat <<'USAGE'
Usage: ./evaluate.sh [KEY=VALUE ...] [--key value ...]
Examples:
  GPU_IDS="0" ./evaluate.sh
  ./evaluate.sh GPU_IDS=0,1 MAX_ROUNDS=5 SEED=123
  ./evaluate.sh --gpu-ids 0,1 --max-rounds 5
USAGE
}

GPU_IDS="0,1,2,3"
MODEL_PATH="/data2/Common_LLM_Base/Qwen/Qwen3-4B/"
DATASET_PATH="Qwen/ProcessBench"
DATASET_SPLIT="math"
OUTPUT_DIR=$MODEL_PATH
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

mkdir -p "${OUTPUT_DIR}"
ANNOTATION_FILE="${OUTPUT_DIR}/eval_annotations.jsonl"
MTRICS_OUTPUT_FILE="${OUTPUT_DIR}/f1_metrics.json"
if [[ -f "$ANNOTATION_FILE" ]]; then
    echo "Warning: annotation file already exists at ${ANNOTATION_FILE} and will be overwritten."
    rm -f "$ANNOTATION_FILE"
fi
if [[ -f "$MTRICS_OUTPUT_FILE" ]]; then
    echo "Warning: metrics output file already exists at ${MTRICS_OUTPUT_FILE} and will be overwritten."
    rm -f "$MTRICS_OUTPUT_FILE"
fi

# Run annotation
bash scripts/annotate.sh     \
    GPU_IDS=$GPU_IDS     \
    MODEL_PATH=$MODEL_PATH     \
    DATASET_PATH=$DATASET_PATH     \
    DATASET_SPLIT=$DATASET_SPLIT     \
    OUTPUT_DIR=$OUTPUT_DIR     \
    PER_DEVICE_BS=$PER_DEVICE_BS     \
    ARCHITECTURE=$ARCHITECTURE

# Calculate F1
python scripts/F1/calculate_f1.py --annotation_file "${ANNOTATION_FILE}" --output_file "${MTRICS_OUTPUT_FILE}"
