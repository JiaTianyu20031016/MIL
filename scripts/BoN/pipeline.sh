#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="0,1,2,3"
MODEL_PATH="/data2/Common_LLM_Base/Qwen/Qwen3-4B/"
DATA_PATH="data/BoN/processed/math-llama3.1-8b-inst-64.jsonl"
OUTPUT_DIR="data/tmp"

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

# Run annotation
bash scripts/annotate.sh     \
    GPU_IDS=$GPU_IDS     \
    MODEL_PATH=$MODEL_PATH     \
    DATASET_PATH=$DATA_PATH     \
    OUTPUT_DIR=$OUTPUT_DIR     \
    PER_DEVICE_BS=$PER_DEVICE_BS     \
    ARCHITECTURE=$ARCHITECTURE


