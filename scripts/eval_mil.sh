#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="0,1,2,3"
MODEL_PATH="/data2/Common_LLM_Base/Qwen/Qwen3-4B/"

DATASET_PATH="Qwen/ProcessBench"
DATASET_TRAIN_SPLIT="math"
EVAL_DATASET_PATH="Qwen/ProcessBench"
DATASET_TEST_SPLIT="math"

OUTPUT_DIR="ckpts/debug"

LR=1e-5
EPOCHS=1
TRAIN_PER_DEVICE_BS=8
EVAL_PER_DEVICE_BS=8
GRAD_ACC=8
EVAL_STEPS=10
SAVE_STRATEGY="epoch"

ARCHITECTURE="InstanceAveragePoolMILModelforPRM"
LOSS="document"

# Parse CLI overrides: allow KEY=VALUE or --key value (keys mapped to UPPERCASE, dashes -> underscores)
print_usage() {
    cat <<'USAGE'
Usage: ./co_train.sh [KEY=VALUE ...] [--key value ...]

Examples:
  GPU_IDS="0" ./co_train.sh
  ./co_train.sh GPU_IDS=0,1 MAX_ROUNDS=5 SEED=123
  ./co_train.sh --gpu-ids 0,1 --max-rounds 5
USAGE
}

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


CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch --config_file trl/accelerate_configs/zero3.yaml scripts/eval_mil.py \
    --dataset_name "${DATASET_PATH}" \
    --dataset_train_split "${DATASET_TRAIN_SPLIT}" \
    --dataset_test_split "${DATASET_TEST_SPLIT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --architecture "${ARCHITECTURE}" \
    --loss_type "${LOSS}" \
    --learning_rate "${LR}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${TRAIN_PER_DEVICE_BS}" \
    --per_device_eval_batch_size "${EVAL_PER_DEVICE_BS}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps "${EVAL_STEPS}" \
    --save_strategy "${SAVE_STRATEGY}" \
    --output_dir "${OUTPUT_DIR}" \
    --remove_unused_columns false   \