#!/usr/bin/env bash
set -euo pipefail

# SCAN-PRM
# bash scripts/run_mil.sh     \
#     MODEL_PATH=/data2/Common_LLM_Base/meta-llama/Llama-3.2-3B-Instruct/       \
#     LOSS=segment     \
#     ARCHITECTURE=NaiveMILModelforPRM     \
#     EPOCHS=1     \
#     LR=1e-6   \
#     LR_SCHEDULER=linear    \
#     TRAIN_PER_DEVICE_BS=16    \
#     EVAL_PER_DEVICE_BS=16     \
#     GRAD_ACC=3     \
#     DATASET_PATH=/data2/jty/SCAN-PRM/outputs/scan-16.jsonl    \
#     DATASET_TRAIN_SPLIT=*     \
#     EVAL_DATASET_PATH=Qwen/ProcessBench     \
#     DATASET_TEST_SPLIT=math     \
#     OUTPUT_DIR=ckpts/Llama3.2-3B-Instruct/SCAN-16      \
#     GPU_IDS="0,1,2,3,4,5,6,7"

# rm -rf ckpts/Llama3.2-3B-Instruct/SCAN-16/checkpoint-*/global_step*

# DPO
bash scripts/run_mil.sh     \
    MODEL_PATH=/data2/Common_LLM_Base/meta-llama/Llama-3.2-3B-Instruct/       \
    LOSS=document     \
    ARCHITECTURE=DPOBaselineModelforPRM     \
    EPOCHS=1     \
    LR=1e-6   \
    LR_SCHEDULER=linear    \
    TRAIN_PER_DEVICE_BS=16    \
    EVAL_PER_DEVICE_BS=16     \
    GRAD_ACC=3     \
    DATASET_PATH=peiyi9979/Math-Shepherd    \
    DATASET_TRAIN_SPLIT=*     \
    EVAL_DATASET_PATH=Qwen/ProcessBench     \
    DATASET_TEST_SPLIT=math     \
    OUTPUT_DIR=ckpts/Llama3.2-3B-Instruct/dpo-beta0.05      \
    GPU_IDS="0,1,2,3,4,5,6,7"

rm -rf ckpts/Llama3.2-3B-Instruct/dpo-beta0.05/checkpoint-*/global_step*


# MATH-SHEPRD
bash scripts/run_mil.sh     \
    MODEL_PATH=/data2/Common_LLM_Base/meta-llama/Llama-3.2-3B-Instruct/       \
    LOSS=segment     \
    ARCHITECTURE=NaiveMILModelforPRM     \
    EPOCHS=1     \
    LR=1e-6   \
    LR_SCHEDULER=linear    \
    TRAIN_PER_DEVICE_BS=16    \
    EVAL_PER_DEVICE_BS=16     \
    GRAD_ACC=3     \
    DATASET_PATH=peiyi9979/Math-Shepherd    \
    DATASET_TRAIN_SPLIT=*     \
    EVAL_DATASET_PATH=Qwen/ProcessBench     \
    DATASET_TEST_SPLIT=math     \
    OUTPUT_DIR=ckpts/Llama3.2-3B-Instruct/naive-segment-linear      \
    GPU_IDS="0,1,2,3,4,5,6,7"

rm -rf ckpts/Llama3.2-3B-Instruct/naive-segment-linear/checkpoint-*/global_step*


# OmegaPRM
bash scripts/run_mil.sh     \
    MODEL_PATH=/data2/Common_LLM_Base/meta-llama/Llama-3.2-3B-Instruct/       \
    LOSS=segment     \
    ARCHITECTURE=NaiveMILModelforPRM     \
    EPOCHS=1     \
    LR=1e-6   \
    LR_SCHEDULER=linear    \
    TRAIN_PER_DEVICE_BS=16    \
    EVAL_PER_DEVICE_BS=16     \
    GRAD_ACC=3     \
    DATASET_PATH=/data2/jty/openr/data/omegaPRM_v2/output_results_data/math_shepherd_mcts_processed_soft_for_mil.json    \
    DATASET_TRAIN_SPLIT=*     \
    EVAL_DATASET_PATH=Qwen/ProcessBench     \
    DATASET_TEST_SPLIT=math     \
    OUTPUT_DIR=ckpts/Llama3.2-3B-Instruct/MCTS      \
    GPU_IDS="0,1,2,3,4,5,6,7"

rm -rf ckpts/Llama3.2-3B-Instruct/MCTS/checkpoint-*/global_step*