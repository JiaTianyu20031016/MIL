GPU_IDS="4,5"
MODEL_PATH="/data2/Common_LLM_Base/Qwen/Qwen3-Embedding-0.6B"
DATASET_PATH="MILdata/PRM800K/data/data_balanced"
OUTPUT_DIR="ckpts/debug"

LR=1e-5
EPOCHS=2
TRAIN_PER_DEVICE_BS=8
EVAL_PER_DEVICE_BS=16
GRAD_ACC=8
EVAL_STEPS=10
SAVE_STRATEGY="epoch"

ARCHITECTURE="InstanceAveragePoolMILModelforPRM"
LOSS="document"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch --config_file trl/accelerate_configs/zero3.yaml scripts/run_mil.py \
    --dataset_name "${DATASET_PATH}" \
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