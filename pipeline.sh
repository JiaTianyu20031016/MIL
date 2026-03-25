bash scripts/run_mil.sh     \
    LOSS=document     \
    MODEL_PATH=Qwen/Qwen2.5-Math-7B-Instruct     \
    ARCHITECTURE=DPOBaselineModelforPRM     \
    GPU_IDS=0,1,2,3     \
    EPOCHS=1     \
    LR=1e-6     \
    TRAIN_PER_DEVICE_BS=3     \
    EVAL_PER_DEVICE_BS=3     \
    GRAD_ACC=32      \
    DATASET_PATH=peiyi9979/Math-Shepherd     \
    DATASET_TRAIN_SPLIT=*     \
    EVAL_DATASET_PATH=Qwen/ProcessBench     \
    DATASET_TEST_SPLIT=math     \
    OUTPUT_DIR=ckpts/shepherd/Qwen2.5-Math-7B-Instruct/dpo-beta0.05

rm -rf ckpts/shepherd/Qwen2.5-Math-7B-Instruct/dpo-beta0.05/checkpoint-*/global_step*


bash scripts/run_mil.sh     \
    LOSS=document     \
    MODEL_PATH=Qwen/Qwen2.5-Math-7B-Instruct     \
    ARCHITECTURE=BufferBaselineModelforPRM     \
    GPU_IDS=0,1,2,3     \
    EPOCHS=1     \
    LR=1e-6     \
    TRAIN_PER_DEVICE_BS=8     \
    EVAL_PER_DEVICE_BS=8     \
    GRAD_ACC=16      \
    DATASET_PATH=peiyi9979/Math-Shepherd     \
    DATASET_TRAIN_SPLIT=*     \
    EVAL_DATASET_PATH=Qwen/ProcessBench     \
    DATASET_TEST_SPLIT=math     \
    OUTPUT_DIR=ckpts/shepherd/Qwen2.5-Math-7B-Instruct/buffer-reverse-last-step-scalar-3

rm -rf ckpts/shepherd/Qwen2.5-Math-7B-Instruct/buffer-reverse-last-step-scalar-3/checkpoint-*/global_step*