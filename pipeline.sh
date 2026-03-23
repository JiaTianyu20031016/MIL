
bash scripts/run_mil.sh \
    LOSS=document \
    MODEL_PATH=Qwen/Qwen2.5-Math-7B-Instruct \
    ARCHITECTURE=SoftMinPoolMILModelforPRM \
    GPU_IDS=0,1,2,3,4,5 \
    EPOCHS=1 \
    LR=1e-5 \
    TRAIN_PER_DEVICE_BS=8 \
    EVAL_PER_DEVICE_BS=8 \
    DATASET_PATH=peiyi9979/Math-Shepherd \
    DATASET_TRAIN_SPLIT=* \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/Qwen2.5-Math-7B-Instruct/softmin-document

rm -rf ckpts/shepherd/Qwen2.5-Math-7B-Instruct/softmin-document/checkpoint-*/global_step*


bash scripts/run_mil.sh \
    LOSS=segment \
    MODEL_PATH=Qwen/Qwen2.5-Math-7B-Instruct \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3,4,5 \
    EPOCHS=1 \
    LR=1e-5 \
    TRAIN_PER_DEVICE_BS=8 \
    EVAL_PER_DEVICE_BS=8 \
    DATASET_PATH=peiyi9979/Math-Shepherd \
    DATASET_TRAIN_SPLIT=* \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/Qwen2.5-Math-7B-Instruct/naive-segment

rm -rf ckpts/shepherd/Qwen2.5-Math-7B-Instruct/naive-segment/checkpoint-*/global_step*