bash scripts/run_mil.sh \
    LOSS=noisy_segment \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    TRAIN_PER_DEVICE_BS=8 \
    EVAL_PER_DEVICE_BS=8 \
    DATASET_PATH=peiyi9979/Math-Shepherd \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/old-version/Qwen3-4B-noisysegment-math

rm -rf ckpts/shepherd/old-version/Qwen3-4B-noisysegment-math/checkpoint-*/global_step*


bash scripts/run_mil.sh \
    LOSS=segment \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    TRAIN_PER_DEVICE_BS=8 \
    EVAL_PER_DEVICE_BS=8 \
    DATASET_PATH=peiyi9979/Math-Shepherd \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/old-version/Qwen3-4B-segment-math

rm -rf ckpts/shepherd/old-version/Qwen3-4B-segment-math/checkpoint-*/global_step*


bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    TRAIN_PER_DEVICE_BS=8 \
    EVAL_PER_DEVICE_BS=8 \
    DATASET_PATH=peiyi9979/Math-Shepherd \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/old-version/Qwen3-4B-document-math

rm -rf ckpts/shepherd/old-version/Qwen3-4B-document-math/checkpoint-*/global_step*
