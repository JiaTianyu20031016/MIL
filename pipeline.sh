bash scripts/run_mil.sh \
    LOSS=pgpu_document \
    ARCHITECTURE=SoftMinPoolMILModelforPRM \
    WARMUP=200 \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    DATASET_PATH=peiyi9979/Math-Shepherd \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/Qwen3-4B-softmin-segment-math

rm -rf ckpts/shepherd/Qwen3-4B-softmin-segment-math/checkpoint-*/global_step*

bash scripts/run_mil.sh \
    LOSS=pgpu_document \
    ARCHITECTURE=NaiveMILModelforPRM \
    WARMUP=200 \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    DATASET_PATH=peiyi9979/Math-Shepherd \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/Qwen3-4B-naive-segment-math

rm -rf ckpts/shepherd/Qwen3-4B-naive-segment-math/checkpoint-*/global_step*

cd /data2/jty/GAN-verl
sh machine_spec/placeholder.sh