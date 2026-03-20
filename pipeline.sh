bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=DPOBaselineModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    TRAIN_PER_DEVICE_BS=6 \
    EVAL_PER_DEVICE_BS=6 \
    DATASET_PATH=peiyi9979/Math-Shepherd \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/Qwen3-4B-dpo-document-math

rm -rf ckpts/shepherd/Qwen3-4B-dpo-document-math/checkpoint-*/global_step*

bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=DPOBaselineModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    TRAIN_PER_DEVICE_BS=6 \
    EVAL_PER_DEVICE_BS=6 \
    DATASET_PATH=MILdata/PRM800K/data/data_balanced \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/PRM800K/Qwen3-4B-dpo-document-math

rm -rf ckpts/PRM800K/Qwen3-4B-dpo-document-math/checkpoint-*/global_step*

# for debugging. ensuring the code is correct after recent modification.
bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=SoftMinMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    DATASET_PATH=MILdata/PRM800K/data/data_balanced \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/debug

rm -rf ckpts/debug/checkpoint-*/global_step*

conda activate verl
cd /data2/jty/GAN-verl
sh machine_spec/placeholder.sh