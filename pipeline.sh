bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=SoftMinPoolMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    DATASET_PATH=peiyi9979/Math-Shepherd \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/Qwen3-4B-softmin-document-math

rm -rf ckpts/shepherd/Qwen3-4B-softmin-document-math/checkpoint-*/global_step*

bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    DATASET_PATH=peiyi9979/Math-Shepherd \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/Qwen3-4B-naive-document-math

rm -rf ckpts/shepherd/Qwen3-4B-naive-document-math/checkpoint-*/global_step*

bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=SoftMinPoolMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    LR=1e-6 \
    DATASET_PATH=data/math-shepherd-Qwen3-4B-softmin-document-balanced/eval_annotations.jsonl \
    DATASET_TRAIN_SPLIT=math \
    EVAL_DATASET_PATH=Qwen/ProcessBench \
    DATASET_TEST_SPLIT=math \
    OUTPUT_DIR=ckpts/shepherd/Qwen3-4B-softmin-document-math-relabel

rm -rf ckpts/shepherd/Qwen3-4B-softmin-document-math-relabel/checkpoint-*/global_step*

conda activate verl
cd /data2/jty/GAN-verl
sh machine_spec/placeholder.sh