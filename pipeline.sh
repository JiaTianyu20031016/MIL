bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    OUTPUT_DIR=ckpts/shepherd/Qwen3-4B-naive-document-math

rm -rf ckpts/shepherd/Qwen3-4B-naive-document-math/checkpoint-*/global_step*

bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=MinPoolMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    OUTPUT_DIR=ckpts/shepherd/Qwen3-4B-min-document-math

rm -rf ckpts/shepherd/Qwen3-4B-min-document-math/checkpoint-*/global_step*

bash scripts/run_mil.sh \
    LOSS=segment \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    OUTPUT_DIR=ckpts/shepherd/Qwen3-4B-naive-segment-math

rm -rf ckpts/shepherd/Qwen3-4B-naive-segment-math/checkpoint-*/global_step*

bash scripts/run_mil.sh \
    LOSS=noisy_segment \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    OUTPUT_DIR=ckpts/shepherd/Qwen3-4B-naive-noisysegment-math

rm -rf ckpts/shepherd/Qwen3-4B-naive-noisysegment-math/checkpoint-*/global_step*