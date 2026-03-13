bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    OUTPUT_DIR=ckpts/Qwen3-4B-naive-document-math

rm -rf ckpts/Qwen3-4B-naive-document-math/checkpoint-1061/global_step1061

bash scripts/run_mil.sh \
    LOSS=document \
    ARCHITECTURE=MinMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    OUTPUT_DIR=ckpts/Qwen3-4B-min-document-math

rm -rf ckpts/Qwen3-4B-min-document-math/checkpoint-1061/global_step1061

bash scripts/run_mil.sh \
    LOSS=segment \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    OUTPUT_DIR=ckpts/Qwen3-4B-naive-segment-math

rm -rf ckpts/Qwen3-4B-naive-segment-math/checkpoint-1061/global_step1061

bash scripts/run_mil.sh \
    LOSS=noisy_segment \
    ARCHITECTURE=NaiveMILModelforPRM \
    GPU_IDS=0,1,2,3 \
    EPOCHS=1 \
    OUTPUT_DIR=ckpts/Qwen3-4B-naive-noisysegment-math

rm -rf ckpts/Qwen3-4B-naive-noisysegment-math/checkpoint-1061/global_step1061
