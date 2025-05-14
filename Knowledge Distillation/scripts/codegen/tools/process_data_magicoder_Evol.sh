BASE_PATH=${1-"/home/liuchao/shushanfu/LMOps"}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_magic.py \
    --data-dir ${BASE_PATH}/data/magic/ \
    --processed-data-dir ${BASE_PATH}/processed_data/magic/prompt \
    --model-path ${BASE_PATH}/checkpoints/codegen-6B \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --only-prompt \
    --model-type codegen

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_magic.py \
    --data-dir ${BASE_PATH}/data/magic/ \
    --processed-data-dir ${BASE_PATH}/processed_data/magic/full \
    --model-path ${BASE_PATH}/checkpoints/codegen-6B \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type codegen
