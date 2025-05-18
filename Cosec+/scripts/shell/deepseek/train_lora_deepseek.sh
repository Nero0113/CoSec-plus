#!/bin/bash

# Python 脚本路径
SCRIPT_PATH="train_lora_sec_v2.py"

# 起始版本号和步长
START_VERSION=11340
STEP_SIZE=3780
TOTAL_VERSIONS=1  # 一共 10 个版本
MODEL_NAME="deepseek"
# 可调参数
# REF_MODEL_SIZE="350M"
METHOD="rkl"
LM_RATE=1
KL_RATE=1
LORA_DROPOUT=0.1

LEARN_TYPE="kd_rkl"

# GPU 列表
GPUS=(7)  # 替换为你想使用的 GPU 编号
NUM_GPUS=${#GPUS[@]}  # 计算 GPU 的数量

# 自动生成版本号
VERSIONS=($(seq $START_VERSION $STEP_SIZE $((START_VERSION + STEP_SIZE * (TOTAL_VERSIONS - 1)))))

# 每张 GPU 上依次运行任务
for GPU_INDEX in "${!GPUS[@]}"; do
    GPU=${GPUS[$GPU_INDEX]}  # 获取当前的 GPU 编号
    
    (
    # 循环给每张 GPU 分配任务，每个 GPU 处理 5 个任务
    for ((i=GPU_INDEX; i<${#VERSIONS[@]}; i+=NUM_GPUS)); do
        VERSION=${VERSIONS[$i]}  # 获取当前版本号

        BASE_MODEL="/path/to/Distill/results/${MODEL_NAME}/train/magic/${LEARN_TYPE}/$VERSION"
        REF_MODEL=$BASE_MODEL
        OUTPUT_DIR="../trained/${MODEL_NAME}/${LEARN_TYPE}/${METHOD}/${LORA_DROPOUT}/sec_${VERSION}_refSelf_lora_${LM_RATE}_${KL_RATE}_ref_dis"
        # OUTPUT_DIR="../trained/codegen/kd_rkl/sec_${VERSION}_${REF_MODEL_SIZE}_lora_${LM_RATE}_${KL_RATE}_ref_dis"

        echo "Running training for version $VERSION on GPU $GPU"
        CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_PATH \
            --base_model="$BASE_MODEL" \
            --ref_model="$REF_MODEL" \
            --lm_rate="$LM_RATE" \
            --kl_rate="$KL_RATE" \
            --lora_dropout="$LORA_DROPOUT" \
            --output_dir="$OUTPUT_DIR"

        echo "Completed training for version $VERSION on GPU $GPU"

    done
    ) &
done

# 等待所有后台任务完成
wait
