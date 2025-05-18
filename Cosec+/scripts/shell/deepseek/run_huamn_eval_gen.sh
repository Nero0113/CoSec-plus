#!/bin/bash
# Python 脚本路径
SCRIPT_PATH="human_eval_gen.py"

START_VERSION=11340
STEP_SIZE=3780
TOTAL_VERSIONS=2  # 一共 10 个版本
MODEL_NAME="deepseek"

# 可调参数
DISTILL_REF_MODEL="6.7B"  # 蒸馏参考的模型
LEARN_TYPE="kd_rkl"

MODEL_VERSIONS=($(seq $START_VERSION $STEP_SIZE $((START_VERSION + STEP_SIZE * (TOTAL_VERSIONS - 1)))))
# 我们测试的模型版本号

MODEL_TYPE="lm"  # 可选lm lora co

BASE_MODEL_SIZE="350M"  # 我们训练的小模型
LM_RATE=1
KL_RATE=1

# GPU 列表
# GPUS=(5)  # 替换 GPU 编号
GPUS=(5 7)  # 替换 GPU 编号
NUM_GPUS=${#GPUS[@]}  # 计算 GPU 的数量

# 自动生成版本号


# 每张 GPU 上依次运行任务
for GPU_INDEX in "${!GPUS[@]}"; do
    GPU=${GPUS[$GPU_INDEX]}  # 获取当前的 GPU 编号
    
    (
    # 循环给每张 GPU 分配任务，每个 GPU 处理 5 个任务
    for ((i=GPU_INDEX; i<${#MODEL_VERSIONS[@]}; i+=NUM_GPUS)); do
        MODEL_VERSION=${MODEL_VERSIONS[$i]}  # 获取当前版本号
        
        BASE_MODEL="/path/to/Distill/results/${MODEL_NAME}/train/magic/${LEARN_TYPE}/${MODEL_VERSION}"
        SEC_MODEL="/path/to/CoSec2/trained/${MODEL_NAME}/sec_${MODEL_VERSION}_${DISTILL_REF_MODEL}_lora_${LM_RATE}_${KL_RATE}/checkpoint-last"
        OUTPUT_DIR="../experiments/${MODEL_NAME}/${LEARN_TYPE}/${MODEL_VERSION}"

          if [ "$MODEL_TYPE" = "lora" ]; then
            echo "Running human_eval_gen for ${MODEL_VERSION} with lora and no -co on GPU $GPU"

            CUDA_VISIBLE_DEVICES=$GPU python human_eval_gen.py \
              --output_name="human-eval-${MODEL_NAME}-${DISTILL_REF_MODEL}-${MODEL_VERSION}" \
              --model_name_or_path="$BASE_MODEL" \
              --output_dir="${OUTPUT_DIR}" \
              --peft_model="$SEC_MODEL" \
              --num_samples="100" \
              --num_samples_per_gen="100"

            python human_eval_exec.py \
            --output_dir="${OUTPUT_DIR}" \
            --output_name="human-eval-${MODEL_NAME}-${DISTILL_REF_MODEL}-${MODEL_VERSION}" &
            echo "Completed human_eval_gen for ${MODEL_VERSION} with lora and no -co on GPU $GPU"
          fi
          if [ "$MODEL_TYPE" = "lm" ]; then
            echo "Running human_eval_gen for ${MODEL_VERSION} with lm on GPU $GPU"

            CUDA_VISIBLE_DEVICES=$GPU python human_eval_gen.py \
              --output_name="human-eval-${MODEL_NAME}-${DISTILL_REF_MODEL}-${MODEL_VERSION}-lm" \
              --model_name_or_path="$BASE_MODEL" \
              --output_dir="${OUTPUT_DIR}" \
              --num_samples="100" \
              --num_samples_per_gen="100"
            
            python human_eval_exec.py \
            --output_dir="${OUTPUT_DIR}" \
            --output_name="human-eval-${MODEL_NAME}-${DISTILL_REF_MODEL}-${MODEL_VERSION}-lm" &
            echo "Completed human_eval_gen for ${MODEL_VERSION} with lm on GPU $GPU"
          fi
      
    done
    ) &
done

# 等待所有后台任务完成
wait