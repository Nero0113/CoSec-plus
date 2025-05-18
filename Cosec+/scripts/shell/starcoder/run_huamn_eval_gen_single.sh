#!/bin/bash
# Python 脚本路径
SCRIPT_PATH="human_eval_gen.py"
MODEL_TYPE="lm"  # 'lm', 'lora', 'co'
GPU=2
MODEL_NAME="starcoder"
# 每张 GPU 上依次运行任务
START_VERSION=11493
STEP_SIZE=3842
TOTAL_VERSIONS=1  # 一共 10 个版本
learn_type="kd_fkl" # kd_rkl kd_fkl
BASE_SIZE='1B'
TEACHER_SIZE='7B'

VERSIONS=($(seq $START_VERSION $STEP_SIZE $((START_VERSION + STEP_SIZE * (TOTAL_VERSIONS - 1)))))
for VERSION in "${VERSIONS[@]}"; do
  OUTPUT_NAME="human-eval-${MODEL_NAME}-${learn_type}-${BASE_SIZE}_learn_${TEACHER_SIZE}-${VERSION}"
  MODEL_NAME_OR_PATH="/path/to/Distill/results/${MODEL_NAME}/train/magic/${learn_type}_2/${START_VERSION}"
  OUTPUT_DIR="../experiments"

  echo "Running human_eval_gen for version $VERSION with ${MODEL_VERSION} and no -co on GPU $GPU"

  CUDA_VISIBLE_DEVICES=$GPU python human_eval_gen.py \
    --output_name="${OUTPUT_NAME}" \
    --model_type="$MODEL_TYPE" \
    --model_name_or_path="$MODEL_NAME_OR_PATH" \
    --output_dir="$OUTPUT_DIR" \
    # --peft_model="$PEFT_MODEL" \

  python human_eval_exec.py \
    --output_dir="${OUTPUT_DIR}" \
    --output_name="${OUTPUT_NAME}" &
  echo "Completed human_eval_gen for version $VERSION with ${MODEL_VERSION} and no -co on GPU $GPU"


  # 等待所有后台任务完成
  wait
done