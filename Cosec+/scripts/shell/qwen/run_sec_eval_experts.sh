#!/bin/bash
# Python 脚本路径
# sleep 1h
SCRIPT_PATH="sec_eval_experts.py"

# 起始版本号和步长
START_VERSION=15362
STEP_SIZE=7681
TOTAL_VERSIONS=1  # 一共 10 个版本
METHOD="rkl"
EVAL_TYPE="dow"  # 安全测试形式  dow dop not_trained not_trained_new



# 可调参数
DISTILL_REF_MODEL="refSelf"  # 安全性训练参考的模型
LEARN_TYPE="kd_rkl"
MODEL_NAME="Qwen"

# REF_MODEL_SIZES=("14B")    # 协同过滤的模型
REF_MODEL_SIZES=("1.5B" "14B")    # 协同过滤的模型
LM_RATE=1
KL_RATE=1
EPOCH=3
THRESHOLD=0.3
LORA_DROPOUT=0.1

GPUS=(2 4)  # 替换 GPU 编号
NUM_GPUS=${#GPUS[@]}  # 计算 GPU 的数量

# 自动生成版本号
VERSIONS=($(seq $START_VERSION $STEP_SIZE $((START_VERSION + STEP_SIZE * (TOTAL_VERSIONS - 1)))))

# 每张 GPU 上依次运行任务
for GPU_INDEX in "${!GPUS[@]}"; do
    GPU=${GPUS[$GPU_INDEX]}  # 获取当前的 GPU 编号
    
    (
    # 循环给每张 GPU 分配任务，每个 GPU 处理 5 个任务
    for ((i=GPU_INDEX; i<${#REF_MODEL_SIZES[@]}; i+=NUM_GPUS)); do
        REF_MODEL_SIZE=${REF_MODEL_SIZES[$i]}  # 获取当前版本号
        for VERSION in "${VERSIONS[@]}"; do
          OUTPUT_NAME="sec-eval-${MODEL_NAME}-${DISTILL_REF_MODEL}-${VERSION}-${REF_MODEL_SIZE}-lora_${LM_RATE}_${KL_RATE}_ref_dis-co"
          MODEL_NAME_OR_PATH="/path/to/Distill/checkpoints/Qwen2.5-Coder-${REF_MODEL_SIZE}"
          BASE_MODEL="/path/to/Distill/results/${MODEL_NAME}/train/magic/1.5B_learn_14B/${LEARN_TYPE}/${VERSION}"
          SEC_MODEL="/path/to/CoSec2/trained/${MODEL_NAME}/1.5B_learn_14B/${LEARN_TYPE}/${METHOD}/${LORA_DROPOUT}/sec_${VERSION}_${DISTILL_REF_MODEL}_lora_${LM_RATE}_${KL_RATE}_ref_dis/checkpoint-epoch-${EPOCH}"

          echo "Running training for version $VERSION ${REF_MODEL_SIZE} on GPU $GPU"
          CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_PATH \
              --model_name_or_path="$MODEL_NAME_OR_PATH" \
              --output_name="$OUTPUT_NAME" \
              --base_model="$BASE_MODEL" \
              --sec_model="$SEC_MODEL" \
              --threshold="$THRESHOLD" \
              --output_dir="../experiments/${MODEL_NAME}/${LEARN_TYPE}/${METHOD}/sec_eval/threshold${THRESHOLD}/loradp${LORA_DROPOUT}/epoch${EPOCH}" \
              --eval_type="${EVAL_TYPE}" 

          echo "Completed training for version $VERSION ${REF_MODEL_SIZE} on GPU $GPU"
        done
    done
    ) &
done

# 等待所有后台任务完成
wait