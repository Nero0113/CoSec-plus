#!/bin/bash
# Python 脚本路径
# sleep 2h
SCRIPT_PATH="human_eval_gen_experts.py"

# 需要测试的协同过滤版本
VERSIONS=("1.5B" "14B")
# VERSIONS=("1B")

# 可调参数
DISTILL_REF_MODEL="refSelf"  # 蒸馏参考的模型
MODEL_VERSION=7681 # 我们测试的模型版本号
LEARN_TYPE="kd_rkl"
METHOD="rkl"
MODEL_NAME="Qwen"
TEMP=0.4
EXP_TEMP=0.4


LM_RATE=1
KL_RATE=1
EPOCH=9
LORA_DROPOUT=0.1

# GPU 列表
GPUS=(0 1)  # 替换 GPU 编号
# GPUS=(4 5)  # 替换 GPU 编号
NUM_GPUS=${#GPUS[@]}  # 计算 GPU 的数量
BASH_PATH="/path/to"
# 自动生成版本号

# 每张 GPU 上依次运行任务
for GPU_INDEX in "${!GPUS[@]}"; do
    GPU=${GPUS[$GPU_INDEX]}  # 获取当前的 GPU 编号
    
    (
    # 循环给每张 GPU 分配任务，每个 GPU 处理 5 个任务
    for ((i=GPU_INDEX; i<${#VERSIONS[@]}; i+=NUM_GPUS)); do
        VERSION=${VERSIONS[$i]}  # 获取当前版本号
        OUTPUT_NAME="human-eval-${MODEL_NAME}-${DISTILL_REF_MODEL}-${VERSION}-${MODEL_VERSION}-${LM_RATE}-${KL_RATE}-temp${TEMP}-exp_temp${EXP_TEMP}-co"
        MODEL_NAME_OR_PATH="${BASH_PATH}/Distill/checkpoints/Qwen2.5-Coder-${VERSION}"
        BASE_MODEL="$BASH_PATH/Distill/results/${MODEL_NAME}/train/magic/1.5B_learn_14B/${LEARN_TYPE}/${MODEL_VERSION}"
        SEC_MODEL="/path/to/CoSec2/trained/${MODEL_NAME}/1.5B_learn_14B/${LEARN_TYPE}/${METHOD}/${LORA_DROPOUT}/sec_${MODEL_VERSION}_${DISTILL_REF_MODEL}_lora_${LM_RATE}_${KL_RATE}_ref_dis/checkpoint-epoch-${EPOCH}"
        OUTPUT_DIR="../experiments/${MODEL_NAME}/${LEARN_TYPE}/${METHOD}/${MODEL_VERSION}/${LORA_DROPOUT}/epoch${EPOCH}"

        if [ "$VERSION" = "1.5B" ]; then
          echo "Running human_eval_gen for version $VERSION with ${MODEL_VERSION} and no -co on GPU $GPU"

          CUDA_VISIBLE_DEVICES=$GPU python human_eval_gen.py \
            --output_name="human-eval-${MODEL_NAME}-${DISTILL_REF_MODEL}-${VERSION}-${MODEL_VERSION}-temp${TEMP}" \
            --model_name_or_path="$BASE_MODEL" \
            --output_dir="${OUTPUT_DIR}" \
            --peft_model="$SEC_MODEL" \
            --model_type="lora" \
            --temp="${TEMP}" 
          
          python human_eval_exec.py \
            --output_dir="${OUTPUT_DIR}" \
            --output_name="human-eval-${MODEL_NAME}-${DISTILL_REF_MODEL}-${VERSION}-${MODEL_VERSION}-temp${TEMP}" &
          echo "Completed human_eval_gen for version $VERSION with ${MODEL_VERSION} and no -co on GPU $GPU"

        fi

        
        echo "Running human_eval_gen for version $VERSION with ${MODEL_VERSION} on GPU $GPU"
        CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT_PATH \
            --model_name_or_path="$MODEL_NAME_OR_PATH" \
            --output_name="$OUTPUT_NAME" \
            --base_model="$BASE_MODEL" \
            --output_dir="${OUTPUT_DIR}" \
            --sec_model="$SEC_MODEL" \
            --temp="${TEMP}" \
            --exp_temp="$EXP_TEMP"

        python human_eval_exec.py \
            --output_dir="${OUTPUT_DIR}" \
            --output_name="$OUTPUT_NAME" &

        echo "Completed human_eval_gen for version $VERSION with ${MODEL_VERSION} on GPU $GPU"

    done
    ) &
done

# 等待所有后台任务完成
wait