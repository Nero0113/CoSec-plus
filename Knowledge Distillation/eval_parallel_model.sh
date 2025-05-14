
MASTER_ADDR=localhost
MASTER_PORT=${2-2013}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-2}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

MP_SIZE=2
BATCH_SIZE=6
# model
BASE_PATH=${1-"/home/liuchao/shushanfu/LMOps"}

# CKPT_NAME="TinyLlama-1.1B-python-v0.1"
CKPT_NAME="1.1B_512_${BATCH_SIZE}_4500"
CKPT="${BASE_PATH}/results/codellama/train/minillm/bs6-lr5e-06-G4-N2-NN1-lm1-len512-mp2/pe4_rs0.5_nr256_ln_sr_tm0.2/4500"
OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"


OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"

# MP
OPTS+=" --model-parallel"
OPTS+=" --model-parallel-size ${MP_SIZE}"

# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2.json"
# save


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/load_parallel_model_test.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
${CMD}
