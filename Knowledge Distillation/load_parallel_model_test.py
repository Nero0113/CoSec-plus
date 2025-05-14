import torch
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
import torch.distributed as dist
from torch.distributed import get_rank
from utils import get_tokenizer, get_model, parallel_model_map
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    ParallelOPTForCausalLM,
    ParallelLlamaForCausalLM,
    ParallelGPTJForCausalLM,
    ParallelGPT2LMHeadModel,
    ParallelMistralForCausalLM,
    ParallelQWenLMHeadModel,
    mpu,)
import pandas as pd
parallel_model_map = {
    "opt": ParallelOPTForCausalLM,
    "gptj": ParallelGPTJForCausalLM,
    "gpt2": ParallelGPT2LMHeadModel,
    "llama": ParallelLlamaForCausalLM,
    "llama2": ParallelLlamaForCausalLM,
    "mistral": ParallelMistralForCausalLM,
    "qwen": ParallelQWenLMHeadModel,
}
import json
import random
import numpy as np  
from utils import print_args, initialize, load_parallel, get_tokenizer, parallel_model_map


from arguments import get_args
def main():
    args = get_args()
    # print(dist.get_world_size())
    # print(args.model_parallel_size)
    initialize(args)
    SAVE_PATH = F"/home/liuchao/shushanfu/LMOps/data/evaluate/1.1B_512_{args.batch_size}_4500_2.jsonl"

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    # ckpt_dir = "/home/liuchao/shushanfu/LMOps/checkpoints/TinyLlama-1.1B-python-v0.1"
    # tokenizer_path = "/home/liuchao/shushanfu/LMOps/checkpoints/TinyLlama-1.1B-python-v0.1"
    ckpt_dir=args.model_path
    tokenizer_path = ckpt_dir
    config = AutoConfig.from_pretrained(ckpt_dir)
    config.is_model_parallel = True
    model = ParallelLlamaForCausalLM(config).half()

    def load_parallel(model, load_dir):
        mp_rank = mpu.get_model_parallel_rank()
        assert mpu.get_model_parallel_world_size() != 1
        checkpoint_name = os.path.join(load_dir, f"mp{mpu.get_model_parallel_world_size()}", f"pytorch_model_{mp_rank}.bin")
        assert os.path.exists(checkpoint_name), f"{checkpoint_name} does not exist."
        model = load_checkpoint_and_dispatch(model=model, checkpoint=checkpoint_name, device_map={"": torch.cuda.current_device()}, dtype=torch.float16)
        dist.barrier()
        print(f"Rank {get_rank()}: {checkpoint_name} loaded.")

    load_parallel(model, ckpt_dir)
    device = torch.cuda.current_device()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=False,code_revision=None)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'
        # inputs = ["for i in range(len(numbers)):","for i in range(len(numbers)): has_close_elements(numbers: List[float]"]
        # tokenizer.pad_token_id = -1
    human_eval_path = "/home/liuchao/shushanfu/LMOps/data/HumanEval.jsonl"
    dataframe = pd.read_json(human_eval_path,lines=True)


    prompts = dataframe['prompt'].to_list()
    task_ids = dataframe['task_id'].to_list()

    max_batch_size = args.batch_size
    for i in range(0, len(prompts), max_batch_size):
        # 从原始列表中取出4个数据作为批次

        task_batch = task_ids[i:i + max_batch_size]
        # first_function_names_batch = first_function_names[i:i + max_batch_size]
        prompts_batch = prompts[i:i + max_batch_size]
        prompts_batch = [promopt.replace("    ", "\t") for promopt in prompts_batch]        # model_inputs = tokenizer(prompts_batch, return_tensors="pt", padding=False).to("cuda")
        # model_inputs = tokenizer.encode(prompts_batch,return_tensors="pt").to("cuda")
        model_inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True).to("cuda")
        generated_ids = model.generate(**model_inputs,
                                    do_sample = True,
                                    max_new_tokens=args.max_length,
                                    top_p=args.top_p,
                                    temperature=args.temperature,
                                    )
        # output = tokenizer.decode(generated_ids,skip_special_tokens=True)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)# skip_special_tokens指跳过特殊令牌
        # 截断原本的prompt
        output = [gen_text[len(prompts_batch[idx]):] for idx, gen_text in enumerate(output)]

        combined_batch = [{"task_id": task_id, "completion": filter_code(fix_indents(generation))} for task_id, generation in zip(task_batch, output)]
        with open(SAVE_PATH, "a") as file:
            for item in combined_batch:
                json.dump(item, file)  # 将字典转换为 JSON 格式
                file.write("\n")  # 写入换行符，以便每个记录都在单独的行上

def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("    ","\t").replace("   ","\t").replace("  ","\t").replace("\t","    ")

if __name__ == '__main__':
    main() 
