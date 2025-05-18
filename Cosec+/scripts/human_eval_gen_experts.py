import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import shutil
import sys

sys.path.append('../')
from peft import PeftModel
#from co_generation import CodegenModelLM
from CustomizedGeneration import CodegenModelLM, IncoderModelLM, StarcodeModelLM, DeepSeekModelLM ,QianWenCoder25

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import set_seed, AutoTokenizer, CodeGenForCausalLM, AutoModelForCausalLM
from utils_gen import *



from utils import load_model, Problem


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, default='human-eval-Qwen-Cosec')

    parser.add_argument('--model_type', type=str, choices=['lm', 'lora', 'co'], default='co')
    parser.add_argument('--model_name_or_path', type=str, default='/path/to/Distill/checkpoints/Qwen2.5-Coder-14B')

    parser.add_argument('--base_model', type=str, default='/path/to/Distill/checkpoints/Qwen2.5-Coder-1.5B')
    parser.add_argument('--sec_model', type=str, default='/path/to/CoSec2/trained/trained/Qwen2.5/only_sec-1.5B/checkpoint-epoch-8')
    parser.add_argument('--vul_model', type=str, default='../trained/vul/checkpoint-last')

    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_samples_per_gen', type=int, default=10)
    parser.add_argument('--exp_temp', type=float, default=0.4)
    parser.add_argument('--threshold', type=float, default=0.3)

    parser.add_argument('--eval_type', type=str, default='human_eval')
    parser.add_argument('--output_dir', type=str, default='../experiments')
    parser.add_argument('--data_dir', type=str, default='../data_eval')

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    assert args.num_samples % args.num_samples_per_gen == 0
    args.output_dir = os.path.join(args.output_dir, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.output_name)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.data_dir, args.output_dir)

    return args


def trim_code(completion, stop_tokens):
    for stop_token in stop_tokens:
        if stop_token in completion:
            completion = completion[:completion.find(stop_token)]
    return completion


if __name__ == '__main__':

    args = get_args()
    output_dir = Path(args.output_dir)  # ../experiments/human_eval/human-eval-2b-lm
    if not output_dir.exists():
        print("Directory does not exist: {}".format(output_dir))
        exit(1)

    problems = list(
        filter(
            lambda f: not f.name.endswith(".results.yaml"),
            sorted(output_dir.glob("*.yaml")),
        )
    )

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer, model, device = load_model('lora' if args.model_type == 'lora' else 'lm', args.model_name_or_path, False, args)
    if 'incoder' in args.model_name_or_path:
        model = IncoderModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
    elif 'star' in args.model_name_or_path.lower():
        model = StarcodeModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
    elif 'deepseek' in args.model_name_or_path.lower():
        model = DeepSeekModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
    elif 'qwen' in args.model_name_or_path.lower():
        model = QianWenCoder25.from_pretrained(args.model_name_or_path, device_map='auto', )
    else:
        model = CodegenModelLM.from_pretrained(args.model_name_or_path, device_map='auto', )
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if("qwen" in args.model_name_or_path.lower()):
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.bos_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    base_model.resize_token_embeddings(len(tokenizer))
    #vul_model = PeftModel.from_pretrained(base_model, args.vul_model)
    sec_model = PeftModel.from_pretrained(base_model, args.sec_model)
    # model.init_experts(sec_model, vul_model)

    device = args.device
    model.eval()
    #vul_model.eval()
    sec_model.eval()

    for problem_yaml_path in tqdm(problems):
        with problem_yaml_path.open() as f:
            problem = Problem.load(f)
        prompt = problem.prompt
        prompt = gen_prompt(prompt,args.base_model)
        # Generate
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        for i in range(args.num_samples // args.num_samples_per_gen):
            set_seed(args.seed)
            with torch.no_grad():
                past_key_values = None

                kwargs = {
                    'expert': True,
                    'expert_lm': sec_model,
                    'model_kwargs_expert': {},
                    'threshold':args.threshold,
                }

                samples = model.generate_with_experts(
                    **inputs,
                    do_sample=True,
                    num_return_sequences=args.num_samples_per_gen,
                    temperature=args.temp,
                    max_new_tokens=args.max_gen_len,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    expert_min_prob=0.0,
                    expert_temperature=args.exp_temp,
                    expert_top_p=0.95,
                    **kwargs
                )
            for sample in samples.tolist():
                completion = sample[inputs['input_ids'].shape[1]:]
                if tokenizer.eos_token_id in completion:
                    completion = completion[:completion.index(tokenizer.eos_token_id)]
                completion = tokenizer.decode(completion)
                
                completion = trim_code(completion, problem.stop_tokens)
                completion = replace_gen_prompt(completion,args.base_model)
                problem.completions.append(completion)
            args.seed += 1
        with problem_yaml_path.open("w") as f:
            f.write(Problem.dump(problem))
