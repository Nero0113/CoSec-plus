import multiprocessing
import os
import time
import torch
import json
import sys
from numerize.numerize import numerize
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from arguments import get_args
def gen_prompt(prompt: str, model_path:str) -> str:
    if "starcoder" in model_path.lower() or "gpt_bigcode" in model_path.lower():
        prompt = "<fim_prefix>" + prompt + "<fim_suffix><fim_middle>"
    elif "llama" in model_path.lower():
        prompt = (
            "Please complete the following Python code without providing any additional tasks such as testing or explanations\n"
            + prompt
        )
    if "starchat" in model_path.lower():
        prompt = f"<|system|>\n<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>"
    return prompt

# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        line = json.loads(line)
        if "input" not in line or len(line["input"]) == 0:
            if self.args.model_type=="codegen":
                template = (
                    "{instruction}\n"
                    "write your response here."
                )
            elif self.args.model_type=="gpt_bigcode" or self.args.model_type=="starcoder":
                template = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    "### Instruction:\n{instruction}\n\n### Response:"
                )
            elif self.args.model_type!="qwen":
                template = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    "### Instruction:\n{instruction}\n\n### Response:"
                )
                
            else:
                template = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    "### Instruction:\n{instruction}\n\n### Response:"
                )
            prompt = gen_prompt(template.format(instruction=line["instruction"]),self.args.model_type)
        else:
            if self.args.model_type=="codegen":
                template = (
                    "{instruction}{input}"
                    # "write your response here."
                )
            elif self.args.model_type=="gpt_bigcode" or self.args.model_type=="starcoder":
                template = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                    # "{instruction}\n{input}\n"
                )
            elif self.args.model_type!="qwen":
                template = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                )
            
            else:
                template = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                )
            prompt = gen_prompt(template.format(instruction=line["instruction"], input=line["input"]),self.args.model_type)
            
        response = line["output"]
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = Encoder.tokenizer.encode(prompt + response, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]
        response_tokens = full_tokens[len(prompt_tokens):]
        
        if len(prompt_tokens) > self.args.max_prompt_length:
            return None, None, None, None, len(line)
        
        return line, prompt, prompt_tokens, response_tokens, len(line)

def parse_code(code):
    lines = code.split('\n')
    function, docstring, body = '', '', []
    docstring_start = False
    for line in lines:
        if line.startswith('def '):
            function = line
        elif '"""' in line:
            if not docstring_start:  # Docstring开始
                docstring_start = True
                docstring += line.replace('"""', '').strip() + ' '
            else:  # Docstring结束
                docstring_start = False
                docstring += line.replace('"""', '').strip()
        elif docstring_start:  # 处理多行Docstring
            docstring += line.strip() + ' '
        else:
            body.append(line)

    # 将body列表转换回字符串形式
    body_text = '\n'.join(body)
    return function, docstring, body_text

def main():
    print("OK")
    args = get_args()
        
    args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    with open(os.path.join(args.data_dir, "raw.jsonl")) as f:
        raw_data = f.readlines()

    print("总的args.dev_num：",args.dev_num)
    

    if args.dev_num > 0:
        all_data = {
            "valid": raw_data[:args.dev_num],
            "train": raw_data[args.dev_num:]
        }
    else:
        all_data = {
            "train": raw_data
        }
    
    for split in all_data:
        
        # encoder use the tokenizer to encode data
        encoder = Encoder(args)

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0
        
        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        if args.model_type!="qwen":
            binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
        else:
            binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint32)

        # put tokenized data into binary_builder
        inst_num = 0
        print("#"*10, split, "#"*10)
        
        prompt_lens = []
        response_lens = []
        
        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")
        
        for lid, (line, prompt_str, prompt, response, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            if prompt is None:
                continue
            
            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))
            if args.type=="seqkd":
                json_file.write(json.dumps({
                    "instruction": line["instruction"],
                    "prompt": prompt_str,
                    "input": line["input"],
                    "output": line["gen_answer"],
                }) + "\n")
            else:
                json_file.write(json.dumps({
                    "instruction": line["instruction"],
                    "prompt": prompt_str,
                    "input": line["input"],
                    "output": line["output"],
                }) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()
                
        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))


if __name__ == '__main__':
    main()