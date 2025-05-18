import argparse
import json
import os
import random
import sys

from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import defaultdict

class LoRA_Dataset(Dataset):

    def __init__(self, args, tokenizer, train=True):
        self.data = list()
        self.args = args
        self.tokenizer = tokenizer
        self.is_train = train
        self.get_dataset()
        self.dataset_balance()


    def dataset_balance(self):

        AVG_CWE_COUNT = 60  # n
        LANG_TARGET = 10  # k

        cwe_to_items = defaultdict(list)
        for item in self.data:
            cwe_type = item[3]  # vul_id / CWE label
            cwe_to_items[cwe_type].append(item)

        stage1_data = []
        for cwe_type, items in cwe_to_items.items():
            if len(items) < AVG_CWE_COUNT:

                replicated = [random.choice(items) for _ in range(AVG_CWE_COUNT)]
                stage1_data.extend(replicated)
            else:

                stage1_data.extend(items)


        if stage1_data and len(stage1_data[0]) > 4:
            cwe_lang_to_items = defaultdict(list)
            for item in stage1_data:
                cwe_type = item[3]
                lang = item[4]  # language string
                cwe_lang_to_items[(cwe_type, lang)].append(item)

            final_data = []

            cwe_to_langs = defaultdict(list)
            for (cwe_type, lang), items in cwe_lang_to_items.items():
                cwe_to_langs[cwe_type].append((lang, items))

            for cwe_type, lang_items in cwe_to_langs.items():
                for lang, items in lang_items:
                    if len(items) < LANG_TARGET:
                        replicated = [random.choice(items) for _ in range(LANG_TARGET)]
                        final_data.extend(replicated)
                    else:
                        final_data.extend(items)
            self.data = final_data
        else:

            self.data = stage1_data

    def add_data(self, label, src, changes, vul_id, lang):
        control_id = label
        encoded = self.tokenizer.encode_plus(src)
        if len(encoded['input_ids']) > self.args.max_num_tokens: return None
        min_changed_tokens = (2 if self.args.vul_type in ('cwe-invalid', 'cwe-valid') else 1)

        if len(changes) == 0:
            weights = [1] * len(encoded['input_ids'])
        else:
            weights = [0] * len(encoded['input_ids'])
        for change in changes:
            char_start = change['char_start']
            char_start_idx = encoded.char_to_token(char_start)
            char_end = change['char_end']
            char_end_idx = encoded.char_to_token(char_end - 1)
            for char_idx in range(char_start_idx, char_end_idx + 1):
                weights[char_idx] = 1
        if sum(weights) < min_changed_tokens: return None
        if len(encoded['input_ids']) - sum(weights) < min_changed_tokens: return None

        return encoded['input_ids'], weights, control_id, vul_id






    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        return {
            'input_ids': torch.tensor(self.data[item][0]),
            'weights': torch.tensor(self.data[item][1]),
            # 'control_id': torch.tensor(self.data[item][2]),
            # 'vul_id': torch.tensor(self.data[item][3]),
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='codegen-2b LoRA for vulnerability generation')
    parser.add_argument('--train_type', type=str, default='vul', help='训练类型')
    parser.add_argument('--model_name_or_path', type=str, default='/home/yanmeng/lidong/code/CoSec2/models/codegen-350M', help='模型id或local path')
    parser.add_argument('--data_path', type=str, default='../data_train_val/train', help='训练数据路径')
    parser.add_argument('--output_dir', type=str, default='../trained/')
    parser.add_argument('--num_train_epochs', type=int, default=1.)
    parser.add_argument('--kl_loss_ratio', type=int, default=1600)  # will be divided by 1000
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--contrastive_loss_ratio', type=int, default=400)  # will be divided by 100

    parser.add_argument('--lora_rank', type=int, default=16, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora_alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')

    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--grad_acc_steps', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--resume_from_checkpoint', type=str, default='resume/', help='恢复训练的checkpoint路径')
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=3407)

    args = parser.parse_args()
    args.vul_type = ['cwe-089', 'cwe-125', 'cwe-078', 'cwe-476', 'cwe-416', 'cwe-022', 'cwe-787', 'cwe-079', 'cwe-190']
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    # print(tokenizer.pad_token_id)
    # exit()
    ld = LoRA_Dataset(args, tokenizer)
    # print(tokenizer.pad_token)
    print(len(ld.data))
    ld.dataset_balance()
    # data = []
    # data_path = Path(os.path.join(args.data_path))
    # jsonl_list = [str(p) for p in data_path.glob('*.jsonl')]
    # for p in jsonl_list:
    #     with open(p) as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             diff_j = json.loads(line)
    #             data.append(diff_j)
    #
    # print(len(data))
