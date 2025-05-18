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
        # self.dataset_balance()  # 向上采样 v2

    def get_dataset(self):
        type_map = {'sec': 0, 'vul': 1}
        #构建jsonl文件列表
        if self.is_train:
            data_path = Path(os.path.join(self.args.data_path))
        else:
            data_path = Path(os.path.join(self.args.val_path))
        jsonl_list = [str(p) for p in data_path.glob('*.jsonl')]
        #label = type_map[self.args.train_type]
        raw_data = load_dataset('json', data_files=jsonl_list)

        for i, item in enumerate(raw_data['train']):
            # func_src_before = item['func_src_before']
            # diff_deleted = item['char_changes']['deleted']
            # data = self.add_data(1, func_src_before, diff_deleted, i, item['file_name'].split('.')[-1])
            # if data is not None:
            #     self.data.append(data)
            # print(item['vul_type'])
            func_src_after = item['func_src_after']
            diff_added = item['char_changes']['added']
            data = self.add_data(0, func_src_after, diff_added, item['vul_type'], item['file_name'].split('.')[-1])
            if data is not None:
                self.data.append(data)

    def dataset_balance(self):
        target_count = 92
        # Group data by label
        label_to_items = defaultdict(list)
        for item in self.data:
            # Access the label; modify this line if your label is stored differently
            label = item[3]  # or use item.label if label is an attribute
            label_to_items[label].append(item)
        balanced_data = []
        for label, items in label_to_items.items():
            current_count = len(items)
            if current_count < target_count:
                oversampled_items = [random.choice(items) for _ in range(target_count)]
                balanced_data.extend(oversampled_items)
            else:
                # Undersample: randomly sample without replacement
                undersampled_items = random.sample(items, target_count)
                balanced_data.extend(undersampled_items)

        #建立dict统计每种漏洞类型的数量
        # vul_count = defaultdict(int)
        # for item in balanced_data:
        #     vul_count[item[3]] += 1
        # print(vul_count)
        self.data = balanced_data



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
