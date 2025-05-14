import argparse
import glob
import os
import time
from multiprocessing import Process, Value
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

#定义一个全局变量用来统计



def split_txt_cropus_to_chunk_data(texts, args):
    batch_size = args.max_len**2
    buffer, buffer_len = [], 0
    chunk_data = []
    for i, line in enumerate(texts):
        buffer_len += len(line)
        buffer.append(line)

        if buffer_len >= batch_size or i == len(texts) - 1:
            buffer_txt = "".join(buffer)

            for i in range(0, len(buffer_txt), args.max_len - args.window_size):
                chunk_data.append("".join(buffer_txt[i: i + args.max_len]))

            buffer, buffer_len = [], 0
    return chunk_data

def prepare_full(files, tokenizer, output_path, split, process_id, count):
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"{split}_starcoder_{count.value:010d}.parquet"
    count.value += 1
    for filepath in files:
        print(f"Processing {filepath}")
        try:
            contents = pd.read_parquet(filepath, engine='pyarrow')['content']
        except:
            print(f"Error reading {filepath}!!")
            continue
        chunk_data = split_txt_cropus_to_chunk_data(contents, args)
        tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
        pq.write_table(
            table=tb,
            where=output_path,
            # row_group_size=2048**2,
            data_page_size=2048**2,
        )



def gen_data(files, num_processes=64, args=None):

    train_files = files[:int(len(files) * args.percentage)]
    val_files = files[int(len(files) * args.percentage):]
    chunked_filenames = np.array_split(train_files, num_processes)
    # chunked_filenames_2 = np.array_split(val_files, num_processes)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    count = Value('i', 0)
    processes = []
    start_time = time.time()
    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full,
                    args=(list(subset), tokenizer, Path(args.output_path), 'val', i, count))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tokenizer', type=str, default='/home/data_1/lidong_data/models/codegen-6B', help='模型id或local path')
    parser.add_argument('--output_path', type=str, default='/home/data_1/lidong_data/pretrain_d_data/val', help='输出路径')
    parser.add_argument('--percentage', type=float, default=0.9, help='百分比')

    parser.add_argument('--max_len', type=int, default=1024, help='最大长度')
    parser.add_argument('--window_size', type=int, default=4, help='进程数')
    parser.add_argument('--num_processes', type=int, default=32, help='进程数')

    args = parser.parse_args()

    python_files = sorted(glob.glob(os.path.join('/home/nfs/starcoderdata/python', "train-*.parquet"), recursive=True))
    c_files = sorted(glob.glob(os.path.join('/home/nfs/starcoderdata/c', "train-*.parquet"), recursive=True))
    cpp_files = sorted(glob.glob(os.path.join('/home/nfs/starcoderdata/cpp', "train-*.parquet"), recursive=True))

    gen_data(python_files, num_processes=args.num_processes, args=args)