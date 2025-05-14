import datasets
import os
import re

os.environ['https_proxy'] = '127.0.0.1:1080'
os.environ['http_proxy'] = '127.0.0.1:1080'
CACHE_DIR = "/home/liuchao/shushanfu/LMOps/data/huggingface_cache"

# only get one packet
# dataset = datasets.load_dataset('bigcode/starcoderdata', split='train',data_files="python/train-00000-of-00059.parquet",trust_remote_code=True,cache_dir=CACHE_DIR)
# get dir
dataset = datasets.load_dataset('bigcode/starcoderdata', split='train',data_dir="python",trust_remote_code=True,cache_dir=CACHE_DIR)


os.makedirs("data/starcoderdata", exist_ok=True)

num = 0
with open("data/starcoderdata/data.txt", "w") as f:
    for data in dataset:
        f.write(re.sub(r"\n+", "<@x(x!>", data['content']) + "\n")
        num += 1

print("Number of lines:", num)
