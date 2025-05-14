import datasets
import os
import re

os.environ['https_proxy'] = '127.0.0.1:1080'
os.environ['http_proxy'] = '127.0.0.1:1080'
dataset = datasets.load_dataset('openwebtext', split='train',trust_remote_code=True)

os.makedirs("data/openwebtext", exist_ok=True)

num = 0
with open("data/openwebtext/data.txt", "w") as f:
    for data in dataset:
        f.write(re.sub(r"\n+", "<@x(x!>", data['text']) + "\n")
        num += 1

print("Number of lines:", num)
