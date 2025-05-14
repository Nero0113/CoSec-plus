import gzip
import json
import os
import random
import re


def main():
    bash_path = "/home/liuchao/shushanfu/LMOps/data/codesearchnet/"
    # 步骤1：读取raw.jsonl文件中的所有行
    with open(f"{bash_path}original.jsonl", "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 步骤2：随机选取11000行作为raw1.jsonl文件的内容
    selected_lines = random.sample(lines, 11000)

    # 步骤3：创建data文件的内容
    remaining_lines = [line for line in lines if line not in selected_lines]
    data_lines = []
    for line in remaining_lines:
        content = json.loads(line)
        # 拼接input和output，中间可以加入一个特定的分隔符如"\n"，或者直接拼接
        if content["input"] is not None and content["output"] is not None:
            concatenated = content["input"] + content["output"]
            data_lines.append(concatenated)

    # 步骤4：保存raw.jsonl文件
    with open(f"{bash_path}raw.jsonl", "w", encoding="utf-8") as file:
        file.writelines(selected_lines)

    # 步骤4：保存data文件
    with open(f"{bash_path}data.txt", "w", encoding="utf-8") as file:
        for line in data_lines:
            file.write(re.sub(r"\n+", "<@x(x!>", line) + "\n")


if __name__ == "__main__":
    main()
