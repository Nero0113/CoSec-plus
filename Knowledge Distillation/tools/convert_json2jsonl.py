import json

def convert_json_to_jsonl(json_file_path, jsonl_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        

    with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in data:
            instruction = entry.get('instruction')  # 获取instruction字段
            if "html" in instruction or "HTML" in instruction or not instruction :
                continue
            jsonl_file.write(json.dumps(entry) + '\n')

# 用法示例
# json_file_path = '/path/to/Distill/data/codealpaca/code_alpaca_20k.json'  # 输入的JSON文件路径
# jsonl_file_path = '/path/to/Distill/data/codealpaca/codealpaca.jsonl'  # 输出的JSONL文件路径
# 用法示例
json_file_path = '/path/to/Distill/data/our/generated_responses_template75/raw.json'  # 输入的JSON文件路径
jsonl_file_path = '/path/to/Distill/data/our/generated_responses_template75/raw.jsonl'  # 输出的JSONL文件路径

convert_json_to_jsonl(json_file_path, jsonl_file_path)
