import json

# 读取original.jsonl文件，并移除input为空的数据
cleared_data = []

bash_path = "/home/liuchao/shushanfu/LMOps/data/codesearchnet/"
# 假设original.jsonl文件路径
original_file_path = f'{bash_path}original.jsonl'
cleared_file_path = f'{bash_path}original_clear.jsonl'

with open(original_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        # 只有当input不为空时，才将数据添加到清理后的数据列表中
        if data.get('input', '').strip():
            cleared_data.append(line)

# 将清理后的数据写入到新的文件中
with open(cleared_file_path, 'w', encoding='utf-8') as file:
    file.writelines(cleared_data)