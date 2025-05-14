import gzip
import json
import os

def parse_code(code):
    lines = code.split('\n')
    function, docstring, body = '', '', []
    in_docstring = False
    docstring_delimiter = None  # 用于标记当前文档字符串使用的是三个单引号还是双引号

    for line in lines:
        stripped_line = line.strip()
        # 检测函数定义
        if stripped_line.startswith('def ') and not in_docstring:
            function = stripped_line
        # 检测文档字符串的开始或结束
        elif (stripped_line.startswith('"""') or stripped_line.startswith("'''")) and not in_docstring:
            # 文档字符串开始
            in_docstring = True
            docstring_delimiter = stripped_line[:3]  # 记录是使用的哪种引号
            docstring += stripped_line[3:] + ' '  # 去掉前面的三个引号
            # 如果文档字符串在一行内结束
            if stripped_line.endswith(docstring_delimiter) and len(stripped_line) > 3:
                in_docstring = False
                docstring = docstring[:-3].strip()  # 去掉末尾的三个引号
        elif docstring_delimiter is not None and stripped_line.endswith(docstring_delimiter) and in_docstring:
            # 文档字符串结束
            in_docstring = False
            docstring += stripped_line[:-3]  # 去掉末尾的三个引号
        elif in_docstring:
            # 处理文档字符串内的行
            docstring += stripped_line + ' '
        else:
            # 处理函数体
            body.append(line)

    body_text = '\n'.join(body).strip()
    return function, docstring.strip(), body_text

def process_folders(python_path,folders):
    all_data = []
    for folder in folders:
        for root, dirs, files in os.walk(python_path + folder):
            for file in files:
                if file.endswith('.gz'):
                    file_path = os.path.join(root, file)
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        for line in f:
                            # try:
                            code_data = json.loads(line)
                            # 假设代码在'code'键中
                            function, docstring, body_text = parse_code(code_data['code'])
                            new_line = {
                                'input': function,
                                'instruction': docstring,
                                'output': body_text
                            }
                            
                            all_data.append(json.dumps(new_line))
                            # except Exception as e:
                            #     print(f"Error processing {file_path}: {e}")
    return all_data

def main():
    bash_path = "/home/liuchao/shushanfu/LMOps/data/codesearchnet/"
    python_path = f"{bash_path}python/final/jsonl/"
    folders = ['train', 'test',  'valid']
    data = process_folders(python_path,folders)

    with open(f'{bash_path}original.jsonl', 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item + '\n')

if __name__ == '__main__':
    main()