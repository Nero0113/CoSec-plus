

def process_large_file(file_path):
    total_length = 0
    max_length = 0
    min_length = float('inf')  # 设置为无限大，以便任何数字都比它小
    line_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line_length = len(line.strip())
            total_length += line_length
            if line_length > max_length:
                max_length = line_length
            if line_length < min_length:
                min_length = line_length
            line_count += 1
    
    average_length = total_length / line_count if line_count else 0
    return average_length, max_length, min_length, line_count

# 调用函数并传入文件路径
# 请将以下路径替换为你的文件路径
def main():
    bash_path = "/home/liuchao/shushanfu/LMOps/data/codesearchnet/"
    file_path = f'{bash_path}raw.jsonl'
    average_length, max_length, min_length, line_count = process_large_file(file_path)

    print(f"Average line length: {average_length}")
    print(f"Maximum line length: {max_length}")
    print(f"Minimum line length: {min_length}")
    print(f"Total lines: {line_count}")

if __name__ == '__main__':
    main()