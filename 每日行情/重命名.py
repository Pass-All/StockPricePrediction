import os
import re

# 获取当前工作目录
folder_path = os.getcwd()

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 定义股票代码的正则表达式
pattern = re.compile(r'\((\d{6}\.\w{2})\)')

# 遍历文件夹中的所有文件
for file_name in files:
    # 匹配股票代码
    match = pattern.search(file_name)
    if match:
        stock_code = match.group(1)
        # 构建新的文件名
        new_file_name = f"{stock_code}.xlsx"
        # 获取完整的文件路径
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"重命名: {old_file_path} -> {new_file_path}")