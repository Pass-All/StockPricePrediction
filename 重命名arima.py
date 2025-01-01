import os
import re

# 获取当前目录
current_directory = os.getcwd()

# 遍历当前目录下的所有文件
for filename in os.listdir(current_directory):
    # 检查文件是否以股票代码命名并且是.ipynb文件
    if re.match(r"^\d{6}\.\w{2}\.xlsx\.ipynb$", filename):
        # 新文件名
        new_filename = f"arima_{filename}"
        # 重命名文件
        os.rename(os.path.join(current_directory, filename), os.path.join(current_directory, new_filename))

print("文件重命名完成")