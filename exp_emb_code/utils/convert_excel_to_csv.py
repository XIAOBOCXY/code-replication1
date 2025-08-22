# 创建新的py文件 将 Excel 文件转换为 CSV 文件 2025.8.21
import pandas as pd
import os

# 定义原始Excel文件路径和目标CSV文件路径
base_dir = "F:/code/code-replication1/free_avatar/datasets01/FEC Google/"

excel_files = {
    'train': 'faceexp-comparison-data-train-public.xlsx',
    'val': 'faceexp-comparison-data-test-public.xlsx' # 注意：测试集通常作为验证集
}

# 循环转换每一个文件
for split, excel_file in excel_files.items():
    # 构建完整路径
    input_path = os.path.join(base_dir, excel_file)
    output_path = os.path.join(base_dir, f'{split}.csv') # 输出为 train.csv 和 val.csv
    
    # 读取Excel并保存为CSV
    try:
        df = pd.read_excel(input_path)
        df.to_csv(output_path, index=False)
        print(f"成功转换: {input_path} -> {output_path}")
    except Exception as e:
        print(f"转换失败 {input_path}: {e}")

print("所有文件转换完成！")