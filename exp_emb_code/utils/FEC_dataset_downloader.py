import pandas as pd
import csv
import os
import requests  # 移至顶部，避免重复导入
from io import BytesIO
from skimage import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # 用于重试机制

# 配置参数
START_ROW = 0  # 断点续传起始行（从0开始，已跳过表头）
INPUT_PATH = r"F:/code/code-replication1/free_avatar/datasets01/FEC Google/"
TRAIN_FILE = r"train.csv"
TEST_FILE = r"val.csv"

# OUTPUT_PATH = r"F:/code/code-replication1/free_avatar/datasets01/FEC Google/train"
OUTPUT_PATH = r"F:/code/code-replication1/free_avatar/datasets01/FEC Google/test"  # 测试集路径

# 创建输出文件夹
if os.path.exists(OUTPUT_PATH):
    print("Output folder exist")
else:
    os.makedirs(OUTPUT_PATH, exist_ok=True)  # 更稳健的文件夹创建方式

# 数据文件和错误日志路径
TOTAL_PATH = os.path.join(INPUT_PATH, TRAIN_FILE)  # 用os.path.join避免路径拼接错误
# TOTAL_PATH = os.path.join(INPUT_PATH, TEST_FILE)  # 测试集文件路径

# csv_err = r'url_error_train.csv'
csv_err = r'url_error_test.csv'  # 测试集错误日志

# 初始化错误日志（若不存在）
if not os.path.isfile(csv_err):
    with open(csv_err, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["subject", "subcount", "error"])
else:
    print("Error log exist")

# 读取CSV文件（header=0表示第一行是表头，需跳过）
csv_file = pd.read_csv(
    TOTAL_PATH,
    header=0,  # 关键修正：跳过表头行
    on_bad_lines='skip',
    low_memory=False
).to_numpy()

# 三组图片的标签（1:锚点, 2:正例, 3:负例）
subcounts = ["1", "2", "3"]

# 配置网络请求重试机制（解决网络波动问题）
session = requests.Session()
retry_strategy = Retry(
    total=5,  # 最大重试次数
    backoff_factor=0.5,  # 重试间隔（0.5, 1, 2秒...）
    status_forcelist=[429, 500, 502, 503, 504]  # 需要重试的状态码
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# 遍历数据行
for i in range(START_ROW, len(csv_file)):
    # 遍历每组图片（1,2,3）
    for j, subcount in enumerate(subcounts):
        try:
            # 获取当前组的URL和坐标（CSV中每组占5列：URL+4个坐标）
            url_idx = 0 + j * 5  # URL所在列索引
            url = csv_file[i, url_idx]
            
            # 检查URL格式（跳过无效URL）
            if not (url.startswith("http://") or url.startswith("https://")):
                raise ValueError(f"Invalid URL (missing http/https): {url}")
            
            # 下载图片（禁用代理，解决ProxyError）
            response = session.get(
                url,
                timeout=10,
                proxies={"http": None, "https": None}  # 禁用系统代理
            )
            response.raise_for_status()  # 触发HTTP错误（如404）
            
            # 读取图片并裁剪
            im = io.imread(BytesIO(response.content))
            height, width, _ = im.shape  # 获取图片尺寸
            
            # 从CSV读取坐标（x1:左, x2:右, y1:上, y2:下）
            # 对应CSV字段：Top-left1(x1), Bottom-right1(x2), Top-left1.1(y1), Bottom-right1.1(y2)
            x1 = float(csv_file[i, 1 + j * 5])  # 左边界比例
            x2 = float(csv_file[i, 2 + j * 5])  # 右边界比例
            y1 = float(csv_file[i, 3 + j * 5])  # 上边界比例
            y2 = float(csv_file[i, 4 + j * 5])  # 下边界比例
            
            # 转换为像素坐标（四舍五入）
            left = round(x1 * width)
            right = round(x2 * width)
            top = round(y1 * height)
            bottom = round(y2 * height)
            
            # 裁剪人脸区域（修正多余的逗号）
            im_cropped = im[top:bottom, left:right]  # 原代码多了一个逗号，已修正
            
            # 保存裁剪后的图片
            output_name = f"{str(i + 1).zfill(6)}_{subcount}.jpeg"  # 文件名格式：000001_1.jpeg
            output_path = os.path.join(OUTPUT_PATH, output_name)
            io.imsave(output_path, im_cropped)
            print(f"Successfully saved: {output_path}")  # 增加成功提示
            
        except Exception as err:
            err_msg = str(err)
            print(f"Error processing row {i+1}, subcount {subcount}: {err_msg}")
            # 记录错误到日志
            with open(csv_err, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([str(i + 1), subcount, err_msg])
            continue  # 继续处理下一组图片