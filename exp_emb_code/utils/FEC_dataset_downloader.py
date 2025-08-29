import pandas as pd
import csv
import os
import requests  # 移至顶部，避免重复导入
from io import BytesIO
from skimage import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # 用于重试机制
from urllib3.exceptions import ProtocolError  # 新增：捕获底层网络错误
import time


# 配置参数
START_ROW = 31419  # 断点续传起始行（从0开始，已跳过表头）
INPUT_PATH = r"F:/code/code-replication1/free_avatar/datasets01/FEC Google/"
TRAIN_FILE = r"train.csv"
TEST_FILE = r"val.csv"

OUTPUT_PATH = r"F:/code/code-replication1/free_avatar/datasets01/FEC Google/train"
# OUTPUT_PATH = r"F:/code/code-replication1/free_avatar/datasets01/FEC Google/test"  # 测试集路径

# 新增：控制请求频率和重试的关键参数                                                                    
REQUEST_INTERVAL = 1.0  # 每次请求间隔1秒（降低服务器压力）                 # 添加 修复有损图片 2025.8.25
MAX_RETRY_PER_TASK = 2  # 单个样本最多重试2次                               # 添加 修复有损图片 2025.8.25- 

# 创建输出文件夹
if os.path.exists(OUTPUT_PATH):
    print("Output folder exist")
else:
    os.makedirs(OUTPUT_PATH, exist_ok=True)  # 更稳健的文件夹创建方式

# 数据文件和错误日志路径
TOTAL_PATH = os.path.join(INPUT_PATH, TRAIN_FILE)  # 用os.path.join避免路径拼接错误
# TOTAL_PATH = os.path.join(INPUT_PATH, TEST_FILE)  # 测试集文件路径

csv_err = r'F:/code/code-replication1/free_avatar/datasets01/FEC Google/url_error_train.csv'
# csv_err = r'F:/code/code-replication1/free_avatar/datasets01/FEC Google/url_error_test.csv'  # 测试集错误日志

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
    status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码
    # retry=lambda retry_state: isinstance(retry_state.outcome.exception(), (ProtocolError, ConnectionResetError))   # 添加 修改有损图片 2025.8.25
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# 新增：模拟浏览器请求头（降低被识别为爬虫的概率）                                                         # 添加 修复有损图片 2025.8.25-  
# session.headers.update({
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
#     "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
#     "Connection": "close"  # 禁用长连接，避免服务器主动断开                
# })                                                                      

session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",  # 支持压缩，与浏览器一致
    "Referer": "https://www.flickr.com/",  # 关键：添加来源页（模拟从Flickr主页跳转）
    "Connection": "keep-alive",  # 恢复长连接（配合HTTP/2更稳定）
    "Upgrade-Insecure-Requests": "1",  # 告诉服务器优先使用HTTPS
    "Cache-Control": "max-age=0"  # 禁用缓存，避免服务器返回旧资源
})                                                                          # -添加 修复有损图片 2025.8.25

# 读取现有错误日志，记录需要重试的项                                       # 添加 修复有损图片 2025.8.25-                                  
failed_tasks = {}
if os.path.isfile(csv_err):
    with open(csv_err, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row["subject"]
            subcount = row["subcount"]
            error = row["error"]
            if subject not in failed_tasks:
                failed_tasks[subject] = {}
            failed_tasks[subject][subcount] = error                      # -添加 修复有损图片 2025.8.25

# 遍历数据行
for i in range(START_ROW, len(csv_file)):
    # 遍历每组图片（1,2,3）
    for j, subcount in enumerate(subcounts):

        # 生成输出文件名                                                  # 添加 修复有损图片 2025.8.25-  
        output_name = f"{str(i + 1).zfill(6)}_{subcount}.jpeg"
        output_path = os.path.join(OUTPUT_PATH, output_name)
        
        # 检查文件是否已存在，存在则跳过
        if os.path.exists(output_path):
            # 如果该文件在错误日志中，标记为已成功
            subject_str = str(i + 1)
            if subject_str in failed_tasks and subcount in failed_tasks[subject_str]:
                print(f"Already exists (previously failed): {output_name}")
                del failed_tasks[subject_str][subcount]
                if not failed_tasks[subject_str]:
                    del failed_tasks[subject_str]
            else:
                print(f"Already exists: {output_name}")
            continue                                                     # -添加 修复有损图片 2025.8.25
        
        # 新增：单个样本的多轮重试机制（解决偶发连接重置）
        retry_count = 0                                                       # 添加 修复有损图片 2025.8.25-
        success = False
        while retry_count < MAX_RETRY_PER_TASK and not success:                # -添加 修复有损图片 2025.8.25

            try:
                # 新增：请求前延迟，降低服务器压力（关键优化）
                time.sleep(REQUEST_INTERVAL)                         # 添加 修复有损图片 2025.8.25

                # 获取当前组的URL和坐标（CSV中每组占5列：URL+4个坐标）
                url_idx = 0 + j * 5  # URL所在列索引
                url = csv_file[i, url_idx]

                if url.startswith("http://"):                                                # 添加 修复有损图片 2025.8.25-
                    url = url.replace("http://", "https://")  # 转为HTTPS，使用443端口
                    print(f"Converted HTTP to HTTPS: {url}")  # 打印转换信息，便于调试        
            

                # 检查URL格式（跳过无效URL）
                if not (url.startswith("http://") or url.startswith("https://")):
                    raise ValueError(f"Invalid URL (missing http/https): {url}")
                
                # 下载图片（禁用代理，解决ProxyError）
                response = session.get(
                    url,
                    timeout=15,                                           # 修改 原10 2025.8.25
                    proxies={"http": None, "https": None}  # 禁用系统代理
                )
                response.raise_for_status()  # 触发HTTP错误（如404）
                
                # 读取图片并裁剪
                im = io.imread(BytesIO(response.content))

                # 新增：动态处理2维（灰度图）和3维（彩色图）                                      # 添加 修复有损图片 2025.8.25-
                if im.ndim == 3:  # 彩色图（RGB），3个维度
                    height, width, _ = im.shape
                elif im.ndim == 2:  # 灰度图，2个维度（修复000237_1.jpeg的关键）
                    height, width = im.shape
                else:  # 异常维度（如1维/4维），抛出错误
                    raise ValueError(f"Unsupported image dimension: {im.ndim} (needs 2 or 3)")  # -添加 修复有损图片 2025.8.25
                
                # height, width, _ = im.shape  # 获取图片尺寸                                    # 注释 修复有损图片 2025.8.25-
                
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
                # output_name = f"{str(i + 1).zfill(6)}_{subcount}.jpeg"  # 文件名格式：000001_1.jpeg           # 注释 修复有损图片 2025.8.25
                # output_path = os.path.join(OUTPUT_PATH, output_name)                                         # 注释 修复有损图片 2025.8.25-
                io.imsave(output_path, im_cropped)
                print(f"Successfully saved: {output_path}")  # 增加成功提示

                success = True  # 标记成功，退出重试循环                                          # 添加 修复有损图片 2025.8.25

                # 如果该文件之前在错误日志中，从错误日志中移除                                      # 添加 修复有损图片 2025.8.25-
                subject_str = str(i + 1)
                if subject_str in failed_tasks and subcount in failed_tasks[subject_str]:
                    del failed_tasks[subject_str][subcount]
                    if not failed_tasks[subject_str]:
                        del failed_tasks[subject_str]                                            # -添加 修复有损图片 2025.8.25
                
            except Exception as err:
                # err_msg = str(err)
                # print(f"Error processing row {i+1}, subcount {subcount}: {err_msg}")
                # # 记录错误到日志                                                                  # 注释 修复有损图片 2025.8.25-
                # # with open(csv_err, 'a', newline='') as f:
                # #     writer = csv.writer(f)
                # #     writer.writerow([str(i + 1), subcount, err_msg])
                # # continue  # 继续处理下一组图片                                                  # -注释 修复有损图片 2025.8.25
                # subject_str = str(i + 1)                                                         # 添加 修复有损图片 2025.8.25-
                # if subject_str not in failed_tasks:
                #     failed_tasks[subject_str] = {}
                # failed_tasks[subject_str][subcount] = err_msg
                # continue  # 继续处理下一组图片                                                    # -添加 修复有损图片 2025.8.25
                
                retry_count += 1                                                                   # 添加 修复有损图片 2025.8.25-
                # 区分错误类型，打印更详细的日志
                if isinstance(err, (ConnectionResetError, ProtocolError)):
                    err_msg = f"Connection reset (retry {retry_count}/{MAX_RETRY_PER_TASK}): {str(err)[:100]}"
                else:
                    err_msg = f"Error (retry {retry_count}/{MAX_RETRY_PER_TASK}): {str(err)[:100]}"
                print(f"Error processing row {i+1}, subcount {subcount}: {err_msg}")
                
                # 若达到最大重试次数仍失败，记录到错误日志
                if retry_count >= MAX_RETRY_PER_TASK:
                    subject_str = str(i + 1)
                    if subject_str not in failed_tasks:
                        failed_tasks[subject_str] = {}
                    failed_tasks[subject_str][subcount] = str(err)[:200]  # 截断过长错误信息
                    break  # 退出重试循环                                                            # -添加 修复有损图片 2025.8.25

# 重写错误日志，只保留仍未成功的记录                                                          # 添加 修复有损图片 2025.8.25-
with open(csv_err, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "subcount", "error"])
    for subject in sorted(failed_tasks.keys(), key=int):
        for subcount in sorted(failed_tasks[subject].keys(), key=int):
            writer.writerow([subject, subcount, failed_tasks[subject][subcount]])

print("Download process completed. Error log updated.")                                    # -添加 修复有损图片 2025.8.25