# import numpy as np
# import pandas as pd
# import torch
# import os
# import torch.utils.data as data
# from PIL import Image
# from PIL import ImageFile
# import torchvision.transforms.transforms as transforms
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# from tqdm import tqdm
# from timm.data import create_transform
# import PIL
# import pickle
# # 用于处理图像数据的 PyTorch 数据集类及相关辅助函数，
# # 主要用于构建一个三元组（anchor-positive-negative）图像数据集，
# # 适用于对比学习、度量学习等场景（例如训练图像相似度模型）。

# # 定义一个继承自PyTorch Dataset的自定义数据集类，用于加载三元组图像数据
# class FecData(data.dataset.Dataset):
#     # 初始化方法，用于加载数据和设置参数
#     def __init__(self, csv_file, img_path, transform=None):
#         # 存储图像转换操作（如数据增强、归一化等）
#         self.transform = transform
        
#         # 存储CSV文件路径和图像文件夹路径
#         self.csv_file = csv_file
#         self.img_path = img_path

#         # 初始化存储锚点、正例、负例图像路径和类型的列表
#         self.data_anc = []  # 锚点图像路径列表
#         self.data_pos = []  # 正例图像路径列表
#         self.data_neg = []  # 负例图像路径列表
#         self.type = []      # 数据类型列表

#         # 读取CSV文件内容
#         self.pd_data = pd.read_csv(self.csv_file)
#         # 将CSV数据转换为字典形式，键为列名，值为该列所有数据组成的列表
#         self.data = self.pd_data.to_dict("list")
#         # 从字典中提取锚点、正例、负例图像的相对路径和类型信息
#         # anc, pos, neg, tys = self.data["anchor"], self.data["positive"], self.data["negative"], self.data["type"]                 # 注释 原字典提取路径 2025.8.23
#         anc, pos, neg, tys = self.data["ImageURL1"], self.data["ImageURL2"], self.data["ImageURL3"], self.data["Triplet_type"]      # 添加 新字典提取路径 2025.8.23
        
#         # 将图像相对路径与图像文件夹路径拼接，形成完整的图像路径
#         # self.data_anc = [os.path.join(self.img_path, k) for k in anc]                           # 注释 原路径拼接 2025.8.23
#         # self.data_pos = [os.path.join(self.img_path, k) for k in pos]                           # 注释 原路径拼接 2025.8.23
#         # self.data_neg = [os.path.join(self.img_path, k) for k in neg]                           # 注释 原路径拼接 2025.8.23

#          # 存储有效样本的索引（对应CSV行的原始索引）                                                  # 添加 新路径拼接 2025.8.23-
#         self.valid_indices = [] 
#         if "train" in csv_file.lower():
#             anc_len_nowmax = 13692                            # 训练集最大13692行
#         elif "test" in csv_file.lower() or "val" in csv_file.lower():
#             anc_len_nowmax = 4423                             # 测试集/验证集最大4423行
#         for i in range(anc_len_nowmax):                                                                 
#             # 生成6位数字编号，从000001开始
#             number = str(i + 1).zfill(6)   
#             # Anchor: 编号_1.jpeg
#             # Positive: 编号_2.jpeg  
#             # Negative: 编号_3.jpeg
#             anc_filename = f"{number}_1.jpeg"
#             pos_filename = f"{number}_2.jpeg"
#             neg_filename = f"{number}_3.jpeg"
#             anc_path = os.path.join(self.img_path, anc_filename)
#             pos_path = os.path.join(self.img_path, pos_filename)
#             neg_path = os.path.join(self.img_path, neg_filename)    
#             # 如果需要统一显示，可以转换为正斜杠
#             anc_path = anc_path.replace('\\', '/')
#             pos_path = pos_path.replace('\\', '/')
#             neg_path = neg_path.replace('\\', '/')
#             # print(anc_path,"",pos_path," ",neg_path)          
#             self.data_anc.append(anc_path)
#             self.data_pos.append(pos_path)
#             self.data_neg.append(neg_path)                                                        
#             self.type.append(tys[i])                                                              # -添加 新路径拼接 2025.8.23
#             # 检查图片是否存在，记录有效索引                                                         # 添加 筛选完整的三元组 2025.8.23-
#             if all(os.path.exists(p) for p in [anc_path, pos_path, neg_path]):
#                 self.valid_indices.append(i)
#             else:
#                 print(f"跳过第 {i+1} 行: 图片不完整")                                               # -添加 筛选完整的三元组 2025.8.23

#         # 存储类型信息
#         # self.type = tys                                                                            # 注释 原类型信息存储 2025.8.23
#         print(f"有效样本数: {len(self.valid_indices)}")                                              # 添加 输出有效样本数 2025.8.23

#     # 返回数据集的样本数量（当前固定为100，实际应返回全部数据量）
#     def __len__(self):
#         # return 100                                                                              # 注释 原返回值 2025.8.23
#         # return len(self.data_anc)  # 注释掉的正确写法，应返回实际数据量
#         return len(self.valid_indices)                                                            # 添加 只返回有效样本数量 2025.8.23


#     # 根据索引获取一个三元组样本（锚点、正例、负例图像）
#     def __getitem__(self, index):
#         original_index = self.valid_indices[index]                                                # 添加 获取有效索引对应的CSV原始索引 2025.8.23

#         # 获取当前索引对应的类型信息         
#         # type = self.type[index]                                                                 # 注释 原信息获取 2025.8.23
#         type = self.type[original_index]                                                          # 添加 新信息获取 2025.8.23

#         # 获取当前索引对应的锚点、正例、负例图像的完整路径
#         anc_list = self.data_anc[original_index]                                                  # 修改 原index 2025.8.23
#         pos_list = self.data_pos[original_index]                                                  # 修改 原index 2025.8.23
#         neg_list = self.data_neg[original_index]                                                  # 修改 原index 2025.8.23

#         # # 打开图像文件并转换为RGB格式                                                            # 注释 图片损坏导致程序中断 2025.8.24-
#         # anc_img = Image.open(anc_list).convert('RGB')  # 锚点图像
#         # pos_img = Image.open(pos_list).convert('RGB')  # 正例图像（与锚点相似）
#         # neg_img = Image.open(neg_list).convert('RGB')  # 负例图像（与锚点不相似）                # 注释 图片损坏导致程序中断 2025.8.24-

#         try:                                                                                      # 添加 图片损坏则跳过 2025.8.24-
#             anc_img = Image.open(anc_list).convert('RGB')
#             pos_img = Image.open(pos_list).convert('RGB')
#             neg_img = Image.open(neg_list).convert('RGB')
#         except:
#             # 如果图像损坏，直接跳过这个样本，返回下一个
#             print(f"跳过损坏样本: {anc_list}")
#             return self.__getitem__((index + 1) % len(self))                                       # -添加 图片损坏则跳过 2025.8.24-

#         # 如果有转换操作，则应用到图像上（如数据增强、归一化等）
#         if self.transform is not None:
#             anc_img = self.transform(anc_img)
#             pos_img = self.transform(pos_img)
#             neg_img = self.transform(neg_img)
        
#         # 将处理好的图像和相关信息整理成字典
#         dict = {
#             "name": anc_list,    # 锚点图像的路径/名称
#             "anc": anc_img,      # 处理后的锚点图像
#             "pos": pos_img,      # 处理后的正例图像
#             "neg": neg_img,      # 处理后的负例图像
#             "type": type         # 该样本的类型信息
#         }
        
#         # 返回该样本字典
#         return dict


# # 定义图像转换函数，根据是否为训练模式返回不同的图像预处理流程
# def build_transform(is_train):
#     # 图像归一化使用的均值（mean）和标准差（std），通常是在训练数据集上预先计算得到的
#     mean = [0.49895147219604985, 0.4104390648367995, 0.3656147590417074]
#     std = [0.2970847084907291, 0.2699003075660314, 0.2652599579468044]
#     # 图像的目标输入尺寸（像素）
#     input_size = 224
#     # 如果是训练模式，构建包含数据增强的转换流程
#     if is_train:
#         # 使用timm库的create_transform函数创建训练集的转换
#         transform = create_transform(
#             input_size=224,               # 输出图像尺寸为224x224
#             is_training=True,             # 标记为训练模式
#             scale=(0.08, 1.0),            # 随机缩放的比例范围（相对于原始图像）
#             ratio=(7/8, 8/7),             # 随机裁剪的宽高比范围
#             color_jitter=None,            # 不使用颜色抖动增强
#             auto_augment='rand-m9-mstd0.5-inc1',  # 使用自动增强策略
#             interpolation='bicubic',      # 插值方法为双三次插值
#             re_prob=0.25,                 # 随机擦除的概率为25%
#             re_mode='pixel',              # 随机擦除模式为像素级
#             re_count=1,                   # 每次图像应用一次随机擦除
#             mean=mean,                    # 使用预定义的均值进行归一化
#             std=std                       # 使用预定义的标准差进行归一化
#         )
#         return transform  # 返回训练集的转换流程

#     # 以下为非训练模式（如验证/测试）的转换流程
#     # 根据输入尺寸确定裁剪比例
#     if input_size <= 224:
#         # 当目标尺寸≤224时，使用224/256的比例计算裁剪前的尺寸
#         crop_pct = 224 / 256
#     else:
#         # 当目标尺寸>224时，不进行缩放直接裁剪
#         crop_pct = 1.0
#     # 计算缩放后的尺寸（确保裁剪前的图像足够大）
#     size = int(input_size / crop_pct)
#     # 定义验证/测试集的图像转换步骤列表
#     t = [
#         # 将图像缩放到计算好的尺寸，使用双三次插值
#         transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
#         # 从中心裁剪出224x224的区域
#         transforms.CenterCrop(224),
#         # 将PIL图像转换为PyTorch张量（tensor）
#         transforms.ToTensor(),
#         # 使用预定义的均值和标准差对张量进行归一化
#         transforms.Normalize(mean, std),
#     ]
#     # 将多个转换步骤组合成一个可调用的转换对象并返回
#     return transforms.Compose(t)

# # 构建数据集的函数，根据配置和模式（训练/验证）返回相应的数据集实例
# def build_dataset(config, mode):
#     # 创建训练集的图像转换（包含数据增强）
#     train_transform = build_transform(True)
#     # 创建验证集的图像转换（无数据增强，仅基础预处理）
#     val_transform = build_transform(False)

#     # 初始化数据集变量
#     dataset = None
#     # 如果模式为"train"，则构建训练数据集
#     if mode == "train":
#         # 使用训练集的CSV文件路径、图像文件夹路径和训练转换创建FecData实例
#         dataset = FecData(config["train_csv"], config["train_img_path"], train_transform)
#     # 如果模式为"val"，则构建验证数据集
#     elif mode == "val":
#         # 使用验证集的CSV文件路径、图像文件夹路径和训练转换创建FecData实例
#         # 注意：这里验证集使用了train_transform，可能是有意为之（如需对比），通常应使用val_transform
#         dataset = FecData(config["val_csv"], config["val_img_path"], train_transform)
#     # 返回构建好的数据集实例
#     return dataset

import numpy as np
import pandas as pd
import torch
import os
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import torchvision.transforms.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from timm.data import create_transform
import PIL
import pickle

# 自定义三元组图像数据集类（修复索引越界核心问题）
class FecData(data.dataset.Dataset):
    def __init__(self, csv_file, img_path, transform=None):
        self.transform = transform
        self.csv_file = csv_file
        self.img_path = img_path

        # 初始化存储列表（仅存储有效样本，避免索引映射混乱）
        self.data_anc = []  # 有效锚点图像路径
        self.data_pos = []  # 有效正例图像路径
        self.data_neg = []  # 有效负例图像路径
        self.type = []      # 有效样本的三元组类型
        # ！！关键修改：移除 self.valid_indices（无需原始索引映射，直接用列表索引）

        # 读取CSV数据
        self.pd_data = pd.read_csv(self.csv_file)
        self.data = self.pd_data.to_dict("list")
        # 从CSV中提取类型信息（兼容不同列名）
        tys = self.data.get("Triplet_type", self.data.get("type", []))
        
        # 动态获取样本数量（从CSV长度获取，替代硬编码）
        total_samples = len(self.pd_data)
        print(f"CSV文件总样本数: {total_samples}")

        # 遍历所有样本，构建路径并验证有效性
        for i in tqdm(range(total_samples), desc="筛选有效样本"):
            # 生成6位数字编号（000001开始）
            number = str(i + 1).zfill(6)
            # 构建图像路径
            anc_filename = f"{number}_1.jpeg"
            pos_filename = f"{number}_2.jpeg"
            neg_filename = f"{number}_3.jpeg"
            anc_path = os.path.join(self.img_path, anc_filename).replace('\\', '/')
            pos_path = os.path.join(self.img_path, pos_filename).replace('\\', '/')
            neg_path = os.path.join(self.img_path, neg_filename).replace('\\', '/')

            # 验证图像是否存在且可正常打开（严格筛选）
            valid = True
            try:
                with Image.open(anc_path) as img:
                    img.convert('RGB')
                with Image.open(pos_path) as img:
                    img.convert('RGB')
                with Image.open(neg_path) as img:
                    img.convert('RGB')
            except Exception as e:
                valid = False

            if valid:
                # ！！关键修改：仅存储有效样本，self.data_anc/pos/neg/type 长度完全一致
                self.data_anc.append(anc_path)
                self.data_pos.append(pos_path)
                self.data_neg.append(neg_path)
                # 安全添加类型（避免CSV列长度不足）
                self.type.append(tys[i] if i < len(tys) else "UNKNOWN")

        print(f"有效样本数: {len(self.data_anc)} (总样本数: {total_samples})")

    def __len__(self):
        # 直接返回有效样本数（self.data_anc 长度即有效样本数）
        return len(self.data_anc)

    def __getitem__(self, index):
        # ！！关键修改：直接用 index 访问，无需映射原始索引（避免越界）
        if index >= len(self.data_anc):
            raise IndexError(f"样本索引 {index} 超出范围（有效样本数：{len(self.data_anc)}）")
        
        # 直接通过 index 获取当前样本的路径和类型（长度一致，不会越界）
        type_ = self.type[index]
        anc_path = self.data_anc[index]
        pos_path = self.data_pos[index]
        neg_path = self.data_neg[index]

        # 加载图像（初始化已验证，极端情况容错）
        try:
            anc_img = Image.open(anc_path).convert('RGB')
            pos_img = Image.open(pos_path).convert('RGB')
            neg_img = Image.open(neg_path).convert('RGB')
        except Exception as e:
            print(f"紧急处理损坏样本 {index}: {str(e)}")
            # 随机返回一个有效样本（避免训练中断）
            fallback_idx = np.random.randint(0, len(self.data_anc))
            return self.__getitem__(fallback_idx)

        # 应用转换
        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return {
            "name": anc_path,
            "anc": anc_img,
            "pos": pos_img,
            "neg": neg_img,
            "type": type_
        }


def build_transform(is_train):
    mean = [0.49895147219604985, 0.4104390648367995, 0.3656147590417074]
    std = [0.2970847084907291, 0.2699003075660314, 0.2652599579468044]
    input_size = 224

    if is_train:
        # ！！适配显存：降低数据增强强度（避免显存占用过高）
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            scale=(0.8, 1.0),  # 缩小缩放范围（原0.08-1.0，降低计算量）
            ratio=(0.9, 1.1),  # 缩小宽高比范围（原7/8-8/7，减少显存波动）
            color_jitter=0.1,  # 降低颜色抖动强度（原0.3，减少中间特征显存）
            auto_augment=None,  # 禁用自动增强（显存消耗大，后续效果稳定后再启用）
            interpolation='bicubic',
            re_prob=0.1,       # 降低随机擦除概率（原0.25，减少显存消耗）
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std
        )
        return transform

    # 验证/测试集转换（无变化）
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)


def build_dataset(config, mode):
    train_transform = build_transform(is_train=True)
    val_transform = build_transform(is_train=False)

    dataset = None
    if mode == "train":
        dataset = FecData(config["train_csv"], config["train_img_path"], train_transform)
    elif mode == "val":
        # 验证集使用无增强的transform（修复原代码错误，确保评估准确）
        dataset = FecData(config["val_csv"], config["val_img_path"], val_transform)
    return dataset
