# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# 导入argparse模块，用于解析解析命令行参数解析
import argparse
# 导入datetime模块，用于处理日期时间
import datetime
# 导入json模块，用于JSON数据的读写
import json
# 导入numpy库，用于数值计算
import numpy as np
# 导入os模块，用于操作系统相关功能（如路径处理）
import os
# 导入time模块，用于时间相关操作
import time
# 从pathlib导入Path，用于更便捷的路径处理
from pathlib import Path

# 导入PyTorch库
import torch
# 导入PyTorch的CUDA后端相关功能，用于CUDA加速配置
import torch.backends.cudnn as cudnn
# 从torch.utils.tensorboard导入SummaryWriter，用于TensorBoard日志记录
from torch.utils.tensorboard import SummaryWriter
# 导入torchvision的transforms模块，用于图像预处理
import torchvision.transforms as transforms
# 导入torchvision的datasets模块，用于加载数据集
import torchvision.datasets as datasets

# 导入timm库（PyTorch Image Models），用于加载预训练模型
import timm

# 断言timm版本为0.3.2，确保版本兼容性
# assert timm.__version__ == "0.3.2"  # version check                                                     报错 注释 2025.8.16
# 从timm.optim.optim_factory导入优化器相关工具
import timm.optim.optim_factory as optim_factory

# 导入自定义的工具模块misc
import util.misc as misc
# 从util.misc导入NativeScalerWithGradNormCount并命名为NativeScaler，用于梯度缩放和梯度范数计算
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# 导入自定义的mae模型模块
import models_mae

# 从engine_pretrain导入train_one_epoch函数，用于执行单轮训练
from engine_pretrain import train_one_epoch

# 预训练任务的命令行参数解析器，用于定义和管理训练过程中需要的各种参数
def get_args_parser():
    # 创建命令行参数解析器，用于解析MAE预训练的相关参数，add_help=False表示不添加默认的帮助选项
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # 训练批次相关参数
    # 每个GPU的批次大小（实际批次大小 = batch_size * accum_iter * GPU数量）
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # 训练的总轮次
    parser.add_argument('--epochs', default=400, type=int)
    # 梯度累积的迭代次数（在内存有限时用于增大有效批次大小）
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    # 模型相关参数
    # 要训练的模型名称（如mae_vit_large_patch16）
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # 输入图像的尺寸
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    # 掩码比例（被移除的图像块的百分比）
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    # 使用（每个图像块的）归一化像素作为计算损失的目标
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    # 优化器相关参数
    # 权重衰减系数（默认：0.05）
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # 学习率（绝对学习率）
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    # 基础学习率：绝对学习率 = 基础学习率 * 总批次大小 / 256
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # 循环学习率调度器的最低学习率下限（达到0时）
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    # 学习率预热的轮次
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    # # 数据集相关参数
    # 数据集的路径
    parser.add_argument('--data_path', default='/root/autodl-tmp/FreeAvatar-1/free_avatar/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')                                              # 修改 原/datasets01/imagenet_full_size/061417/ 2025.8.16
    # 保存模型和结果的路径，为空则不保存
    parser.add_argument('--output_dir', default='/root/autodl-tmp/FreeAvatar-1/free_avatar/mae-main/output_dir',
                        help='path where to save, empty for no saving')                   # 修改 原./output_dir 2025.8.16
    # TensorBoard日志的保存路径
    parser.add_argument('--log_dir', default='/root/autodl-tmp/FreeAvatar-1/free_avatar/mae-main/output_dir',                       
                        help='path where to tensorboard log')                             # 修改 原./output_dir 2025.8.16
    # 用于训练/测试的设备（如cuda或cpu）
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # 随机种子（用于保证实验可重复性）
    parser.add_argument('--seed', default=0, type=int)
    # 从检查点恢复训练的路径
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    # 开始训练的轮次（用于断点续训）
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # 数据加载器的工作进程数量
    parser.add_argument('--num_workers', default=10, type=int)
    # 在DataLoader中锁定CPU内存，以更高效地（有时）传输到GPU
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True) # 默认启用内存锁定

    # distributed training parameters
    # 分布式训练相关参数
    # 分布式进程的数量
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # 本地进程的排名（用于分布式训练）
    parser.add_argument('--local_rank', default=-1, type=int)
    # 是否在ITP上进行分布式训练
    parser.add_argument('--dist_on_itp', action='store_true')
    # 用于设置分布式训练的URL
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser # 返回解析器对象

# MAE（掩码自编码器）预训练的主函数
def main(args):
    # 初始化分布式训练模式（设置进程组、排名等，支持多GPU训练）
    misc.init_distributed_mode(args)

    # 打印当前工作目录（脚本所在的文件夹路径）
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # 打印所有参数（格式化显示，每个参数一行）
    print("{}".format(args).replace(', ', ',\n'))

    # 定义训练使用的设备（GPU或CPU）
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # 固定随机种子，保证实验可重复性（不同进程使用不同种子，避免同步偏差）
    seed = args.seed + misc.get_rank()  # 结合全局排名生成种子，确保多进程种子不同
    torch.manual_seed(seed)  # 设置PyTorch随机种子
    np.random.seed(seed)     # 设置NumPy随机种子

    # 启用CuDNN的基准模式（自动寻找最佳卷积算法，加速训练）
    cudnn.benchmark = True

    # simple augmentation
    # 定义训练集的数据增强管道（简单的数据预处理和增强）
    transform_train = transforms.Compose([
            # 随机裁剪并调整为输入尺寸，缩放范围0.2-1.0，使用双三次插值
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
            transforms.ToTensor(),  # 转换为PyTorch张量
            # 标准化（使用ImageNet的均值和标准差）
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # 加载训练数据集（使用ImageFolder，假设数据按类别分文件夹存放）
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)  # 打印数据集信息（如样本数量、类别数）

    # 配置分布式采样器（多GPU训练时，将数据分配到不同进程）
    if True:  # args.distributed: # 实际应为判断是否分布式训练，这里简化为True
        num_tasks = misc.get_world_size()  # 获取总进程数
        global_rank = misc.get_rank()      # 获取当前进程的全局排名
        # 分布式采样器：将数据集分割到不同进程，确保每个进程处理不同样本
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        # 非分布式训练时使用随机采样器
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # 初始化TensorBoard日志（仅主进程创建，避免重复写入）
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)  # 创建日志目录（若不存在）
        log_writer = SummaryWriter(log_dir=args.log_dir)  # 实例化日志写入器
    else:
        log_writer = None  # 非主进程不写日志

    # 创建训练数据加载器（批量加载数据）
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,  # 使用上面定义的采样器
        batch_size=args.batch_size,            # 每个进程的批次大小
        num_workers=args.num_workers,          # 数据加载的工作进程数
        pin_memory=args.pin_mem,               # 是否锁定内存（加速GPU传输）
        drop_last=True,                        # 丢弃最后一个不完整的批次
    )
    
    # define the model
    # 定义MAE模型（从models_mae中加载指定名称的模型）
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    # 将模型移动到指定设备（GPU/CPU）
    model.to(device)

    # 保存未包装的模型引用（分布式训练时用于访问原始模型参数）
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp)) # 打印模型结构

    # 计算有效批次大小（考虑梯度累积和多GPU）
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    # 计算学习率（若未指定lr，则根据基础学习率和有效批次大小计算）
    if args.lr is None:  # only base_lr is specified # 仅指定了基础学习率blr
        args.lr = args.blr * eff_batch_size / 256 # 按256的基准批次大小缩放

    # 打印学习率信息
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))  # 基准学习率
    print("actual lr: %.2e" % args.lr)  # 实际使用的学习率

    # 打印训练配置信息
    print("accumulate grad iterations: %d" % args.accum_iter)  # 梯度累积次数
    print("effective batch size: %d" % eff_batch_size)         # 有效批次大小

    # 配置分布式训练（多GPU数据并行）
    if args.distributed:
        # 将模型包装为分布式数据并行（DDP），实现多GPU协同训练
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module  # 获取原始模型（去除DDP包装）
    
    # following timm: set wd as 0 for bias and norm layers
    # 配置优化器的参数组（对偏置和归一化层不应用权重衰减）

    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)                    #报错 修改注释 2025.8.16
    # 手动实现参数分组（区分需要权重衰减的参数和不需要的参数）
    param_groups = [                                                                                         #报错 修改添加 2025.8.16
        {'params': [], 'weight_decay': args.weight_decay},  # 应用权重衰减的参数
        {'params': [], 'weight_decay': 0.0}  # 不应用权重衰减的参数（偏置、归一化层等）
    ]
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue  # 跳过不需要梯度的参数
        # 偏置参数和归一化层参数不应用权重衰减
        if name.endswith('.bias') or 'norm' in name:
            param_groups[1]['params'].append(param)
        else:
            param_groups[0]['params'].append(param)

    # 初始化AdamW优化器
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)  # 打印优化器配置
    # 初始化梯度缩放器（用于混合精度训练，防止梯度下溢）
    loss_scaler = NativeScaler()

    # 加载预训练模型或断点续训（从检查点恢复模型、优化器、缩放器状态）
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # 开始训练循环
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()  # 记录训练开始时间
    # 遍历每个训练轮次（从start_epoch到epochs-1）
    for epoch in range(args.start_epoch, args.epochs):
        # 分布式训练时，更新采样器的epoch（确保每个epoch数据打乱方式不同）
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # 训练一个轮次，返回训练统计信息（损失、学习率等）
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        # 定期保存模型（每20个epoch或最后一个epoch）
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # 整理本轮的日志信息（训练指标+轮次）
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        # 主进程写入日志到文件
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()  # 刷新TensorBoard日志
            # 写入文本日志（JSON格式）
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # 计算总训练时间并打印
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

# 解析用户输入的参数、准备输出环境，最终调用主函数启动训练
if __name__ == '__main__':
    # 当脚本被直接运行时执行以下逻辑（而非被导入为模块时）
    # 获取命令行参数解析器（由get_args_parser()函数定义的参数配置）
    args = get_args_parser()
    # 解析命令行输入的参数，得到参数对象（包含所有配置的参数值）
    args = args.parse_args()
    # 若指定了输出目录（output_dir），则创建该目录（包括父目录，已存在则不报错）
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # 调用主函数main()，传入解析后的参数对象，启动MAE预训练流程
    main(args)
