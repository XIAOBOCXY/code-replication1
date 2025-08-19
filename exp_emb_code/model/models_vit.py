# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
# 定义了一个改进版的Vision Transformer（视觉 Transformer，简称 ViT） 模型架构，
# 并提供了多种预设配置（如不同尺寸的模型），用于图像分类等计算机视觉任务。
# 代码基于timm库（PyTorch 图像模型库）的 ViT 实现进行扩展，
# 核心是支持全局平均池化（global average pooling），并简化了不同规格 ViT 模型的创建流程。
from functools import partial

import torch

import timm.models.vision_transformer
#import model.vision_transformer

# 定义一个继承自timm库VisionTransformer的子类，扩展了全局平均池化功能
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    # 初始化方法，新增global_pool参数控制是否使用全局平均池化
    def __init__(self, global_pool=False, **kwargs):
        # 调用父类（timm的VisionTransformer）的初始化方法，传入所有其他参数
        super(VisionTransformer, self).__init__(** kwargs)
        # 标记模型是否为预训练（此处固定为True，可能用于后续逻辑）
        self.pretrained = True
        # 存储是否使用全局平均池化的标志
        self.global_pool = global_pool
        # 如果启用全局平均池化
        if self.global_pool:
            # 从传入的参数中获取归一化层类型（如LayerNorm）
            norm_layer = kwargs['norm_layer']
            # 获取嵌入维度（特征维度）
            embed_dim = kwargs['embed_dim']
            # 创建用于全局池化后归一化的层
            self.fc_norm = norm_layer(embed_dim)

            # 删除父类中原有的归一化层（因为全局池化用fc_norm替代）
            del self.norm  # remove the original norm

    # 特征提取的前向传播逻辑（处理输入并生成特征）
    def forward_features(self, x):
        # 获取输入批次大小（B为batch size）
        B = x.shape[0]
        # 将输入图像通过补丁嵌入层（patch_embed）转换为补丁向量序列
        x = self.patch_embed(x)

        # 扩展类别令牌（cls_token）到批次维度（形状从[1,1,D]变为[B,1,D]）
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 借鉴了Phil Wang的实现，感谢
        # 将类别令牌与补丁向量序列拼接（在序列维度，即dim=1）
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码（pos_embed）
        x = x + self.pos_embed
        # 应用位置dropout（防止过拟合）
        x = self.pos_drop(x)

        # 依次通过所有Transformer块（blocks）进行特征提取
        for blk in self.blocks:
            x = blk(x)

        # 如果启用全局平均池化
        if self.global_pool:
            # 移除类别令牌，只保留补丁特征（形状从[N, L+1, D]变为[N, L, D]，L为补丁数量）
            x = x[:, 1:, :]  # without cls token (N, L=14*14, D=768=16*16*3)
            # 对补丁序列做全局平均池化（在序列维度取均值，得到[N, D]的特征）
            x = x.mean(dim=1)  # global average pooling (N, D=768)
            # 通过fc_norm进行归一化，得到最终特征
            outcome = self.fc_norm(x)  # Layer Normalization (N, D=768)
        else:
            # 不启用全局池化时，使用父类的归一化层
            x = self.norm(x)
            # 取类别令牌对应的特征作为输出（即序列的第0个元素）
            outcome = x[:, 0]
        # 返回提取到的特征
        return outcome

    # borrow from timm
    # 完整的前向传播方法（调用特征提取并通过分类头输出结果），借鉴自timm库
    def forward(self, x, ret_feature=False):
        # 先调用forward_features获取特征
        x = self.forward_features(x)
        # 保存特征（用于后续返回）
        feature = x
        # 检查是否有分布式分类头（head_dist，用于某些训练策略）
        if getattr(self, 'head_dist', None) is not None:
            # 通过主分类头和分布式分类头分别计算输出
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple # x必须是元组格式
            # 如果是训练模式且非脚本模式
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                # 训练时返回两个分类头的输出
                return x, x_dist
            else:
                # 推理时返回两个输出的平均值
                return (x + x_dist) / 2
        else:
            # 普通分类头：将特征通过分类头（head）映射到类别空间
            x = self.head(x)
        # return
        # 根据ret_feature参数决定返回内容
        if ret_feature:
            # 返回分类结果和原始特征
            return x, feature
        else:
            # 只返回分类结果
            return x


# setup model archs
# 定义Vision Transformer的基础参数配置（所有模型共享的通用参数）
VIT_KWARGS_BASE = dict(mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
# MLP层的扩展比例（中间层维度是输入的4倍）、在Q、K、V投影层中使用偏置项、归一化层配置：LayerNorm，精度为1e-6

# 定义不同规格ViT模型的预设参数（按模型规模从小到大排列）
VIT_KWARGS_PRESETS = {
    'micro': dict(patch_size=16, embed_dim=96, depth=12, num_heads=2), # 微型模型、图像补丁大小为16x16像素、补丁嵌入后的特征维度为96、Transformer块的数量为12（网络深度）、多头注意力的头数为2
    'mini': dict(patch_size=16, embed_dim=128, depth=12, num_heads=2), # 小型模型、特征维度128（比micro大）
    'tiny_d6': dict(patch_size=16, embed_dim=192, depth=6, num_heads=3), # tiny模型（深度6）、深度仅6（比同规格的tiny浅）
    'tiny': dict(patch_size=16, embed_dim=192, depth=12, num_heads=3), # tiny模型（标准深度）、深度12（比tiny_d6深）
    'small': dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),# 小型模型、特征维度384、注意力头数6（比tiny多）
    'base': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),# 基础模型、特征维度768、注意力头数12
    'large': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),# 大型模型、特征维度1024、深度24（比base深一倍）、注意力头数16
    'huge': dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),# 巨型模型、补丁大小14x14（比之前的16小，更精细）、特征维度1280、深度32
    'giant': dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11),# 超大型模型、特征维度1408、深度40、自定义MLP比例（覆盖基础参数的4）
    'gigantic': dict(patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=64/13),# 超大巨型模型、特征维度1664、深度48（最深）、自定义MLP比例
}

# 创建ViT模型的通用函数，整合参数并实例化模型
def create_vit_model(preset=None, creator=None, **kwargs):
    # 如果未指定预设，默认使用'base'规格
    preset = 'base' if preset is None else preset.lower()
    # 初始化参数字典
    all_kwargs = dict()
    # 合并基础参数（VIT_KWARGS_BASE）
    all_kwargs.update(VIT_KWARGS_BASE)
    # 合并指定预设的参数（如'base'的参数）
    all_kwargs.update(VIT_KWARGS_PRESETS[preset])
    # 合并用户传入的自定义参数（优先级最高，可覆盖上述参数）
    all_kwargs.update(kwargs)
    # 如果未指定模型创建类，默认使用自定义的VisionTransformer类
    if creator is None:
        creator = VisionTransformer
    # 用整合后的参数实例化模型并返回
    return creator(** all_kwargs)

# 以下通过partial函数创建便捷的模型构造函数，固定预设参数
# 每个函数对应一种规格的ViT模型，直接调用即可创建对应模型
vit_micro_patch16 = partial(create_vit_model, preset='micro')  # 微型模型（16x16补丁）
vit_mini_patch16 = partial(create_vit_model, preset='mini')    # 小型模型（16x16补丁）
vit_tiny_d6_patch16 = partial(create_vit_model, preset='tiny_d6')  # tiny模型（深度6，16x16补丁）
vit_tiny_patch16 = partial(create_vit_model, preset='tiny')    # tiny模型（标准深度，16x16补丁）
vit_small_patch16 = partial(create_vit_model, preset='small')  # 小型模型（16x16补丁）
vit_base_patch16 = partial(create_vit_model, preset='base')    # 基础模型（16x16补丁）
vit_large_patch16 = partial(create_vit_model, preset='large')  # 大型模型（16x16补丁）
vit_huge_patch14 = partial(create_vit_model, preset='huge')    # 巨型模型（14x14补丁）
vit_giant_patch14 = partial(create_vit_model, preset='giant')  # 超大型模型（14x14补丁）
vit_gigantic_patch14 = partial(create_vit_model, preset='gigantic')  # 超大巨型模型（14x14补丁）



