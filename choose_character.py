import numpy as np

# 根据输入的角色名称，返回该角色的专属配置信息（用于多角色训练/推理）
def character_choice(character):
    # 参数：character - 角色名称（字符串），如'l36_233'、'l36_234'等
    print(f'=> Choose character: {character}') # 打印当前选择的角色，便于调试

    # 若角色为'l36_233'，返回其专属配置
    if character.lower() == 'l36_233':
        img_postfix = '.jpg'  # 该角色图像文件的后缀（.jpg格式）
        n_rig = 61  # 该角色的rig参数维度（61个控制参数）
        data_path = '/project/qiuf/DJ01/L36/images'  # 该角色训练数据的存放路径
        # 计算嘴部和眼部区域的裁剪坐标（从512x512图像映射到256x256）
        # 原始坐标（512x512）→ 缩放至256x256：乘以256/512
        mouth_left, mouth_right, mouth_top, mouth_bottom = map(int, np.array([150, 350, 350, 450]) * 256 / 512.)
        eye_left, eye_right, eye_top, eye_bottom = map(int, np.array([106, 404, 161, 266]) * 256 / 512.)
        ckpt_img2rig = None  # 该角色的"图像到rig"模型权重路径（未提供，可能无需使用）
        # 该角色的"rig到图像"模型预训练权重路径（用于将rig参数生成图像）
        ckpt_rig2img = '/project/qiuf/expr-capture/ckpt/rig2img_20240211-045412.pt'
    # 若角色为'l36_234'，返回其专属配置
    elif character.lower() == 'l36_234':
        img_postfix = '.jpg'  # 图像文件后缀为.jpg
        n_rig = 61  # rig参数维度为61
        data_path = '/project/qiuf/DJ01/L36_234/images'  # 数据存放路径
        # 嘴部和眼部区域坐标（512x512→256x256）
        mouth_left, mouth_right, mouth_top, mouth_bottom = map(int, np.array([178, 360, 363, 451]) * 256 / 512.)
        eye_left, eye_right, eye_top, eye_bottom = map(int, np.array([119, 415, 136, 250]) * 256 / 512.)
        ckpt_img2rig = None  # 无"图像到rig"模型权重
        # "rig到图像"模型权重路径
        ckpt_rig2img = '/project/qiuf/expr-capture/ckpt/rig2img_20240324-184156.pt'
    # 若角色为'l36_230_61'，返回其专属配置
    elif character.lower() == 'l36_230_61':
        img_postfix = '.png'  # 图像文件后缀为.png（与前两个角色不同）
        n_rig = 67  # rig参数维度为67（与前两个角色不同，需单独适配）
        data_path = '/project/qiuf/DJ01/L36_230_61/images'  # 数据存放路径
        # 嘴部和眼部区域坐标（512x512→256x256）
        mouth_left, mouth_right, mouth_top, mouth_bottom = map(int, np.array([194, 295, 312, 367]) * 256 / 512.)
        eye_left, eye_right, eye_top, eye_bottom = map(int, np.array([151, 337, 185, 261]) * 256 / 512.)
        # 该角色的"图像到rig"模型权重路径（存在预训练模型）
        ckpt_img2rig = '/project/qiuf/expr-capture/ckpt/img2rig_20240522-153804.pt'
        # "rig到图像"模型权重路径
        ckpt_rig2img = '/project/qiuf/expr-capture/ckpt/rig2img_20240425-180631.pt'
    # 若角色名称不在上述列表中，抛出未实现错误
    else:
        raise NotImplementedError

    # 生成嘴部和眼部区域的掩码（用于后续损失计算时聚焦关键区域）
    # 初始化3通道（RGB）256x256的掩码，初始值为0
    mouth_crop = np.zeros((3, 256, 256))
    # 将嘴部区域设为1（掩码值1表示该区域在损失计算中需要被关注）
    mouth_crop[:, mouth_top:mouth_bottom, mouth_left:mouth_right] = 1
    # 将眼部区域设为1（同样作为关键区域）
    mouth_crop[:, eye_top:eye_bottom, eye_left:eye_right] = 1
    # 返回该角色的完整配置字典
    return {
        'data_path': data_path,  # 数据存放路径
        'mouth_crop': mouth_crop,  # 嘴部和眼部的掩码（用于损失计算）
        'n_rig': n_rig,  # rig参数维度
        'ckpt_img2rig': ckpt_img2rig,  # "图像到rig"模型权重路径
        'ckpt_rig2img': ckpt_rig2img,  # "rig到图像"模型权重路径
        'img_postfix': img_postfix  # 图像文件后缀
    }
    