import cv2
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.common import *
from models.DCGAN import Generator
from models.gan_loss import GANLoss

# 训练生成对抗网络（GAN）中的生成器（model）和判别器（model_D）
def Train(epoch, loader, model, model_D):
    # 参数说明：
    #   epoch：当前训练轮次
    #   loader：训练数据加载器
    #   model：生成器模型（此处用于从图像嵌入生成rig参数）
    #   model_D：判别器模型（用于区分生成图像与真实图像）
    # 获取生成器优化器当前的学习率
    lr = optimizer.param_groups[0]['lr']
    # 打印当前训练轮次、学习率和时间戳（用于跟踪训练过程）
    print(f"*** Epoch {epoch}, lr:{lr:.5f}, timestamp:{timestamp}")
    # 初始化生成器总损失和判别器总损失计数器
    loss_sum = 0.0
    loss_sum_D = 0.
    # 将判别器设置为训练模式
    model_D = model_D.train()
    # 确定每轮训练的最大步数（取配置的最大步数或数据加载器长度的较小值）
    train_step = min(args.train_step_per_epoch, len(loader))
    # 定义进度条格式
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    # 创建进度条（遍历数据加载器，显示训练进度）
    pbar = tqdm(enumerate(loader), bar_format=b, total=train_step)
    # 用于存储训练日志
    logger = []
    # 记录训练开始时间
    time0 = time.time()
    # 遍历数据加载器中的每个批次（带索引i）
    for i, data in pbar:
        # 若达到设定的最大训练步数，提前结束本轮训练
        if i > train_step:
            break
        # 清零生成器优化器的梯度（避免上一轮梯度累积）
        optimizer.zero_grad()
        # 将生成器设置为训练模式
        model.train()
        # 将判别器设置为评估模式（更新生成器时固定判别器参数）
        model_D.eval()
        # 初始化损失字典（存储生成器的各类损失）
        loss = dict()

        # 从数据中提取各类输入，转移到GPU并转换为float类型
        sources = data['img'].cuda().float()  # 源图像（如真实人脸图像）
        targets = data['target'].cuda().float()  # 目标图像（如虚拟角色渲染图像）
        target_rigs = data['rigs'].cuda().float()  # 目标rig参数（作为生成器的监督信号）
        is_render = data['is_render'].cuda().float()  # 标记是否为渲染图像（1=渲染图，0=真实图）
        ldmk = data['ldmk'].cuda().float()  # 人脸关键点（可能用于辅助损失计算，此处未直接使用）
        role_id = data['role_id'].cuda().long()  # 角色ID（区分不同虚拟角色）
        do_pixel = data['do_pixel'].cuda().int()  # 标记是否计算像素级损失（1=需要计算）
        bs_input = data['bs'].cuda().float()  # 可能是blendshape参数（此处未直接使用）
        has_rig = data['has_rig'].cuda().float()  # 标记是否有对应的rig参数（1=有）
        
        
        # 提取各类索引（用于筛选特定数据计算损失）
        real_idx = ((is_render == 0).nonzero(as_tuple=True)[0])  # 真实图像的索引
        render_idx = ((is_render == 1).nonzero(as_tuple=True)[0])  # 渲染图像的索引
        has_rig = ((has_rig == 1).nonzero(as_tuple=True)[0])  # 有rig参数的样本索引
        do_pixel_idx = ((do_pixel == 1).nonzero(as_tuple=True)[0])  # 需要计算像素损失的样本索引
        
        # 按角色ID分组的索引（区分不同虚拟角色的样本）
        role_idxes = [((role_id == CHARACTER_NAMES.index(name_e)).nonzero(as_tuple=True)[0]) for name_e in characters]
        
        # source image to emb
        # 源图像通过嵌入模型（model_emb）生成嵌入向量（用于后续生成rig参数）
        with torch.no_grad():  # 嵌入模型参数不更新，禁用梯度计算
            emb_hidden_in, emb_in = model_emb(resize(sources))  # 源图像的隐藏嵌入和输出嵌入
            # 若启用对称损失，通过对称嵌入模型（model_symm）处理源图像
            if args.weight_symm:
                emb_hidden_symm_in, emb_symm_in = model_symm(resize_symm(sources))
                # 拼接原始嵌入和对称嵌入（增强特征表达）
                emb_hidden_in = torch.cat((emb_hidden_in, emb_hidden_symm_in), dim=1)
            
        # 生成器根据源图像的嵌入向量和角色ID，输出预测的rig参数
        output_rig = model(emb_hidden_in, role_id)
        
        # 用每个角色对应的rig2img模型，将生成的rig参数转换为目标图像（虚拟角色渲染图）
        with torch.no_grad(): # rig2img模型参数不更新，禁用梯度计算
            output_imgs_c = [] # 存储每个角色的生成图像
            for c_i, cname in enumerate(characters):
                # 取出当前角色的rig参数，调整形状后输入rig2img模型生成图像
                output_imgs_c.append(configs_characters[cname]["model_rig2img"](output_rig[role_idxes[c_i]][:,:configs_characters[cname]['n_rig']].reshape(-1, configs_characters[cname]['n_rig'], 1, 1)))
            
            # 拼接不同角色的生成图像，形成完整的输出图像张量
            C, H, W = output_imgs_c[0].shape[1:]  # 获取图像通道数和尺寸
            B = sources.shape[0]  # 获取批次大小
            output_img = torch.empty((B, C, H, W), dtype=torch.float32).cuda()  # 初始化输出图像张量
            for c_i, cname in enumerate(characters):
                output_img[role_idxes[c_i]] = output_imgs_c[c_i]  # 按角色索引填充生成图像
            
        # 计算rig参数损失（L2损失）：若启用权重且存在有效样本
        if args.weight_rig:
            if len(do_pixel_idx) > 0:
                loss['rig'] = criterion_l2(output_rig[has_rig], target_rigs[has_rig])
        
        # 计算嵌入一致性损失（L2损失）：生成图像的嵌入应与源图像的嵌入一致
        if args.weight_emb:
            emb_hidden_out, emb_out = model_emb(resize(output_img))  # 生成图像的嵌入
            loss['emb'] = criterion_l2(emb_out, emb_in)  # 与源图像嵌入对比
        
        # 计算像素级损失（L1损失）：生成图像与目标图像的像素差异
        if args.weight_img:
            if len(do_pixel_idx) > 0:  # 仅对标记需要计算的样本
                loss['image'] = criterion_l1(output_img[do_pixel_idx], targets[do_pixel_idx]) * args.weight_img
        
        # 计算生成器的对抗损失：希望判别器将生成图像判定为真实
        if args.weight_D:
            output_D_G = model_D(output_img)  # 判别器对生成图像的输出
            loss['G_D'] = criterion_gan(output_D_G, True, False)[0] * args.weight_D  # 对抗损失（标签为真）
        
        # 计算对称损失：生成图像的对称嵌入应与源图像的对称嵌入一致
        if args.weight_symm:
            with torch.no_grad():  # 对称嵌入模型参数不更新
                emb_hidden_symm_out, emb_symm_out = model_symm(resize_symm(output_img))  # 生成图像的对称嵌入
            loss['symm'] = criterion_l2(emb_symm_out, emb_symm_in) * args.weight_symm  # 与源图像对称嵌入对比
        # 若没有损失项，跳过当前批次
        if not loss:
            continue

        
        # 计算生成器的总损失（所有损失项之和）
        loss_value = sum([v for k, v in loss.items()])
        
        # 累加生成器总损失（用于计算平均损失）
        loss_sum += loss_value.item()
        # 生成器损失反向传播（retain_graph=True保留计算图，供后续可能的梯度计算）
        loss_value.backward(retain_graph=True)
        # 优化器更新生成器参数
        optimizer.step()
        # 生成器学习率调度器更新
        scheduler.step()
        
        # discriminator
        # 训练判别器
        # 清零判别器优化器的梯度
        optimizer_D.zero_grad()
        # 将判别器设置为训练模式
        model_D.train()
        # 将生成器设置为评估模式（更新判别器时固定生成器参数）
        model.eval()
        # 判别器对生成图像的判断（detach()切断与生成器的梯度连接，避免更新生成器）
        outputs_fake = model_D(output_img.detach())
        # 判别器对真实目标图像的判断（仅对需要计算像素损失的样本）
        outputs_real = model_D(targets[do_pixel_idx])
        # 计算判别器对假样本的损失（标签为假）
        loss_fake = criterion_gan(outputs_fake, False, True)[0]
        # 计算判别器对真样本的损失（标签为真）
        loss_real = criterion_gan(outputs_real, True, True)[0]
        # 判别器总损失（真假样本损失的平均值）
        loss_D_train = (loss_fake + loss_real) / 2.
        # 判别器损失反向传播
        loss_D_train.backward()
        # 优化器更新判别器参数
        optimizer_D.step()
        # 判别器学习率调度器更新
        scheduler_D.step()

        # 将生成器的各类损失写入TensorBoard（用于可视化）
        writer.add_scalars(f'train/loss_G', loss, epoch * train_step + i)
        # 将生成器总损失写入TensorBoard
        writer.add_scalar(f'train/loss_G_total', loss_value.item(), epoch * train_step + i)
        # 将判别器损失写入TensorBoard
        writer.add_scalar(f'train/loss_D', loss_D_train.item(), epoch * train_step + i)
        
        # 格式化生成器损失字符串（保留4位小数）
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        # 构建训练日志信息（包含轮次、时间戳、学习率、步数、生成器和判别器损失）
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}, loss_D: {loss_D_train.item():.04f}"
        # 将日志添加到日志列表
        logger.append(_log + '\n')
        # 更新进度条描述（显示当前日志）
        pbar.set_description(_log)
        
    # 将源图像、目标图像和生成图像拼接后，写入TensorBoard（每4个样本取1个，用于可视化效果）
    writer.add_images(f'train/img', torch.cat([sources, targets, output_img], dim=-2)[::4], epoch * train_step + i)
    # 计算本轮训练生成器的平均损失
    avg_loss = loss_sum / train_step
    # 构建本轮训练的总结日志
    _log = "==> [Train] Epoch {} ({}), training loss={}".format(epoch, timestamp, avg_loss)
    # 打印总结日志
    print(_log)
    # 将本轮所有训练日志写入日志文件
    with open(os.path.join(log_save_path, f'emb2render_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    # 若当前轮次是设定的保存轮次，保存生成器和判别器的模型权重（文件名添加轮次标识）
    if epoch % args.save_step == 0:
        torch.save({'state_dict': model.state_dict()}, model_path.replace('.pt', f'_{epoch}.pt'))
        torch.save({'state_dict': model_D.state_dict()}, model_path.replace('.pt', f'_{epoch}_D.pt'))
    # 返回本轮训练生成器的平均损失
    return avg_loss

# 在验证阶段评估生成对抗网络（GAN）中生成器和判别器的性能
# 核心目的：通过验证集数据计算模型损失，判断是否为当前最优模型，辅助早停机制
def Eval(epoch, loader, model, model_D, best_score):
    # 参数说明：
    #   epoch：当前验证对应的训练轮次
    #   loader：验证数据加载器
    #   model：生成器模型（需评估的生成器）
    #   model_D：判别器模型（需评估的判别器）
    #   best_score：当前记录的最优验证损失（用于比较是否更新最优模型）
    # 初始化生成器总损失计数器
    loss_sum = 0.0
    # 将生成器和判别器均设置为评估模式（关闭dropout、固定归一化层参数）
    model = model.eval()
    model_D = model_D.eval()

    # 确定每轮验证的最大步数（取配置的最大步数或数据加载器长度的较小值）
    eval_step = min(args.eval_step_per_epoch, len(loader))
    # 定义进度条格式
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    # 创建进度条（遍历验证数据加载器，显示验证进度）
    pbar = tqdm(enumerate(loader), bar_format=b, total=eval_step)
    # 用于存储验证日志
    logger = []
    # 记录验证开始时间
    time0 = time.time()
    # 遍历验证数据加载器中的每个批次（带索引i）
    for i, data in pbar:
        # 若达到设定的最大验证步数，提前结束本轮验证
        if i > eval_step:
            break
        # 初始化损失字典（存储生成器的各类验证损失）
        loss = dict()

        # 从数据中提取各类输入，转移到GPU并转换为float类型
        sources = data['img'].cuda().float()  # 源图像（如真实人脸图像）
        targets = data['target'].cuda().float()  # 目标图像（如虚拟角色渲染图像）
        target_rigs = data['rigs'].cuda().float()  # 目标rig参数（作为评估基准）
        is_render = data['is_render'].cuda().float()  # 标记是否为渲染图像（1=渲染图，0=真实图）
        ldmk = data['ldmk'].cuda().float()  # 人脸关键点（未直接用于损失计算）
        role_id = data['role_id'].cuda().long()  # 角色ID（区分不同虚拟角色）
        do_pixel = data['do_pixel'].cuda().int()  # 标记是否计算像素级损失（1=需要计算）
        bs_input = data['bs'].cuda().float()  # blendshape参数（未直接用于损失计算）
        
        
        # 提取各类索引（用于筛选特定数据计算损失）
        real_idx = ((is_render == 0).nonzero(as_tuple=True)[0])  # 真实图像的索引
        render_idx = ((is_render == 1).nonzero(as_tuple=True)[0])  # 渲染图像的索引
        do_pixel_idx = ((do_pixel == 1).nonzero(as_tuple=True)[0])  # 需要计算像素损失的样本索引
        role_idx0 = ((role_id == 0).nonzero(as_tuple=True)[0])  # 角色0的样本索引（特定角色筛选）
        role_idx1 = ((role_id == 1).nonzero(as_tuple=True)[0])  # 角色1的样本索引（特定角色筛选）
        # 按角色名称分组的索引（区分所有角色的样本）s
        role_idxes = [((role_id == CHARACTER_NAMES.index(name_e)).nonzero(as_tuple=True)[0]) for name_e in characters]
        # source image to emb
        # 禁用梯度计算（验证阶段不更新任何模型参数，节省内存并加速计算）s
        with torch.no_grad():
            
            # 源图像通过嵌入模型（model_emb）生成嵌入向量
            emb_hidden_in, emb_in = model_emb(resize(sources))
            # 若启用对称损失，通过对称嵌入模型（model_symm）处理源图像
            if args.weight_symm:
                emb_hidden_symm_in, emb_symm_in = model_symm(resize_symm(sources))
                # 拼接原始嵌入和对称嵌入
                emb_hidden_in = torch.cat((emb_hidden_in, emb_hidden_symm_in), dim=1)
                
            # 生成器根据源图像的嵌入向量和角色ID，输出预测的rig参数
            output_rig = model(emb_hidden_in, role_id)
            # 用每个角色对应的rig2img模型，将生成的rig参数转换为目标图像
            output_imgs_c = []  # 存储每个角色的生成图像
            for c_i, cname in enumerate(characters):
                output_imgs_c.append(configs_characters[cname]["model_rig2img"](output_rig[role_idxes[c_i]][:,:configs_characters[cname]['n_rig']].reshape(-1, configs_characters[cname]['n_rig'], 1, 1)))
            
            # 拼接不同角色的生成图像，形成完整的输出图像张量
            C, H, W = output_imgs_c[0].shape[1:]  # 获取图像通道数和尺寸
            B = sources.shape[0]  # 获取批次大小s
            output_img = torch.empty((B, C, H, W), dtype=torch.float32).cuda()  # 假设输出的类型为float32  # 初始化输出图像张量
            for c_i, cname in enumerate(characters):
                output_img[role_idxes[c_i]] = output_imgs_c[c_i] # 按角色索引填充生成图像
            
        # 计算rig参数损失（L2损失）：若启用权重且存在有效样本
        if args.weight_rig:
            if len(do_pixel_idx) > 0:
                loss['rig'] = criterion_l2(output_rig[do_pixel_idx], target_rigs[do_pixel_idx])
        
        # 计算嵌入一致性损失（L2损失）：生成图像的嵌入与源图像的嵌入对比
        if args.weight_emb:
            emb_hidden_out, emb_out = model_emb(resize(output_img))  # 生成图像的嵌入
            loss['emb'] = criterion_l2(emb_out, emb_in)
        
        # 计算像素级损失（L1损失）：生成图像与目标图像的像素差异
        if args.weight_img:
            if len(do_pixel_idx) > 0:  # 仅对标记需要计算的样本
                loss['image'] = criterion_l1(output_img[do_pixel_idx], targets[do_pixel_idx]) * args.weight_img
        
        # 计算生成器的对抗损失：判别器对生成图像的判定结果
        if args.weight_D:
            output_D_G = model_D(output_img)  # 判别器对生成图像的输出
            loss['G_D'] = criterion_gan(output_D_G, True, False)[0] * args.weight_D  # 对抗损失（标签为真）
        
        # 计算对称损失：生成图像的对称嵌入与源图像的对称嵌入对比
        if args.weight_symm:
            with torch.no_grad():  # 对称嵌入模型参数不更新
                emb_hidden_symm_out, emb_symm_out = model_symm(resize_symm(output_img))  # 生成图像的对称嵌入
            loss['symm'] = criterion_l2(emb_symm_out, emb_symm_in) * args.weight_symm
        
        # 若没有损失项，跳过当前批次
        if not loss:
            continue
        
        # 计算生成器的总验证损失（所有损失项之和）
        loss_value = sum([v for k, v in loss.items()])
        
        # 累加生成器总损失（用于计算平均损失）
        loss_sum += loss_value.item()

        # 将生成器的各类验证损失写入TensorBoard（用于可视化对比训练损失）
        writer.add_scalars(f'eval/loss_G', loss, epoch * eval_step + i)
        # 将生成器总验证损失写入TensorBoard
        writer.add_scalar(f'eval/loss_G_total', loss_value.item(), epoch * eval_step + i)
        
        # 格式化生成器损失字符串（保留4位小数）
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        # 构建验证日志信息（包含轮次、时间戳、学习率、步数和生成器损失）
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}"
        # 将日志添加到日志列表
        logger.append(_log + '\n')
        # 更新进度条描述（显示当前日志）
        pbar.set_description(_log)
        
    # 将源图像、目标图像和生成图像拼接后，写入TensorBoard（每4个样本取1个，用于可视化验证效果）
    writer.add_images(f'eval/img', torch.cat([sources, targets, output_img], dim=-2)[::4], epoch * eval_step + i)
    # 计算本轮验证生成器的平均损失
    avg_loss = loss_sum / eval_step
    # 构建本轮验证的总结日志
    _log = "==> [Eval] Epoch {} ({}), evaluation loss={}".format(epoch, timestamp, avg_loss)

    
    # 判断当前平均验证损失是否优于历史最优损失（早停机制相关）
    if avg_loss < best_score:
        # 重置耐心计数器（早停等待轮次）
        patience_cur = args.patience
        # 更新最优损失
        best_score = avg_loss        
        # 保存当前生成器和判别器作为最优模型（覆盖之前的最优模型文件）
        torch.save({'state_dict': model.state_dict()}, model_path)
        torch.save({'state_dict': model_D.state_dict()}, model_path.replace('.pt', f'_D.pt'))
        # 日志中标记找到新的最优模型
        _log += '\n Found new best model!\n'
    else:
        # 未找到更优模型，减少耐心计数器
        patience_cur -= 1
    # 打印总结日志
    print(_log)
    # 将总结日志添加到日志列表
    logger.append(_log)
    # 将本轮所有验证日志写入日志文件（与训练日志同文件）
    with open(os.path.join(log_save_path, f'emb2render_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)   
    # 返回本轮验证的平均损失
    return avg_loss

# 使用训练好的模型进行测试，生成虚拟角色图像并保存结果（包括图像、视频和rig参数）
def Test(signature, model, model_emb, model_rig2img, resize):
    # 参数说明：
    #   signature：测试标识（用于区分不同测试任务的结果文件夹）
    #   model：训练好的生成器模型（用于从图像嵌入生成rig参数）
    #   model_emb：图像嵌入模型（用于提取图像特征）
    #   model_rig2img：rig参数转图像的模型（用于将rig参数生成虚拟角色图像）
    #   resize：图像尺寸调整函数（预处理图像以匹配模型输入要求）
    # 将生成器设置为评估模式（关闭dropout等训练特有的层，确保随机因素影响）
    model = model.eval()
    # 测试结果保存根目录
    save_root = '/project/qiuf/expr-capture/test'
    # 测试数据根目录（存放输入图像）
    root = '/data/Workspace/Rig2Face/data'
    # 测试数据文件夹列表（此处仅测试'ziva'文件夹下的数据）
    folders = ['ziva']
    
    # 遍历每个测试文件夹
    for fold in folders:
        # 构建当前当前测试任务+文件夹的结果保存路径
        save_fold = os.path.join(save_root, f'{signature}_{fold}')
        # 创建结果文件夹（若不存在则自动创建，避免路径不存在报错）
        os.makedirs(save_fold, exist_ok=True)
        # 获取测试文件夹下所有图像文件名，并排序（确保处理顺序一致）
        imgnames = os.listdir(os.path.join(root, fold))
        imgnames.sort()
        imgnames = imgnames[:]  # 取所有图像（可通过切片限制数量）

        # 初始化字典，用于存储每个角色的rig参数序列
        rigs = {}
        for charname in characters:
            rigs[charname] = []
        # 遍历每张测试图像（带进度条显示处理进度）
        for i, img in tqdm(enumerate(imgnames), total=len(imgnames)):
            # 读取图像并调整尺寸为256×256
            img = cv2.resize(cv2.imread(os.path.join(root, fold, img)), (256, 256))
            # 将图像从BGR格式（OpenCV默认）转换为RGB格式（模型输入要求）
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 将图像转换为PyTorch张量：调整维度（HWC→CHW）、添加批次维度、转移到GPU、归一化到[0,1]
            img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).cuda() / 255.
            
            # 禁用梯度计算（测试阶段不更新模型参数，节省内存并加速计算）
            with torch.no_grad():
                # 提取输入图像的特征嵌入（用于生成rig参数）
                emb_hidden, emb = model_emb(resize(img_tensor))
                # 若启用对称损失，补充对称特征嵌入并拼接
                if args.weight_symm:
                    emb_hidden_symm_in, emb_symm_in = model_symm(resize_symm(img_tensor))
                    emb_hidden = torch.cat((emb_hidden, emb_hidden_symm_in), dim=1)
                
                # 存储每个角色的生成图像和嵌入距离
                img_outs = []  # 生成的虚拟角色图像列表
                emb_dists = []  # 生成图像与输入图像的嵌入距离（衡量语义一致性）
                # 为每个角色生成对应的虚拟图像
                for c_i, cname in enumerate(characters):
                    # 根据角色ID和输入图像的嵌入，生成该角色的rig参数
                    rig = model(emb_hidden, torch.LongTensor([CHARACTER_NAMES.index(cname), ]).cuda())
                    # 保存当前帧的rig参数（用于后续分析或复用）
                    rigs[cname].append(rig.cpu().numpy())
                    # 将rig参数转换为该角色的虚拟图像（通过角色专属的rig2img模型）
                    img_outs.append(configs_characters[cname]["model_rig2img"](
                        rig[:, :configs_characters[cname]['n_rig']].reshape(1, -1, 1, 1)
                    ))
                    # 计算生成图像与输入图像的嵌入距离（用于量化评估生成质量）
                    emb_hidden_out0, emb_out0 = model_emb(resize(img_outs[-1]))
                    emb_dists.append(torch.dist(emb, emb_out0))  # 欧氏距离
                
                # 可视化结果拼接：输入图像 + 各角色生成图像（横向拼接）
                # 转换为uint8格式（0-255），调整维度为HWC，转移到CPU
                img_vis = torch.cat((img_tensor, *img_outs), dim=-1).squeeze() * 255.
                img_vis = img_vis.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                # 将RGB转回BGR（OpenCV保存图像需要BGR格式）
                img_vis = np.ascontiguousarray(img_vis[..., ::-1], dtype=np.uint8)
                # 在生成图像上方标注嵌入距离（量化生成质量的指标）
                for c_i, cname in enumerate(characters):
                    # 标注的图像、标注文本（保留6位小数）、标注位置（x=角色图像起始x坐标，y=20）、字体、字体大小、颜色（红色）、线宽
                    cv2.putText(img_vis, str(np.round(emb_dists[c_i].item(), 6)), (256*(c_i+1), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # 保存可视化结果图像（文件名按序号递增，如00000.jpg）
                cv2.imwrite(os.path.join(save_fold, f'{i:05d}.jpg'), img_vis)
                
        # 将每个角色的rig参数序列保存为txt文件（便于后续分析或驱动其他模型）
        for c_i, cname in enumerate(characters):
            np.savetxt(os.path.join(save_root, f'ziva_{cname}.txt'), np.array(rigs[cname]).squeeze())

        # 将当前文件夹下的所有生成图像合成为视频（便于直观查看连续帧效果）
        imgs2video(save_fold)

# 主程序入口：当脚本被直接运行时执行以下逻辑，用于启动多角色embedding到rig参数的GAN训练/测试流程
if __name__ == '__main__':
    import time  # 导入时间模块（生成时间戳、记录时间）
    from choose_character import character_choice  # 导入角色配置选择工具（加载不同角色的参数）
    from models.load_emb_model import load_emb_model  # 导入嵌入模型加载工具（加载图像特征提取模型）
    from models.CascadeNet import get_model  # 导入生成器模型构建工具（embedding到rig参数的模型）
    from models.discriminator import MultiscaleDiscriminator, get_parser  # 导入判别器模型及参数解析工具
    from dataset.ABAWData import ABAWDataset2_multichar  # 导入多角色数据集类（加载多角色训练/验证数据）
    from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard工具（可视化训练过程）
    # 从YAML配置文件加载多角色任务参数（如学习率、批次大小、损失权重等）
    args = parse_args_from_yaml('configs_emb2rig_multi.yaml')
    # 设置随机种子（保证实验可重复性，使每次运行结果一致）
    setup_seed(args.seed)
    # 生成时间戳（格式：年月日-时分秒），用于标识当前训练任务（区分不同实验）
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # 调用Git命令：将当前代码修改添加到暂存区（记录训练时的代码状态，便于追溯）
    os.system("git add .")
    # 提交Git暂存区修改，提交信息为时间戳（标记本次训练对应的代码版本）
    os.system("git commit -m" + timestamp)
    # 将本地提交推送到远程Git仓库（备份代码，确保实验可复现）
    os.system("git push")
    
    # emb_model
    # 初始化多角色配置及相关模型
    # 解析配置中的角色列表（将字符串按逗号分割为角色名称列表，如"char1,char2"→["char1","char2"]）
    characters = args.character.replace(' ','').split(',')
    # 角色名称列表（从配置中获取，用于匹配角色ID）
    CHARACTER_NAMES = args.CHARACTER_NAME
    # 为每个角色加载专属配置（如数据路径、rig参数数量、嘴部掩码等），存储为字典（键：角色名，值：配置）
    configs_characters = {e: character_choice(e) for e in characters}
    # 计算所有角色中最大的rig参数维度（生成器输出需覆盖所有角色的rig参数长度）
    n_rig = max([e['n_rig'] for e in configs_characters.values()])
    # 为每个角色初始化并加载"rig转图像"模型（用于将生成的rig参数转换为可视化图像，辅助评估）
    for character in configs_characters:
        # 将当前角色的嘴部区域掩码转换为Tensor并转移到GPU（用于后续嘴部损失计算）
        configs_characters[character]['mouth_crop'] = torch.tensor(configs_characters[character]['mouth_crop']).cuda().float()
        # 定义"rig转图像"模型的参数（输入维度为当前角色的rig数量，输出为3通道RGB图像）
        params = {'nz': configs_characters[character]['n_rig'], 'ngf': 64*2, 'nc': 3}
        model_rig2img = Generator(params)  # 初始化生成器（rig→图像）
        model_rig2img = model_rig2img.eval().cuda()  # 设置为评估模式并转移到GPU
        # 加载预训练的"rig转图像"模型权重（从角色配置中指定的路径加载）
        ckpt_generator = torch.load(configs_characters[character]['ckpt_rig2img'])
        model_rig2img.load_state_dict(ckpt_generator['state_dict'])
        # 将加载好的模型存入角色配置，供后续生成图像使用
        configs_characters[character]['model_rig2img'] = model_rig2img
        print(f'load generator model from {configs_characters[character]["ckpt_rig2img"]}')  # 打印加载提示
        
    # 加载图像嵌入模型（用于提取输入图像的特征，作为生成器的输入）
    # 返回：嵌入模型、嵌入维度（特征长度）、图像尺寸调整函数（预处理图像以匹配模型输入）
    model_emb, emb_dim, resize = load_emb_model(args.emb_backbone)
    model_emb = model_emb.eval().cuda()  # 设置为评估模式并转移到GPU（特征提取时不更新参数）
    model_emb_params = count_parameters(model_emb)  # 计算嵌入模型的参数数量
    print(f'emb model: {model_emb_params}')  # 打印嵌入模型参数数量（评估模型复杂度）
    
    # dissymm model 
    # 若启用对称损失（通过对称特征增强生成图像的结构一致性），加载对称嵌入模型
    if args.weight_symm:
        # 加载对称特征嵌入模型（如用于人脸左右对称特征提取的模型）
        model_symm, emb_dim2, resize_symm = load_emb_model('dissymm_repvit')
        model_symm.cuda().eval()  # 设置为评估模式并转移到GPU
        emb_dim += emb_dim2  # 总嵌入维度 = 原始嵌入维度 + 对称嵌入维度（拼接特征）

    # img2rig model
    # 占位：预留"图像到rig"模型的初始化位置（当前流程未使用，可能为预留扩展）
    pass

    # emb2rig model
    # 初始化生成器和判别器模型
    # 定义生成器模型（embedding→rig参数）的保存路径（结合保存根目录、任务名和时间戳）
    model_path = os.path.join(args.save_root, 'ckpt', "emb2rig_multi_{}.pt".format(timestamp))
    # 构建生成器模型（CascadeNet）：从图像嵌入生成rig参数
    model = get_model(
        1,  # 输入通道数（固定为1，此处无实际意义）
        refine_3d=False,  # 不启用3D精修（针对2D任务）
        norm_twoD=False,  # 不启用2D归一化
        num_blocks=2,  # 网络块数量（控制模型复杂度）
        input_size=emb_dim,  # 输入维度（图像嵌入的总维度）
        output_size=n_rig,  # 输出维度（最大rig参数数量，兼容所有角色）
        linear_size=512,  # 线性层维度（控制中间特征维度）
        dropout=0.1,  # dropout概率（防止过拟合）
        leaky=False,  # 不使用LeakyReLU激活函数
        use_multichar=args.use_multichar,  # 启用多角色模式（输入包含角色ID）
        id_embedding_dim=args.id_embedding_dim  # 角色ID嵌入维度（区分不同角色）
    )
    model = model.cuda()  # 将生成器转移到GPU
    model_params = count_parameters(model)  # 计算生成器参数数量
    print(f'emb2rig model: {model_params}')  # 打印生成器参数数量（评估复杂度）
    
    # D_model
    # 初始化判别器模型（多尺度判别器，用于区分真实/生成的rig参数对应的图像）
    opt = get_parser()  # 获取判别器参数配置
    model_D = MultiscaleDiscriminator(opt).cuda()  # 初始化并转移到GPU

    # 若使用预训练模型，加载生成器和判别器的权重
    if args.pretrained:
        # 预训练生成器权重路径
        ckpt_pretrained = os.path.join(args.save_root, 'ckpt', f"emb2rig_multi_{args.pretrained}.pt")
        # 预训练判别器权重路径
        ckpt_pretrained_D = os.path.join(args.save_root, 'ckpt', f"emb2rig_multi_{args.pretrained}_D.pt")
        # 加载生成器权重
        checkpoint = torch.load(ckpt_pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        # 加载判别器权重
        checkpoint_D = torch.load(ckpt_pretrained_D)
        model_D.load_state_dict(checkpoint_D['state_dict'])
        print(f"load pretrained model {ckpt_pretrained}")  # 打印加载提示
    else:
        # 不使用预训练模型时，初始化生成器和判别器的权重
        model.apply(init_weights)
        model_D.apply(init_weights)
        print("Model initialized")  # 打印初始化提示              
    
    # 定义数据预处理和优化器
    # transforms
    # 训练集图像预处理管道：包含颜色抖动（数据增强）、尺寸调整、转为Tensor
    transform1 = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 随机调整亮度、对比度等（增强泛化能力）
        transforms.Resize([256, 256]),  # 调整图像尺寸为256×256
        transforms.ToTensor(),  # 转为PyTorch张量
    ])
    
    # 验证集图像预处理管道：仅调整尺寸和转为Tensor（无数据增强，保证评估稳定）
    transform2 = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    
    # optimizer
    # 定义生成器优化器（Adam）：过滤需要梯度更新的参数，设置学习率和动量参数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.0, 0.99))
    # 生成器学习率调度器（余弦退火重启）：动态调整学习率，提升训练效果
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, 2, 1e-6)
    # 定义判别器优化器（Adam）：设置独立的学习率和权重衰减（防止过拟合）
    optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, model_D.parameters()), lr=args.lr_D, betas=(0.0, 0.99), weight_decay=1e-6)
    # 判别器学习率调度器（余弦退火重启）：与生成器调度器参数略有差异
    scheduler_D = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, 500, 2, 1e-7)
    
    # loss function 
    # 定义损失函数
    criterion_l1 = nn.L1Loss()  # L1损失（用于像素级/参数级误差）
    criterion_l2 = nn.MSELoss()  # L2损失（用于嵌入特征误差）
    criterion_BCE = nn.BCELoss()  # BCE损失（二分类交叉熵，预留）
    criterion_gan = GANLoss('hinge')  # GAN损失（hinge损失，用于对抗训练）
    
    # Test
    # 处理测试模式（若模式为test，直接运行测试）
    if args.mode == 'test':
        # 调用Test函数：使用预训练模型生成多角色图像并保存结果
        Test(args.pretrained, model, model_emb, model_rig2img, resize)
        exit()  # 测试完成后退出程序
    
    # datasets
    # 初始化数据集和训练辅助工具
    # 初始化训练数据集（多角色）：加载训练集数据，使用训练预处理
    train_dataset = ABAWDataset2_multichar(configs_characters, data_split='train', CHARACTER_NAME=CHARACTER_NAMES, transform=transform1, return_rigs=True)
    # 初始化验证数据集（多角色）：加载测试集数据，使用验证预处理
    test_dataset = ABAWDataset2_multichar(configs_characters, data_split='test', CHARACTER_NAME=CHARACTER_NAMES, transform=transform2, return_rigs=True)

    # 初始化训练数据加载器：批量加载训练数据，打乱顺序，8个工作进程加速加载
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
    # 初始化验证数据加载器：批量加载验证数据，不打乱顺序，8个工作进程
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)
    
    # save files
    # 定义各类文件的保存路径
    ck_save_path = f'{args.save_root}/ckpt'  # 模型权重保存路径
    pred_save_path = f'{args.save_root}/test'  # 预测结果保存路径
    log_save_path = f'{args.save_root}/logs'  # 训练日志保存路径
    tensorboard_path = f'{args.save_root}/tensorboard/{timestamp}'  # TensorBoard记录路径（按时间戳区分）
    
    # 创建保存路径（若不存在则自动创建，exist_ok=True避免路径已存在时报错）
    os.makedirs(ck_save_path,exist_ok=True)
    os.makedirs(pred_save_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)

    # 初始化TensorBoard写入器（用于记录损失曲线、生成图像等）
    writer = SummaryWriter(log_dir=tensorboard_path)
    
    # 初始化早停机制参数：当前耐心值（连续多少轮验证损失不下降则终止训练）
    patience_cur = args.patience
    # 初始化最优验证分数（用损失衡量，初始为无穷大）
    best_score = float('inf')

    # 启动训练循环
    # 无限训练循环（实际通过早停机制终止）
    for epoch in range(500000000):
        # 调用Train函数：在训练集上训练生成器和判别器，返回本轮训练平均损失
        avg_loss = Train(epoch, train_dataloader, model, model_D)
        # 调用Eval函数：在验证集上评估模型，返回本轮验证平均损失，更新最优模型
        avg_loss_eval = Eval(epoch, val_dataloader, model, model_D, best_score)
        # 按固定轮次执行测试：生成当前模型的多角色图像结果并保存（用于中间效果查看）
        if epoch % args.save_step == 0:
            Test(timestamp + '_' + str(epoch), model, model_emb, model_rig2img, resize)
