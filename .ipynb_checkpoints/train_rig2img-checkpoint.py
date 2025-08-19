from tqdm import tqdm

# 执行模型的训练过程，计算损失并更新参数，同时记录训练日志和保存模型
def Train(epoch, loader, model):
    # 参数说明：
    #   epoch：当前训练轮次
    #   loader：训练数据加载器（提供批量数据）
    #   model：待训练的模型
    # 获取当前优化器的学习率
    lr = optimizer.param_groups[0]['lr']
    # 打印当前训练轮次、学习率和时间戳（用于标识训练过程）
    print(f"*** Epoch {epoch}, lr:{lr:.5f}, timestamp:{timestamp}")
    # 初始化总损失计数器
    loss_sum = 0.0
    # 将模型设置为训练模式（启用dropout、批归一化等训练特有的层）
    model.train()    
    # 确定每轮训练的步数：取设定的最大步数或数据加载器总长度中的较小值
    train_step = min(args.train_step_per_epoch, len(loader))
    # 定义进度条格式
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    # 创建进度条（遍历数据加载器，显示训练进度）
    pbar = tqdm(enumerate(loader), bar_format=b, total=train_step)
    # 用于存储训练日志
    logger = []
    # 记录训练开始时间（用于计算耗时）
    time0 = time.time()
    # 遍历数据加载器中的每个批次（带索引i）
    for i, data in pbar:
        # 若达到设定的最大训练步数，则提前结束本轮训练
        if i > train_step:
            break
        # 清零优化器的梯度（避免上一轮梯度累积）
        optimizer.zero_grad()
        # 初始化损失字典（存储不同类型的损失）
        loss = dict()

        # 从数据中提取目标图像（虚拟角色渲染图），转移到GPU并转为float类型
        targets = data['img'].cuda().float()
        # 从数据中提取rig参数（面部驱动参数），转移到GPU并转为float类型
        rigs = data['rigs'].cuda().float()
        # 断言验证：确保当前批次所有数据都有rig参数（避免无驱动数据混入）
        assert (data['has_rig'] == 1).all()
        # 将rig参数调整为模型输入格式（添加空间维度），并通过模型生成输出图像
        outputs = model(rigs.reshape(-1, configs_character['n_rig'], 1, 1))
        # 计算图像重建损失（L1损失），并乘以权重参数
        loss['image'] = criterion_l1(outputs, targets) * args.weight_img
        # 计算嘴部区域的重建损失（仅关注嘴部），乘以对应权重
        loss['mouth'] = criterion_l1(outputs * mouth_crop, targets * mouth_crop) * args.weight_mouth
        
        # 计算总损失（所有损失项之和）
        loss_value = sum([v for k, v in loss.items()])
        
        # 累加总损失（用于计算平均损失）
        loss_sum += loss_value.item()
        # 反向传播计算梯度（retain_graph=True保留计算图，便于多次反向传播）
        loss_value.backward(retain_graph=True)
        # 优化器更新模型参数
        optimizer.step()
        # 学习率调度器更新学习率（按步数调整）
        scheduler.step()
        

        # 将各损失项写入TensorBoard（用于可视化训练过程）
        writer.add_scalars(f'train/loss', loss, epoch * train_step + i)
        # 将总损失写入TensorBoard
        writer.add_scalar(f'train/loss_total', loss_value.item(), epoch * train_step + i)
        
        # 格式化损失字符串（保留4位小数）
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        # 构建训练日志信息（包含轮次、时间戳、学习率、当前步数和损失）
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}"
        # 将日志添加到日志列表
        logger.append(_log + '\n')
        # 更新进度条描述（显示当前日志）
        pbar.set_description(_log)
        
    # 将生成的输出图像和目标图像拼接后，写入TensorBoard（每4个样本取1个，用于可视化效果）
    writer.add_images(f'train/img', torch.cat([outputs, targets], dim=-2)[::4], epoch * train_step + i)
    # 计算本轮训练的平均损失
    avg_loss = loss_sum / train_step
    # 构建本轮训练的总结日志
    _log = "==> [Train] Epoch {} ({}), training loss={}".format(epoch, timestamp, avg_loss)
    # 打印总结日志
    print(_log)
    # 将本轮所有训练日志写入日志文件
    with open(os.path.join(log_save_path, f'{task}_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    # 若当前轮次是设定的保存轮次，则保存模型权重（文件名添加轮次标识）
    if epoch % args.save_step == 0:
        torch.save({'state_dict': model.state_dict()}, model_path.replace('.pt', f'_{epoch}.pt'))
    # 返回本轮训练的平均损失
    return avg_loss

# 在每个训练轮次后对模型进行评估（验证/测试），计算评估损失并判断是否保存最优模型
def Eval(epoch, loader, model, best_score):
    # 参数说明：
    #   epoch：当前评估对应的训练轮次
    #   loader：评估数据加载器（通常是验证集数据）
    #   model：待评估的模型（与训练阶段的模型一致）
    #   best_score：当前最优的评估指标（此处为最小损失值）
    # 初始化总损失计数器
    loss_sum = 0.0
    # 将模型设置为评估模式（关闭dropout、固定批归一化参数等）
    model.eval()    
    # 确定每轮评估的步数：取设定的最大评估步数或数据加载器总长度中的较小值
    eval_step = min(args.eval_step_per_epoch, len(loader))
    # 定义进度条格式
    b = '{l_bar}{bar:20}{r_bar}{bar:10b}'
    # 创建进度条（遍历评估数据加载器，显示评估进度）
    pbar = tqdm(enumerate(loader), bar_format=b, total=eval_step)
    # 用于存储评估日志
    logger = []
    # 记录评估开始时间（用于计算耗时）
    time0 = time.time()
    # 遍历评估数据加载器中的每个批次（带索引i）
    for i, data in pbar:
        # 若达到设定的最大评估步数，则提前结束本轮评估
        if i > eval_step:
            break
        # 初始化损失字典（存储不同类型的损失）
        loss = dict()

        # 从数据中提取目标图像，转移到GPU并转为float类型
        targets = data['img'].cuda().float()
        # 从数据中提取rig参数，转移到GPU并转为float类型
        rigs = data['rigs'].cuda().float()
        # 断言验证：确保当前批次所有数据都有rig参数
        assert (data['has_rig'] == 1).all()
        # 禁用梯度计算（评估阶段不更新参数，节省内存并加速计算）
        with torch.no_grad():
            # 将rig参数调整为模型输入格式，通过模型生成输出图像
            outputs = model(rigs.reshape(-1, configs_character['n_rig'], 1, 1))
            # 计算图像重建损失（L1损失）并乘以权重
            loss['image'] = criterion_l1(outputs, targets) * args.weight_img
            # 计算嘴部区域重建损失并乘以权重
            loss['mouth'] = criterion_l1(outputs * mouth_crop, targets * mouth_crop) * args.weight_mouth

        # 计算总损失（所有损失项之和）
        loss_value = sum([v for k, v in loss.items()])
        
        # 累加总损失（用于计算平均损失）
        loss_sum += loss_value.item()

        # 将各损失项写入TensorBoard（与训练损失对比）
        writer.add_scalars(f'train/loss', loss, epoch * eval_step + i)
        # 将总损失写入TensorBoard
        writer.add_scalar(f'train/loss_total', loss_value.item(), epoch * eval_step + i)
        
        # 格式化损失字符串（保留4位小数）
        _loss_str = str({k: "{:.4f}".format(v/(i+1)) for k, v in loss.items()})
        # 构建评估日志信息（包含轮次、时间戳、学习率、当前步数和损失）
        _log = f"Epoch {epoch}({timestamp}) (lr:{optimizer.param_groups[0]['lr']:05f}): [{i}/{len(train_dataloader)}] loss_G:{_loss_str}"
        # 将日志添加到日志列表
        logger.append(_log + '\n')
        # 更新进度条描述（显示当前日志）
        pbar.set_description(_log)
        
    # 将生成的输出图像和目标图像拼接后，写入TensorBoard（用于可视化评估效果）
    writer.add_images(f'train/img', torch.cat([outputs, targets], dim=-2)[::4], epoch * eval_step + i)
    # 计算本轮评估的平均损失
    avg_loss = loss_sum / eval_step
    # 构建本轮评估的总结日志
    _log = "==> [Eval] Epoch {} ({}), training loss={}".format(epoch, timestamp, avg_loss)
    
    # 判断当前平均损失是否优于历史最优损失（早停机制相关）
    if avg_loss < best_score:
        # 重置耐心计数器（早停等待轮次）
        patience_cur = args.patience
        # 更新最优损失
        best_score = avg_loss        
        # 保存当前模型作为最优模型（覆盖之前的最优模型文件）
        torch.save({'state_dict': model.state_dict()}, model_path)
        # 日志中标记找到新的最优模型
        _log += '\n Found new best model!\n'
    else:
        # 未找到更优模型，减少耐心计数器
        patience_cur -= 1
        
    # 打印总结日志
    print(_log)
    # 将本轮所有评估日志写入日志文件（与训练日志同文件）
    with open(os.path.join(log_save_path, f'{task}_{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    # 返回本轮评估的平均损失
    return avg_loss

# 主程序入口：当脚本被直接运行时执行以下代码，作为模块导入时不执行
if __name__ == '__main__':
    import time  # 导入时间模块（用于生成时间戳、计算耗时等）
    import os  # 导入操作系统模块（用于文件路径操作、系统命令调用）
    import torch  # 导入PyTorch深度学习框架（核心计算库）
    from choose_character import character_choice  # 导入角色配置选择工具（根据角色加载对应参数）
    from utils.common import parse_args_from_yaml, setup_seed, init_weights  # 导入通用工具：解析配置、设置随机种子、初始化模型权重
    from models.DCGAN import Generator  # 导入DCGAN生成器模型（用于从rig参数生成图像）
    import torchvision.transforms as transforms  # 导入图像预处理工具
    import torch.nn as nn  # 导入PyTorch神经网络模块（定义损失函数、网络层等）
    from dataset.ABAWData import ABAWDataset2  # 导入自定义数据集类（加载训练/验证数据）
    from torch.utils.data import DataLoader  # 导入数据加载器（批量加载数据）
    from torch.optim import lr_scheduler  # 导入学习率调度器（动态调整学习率）
    from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard工具（可视化训练过程）
    task = 'rig2img'  # 定义任务名称：从rig参数生成图像（rig to image）
    args = parse_args_from_yaml(f'configs_{task}.yaml')  # 从YAML配置文件加载任务参数（如学习率、批次大小等）
    setup_seed(args.seed)  # 设置随机种子（保证实验可重复性，每次运行结果一致）
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # 生成时间戳（格式：年月日-时分秒），用于标识当前训练过程
    os.system("git add .")  # 调用系统命令：将当前代码所有修改添加到Git暂存区（记录训练时的代码状态）
    os.system("git commit -m" + timestamp)  # 调用系统命令：提交Git暂存区的修改，提交信息为当前时间戳
    os.system("git push")  # 调用系统命令：将本地提交推送到远程Git仓库（备份代码，确保可追溯）
    
    # 根据选择的角色（args.character）加载对应的配置（如数据路径、rig参数数量、嘴部区域掩码等）
    configs_character = character_choice(args.character)
    # 加载嘴部区域掩码（用于后续损失计算中重点优化嘴部细节），并转移到GPU转为float类型
    mouth_crop = torch.tensor(configs_character['mouth_crop']).cuda().float()

    # 定义模型权重保存路径：结合保存根目录、任务名和时间戳（确保每个训练过程的模型文件唯一）
    model_path = os.path.join(args.save_root,'ckpt', f"{task}_{timestamp}.pt")
    # 生成器模型参数：nz（输入维度，即rig参数数量）、ngf（特征图数量）、nc（输出通道数，3对应RGB图像）
    params = {'nz': configs_character['n_rig'], 'ngf': 64*2, 'nc': 3}
    model = Generator(params)  # 初始化DCGAN生成器模型
    model = model.cuda()  # 将模型转移到GPU（加速计算）
    
    # 若使用预训练模型，则加载权重；否则初始化模型权重
    if args.pretrained:
        # 预训练模型路径：从保存根目录加载指定版本的模型
        ckpt_pretrained = os.path.join(args.save_root, 'ckpt', f"{task}_{args.pretrained}.pt")
        checkpoint = torch.load(ckpt_pretrained)  # 加载预训练模型权重文件
        model.load_state_dict(checkpoint['state_dict'])  # 将预训练权重加载到当前模型
        print("load pretrained model {}".format(ckpt_pretrained))  # 打印提示：已加载预训练模型
    else:
        model.apply(init_weights)  # 对模型参数进行初始化（使用自定义初始化方法）
        print("Model initialized")  # 打印提示：模型已完成初始化  
    # 定义图像预处理管道：将图像resize到256×256，再转为PyTorch张量（Tensor）      
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor()])
    
    criterion_l1 = nn.L1Loss()  # 定义L1损失函数（用于计算生成图像与目标图像的像素差异）
    # 定义Adam优化器：过滤需要梯度更新的参数，设置初始学习率（args.lr）和动量参数（betas）
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.0, 0.99))
    # 定义学习率调度器：余弦退火重启（CosineAnnealingWarmRestarts），周期500步，重启系数2，最小学习率1e-6
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, 2, 1e-6)
 
    # 初始化训练数据集：指定数据路径、角色、只使用渲染图像、训练集分割、预处理、返回rig参数及数量
    train_dataset = ABAWDataset2(root_path=configs_character['data_path'],character=args.character, only_render=True,
                                 data_split='train', transform=transform, return_rigs=True, n_rigs=configs_character['n_rig'])
    # 初始化验证数据集：参数与训练集类似，但数据分割为测试集（实际为验证集）
    test_dataset = ABAWDataset2(root_path=configs_character['data_path'],character=args.character,only_render=True,
                                data_split='test', transform=transform, return_rigs=True, n_rigs=configs_character['n_rig'])
    # 训练数据加载器：批量大小args.batch_size，打乱数据，丢弃最后不完整批次，12个工作进程（加速数据加载）
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=12)
    # 验证数据加载器：不打乱数据（保证验证结果稳定），其他参数同训练加载器
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True, num_workers=12)

    # 定义各种保存路径
    ck_save_path = f'{args.save_root}/ckpt'  # 模型检查点（权重文件）保存路径
    pred_save_path = f'{args.save_root}/test'  # 预测结果（生成图像）保存路径
    log_save_path = f'{args.save_root}/logs'  # 训练日志保存路径
    tensorboard_path = f'{args.save_root}/tensorboard/{timestamp}'  # TensorBoard记录路径（按时间戳区分）
    
    # 创建上述路径（若不存在则自动创建，exist_ok=True避免路径已存在时报错）
    os.makedirs(ck_save_path,exist_ok=True)
    os.makedirs(pred_save_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_path)  # 初始化TensorBoard写入器（用于记录损失曲线、生成图像等）
    
    patience_cur = args.patience  # 早停机制当前耐心值（初始化为配置中的耐心值，用于控制过拟合）
    best_score = float('inf')  # 初始化最佳验证分数（用损失衡量，初始为无穷大）


    # 训练循环：设置极大的epoch数（实际通过早停机制终止）
    for epoch in range(500000000):
        avg_loss = Train(epoch, train_dataloader, model)  # 调用训练函数，返回本轮训练平均损失
        avg_loss_eval = Eval(epoch, val_dataloader, model, best_score)  # 调用验证函数，返回本轮验证平均损失
