from dataset import build_dataset
from model.mae_pipeline import Pipeline 
from torch import optim
from utils.metrics import triplet_prediction_accuracy
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.distance import PairwiseDistance
from tqdm import tqdm
import yaml
import argparse
import utils.misc as misc
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import os
# 一个基于三元组损失（Triplet Loss）的图像特征嵌入训练与评估脚本，
# 用于训练模型学习图像的判别性特征表示（嵌入向量），使得同类图像（正例）的特征距离更近，不同类图像（负例）的特征距离更远。
# 其核心应用场景包括图像相似度匹配、检索或细粒度分类等任务。

# 读取YAML格式文件，并将其内容解析为Python字典返回
def read_yaml_to_dict(yaml_path):
    # 打开指定路径的YAML文件（使用with语句确保文件操作完成后自动关闭）
    with open(yaml_path) as file:
        # 读取文件内容，通过yaml.FullLoader安全解析YAML格式
        # 将解析结果转换为Python字典（键值对结构）
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        # 返回解析后的字典
        return dict_value

# 计算三元组损失（Triplet Loss）并返回特征间的距离
def compute_loss(anc_fea, pos_fea, neg_fea, type):
    # 输入：
    #   anc_fea: 锚点图像的特征向量（[batch_size, feature_dim]）
    #   pos_fea: 正例图像的特征向量（与锚点同类，[batch_size, feature_dim]）
    #   neg_fea: 负例图像的特征向量（与锚点不同类，[batch_size, feature_dim]）
    #   type: 三元组类型列表（用于区分不同类型的三元组损失计算）
    # 输出：
    #   loss: 平均三元组损失
    #   dists1: 锚点与正例的L2距离（[batch_size,]）
    #   dists2: 锚点与负例的L2距离（[batch_size,]）
    #   dists3: 正例与负例的L2距离（[batch_size,]）
    # 定义三个三元组损失函数，margin（边界值）分别为0.1、0.2、0.1
    # margin用于控制正负例距离的差异要求（希望正例距离 < 负例距离 - margin）
    criterion1 = nn.TripletMarginLoss(margin=0.1)
    criterion2 = nn.TripletMarginLoss(margin=0.2)
    criterion3 = nn.TripletMarginLoss(margin=0.1)
    # 定义L2距离计算器（用于后续计算特征间的具体距离）
    l2_dist = PairwiseDistance(2)
    # 初始化损失值
    loss = 0
    # 遍历每个样本，按三元组类型计算损失
    for i in range(len(type)):
        # 为单个样本特征添加批次维度（从[feature_dim]转为[1, feature_dim]，适配损失函数输入要求）
        anc_ = anc_fea[i].unsqueeze(0)
        pos_ = pos_fea[i].unsqueeze(0)
        neg_ = neg_fea[i].unsqueeze(0)

        # 根据三元组类型选择损失函数：
        # 若为"ONE_CLASS_TRIPLET"类型，使用margin=0.1的损失函数
        # 同时计算（锚点,正例,负例）和（正例,锚点,负例）两种组合的损失，增强约束
        if type[i] == "ONE_CLASS_TRIPLET":
            loss += criterion1(anc_, pos_, neg_) + criterion1(pos_, anc_, neg_)
        # 其他类型三元组使用margin=0.2的损失函数，同样计算两种组合的损失
        else:
            loss += criterion2(anc_, pos_, neg_) + criterion2(pos_, anc_, neg_)

    # 计算批次平均损失（除以样本数量）
    loss = loss / anc_fea.shape[0]
    # 计算锚点与正例的L2距离，转为numpy数组（用于后续准确率评估）
    dists1 = l2_dist.forward(anc_fea, pos_fea).data.cpu().numpy()
    # 计算锚点与负例的L2距离
    dists2 = l2_dist.forward(anc_fea, neg_fea).data.cpu().numpy()
    # 计算正例与负例的L2距离
    dists3 = l2_dist.forward(pos_fea, neg_fea).data.cpu().numpy()
    # 返回平均损失和三个距离数组
    return loss, dists1, dists2, dists3

# 主函数，协调整个模型训练与验证流程，包括数据加载、模型初始化、训练循环等
def main(config):
    # 根据配置构建训练集和验证集
    trainset = build_dataset(config, "train")
    valset = build_dataset(config, "val")
    # 创建训练集和验证集的数据加载器（DataLoader）
    # 配置包括批次大小、工作进程数，训练集开启数据打乱（shuffle=True）
    trainloader = DataLoader(trainset, batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)
    valloader = DataLoader(valset, batch_size=config["batch_size"],num_workers=config["num_workers"],shuffle=True)

    # 初始化模型（Pipeline类），并将模型迁移到GPU上
    model = Pipeline(config).cuda()

    # 如果配置了预训练权重路径（resume），加载权重到模型
    if config["resume"] != None:
        state_dict = torch.load(config["resume"])  # 加载权重文件
        model.load_state_dict(state_dict)  # 加载权重到模型
    
    # 如果配置了数据并行（use_dp=True），使用多GPU训练
    if config["use_dp"] == True:
        gpus = config["device"]  # 获取GPU设备列表
        model = torch.nn.DataParallel(model, device_ids=gpus)  # 包装模型为数据并行模式

    # 从配置中获取训练的总轮次（epochs）
    num_epochs = config["num_epochs"]

    # 根据配置选择优化器（SGD或AdamW）
    # 仅优化requires_grad=True的参数（可训练参数）
    if config["optim"] == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], momentum=config["momentum"],
                      weight_decay=config["weight_decay"]) # 学习率、动量（SGD特有参数）、权重衰减（正则化）
    elif config["optim"] == "AdamW":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], betas=(0.9,0.999),
                      weight_decay=config["weight_decay"]) # 筛选可训练参数、AdamW的动量参数、权重衰减

    # 初始化训练和验证的指标记录器（用于记录损失、准确率等）
    metric_logger = misc.MetricLogger(delimiter="  ")
    test_metric_logger = misc.MetricLogger(delimiter="  ")
    # 初始化TensorBoard日志写入器，用于可视化训练过程
    log_writer = SummaryWriter(log_dir=config["log_dir"])

    # 创建日志和模型 checkpoint 保存目录（若不存在则创建）
    os.makedirs(config["log_dir"],exist_ok=True)
    os.makedirs(config["checkpoint_dir"],exist_ok=True)
    

    # 主训练循环：遍历每个epoch
    for epoch in range(num_epochs):
        # 训练一轮：更新模型参数
        train_one_epoch(epoch, model, trainloader, metric_logger, log_writer, optimizer, config)
        # 验证一轮：评估模型性能，不更新参数
        evaluate(epoch, model, valloader, test_metric_logger, config)

# 执行单个epoch的训练过程，包括数据加载、前向传播、损失计算、反向传播和参数更新
def train_one_epoch(epoch, model, data_loader, metric_logger, log_writer, optimizer, config):
    # 输入：
    #   epoch: 当前训练轮次
    #   model: 待训练的模型
    #   data_loader: 训练数据集的数据加载器
    #   metric_logger: 用于记录训练指标（如损失、学习率）的日志器
    #   log_writer: TensorBoard日志写入器，用于可视化训练过程
    #   optimizer: 优化器，用于更新模型参数
    #   config: 配置字典，包含训练相关参数（如打印频率、梯度累积次数等）
    # 将模型设置为训练模式（启用 dropout、批量归一化的训练模式等）
    model.train(True)
    # 从配置中获取日志打印频率（每多少个批次打印一次训练状态）
    print_freq = config["print_freq"]
    # 从配置中获取梯度累积次数（用于在内存有限时模拟大批次训练）
    accum_iter = config["accum_iter"]
    # 定义训练日志的标题，包含当前epoch信息
    header = 'Training Epoch: [{}]'.format(epoch)
    # 包装数据加载器，实现按频率打印日志的功能，返回枚举器（迭代索引和样本）
    t = enumerate(metric_logger.log_every(data_loader, print_freq, header))
    # 遍历每个批次的训练数据
    for step, samples in t:
        # 从样本字典中提取锚点图像、正例图像、负例图像、锚点名称和三元组类型
        anc_img, pos_img, neg_img, anc_list, type = samples["anc"], samples["pos"], samples["neg"], samples["name"], samples["type"]
        # 将图像数据迁移到GPU（若使用GPU训练）
        anc_img, pos_img, neg_img = anc_img.cuda(), pos_img.cuda(), neg_img.cuda()
        # 清空模型当前的梯度（避免与上一批次的梯度累积）
        model.zero_grad()
        # 将锚点、正例、负例图像在批次维度拼接（便于一次性输入模型提取特征）
        vec = torch.cat((anc_img, pos_img, neg_img), dim=0)
        # 前向传播：输入拼接的图像，获取所有图像的特征嵌入
        emb = model.forward(vec)
        # 计算单个类别的样本数量（总样本数为3倍单类别数量，因包含锚点、正例、负例）
        ll = int(emb.shape[0] / 3)
        # 将特征嵌入按锚点、正例、负例拆分（恢复为三个独立的特征矩阵）
        anc_fea, pos_fea, neg_fea = torch.split(emb, ll, dim=0)

        # 计算三元组损失和特征间的距离（锚点-正例、锚点-负例、正例-负例）
        loss, dists1, dists2, dists3 = compute_loss(anc_fea, pos_fea, neg_fea, type)
        
        # 获取损失值（转为Python数值）
        loss_value = loss.item()
        # 检查损失值是否合法（若为无穷大或NaN，停止训练）
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 反向传播：计算梯度
        loss.backward()
        # 优化器更新模型参数
        optimizer.step()

        # 更新训练指标日志：记录当前批次的损失值和学习率
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.state_dict()['param_groups'][0]['lr'])

        # 在分布式训练中，同步所有进程的损失值（取平均值）
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        # 当达到梯度累积次数时，记录损失到TensorBoard
        if (step + 1) % accum_iter == 0:
            # 计算当前全局迭代次数（用于TensorBoard的x轴）
            iter = epoch * len(data_loader) + step + 1
            # 写入损失值到TensorBoard
            log_writer.add_scalar("loss", loss_value, iter)

    # 打印当前epoch的平均训练指标（如平均损失、平均学习率）
    print("Averaged stats:", metric_logger)

# 执行单个epoch的验证过程，评估模型性能（损失、准确率等），并按条件保存模型
def evaluate(epoch, model, data_loader, test_metric_logger, config):
    # 输入：
    #   epoch: 当前验证轮次
    #   model: 待评估的模型
    #   data_loader: 验证数据集的数据加载器
    #   test_metric_logger: 用于记录验证指标的日志器
    #   config: 配置字典，包含验证相关参数（如打印频率、保存间隔等）
    # 输出：
    #   avg_loss: 验证集平均损失
    #   res: 验证结果（包含整体准确率及各类别准确率）
    # 将模型设置为评估模式（关闭 dropout、固定批量归一化参数等）
    model.eval()
    # 从配置中获取验证日志打印频率
    print_freq = config["print_freq"]
    # 定义验证日志的标题，包含当前epoch信息
    header = 'Validation Epoch: [{}]'.format(epoch)
    # 初始化三元组评估日志器，用于记录距离和准确率，日志保存到指定JSON文件
    acc_logger = misc.Triplet_Logger(os.path.join(config["log_dir"], "test_log.json"))

    # 包装数据加载器，实现按频率打印验证日志的功能，返回枚举器
    t = enumerate(test_metric_logger.log_every(data_loader, print_freq, header))

    # 遍历每个批次的验证数据
    for step, samples in t:
        # 从样本字典中提取锚点图像、正例图像、负例图像、锚点名称和三元组类型
        anc_img, pos_img, neg_img, anc_list, type = samples["anc"], samples["pos"], samples["neg"], samples["name"], samples["type"]
        # 将图像数据迁移到GPU
        anc_img, pos_img, neg_img = anc_img.cuda(), pos_img.cuda(), neg_img.cuda()

        # 将锚点、正例、负例图像在批次维度拼接（一次性输入模型）
        vec = torch.cat((anc_img, pos_img, neg_img), dim=0)
        # 禁用梯度计算（验证阶段不更新参数，节省内存并加速计算）
        with torch.no_grad():
            # 前向传播获取所有图像的特征嵌入
            emb = model.forward(vec)
            # 计算单个类别的样本数量（总样本数为3倍单类别数量）
            ll = int(emb.shape[0] / 3)
            # 拆分特征为锚点、正例、负例特征矩阵
            anc_fea, pos_fea, neg_fea = torch.split(emb, ll, dim=0)

        # 计算验证损失和特征间的距离
        loss, dists1, dists2, dists3 = compute_loss(anc_fea, pos_fea, neg_fea, type)
        # 获取损失值（转为Python数值）
        loss_value = loss.item()
        
        # 更新验证指标日志：记录当前批次的损失值
        test_metric_logger.update(loss=loss_value)
        # 更新三元组评估日志器：记录距离、损失和三元组类型（用于后续计算准确率）
        acc_logger.update(dists1, dists2, dists3, loss, type)
    
    # 汇总当前epoch的验证结果：计算平均损失和各类准确率
    avg_loss, res = acc_logger.summary()
    # 打印验证结果详情
    print(res)
    # 更新验证日志器的平均损失和整体准确率指标
    test_metric_logger.meters['loss_avg'].update(avg_loss, n=1)
    test_metric_logger.meters['overall_accuracy'].update(res[0], n=1)

    # 若结果包含多类别的准确率，更新对应类别的日志指标
    if len(res) > 1:
        test_metric_logger.meters['CLASS1_1_accuracy'].update(res[1], n=1)
        test_metric_logger.meters['CLASS1_2_accuracy'].update(res[2], n=1)
        test_metric_logger.meters['CLASS1_3_accuracy'].update(res[3], n=1)
    
    # 打印当前epoch的关键验证指标：整体准确率和平均损失
    print('* Overall Accuracy: {overall_accuracy.avg:.3f}  loss {loss_avg.global_avg:.3f}'
        .format(overall_accuracy = test_metric_logger.overall_accuracy, loss_avg = test_metric_logger.meters["loss_avg"]))
    
    # 若当前epoch是配置中指定的保存间隔（save_epoch），保存模型权重
    if epoch % config["save_epoch"] == 0:
        # 构建模型保存路径（包含epoch和准确率信息，便于追溯）
        save_path = os.path.join(config["checkpoint_dir"], "epoch_" + str(epoch) + "_acc_" + str(res[0]) + ".pth")
        # 保存模型状态字典
        torch.save(model.state_dict(), save_path)
    
    # 返回当前epoch的平均损失和验证结果
    return avg_loss, res

# 当该脚本作为主程序运行时（而非被导入为模块），执行以下代码
if __name__ == '__main__':
    # 创建命令行参数解析器（用于解析用户输入的配置文件路径）
    parser = argparse.ArgumentParser()
    # 添加命令行参数：--config，指定YAML配置文件路径，默认值为"configs/mae_train_expemb.yaml"
    parser.add_argument("--config", default="configs/mae_train_expemb.yaml")
    # 解析命令行参数，得到包含参数值的对象args
    args = parser.parse_args()
    # 从解析结果中获取配置文件路径
    yml_path = args.config
    # 调用read_yaml_to_dict函数读取YAML文件，将内容解析为Python字典（配置参数）
    config = read_yaml_to_dict(yml_path)

    # 调用主函数main，传入解析后的配置字典，启动训练流程
    main(config)