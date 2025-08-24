# 用训练好的模型提取验证集三元组样本（锚点 / 正例 / 负例）的高维特征，
# 通过 t-SNE 算法将高维特征降维到 2D 并绘制成散点图直观展示样本聚类效果，
# 同时计算三元组准确率、样本间平均距离等指标，
# 量化验证模型是否能让锚点与正例特征相近、与负例特征疏远。
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os

# 复用项目已有模块
from dataset import build_dataset, build_transform, FecData
from train import read_yaml_to_dict
from model.mae_pipeline import Pipeline
from torch.nn.modules.distance import PairwiseDistance
from utils.metrics import triplet_prediction_accuracy


def extract_triplet_features(model, dataloader, device):
    """提取取三元组特征提取函数
    功能：提取锚点、正例、负例的特征向量，并计算三者之间的距离
    返回：所有样本的特征、标签（0=锚点，1=正例，2=负例）、距离列表
    """
    model.eval()  # 设为评估模式，关闭 dropout 等训练特有的层
    all_features = []  # 存储所有样本的特征向量
    all_labels = []    # 存储所有样本的标签（区分锚点/正例/负例）
    all_dists = []     # 存储距离元组：(锚点-正例距离, 锚点-负例距离, 三元组类型, 正例-负例距离)

    # 初始化L2距离计算器（用于计算特征向量间的欧氏距离）
    l2_dist = PairwiseDistance(2)

    # 禁用梯度计算，节省显存并加速
    with torch.no_grad():
        for batch in dataloader:
            # 从批次数据中提取图像和三元组类型
            anc_img = batch["anc"].to(device)
            pos_img = batch["pos"].to(device)
            neg_img = batch["neg"].to(device)
            triplet_type = batch["type"]

            # 将三类图像在批次维度拼接，一次性输入模型提取特征（提高效率）
            batch_imgs = torch.cat((anc_img, pos_img, neg_img), dim=0)
            batch_features = model.forward(batch_imgs)

            # 将提取的特征按锚点、正例、负例拆分（每类样本数量等于批次大小）
            batch_size = anc_img.shape[0]
            anc_fea, pos_fea, neg_fea = torch.split(batch_features, batch_size, dim=0)

            # 计算三个关键距离（转为numpy数组便于后续处理）
            dist_anc_pos = l2_dist(anc_fea, pos_fea).cpu().numpy()  # 锚点-正例距离
            dist_anc_neg = l2_dist(anc_fea, neg_fea).cpu().numpy()  # 锚点-负例距离
            dist_pos_neg = l2_dist(pos_fea, neg_fea).cpu().numpy()  # 正例-负例距离

            # 存储当前批次的距离和类型信息
            all_dists.append((dist_anc_pos, dist_anc_neg, triplet_type, dist_pos_neg))

            # 收集特征和标签（每类样本数量为batch_size）
            all_features.extend(anc_fea.cpu().numpy())
            all_features.extend(pos_fea.cpu().numpy())
            all_features.extend(neg_fea.cpu().numpy())
            all_labels.extend([0]*batch_size + [1]*batch_size + [2]*batch_size)

    # 转换为numpy数组格式返回
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    return all_features, all_labels, all_dists


def visualize_triplet(features_2d, labels, save_path):
    """三元组特征可视化函数
    功能：使用t-SNE降维后的2D特征绘制散点图，用不同颜色区分锚点/正例/负例
    注意：图表注解使用英文，避免中文乱码
    """
    plt.figure(figsize=(12, 10))  # 设置图像大小

    # 绘制散点图：用颜色区分不同类型样本
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],  # t-SNE降维后的两个维度
        c=labels,                              # 按标签着色（0/1/2）
        cmap="viridis",                        # 颜色映射方案（区分度高）
        alpha=0.7,                             # 点的透明度（避免重叠遮挡）
        s=60                                   # 点的大小
    )

    # 添加图例（英文标签，确保显示正常）
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=["Anchor", "Positive", "Negative"],  # 英文标签：锚点、正例、负例
        fontsize=12
    )

    # 添加标题和坐标轴标签（英文）
    plt.title("Triplet Feature Visualization (t-SNE 2D Projection)", fontsize=14, pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=10)  # t-SNE维度1
    plt.ylabel("t-SNE Dimension 2", fontsize=10)  # t-SNE维度2

    # 保存图像（高清，确保标签完整显示）
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # 关闭图像，释放内存
    print(f"✅ 可视化图像已保存至：{save_path}")


def calculate_metrics(all_dists):
    """评估指标计算函数
    功能：计算三元组准确率、平均距离等指标，判断模型权重质量
    返回：整体准确率和距离差（锚点-负例 减 锚点-正例）
    """
    # 初始化存储各类距离的列表
    all_dist_anc_pos = []  # 锚点-正例距离列表
    all_dist_anc_neg = []  # 锚点-负例距离列表
    all_dist_pos_neg = []  # 正例-负例距离列表
    all_types = []         # 三元组类型列表

    # 从所有批次中收集距离和类型数据
    for dist_ap, dist_an, types, dist_pn in all_dists:
        all_dist_anc_pos.extend(dist_ap)
        all_dist_anc_neg.extend(dist_an)
        all_dist_pos_neg.extend(dist_pn)
        all_types.extend(types)

    # 计算整体三元组准确率（需传入三个距离参数）
    overall_acc = triplet_prediction_accuracy(
        np.array(all_dist_anc_pos),
        np.array(all_dist_anc_neg),
        np.array(all_dist_pos_neg)
    )

    # 计算各类平均距离
    avg_dist_ap = np.mean(all_dist_anc_pos)  # 锚点-正例平均距离
    avg_dist_an = np.mean(all_dist_anc_neg)  # 锚点-负例平均距离
    avg_dist_pn = np.mean(all_dist_pos_neg)  # 正例-负例平均距离
    dist_diff = avg_dist_an - avg_dist_ap    # 距离差（越大说明正负例区分越明显）

    # 按三元组类型计算准确率（分析不同类型样本的模型表现）
    type_acc_dict = {}
    unique_types = list(set(all_types))  # 获取所有独特的三元组类型
    for t in unique_types:
        # 筛选出当前类型的样本
        type_mask = [True if x == t else False for x in all_types]
        type_ap = np.array(all_dist_anc_pos)[type_mask]
        type_an = np.array(all_dist_anc_neg)[type_mask]
        type_pn = np.array(all_dist_pos_neg)[type_mask]
        # 计算该类型的准确率
        type_acc = triplet_prediction_accuracy(type_ap, type_an, type_pn)
        type_acc_dict[t] = type_acc

    # 打印指标汇总（中文说明，方便理解）
    print("\n" + "="*50)
    print("📊 权重质量评估指标")
    print("="*50)
    print(f"整体三元组准确率：{overall_acc:.4f}（越高越好，建议>0.8）")
    print(f"锚点-正例平均距离：{avg_dist_ap:.4f}（越小越好）")
    print(f"锚点-负例平均距离：{avg_dist_an:.4f}（越大越好）")
    print(f"正例-负例平均距离：{avg_dist_pn:.4f}（越大越好）")
    print(f"距离差（an - ap）：{dist_diff:.4f}（越大越好，建议>0.1）")
    # 打印各类型的准确率
    for t, acc in type_acc_dict.items():
        print(f"类型[{t}]的准确率：{acc:.4f}")
    print("="*50 + "\n")

    return overall_acc, dist_diff


def main(config_path):
    """主函数
    流程：加载配置 → 准备数据 → 加载模型 → 提取特征 → 可视化 → 计算指标
    """
    # 1. 加载配置文件（从YAML读取参数）
    config = read_yaml_to_dict(config_path)
    print(f"📌 成功加载配置文件：{config_path}")

    # 2. 检查配置文件中是否包含必要参数
    required_keys = ["weight_path", "save_fig_path", "val_csv", "val_img_path"]
    for key in required_keys:
        if key not in config:
            print(f"❌ 配置文件缺少必要参数：{key}")
            print(f"请在{config_path}中添加：{key}: '你的路径'")
            sys.exit(1)

    # 3. 配置计算设备（优先使用GPU，没有则用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用计算设备：{device}")

    # 4. 加载验证集数据
    val_transform = build_transform(is_train=False)  # 用验证模式的图像预处理
    val_dataset = FecData(
        csv_file=config["val_csv"],
        img_path=config["val_img_path"],
        transform=val_transform
    )
    # 创建数据加载器（批次大小取配置中的一半，避免显存不足）
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32) // 2,
        num_workers=config.get("num_workers", 4),
        shuffle=False  # 验证时不需要打乱数据
    )
    print(f"📥 加载验证集完成：共{len(val_dataset)}个有效三元组样本")

    # 5. 加载模型和训练好的权重
    model = Pipeline(config).to(device)  # 初始化模型并移到指定设备
    try:
        # 加载权重文件（自动处理CPU/GPU兼容）
        state_dict = torch.load(config["weight_path"], map_location=device)
        # 处理多GPU训练的权重（去除可能的"module."前缀）
        if next(iter(state_dict.keys())).startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"✅ 成功加载权重文件：{config['weight_path']}")
    except Exception as e:
        print(f"❌ 权重加载失败：{e}")
        sys.exit(1)

    # 6. 提取所有样本的特征
    print("\n🔍 开始提取特征...")
    all_features, all_labels, all_dists = extract_triplet_features(model, val_loader, device)
    print(f"✅ 特征提取完成：共{len(all_features)}个样本（每个三元组包含3个样本）")

    # 7. 使用t-SNE进行降维（高维特征→2D，便于可视化）
    print("\n🔄 正在进行t-SNE降维...")
    tsne = TSNE(
        n_components=2,    # 降维到2维
        random_state=42,   # 固定随机种子，确保结果可复现
        perplexity=30,     # t-SNE参数（影响聚类效果，建议20-50）
        n_iter=1000        # 迭代次数（越多效果越好，但耗时更长）
    )
    features_2d = tsne.fit_transform(all_features)  # 执行降维
    print("✅ t-SNE降维完成")

    # 8. 绘制并保存可视化图像
    visualize_triplet(features_2d, all_labels, config["save_fig_path"])

    # 9. 计算并输出评估指标
    calculate_metrics(all_dists)


if __name__ == "__main__":
    # 解析命令行参数（只需要配置文件路径）
    import argparse
    parser = argparse.ArgumentParser(description="三元组特征可视化与权重评估工具")
    parser.add_argument(
        "--config", 
        required=True,
        help="配置文件路径（例如：configs/mae_train_expemb.yaml）"
    )
    args = parser.parse_args()

    # 启动主流程
    main(args.config)
