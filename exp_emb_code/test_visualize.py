import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os

# 复用你已有的模块（确保脚本路径能找到dataset.py和train.py）
# 如果脚本和dataset.py不在同一目录，需要添加路径（示例：sys.path.append("/root/your_project_dir")）
from dataset import build_dataset, build_transform
from train import read_yaml_to_dict  # 复用配置读取函数
from model.mae_pipeline import Pipeline  # 复用模型结构
from torch.nn.modules.distance import PairwiseDistance
from utils.metrics import triplet_prediction_accuracy  # 复用三元组准确率计算


def extract_triplet_features(model, dataloader, device):
    """提取三元组（anc/pos/neg）的特征和标签"""
    model.eval()  # 模型设为评估模式
    all_features = []  # 存储所有特征（anc+pos+neg）
    all_labels = []    # 标签：0=锚点，1=正样本，2=负样本
    all_dists = []     # 存储距离（anc-pos, anc-neg）用于定量指标

    l2_dist = PairwiseDistance(2)  # 计算L2距离（和train.py一致）

    with torch.no_grad():  # 禁用梯度，节省显存
        for batch in dataloader:
            # 1. 加载批次数据（复用dataset.py输出的字典格式）
            anc_img = batch["anc"].to(device)
            pos_img = batch["pos"].to(device)
            neg_img = batch["neg"].to(device)
            triplet_type = batch["type"]  # 三元组类型（用于后续分析）

            # 2. 提取特征（和train.py逻辑一致：拼接图像一次性前向传播）
            batch_imgs = torch.cat((anc_img, pos_img, neg_img), dim=0)
            batch_features = model.forward(batch_imgs)

            # 3. 拆分特征（anc/pos/neg）
            batch_size = anc_img.shape[0]
            anc_fea, pos_fea, neg_fea = torch.split(batch_features, batch_size, dim=0)

            # 4. 计算距离（用于定量指标）
            dist_anc_pos = l2_dist(anc_fea, pos_fea).cpu().numpy()  # 锚点-正例距离
            dist_anc_neg = l2_dist(anc_fea, neg_fea).cpu().numpy()  # 锚点-负例距离
            all_dists.append( (dist_anc_pos, dist_anc_neg, triplet_type) )

            # 5. 收集特征和标签（给不同样本打标签）
            all_features.extend(anc_fea.cpu().numpy())  # 锚点：标签0
            all_features.extend(pos_fea.cpu().numpy())  # 正样本：标签1
            all_features.extend(neg_fea.cpu().numpy())  # 负样本：标签2
            all_labels.extend([0]*batch_size + [1]*batch_size + [2]*batch_size)

    # 转换为numpy数组（便于后续处理）
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    return all_features, all_labels, all_dists


def visualize_triplet(features_2d, labels, save_path):
    """绘制三元组特征的t-SNE可视化图"""
    plt.figure(figsize=(12, 10))
    # 绘制散点图：不同颜色代表不同样本类型
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=labels,          # 按标签着色（0=anc,1=pos,2=neg）
        cmap="viridis",    # 颜色映射（3种颜色清晰区分）
        alpha=0.7,         # 点的透明度（避免重叠遮挡）
        s=60               # 点的大小
    )
    # 添加图例和标题
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=["锚点(anc)", "正样本(pos)", "负样本(neg)"],
        fontsize=12
    )
    plt.title("三元组特征t-SNE可视化（降维至2D）", fontsize=14, pad=20)
    plt.xlabel("t-SNE维度1", fontsize=10)
    plt.ylabel("t-SNE维度2", fontsize=10)
    # 保存图片（高清）
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 可视化图已保存至：{save_path}")


def calculate_metrics(all_dists):
    """计算定量指标，判断权重好坏（和train.py的评估逻辑一致）"""
    all_dist_anc_pos = []
    all_dist_anc_neg = []
    all_types = []

    # 整理所有批次的距离和类型
    for dist_ap, dist_an, types in all_dists:
        all_dist_anc_pos.extend(dist_ap)
        all_dist_anc_neg.extend(dist_an)
        all_types.extend(types)

    # 1. 三元组准确率（核心指标）：锚点-正例距离 < 锚点-负例距离 的比例
    # （准确率越高，说明模型越能区分正负样本）
    overall_acc = triplet_prediction_accuracy(
        np.array(all_dist_anc_pos),
        np.array(all_dist_anc_neg)
    )

    # 2. 平均距离分析（辅助指标）
    avg_dist_ap = np.mean(all_dist_anc_pos)  # 锚点-正例平均距离（越小越好）
    avg_dist_an = np.mean(all_dist_anc_neg)  # 锚点-负例平均距离（越大越好）
    dist_diff = avg_dist_an - avg_dist_ap    # 距离差（越大越好，说明正负区分越明显）

    # 3. 按三元组类型统计准确率（可选，看不同类型的表现）
    type_acc_dict = {}
    unique_types = list(set(all_types))
    for t in unique_types:
        type_mask = [True if x == t else False for x in all_types]
        type_ap = np.array(all_dist_anc_pos)[type_mask]
        type_an = np.array(all_dist_anc_neg)[type_mask]
        type_acc = triplet_prediction_accuracy(type_ap, type_an)
        type_acc_dict[t] = type_acc

    # 输出指标汇总
    print("\n" + "="*50)
    print("📊 权重质量定量指标")
    print("="*50)
    print(f"整体三元组准确率：{overall_acc:.4f}（越高越好，建议>0.8）")
    print(f"锚点-正例平均距离：{avg_dist_ap:.4f}（越小越好）")
    print(f"锚点-负例平均距离：{avg_dist_an:.4f}（越大越好）")
    print(f"距离差（an - ap）：{dist_diff:.4f}（越大越好，建议>0.1）")
    for t, acc in type_acc_dict.items():
        print(f"类型[{t}]准确率：{acc:.4f}")
    print("="*50 + "\n")

    return overall_acc, dist_diff


def main(config_path, weight_path, save_fig_path="triplet_visualization.png"):
    """主函数：加载配置→加载数据→加载模型→提取特征→可视化→评估"""
    # 1. 加载配置文件（复用train.py的逻辑，保证参数一致）
    config = read_yaml_to_dict(config_path)
    print(f"📌 成功加载配置文件：{config_path}")

    # 2. 配置设备（GPU/CPU，和train.py一致）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备：{device}")

    # 3. 加载验证集（用验证模式的transform，避免数据增强影响特征）
    # 注意：这里用val模式的transform（无增强），和train.py的val逻辑一致
    val_transform = build_transform(is_train=False)  # 关键：禁用训练时的增强
    val_dataset = FecData(  # 直接用dataset.py的FecData类
        csv_file=config["val_csv"],
        img_path=config["val_img_path"],
        transform=val_transform
    )
    # 数据加载器（batch_size可设小些，避免显存不足）
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"] // 2,  # 显存不够可再减半
        num_workers=config["num_workers"],
        shuffle=False  # 验证不需要打乱
    )
    print(f"📥 加载验证集：{len(val_dataset)}个有效三元组样本")

    # 4. 加载模型和权重（和train.py的模型结构完全一致）
    model = Pipeline(config).to(device)
    # 加载权重（处理多GPU训练的情况）
    try:
        state_dict = torch.load(weight_path, map_location=device)
        # 如果训练时用了DataParallel，权重键会有"module."前缀，需要处理
        if next(iter(state_dict.keys())).startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"✅ 成功加载权重文件：{weight_path}")
    except Exception as e:
        print(f"❌ 权重加载失败：{e}")
        sys.exit(1)

    # 5. 提取三元组特征
    print("\n🔍 开始提取特征...")
    all_features, all_labels, all_dists = extract_triplet_features(model, val_loader, device)
    print(f"✅ 特征提取完成：共{len(all_features)}个样本（每个三元组3个样本）")

    # 6. t-SNE降维（高维特征→2D，便于可视化）
    print("\n🔄 正在用t-SNE降维...")
    tsne = TSNE(
        n_components=2,    # 降维到2维
        random_state=42,   # 固定随机种子，结果可复现
        perplexity=30,     # 推荐值（样本数多可设大些，如50）
        n_iter=1000        # 迭代次数，保证降维效果
    )
    features_2d = tsne.fit_transform(all_features)
    print("✅ t-SNE降维完成")

    # 7. 可视化并保存
    visualize_triplet(features_2d, all_labels, save_fig_path)

    # 8. 计算定量指标，判断权重好坏
    calculate_metrics(all_dists)


if __name__ == "__main__":
    # 命令行参数：方便切换不同权重文件
    import argparse
    parser = argparse.ArgumentParser(description="三元组特征可视化与权重评估")
    parser.add_argument(
        "--config", 
        required=True,
        help="配置文件路径（和train.py用同一个，如configs/mae_train_expemb.yaml）"
    )
    parser.add_argument(
        "--weight", 
        required=True,
        help="训练好的权重文件路径（如checkpoint/epoch_10_acc_0.85.pth）"
    )
    parser.add_argument(
        "--save_fig", 
        default="triplet_feature_vis.png",
        help="可视化图保存路径（默认：triplet_feature_vis.png）"
    )
    args = parser.parse_args()

    # 启动流程
    main(
        config_path=args.config,
        weight_path=args.weight,
        save_fig_path=args.save_fig
    )