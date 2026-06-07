#!/usr/bin/env python3
"""
跨作物病害识别训练脚本 v2

改进:
1. 修正类别映射 - 去掉 crop_id=None 噪声类, 修正病害对应
2. 数据增强 - 随机翻转/旋转/颜色抖动
3. BatchNorm + Dropout 提升泛化
4. 类别加权 loss 解决不平衡
5. PyTorch ResNet18 特征提取

用法:
  python train_cross_crop.py --data-dir data/plantvillage --epochs 30
  python train_cross_crop.py --data-dir data/plantvillage --epochs 30 --finetune  # 微调backbone
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_cross_crop")

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / "models"
MODEL_FILE = MODEL_DIR / "world_model.json"

CROP_NAMES = {0: "番茄", 1: "生菜", 2: "辣椒", 3: "黄瓜", 4: "草莓"}
DISEASE_NAMES = {0: "健康", 1: "炭疽病", 2: "灰霉病", 3: "叶灼病", 4: "白粉病"}

# PlantVillage → 世界模型映射 (修正版)
# 只保留能准确映射的类别, 去掉 crop_id=None 的噪声
PLANTVILLAGE_MAPPING = {
    # 番茄 (crop_id=0)
    "Tomato___healthy":                 (0, 0),  # 健康
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (0, 3),  # 叶灼/卷叶
    "Tomato___Tomato_mosaic_virus":     (0, 3),  # 叶灼
    "Tomato___Leaf_Mold":               (0, 2),  # 灰霉/霉菌类
    "Tomato___Early_blight":            (0, 2),  # 灰霉/早疫
    "Tomato___Late_blight":             (0, 1),  # 炭疽/晚疫(坏死斑)
    "Tomato___Septoria_leaf_spot":      (0, 3),  # 叶灼/叶斑
    "Tomato___Bacterial_spot":          (0, 1),  # 炭疽/细菌性斑
    "Tomato___Target_Spot":             (0, 3),  # 叶灼/靶斑
    "Tomato___Spider_mites Two-spotted_spider_mite": (0, 3),  # 叶灼/虫害
    # 辣椒 (crop_id=2)
    "Pepper,_bell___healthy":           (2, 0),  # 健康
    "Pepper,_bell___Bacterial_spot":    (2, 1),  # 炭疽/细菌性斑
    # 草莓 (crop_id=4)
    "Strawberry___healthy":             (4, 0),  # 健康
    "Strawberry___Leaf_scorch":         (4, 3),  # 叶灼
    # 补充白粉病样本 (跨作物, crop_id=0 作为通用)
    "Cherry_(including_sour)___Powdery_mildew": (0, 4),  # 白粉病
    "Squash___Powdery_mildew":          (0, 4),  # 白粉病
}


# ============================================================================
# 数据集
# ============================================================================

class PlantVillageDataset(Dataset):
    """PlantVillage 数据集, 支持数据增强"""

    def __init__(self, image_paths, crop_ids, disease_ids, transform=None):
        self.image_paths = image_paths
        self.crop_ids = crop_ids
        self.disease_ids = disease_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.crop_ids[idx], self.disease_ids[idx]


def get_transforms(train=True, img_size=224):
    """数据增强 transforms"""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ============================================================================
# 数据加载
# ============================================================================

def load_dataset(data_dir):
    """加载 PlantVillage 数据集, 只保留可映射的类别"""
    data_dir = Path(data_dir)
    search_dirs = [data_dir / "color", data_dir / "raw" / "color", data_dir / "train", data_dir]
    img_root = None
    for d in search_dirs:
        if d.exists() and any(x.is_dir() for x in d.iterdir()):
            img_root = d
            break

    if img_root is None:
        logger.error(f"未找到数据集目录: {data_dir}")
        return None

    logger.info(f"扫描图片目录: {img_root}")
    image_paths, crop_ids, disease_ids = [], [], []
    skipped = 0

    for class_dir in sorted(img_root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        crop_id, disease_id = None, None

        # 精确匹配
        for pattern, (c, d) in PLANTVILLAGE_MAPPING.items():
            p = pattern.replace("___", "_").replace(",_", "_").replace(" ", "_").lower()
            n = class_name.replace("___", "_").replace(",_", "_").replace(" ", "_").lower()
            if p == n or p in n or n in p:
                crop_id, disease_id = c, d
                break

        # 跳过无法映射的类别
        if crop_id is None or disease_id is None:
            skipped += 1
            logger.info(f"  跳过: {class_name}")
            continue

        count = 0
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            image_paths.append(str(img_path))
            crop_ids.append(crop_id)
            disease_ids.append(disease_id)
            count += 1

        logger.info(f"  {class_name} → crop={CROP_NAMES[crop_id]}, disease={DISEASE_NAMES[disease_id]}, {count}张")

    logger.info(f"共加载 {len(image_paths)} 张图片, 跳过 {skipped} 个类别")
    if len(image_paths) == 0:
        return None

    return (np.array(image_paths),
            np.array(crop_ids, dtype=np.int64),
            np.array(disease_ids, dtype=np.int64))


# ============================================================================
# 模型
# ============================================================================

class DiseaseClassifier(nn.Module):
    """ResNet18 + BN + Dropout 分类器"""

    def __init__(self, n_crops=5, n_diseases=5, pretrained=True, freeze_backbone=True):
        super().__init__()

        # ResNet18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # 去掉最后的 fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 512, 1, 1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 作物嵌入
        self.crop_embed = nn.Embedding(n_crops, 16)

        # 分类头: 512 + 16 = 528 → 256 → n_diseases
        self.classifier = nn.Sequential(
            nn.Linear(512 + 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_diseases),
        )

        # 环境分支 (轻量)
        self.env_branch = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_diseases),
        )

    def forward(self, images, crop_ids):
        # backbone 特征
        feat = self.backbone(images).squeeze(-1).squeeze(-1)  # (B, 512)

        # 作物嵌入
        crop_emb = self.crop_embed(crop_ids)  # (B, 16)

        # 主分支
        x = torch.cat([feat, crop_emb], dim=1)  # (B, 528)
        logits_main = self.classifier(x)

        # 环境分支
        logits_env = self.env_branch(feat)

        # 融合
        logits = logits_main * 0.7 + logits_env * 0.3
        return logits

    def unfreeze_backbone(self, unfreeze_layers=2):
        """解冻 backbone 最后几层进行微调"""
        children = list(self.backbone.children())
        for layer in children[-unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        logger.info(f"已解冻 backbone 最后 {unfreeze_layers} 层")


# ============================================================================
# 训练
# ============================================================================

def compute_class_weights(disease_ids, n_classes=5):
    """计算类别权重 (反频率)"""
    counts = np.bincount(disease_ids, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.min()  # 归一化, 最小权重=1
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, crop_ids, labels in loader:
        images = images.to(device)
        crop_ids = crop_ids.to(device)
        labels = labels.to(device)

        logits = model(images, crop_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(images)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_crops = [], [], []

    for images, crop_ids, labels in loader:
        images = images.to(device)
        crop_ids = crop_ids.to(device)
        labels = labels.to(device)

        logits = model(images, crop_ids)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(images)

        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_crops.extend(crop_ids.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels), np.array(all_crops)


def export_to_numpy(model, device, img_size=224):
    """将 PyTorch 模型权重导出为 numpy 格式 (供 world_model.py 推理用)"""
    model.eval()

    params = {}

    # Backbone: 提取最后几层的特征 (用于 numpy 推理时的简化版本)
    # 我们导出 classifier 的权重
    clf = model.classifier
    params["classifier_W1"] = clf[0].weight.data.cpu().numpy().tolist()
    params["classifier_b1"] = clf[0].bias.data.cpu().numpy().tolist()
    params["classifier_bn1_mean"] = clf[1].running_mean.data.cpu().numpy().tolist()
    params["classifier_bn1_var"] = clf[1].running_var.data.cpu().numpy().tolist()
    params["classifier_bn1_weight"] = clf[1].weight.data.cpu().numpy().tolist()
    params["classifier_bn1_bias"] = clf[1].bias.data.cpu().numpy().tolist()

    params["classifier_W2"] = clf[4].weight.data.cpu().numpy().tolist()
    params["classifier_b2"] = clf[4].bias.data.cpu().numpy().tolist()
    params["classifier_bn2_mean"] = clf[5].running_mean.data.cpu().numpy().tolist()
    params["classifier_bn2_var"] = clf[5].running_var.data.cpu().numpy().tolist()
    params["classifier_bn2_weight"] = clf[5].weight.data.cpu().numpy().tolist()
    params["classifier_bn2_bias"] = clf[5].bias.data.cpu().numpy().tolist()

    params["classifier_W3"] = clf[8].weight.data.cpu().numpy().tolist()
    params["classifier_b3"] = clf[8].bias.data.cpu().numpy().tolist()

    # 环境分支
    env = model.env_branch
    params["env_W1"] = env[0].weight.data.cpu().numpy().tolist()
    params["env_b1"] = env[0].bias.data.cpu().numpy().tolist()
    params["env_W2"] = env[2].weight.data.cpu().numpy().tolist()
    params["env_b2"] = env[2].bias.data.cpu().numpy().tolist()

    # 作物嵌入
    params["crop_embed"] = model.crop_embed.weight.data.cpu().numpy().tolist()

    # Backbone 最后一层 (layer4) 的权重用于特征提取
    # 为了 numpy 推理, 我们用 ResNet18 的 avgpool 输出做简化
    # 实际推理时需要 PyTorch, 这里保存完整模型路径
    params["model_type"] = "resnet18_pytorch"
    params["img_size"] = 224
    params["fusion_weights"] = [0.7, 0.3]

    return params


def train(args):
    logger.info("=" * 60)
    logger.info("跨作物病害识别训练 v2 (PyTorch ResNet18)")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # 加载数据
    data_dir = Path(args.data_dir)
    result = load_dataset(data_dir)
    if result is None:
        logger.error("数据加载失败")
        return

    image_paths, crop_ids, disease_ids = result
    n_samples = len(image_paths)
    logger.info(f"数据集: {n_samples} 样本")

    # 类别统计
    for d in range(5):
        count = (disease_ids == d).sum()
        if count > 0:
            logger.info(f"  {DISEASE_NAMES[d]}: {count} 样本")

    # 类别权重
    class_weights = compute_class_weights(disease_ids, 5).to(device)
    logger.info(f"类别权重: {class_weights.cpu().numpy().round(2)}")

    # 划分 train/val (分层采样)
    np.random.seed(42)
    train_idx, val_idx = [], []
    for d in range(5):
        idx = np.where(disease_ids == d)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * 0.8)
        train_idx.extend(idx[:split])
        val_idx.extend(idx[split:])
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    logger.info(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}")

    # Dataset + DataLoader
    train_transform = get_transforms(train=True, img_size=224)
    val_transform = get_transforms(train=False, img_size=224)

    train_ds = PlantVillageDataset(
        image_paths[train_idx], crop_ids[train_idx], disease_ids[train_idx],
        transform=train_transform,
    )
    val_ds = PlantVillageDataset(
        image_paths[val_idx], crop_ids[val_idx], disease_ids[val_idx],
        transform=val_transform,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 模型
    model = DiseaseClassifier(
        n_crops=5, n_diseases=5,
        pretrained=True,
        freeze_backbone=not args.finetune,
    ).to(device)

    # 优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"可训练参数: {sum(p.numel() for p in trainable_params):,}")

    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 训练
    best_acc = 0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    logger.info("-" * 60)

    for epoch in range(args.epochs):
        # 微调: 在第10轮解冻 backbone
        if args.finetune and epoch == 10:
            model.unfreeze_backbone(unfreeze_layers=2)
            optimizer = optim.Adam([
                {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr},
                {"params": model.crop_embed.parameters(), "lr": args.lr},
                {"params": model.env_branch.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)

        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_preds, v_labels, v_crops = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch+1:3d}/{args.epochs} | "
                        f"Train L:{t_loss:.4f} A:{t_acc:.4f} | "
                        f"Val L:{v_loss:.4f} A:{v_acc:.4f} | lr:{lr_now:.6f}")

    logger.info("-" * 60)
    logger.info(f"最佳验证准确率: {best_acc:.4f}")

    # 加载最佳模型
    model.load_state_dict(best_state)

    # 最终评估
    v_loss, v_acc, v_preds, v_labels, v_crops = evaluate(model, val_loader, criterion, device)

    # 各作物准确率
    logger.info("\n各作物准确率:")
    for c in range(5):
        mask = v_crops == c
        if mask.sum() > 0:
            acc = (v_preds[mask] == v_labels[mask]).mean()
            logger.info(f"  {CROP_NAMES[c]}: {acc:.4f} ({mask.sum()} 样本)")

    # 各病害准确率
    logger.info("\n各病害准确率:")
    for d in range(5):
        mask = v_labels == d
        if mask.sum() > 0:
            acc = (v_preds[mask] == v_labels[mask]).mean()
            logger.info(f"  {DISEASE_NAMES[d]}: {acc:.4f} ({mask.sum()} 样本)")

    # 混淆矩阵
    logger.info("\n混淆矩阵:")
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(v_labels, v_preds):
        cm[t][p] += 1
    header = "       " + " ".join(f"{DISEASE_NAMES[i][:4]:>5}" for i in range(5))
    logger.info(header)
    for i in range(5):
        row = f"{DISEASE_NAMES[i][:6]:>6} " + " ".join(f"{cm[i][j]:5d}" for j in range(5))
        logger.info(row)

    # 保存模型
    logger.info("\n保存模型...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 保存 PyTorch 完整模型
    torch.save(best_state, MODEL_DIR / "disease_classifier.pth")

    # 导出 numpy 权重 (兼容 world_model.py)
    np_params = export_to_numpy(model, device)
    np_params["best_val_acc"] = float(best_acc)
    np_params["train_samples"] = len(train_idx)
    np_params["train_epochs"] = args.epochs
    np_params["data_mode"] = "real_pytorch"

    existing = {}
    if MODEL_FILE.exists():
        with open(MODEL_FILE, 'r') as f:
            existing = json.load(f)
    existing["disease"] = np_params
    existing["step_count"] = existing.get("step_count", 0) + args.epochs

    with open(MODEL_FILE, 'w') as f:
        json.dump(existing, f, indent=2)

    logger.info(f"PyTorch 模型: {MODEL_DIR / 'disease_classifier.pth'}")
    logger.info(f"JSON 模型: {MODEL_FILE} ({MODEL_FILE.stat().st_size / 1024:.1f} KB)")

    # 训练曲线
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["val_loss"], label="Val")
        ax1.set_title("Loss"); ax1.legend()
        ax2.plot(history["train_acc"], label="Train")
        ax2.plot(history["val_acc"], label="Val")
        ax2.set_title("Accuracy"); ax2.legend()
        plt.tight_layout()
        p = MODEL_DIR / "training_curve.png"
        plt.savefig(p)
        logger.info(f"训练曲线: {p}")
    except Exception as e:
        logger.warning(f"绘图失败: {e}")

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="PlantVillage 数据目录")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune", action="store_true", help="微调 backbone (更慢但更准)")
    args = parser.parse_args()
    train(args)
