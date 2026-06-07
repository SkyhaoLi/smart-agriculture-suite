#!/usr/bin/env python3
"""
跨作物病害识别训练脚本

支持两种模式:
1. 合成数据模式 (默认): 用模拟特征快速训练, 验证架构
2. 真实数据模式: 用 PlantVillage 图片训练 (需先下载)

用法:
  python train_cross_crop.py                  # 合成数据训练
  python train_cross_crop.py --data-dir data/plantvillage  # 真实数据训练

训练完成后权重保存到 models/world_model.json
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_cross_crop")

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / "models"
MODEL_FILE = MODEL_DIR / "world_model.json"

CROP_NAMES = {0: "番茄", 1: "生菜", 2: "辣椒", 3: "黄瓜", 4: "草莓"}
DISEASE_NAMES = {0: "健康", 1: "炭疽病", 2: "灰霉病", 3: "叶灼病", 4: "白粉病"}

# PlantVillage 38类 → 世界模型映射
PLANTVILLAGE_MAPPING = {
    "Apple___Apple_scab":               (None, 1),
    "Apple___Black_rot":                (None, 1),
    "Apple___Cedar_apple_rust":         (None, 2),
    "Apple___healthy":                  (None, 0),
    "Blueberry___healthy":              (None, 0),
    "Cherry_(including_sour)___Powdery_mildew": (None, 4),
    "Cherry_(including_sour)___healthy":        (None, 0),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (None, 3),
    "Corn_(maize)___Common_rust_":               (None, 2),
    "Corn_(maize)___Northern_Leaf_Blight":       (None, 3),
    "Corn_(maize)___healthy":                    (None, 0),
    "Grape___Black_rot":                (None, 1),
    "Grape___Esca_(Black_Measles)":     (None, 2),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (None, 3),
    "Grape___healthy":                  (None, 0),
    "Orange___Haunglongbing_(Citrus_greening)": (None, 3),
    "Peach___Bacterial_spot":           (None, 1),
    "Peach___healthy":                  (None, 0),
    "Pepper,_bell___Bacterial_spot":    (2, 1),
    "Pepper,_bell___healthy":           (2, 0),
    "Potato___Early_blight":            (None, 2),
    "Potato___Late_blight":             (None, 2),
    "Potato___healthy":                 (None, 0),
    "Raspberry___healthy":              (None, 0),
    "Soybean___healthy":                (None, 0),
    "Squash___Powdery_mildew":          (None, 4),
    "Strawberry___Leaf_scorch":         (4, 3),
    "Strawberry___healthy":             (4, 0),
    "Tomato___Bacterial_spot":          (0, 1),
    "Tomato___Early_blight":            (0, 2),
    "Tomato___Late_blight":             (0, 2),
    "Tomato___Leaf_Mold":               (0, 3),
    "Tomato___Septoria_leaf_spot":      (0, 3),
    "Tomato___Spider_mites Two-spotted_spider_mite": (None, 3),
    "Tomato___Target_Spot":             (None, 3),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (None, 3),
    "Tomato___Tomato_mosaic_virus":      (None, 3),
    "Tomato___healthy":                 (0, 0),
}


# ============================================================================
# 合成数据生成
# ============================================================================

def generate_synthetic_dataset(n_samples=10000, seed=42):
    """
    生成模拟的跨作物病害特征数据

    设计思路: 每种病害有独特的特征模式, 不同作物的相同病害共享核心特征
    这模拟了 PlantVillage 中 "番茄灰霉病" 和 "辣椒灰霉病" 共享病害特征的情况
    """
    rng = np.random.RandomState(seed)

    features_list = []
    crop_ids = []
    disease_ids = []

    # 每种病害的基础特征向量 (128维)
    disease_base = {
        0: rng.randn(128) * 0.3,           # 健康: 低方差, 均匀分布
        1: rng.randn(128) * 0.5 + 0.5,     # 炭疽病: 偏暗, 高对比
        2: rng.randn(128) * 0.4 + 0.3,     # 灰霉病: 中等偏亮
        3: rng.randn(128) * 0.6 - 0.2,     # 叶灼病: 高方差, 偏暗
        4: rng.randn(128) * 0.3 + 0.8,     # 白粉病: 偏亮, 低方差
    }

    # 每种作物的颜色偏移 (模拟不同作物叶片颜色差异)
    crop_offset = {
        0: rng.randn(128) * 0.15,   # 番茄
        1: rng.randn(128) * 0.15,   # 生菜
        2: rng.randn(128) * 0.15,   # 辣椒
        3: rng.randn(128) * 0.15,   # 黄瓜
        4: rng.randn(128) * 0.15,   # 草莓
        -1: np.zeros(128),          # 未知作物 (用于跨作物学习)
    }

    # 为每种作物-病害组合生成样本
    # 已知作物 (有明确 crop_id)
    known_combos = [
        (0, 0), (0, 1), (0, 2), (0, 3),    # 番茄: 健康/炭疽/灰霉/叶灼
        (2, 0), (2, 1),                      # 辣椒: 健康/炭疽
        (4, 0), (4, 3),                      # 草莓: 健康/叶灼
    ]
    # 未知作物 (仅用于跨作物学习, crop_id=-1)
    unknown_combos = [
        (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4),  # 所有病害
    ]

    all_combos = known_combos + unknown_combos

    for crop_id, disease_id in all_combos:
        # 样本数量: 已知作物多采样, 未知作物少采样
        if crop_id >= 0:
            n = n_samples // len(known_combos)
        else:
            n = n_samples // len(unknown_combos) // 2

        for _ in range(n):
            # 特征 = 病害基础 + 作物偏移 + 噪声
            feat = disease_base[disease_id] + crop_offset[crop_id] + rng.randn(128) * 0.1
            features_list.append(feat)
            crop_ids.append(max(crop_id, 0))  # -1 映射到 0 (通用)
            disease_ids.append(disease_id)

    # 打乱
    indices = rng.permutation(len(features_list))
    return (
        np.array(features_list, dtype=np.float32)[indices],
        np.array(crop_ids, dtype=np.int32)[indices],
        np.array(disease_ids, dtype=np.int32)[indices],
    )


# ============================================================================
# 真实数据加载
# ============================================================================

def extract_features(img_bgr):
    """从图片提取128维特征向量"""
    import cv2
    img = cv2.resize(img_bgr, (96, 96))

    # HSV 颜色直方图 (48维)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-8)
    s_hist = s_hist / (s_hist.sum() + 1e-8)
    v_hist = v_hist / (v_hist.sum() + 1e-8)
    color_hist = np.concatenate([h_hist, s_hist, v_hist])

    # 纹理特征 (32维)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    texture = np.array([
        lap.mean(), lap.std(), np.percentile(lap, 25), np.percentile(lap, 75),
        mag.mean(), mag.std(), np.percentile(mag, 50), np.percentile(mag, 90),
    ])
    # Gabor
    gabor = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_64F)
        filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
        gabor.extend([filtered.mean(), filtered.std()])
    # 局部方差
    local_var = []
    for i in range(4):
        for j in range(4):
            patch = gray[i*24:(i+1)*24, j*24:(j+1)*24]
            local_var.append(patch.var())
    texture = np.concatenate([texture, np.array(gabor), np.array([np.mean(local_var), np.std(local_var)])])
    texture = np.pad(texture, (0, 32))[:32]

    # 颜色矩 (9维)
    color_moments = []
    for ch in range(3):
        c = hsv[:, :, ch].astype(np.float64)
        skew_raw = np.mean((c - c.mean())**3)
        skew = float(np.cbrt(skew_raw) / 255) if abs(skew_raw) > 1e-12 else 0.0
        color_moments.extend([c.mean()/255, c.std()/255, skew])

    # 病害区域特征 (39维)
    dark_ratio = (gray < 80).mean()
    bright_ratio = (gray > 200).mean()
    green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 50)
    green_ratio = green_mask.mean()
    edge_density = cv2.Canny(gray, 50, 150).mean() / 255
    channel_stats = []
    for ch in range(3):
        c = img[:, :, ch].astype(np.float64)
        channel_stats.extend([c.mean()/255, c.std()/255])
    sat = hsv[:,:,1].astype(np.float64)
    grid_means = []
    for i in range(3):
        for j in range(3):
            patch = hsv[i*32:(i+1)*32, j*32:(j+1)*32]
            grid_means.extend([patch[:,:,0].mean()/180, patch[:,:,1].mean()/255])
    disease_feat = np.array([dark_ratio, bright_ratio, green_ratio, edge_density] + channel_stats + [sat.mean()/255, sat.std()/255] + grid_means)
    disease_feat = np.pad(disease_feat, (0, 39))[:39]

    return np.concatenate([color_hist, texture, np.array(color_moments), disease_feat]).astype(np.float32)


def download_plantvillage(data_dir):
    """下载 PlantVillage 数据集"""
    data_dir = Path(data_dir)
    color_dir = data_dir / "color"

    # 检查是否已存在
    if color_dir.exists() and any(color_dir.iterdir()):
        logger.info(f"数据集已存在: {color_dir}")
        return True

    import subprocess
    import shutil

    repo_url = "https://github.com/spMohanty/PlantVillage-Dataset.git"
    clone_dir = data_dir / "_repo"

    logger.info(f"正在下载 PlantVillage 数据集...")
    logger.info(f"仓库: {repo_url}")
    logger.info("(首次约 500MB, 请耐心等待)")

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
            capture_output=True, text=True, timeout=900
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone 失败: {result.stderr[:500]}")

        # 移动 color 目录
        src = clone_dir / "raw" / "color"
        if src.exists():
            shutil.move(str(src), str(color_dir))
            logger.info(f"已移动到 {color_dir}")
        else:
            raise RuntimeError(f"未找到 {src}")

        # 清理
        shutil.rmtree(str(clone_dir), ignore_errors=True)
        logger.info("下载完成!")
        return True

    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.info("请手动下载:")
        logger.info(f"  git clone --depth 1 {repo_url}")
        logger.info(f"  然后将 raw/color/ 复制到 {color_dir}/")
        return False


def load_real_dataset(data_dir):
    """从磁盘加载 PlantVillage 图片"""
    import cv2

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
    features_list, crop_ids, disease_ids = [], [], []
    total = 0

    for class_dir in sorted(img_root.iterdir()):
        if not class_dir.is_dir():
            continue

        # 匹配映射
        crop_id, disease_id = None, None
        class_name = class_dir.name
        for pattern, (c, d) in PLANTVILLAGE_MAPPING.items():
            p = pattern.replace("___", "_").replace(",_", "_").replace(" ", "_").lower()
            n = class_name.replace("___", "_").replace(",_", "_").replace(" ", "_").lower()
            if p in n or n in p:
                crop_id, disease_id = c, d
                break

        if disease_id is None:
            parts = class_name.lower().split("___")
            if len(parts) >= 2:
                if "tomato" in parts[0]: crop_id = 0
                elif "pepper" in parts[0]: crop_id = 2
                elif "strawberry" in parts[0]: crop_id = 4
                d = parts[1]
                if "healthy" in d: disease_id = 0
                elif "bacterial" in d or "black" in d: disease_id = 1
                elif "blight" in d or "rust" in d or "mold" in d: disease_id = 2
                elif "scorch" in d or "spot" in d or "leaf_curl" in d: disease_id = 3
                elif "mildew" in d: disease_id = 4

        if disease_id is None:
            continue

        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                features_list.append(extract_features(img))
                crop_ids.append(crop_id if crop_id is not None else 0)
                disease_ids.append(disease_id)
                total += 1
                if total % 2000 == 0:
                    logger.info(f"  已处理 {total} 张...")
            except Exception:
                continue

    logger.info(f"共加载 {total} 张图片")
    if total == 0:
        return None
    return np.array(features_list, dtype=np.float32), np.array(crop_ids, dtype=np.int32), np.array(disease_ids, dtype=np.int32)


# ============================================================================
# 模型 (纯 numpy)
# ============================================================================

class ImageEncoder:
    """128维特征 → 32维潜在表示"""
    def __init__(self, input_dim=128, latent_dim=32):
        self.W1 = np.random.randn(input_dim, 64) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, latent_dim) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(latent_dim)

    def forward(self, x):
        self._x = x
        self._h1 = np.maximum(x @ self.W1 + self.b1, 0)
        self._out = np.tanh(self._h1 @ self.W2 + self.b2)
        return self._out

    def backward(self, grad_out, lr):
        grad_h2 = grad_out * (1 - self._out**2)
        self.W2 -= lr * (self._h1.T @ grad_h2 / len(grad_out))
        self.b2 -= lr * grad_h2.mean(axis=0)
        grad_h1 = (grad_h2 @ self.W2.T) * (self._h1 > 0)
        self.W1 -= lr * (self._x.T @ grad_h1 / len(grad_out))
        self.b1 -= lr * grad_h1.mean(axis=0)

    def get_params(self):
        return {"W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist()}


class CrossCropClassifier:
    """潜在表示 + 作物嵌入 → 病害分类"""
    def __init__(self, latent_dim=32, n_crops=5, n_diseases=5):
        self.n_crops = n_crops
        self.n_diseases = n_diseases
        self.crop_embed = np.random.randn(n_crops, 16) * 0.1
        inp = latent_dim + 16
        self.W1 = np.random.randn(inp, 64) * np.sqrt(2.0 / inp)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, n_diseases) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(n_diseases)
        self.env_W = np.random.randn(latent_dim, n_diseases) * 0.1
        self.env_b = np.zeros(n_diseases)

    def forward(self, latent, crop_ids):
        self._latent = latent
        self._crop_ids = crop_ids
        crop_emb = self.crop_embed[np.clip(crop_ids, 0, self.n_crops - 1)]
        self._inp = np.concatenate([latent, crop_emb], axis=1)
        self._h1 = np.maximum(self._inp @ self.W1 + self.b1, 0)
        logits_main = self._h1 @ self.W2 + self.b2
        logits_env = latent @ self.env_W + self.env_b
        self._logits = logits_main * 0.7 + logits_env * 0.3
        e = np.exp(self._logits - self._logits.max(axis=1, keepdims=True))
        self._probs = e / (e.sum(axis=1, keepdims=True) + 1e-8)
        return self._probs

    def backward(self, targets, lr):
        n = len(targets)
        grad_logits = (self._probs - targets) / n
        # env branch
        self.env_W -= lr * (self._latent.T @ (grad_logits * 0.3) / n)
        self.env_b -= lr * (grad_logits * 0.3).mean(axis=0)
        # main branch
        gl = grad_logits * 0.7
        self.W2 -= lr * (self._h1.T @ gl / n)
        self.b2 -= lr * gl.mean(axis=0)
        grad_h1 = (gl @ self.W2.T) * (self._h1 > 0)
        self.W1 -= lr * (self._inp.T @ grad_h1 / n)
        self.b1 -= lr * grad_h1.mean(axis=0)
        # crop embed
        grad_inp = grad_h1 @ self.W1.T
        grad_ce = grad_inp[:, 32:]
        for i in range(self.n_crops):
            mask = self._crop_ids == i
            if mask.any():
                self.crop_embed[i] -= lr * grad_ce[mask].mean(axis=0)

    def get_params(self):
        return {
            "crop_embed": self.crop_embed.tolist(),
            "disease_W1": self.W1.tolist(), "disease_b1": self.b1.tolist(),
            "disease_W2": self.W2.tolist(), "disease_b2": self.b2.tolist(),
            "classifier_W": self.W2.tolist(), "classifier_b": self.b2.tolist(),
            "env_disease_W": self.env_W.tolist(), "env_disease_b": self.env_b.tolist(),
        }


# ============================================================================
# 训练
# ============================================================================

def train(args):
    logger.info("=" * 60)
    logger.info("跨作物病害识别训练")
    logger.info("=" * 60)

    # 加载数据
    if args.data_dir:
        data_dir = Path(args.data_dir)
        # 检查数据是否已存在 (支持 color/ 或 raw/color/ 路径)
        has_data = False
        for check in [data_dir / "color", data_dir / "raw" / "color"]:
            if check.exists() and any(check.iterdir()):
                has_data = True
                break
        if not has_data:
            logger.info("数据集不存在, 尝试下载...")
            download_plantvillage(data_dir)

        logger.info(f"使用真实数据: {data_dir}")
        result = load_real_dataset(data_dir)
        if result is None:
            logger.error("真实数据加载失败, 回退到合成数据")
            features, crop_ids, disease_ids = generate_synthetic_dataset(args.samples)
        else:
            features, crop_ids, disease_ids = result
    else:
        logger.info(f"使用合成数据 (n={args.samples})")
        features, crop_ids, disease_ids = generate_synthetic_dataset(args.samples)

    logger.info(f"数据集: {len(features)} 样本")
    for d in range(5):
        logger.info(f"  {DISEASE_NAMES[d]}: {(disease_ids == d).sum()}")

    # 划分
    n = len(features)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, X_val = features[train_idx], features[val_idx]
    C_train, C_val = crop_ids[train_idx], crop_ids[val_idx]
    D_train, D_val = disease_ids[train_idx], disease_ids[val_idx]

    # one-hot
    def oh(ids, n=5):
        o = np.zeros((len(ids), n), dtype=np.float32)
        o[np.arange(len(ids)), ids] = 1.0
        return o
    D_train_oh, D_val_oh = oh(D_train), oh(D_val)

    # 标准化
    mu, sigma = X_train.mean(0), X_train.std(0) + 1e-8
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma

    # 模型
    enc = ImageEncoder(128, 32)
    cls = CrossCropClassifier(32, 5, 5)

    # 训练
    bs = args.batch_size
    lr = args.lr
    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    logger.info(f"训练: epochs={args.epochs}, bs={bs}, lr={lr}")
    logger.info("-" * 60)

    for epoch in range(args.epochs):
        perm = np.random.permutation(len(X_train))
        Xs, Cs, Ds = X_train[perm], C_train[perm], D_train_oh[perm]
        total_loss, correct, total, nb = 0, 0, 0, 0

        for i in range(0, len(Xs), bs):
            xb, cb, db = Xs[i:i+bs], Cs[i:i+bs], Ds[i:i+bs]
            latent = enc.forward(xb)
            probs = cls.forward(latent, cb)
            loss = -np.mean(np.sum(db * np.log(probs + 1e-8), axis=1))
            total_loss += loss
            correct += (probs.argmax(1) == db.argmax(1)).sum()
            total += len(xb)
            cls.backward(db, lr)
            grad_latent = ((probs - db) * 0.7 @ cls.env_W.T) + \
                          (((probs - db) @ cls.W2.T) * (cls._h1 > 0) @ cls.W1.T)[:, :32]
            enc.backward(grad_latent, lr)
            nb += 1

        tl, ta = total_loss / nb, correct / total
        # 验证
        vlatent = enc.forward(X_val)
        vprobs = cls.forward(vlatent, C_val)
        vl = -np.mean(np.sum(D_val_oh * np.log(vprobs + 1e-8), axis=1))
        va = (vprobs.argmax(1) == D_val_oh.argmax(1)).mean()

        history["train_loss"].append(float(tl))
        history["train_acc"].append(float(ta))
        history["val_loss"].append(float(vl))
        history["val_acc"].append(float(va))

        if va > best_acc:
            best_acc = va
        lr *= args.lr_decay

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{args.epochs} | Train L:{tl:.4f} A:{ta:.4f} | Val L:{vl:.4f} A:{va:.4f} | lr:{lr:.6f}")

    logger.info("-" * 60)
    logger.info(f"最佳验证准确率: {best_acc:.4f}")

    # 各作物/病害准确率
    vpred = vprobs.argmax(1)
    vtrue = D_val_oh.argmax(1)
    logger.info("\n各作物准确率:")
    for c in range(5):
        m = C_val == c
        if m.sum() > 10:
            logger.info(f"  {CROP_NAMES[c]}: {(vpred[m]==vtrue[m]).mean():.4f} ({m.sum()}样本)")
    logger.info("\n各病害准确率:")
    for d in range(5):
        m = vtrue == d
        if m.sum() > 10:
            logger.info(f"  {DISEASE_NAMES[d]}: {(vpred[m]==vtrue[m]).mean():.4f} ({m.sum()}样本)")

    # 保存
    logger.info("\n保存模型...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if MODEL_FILE.exists():
        with open(MODEL_FILE, 'r') as f:
            existing = json.load(f)

    disease_params = cls.get_params()
    disease_params["image_encoder"] = enc.get_params()
    disease_params["feature_mean"] = mu.tolist()
    disease_params["feature_std"] = sigma.tolist()
    disease_params["train_epochs"] = args.epochs
    disease_params["best_val_acc"] = float(best_acc)
    disease_params["train_samples"] = len(X_train)
    disease_params["data_mode"] = "synthetic" if args.data_dir is None else "real"

    existing["disease"] = disease_params
    existing["step_count"] = existing.get("step_count", 0) + args.epochs

    with open(MODEL_FILE, 'w') as f:
        json.dump(existing, f, indent=2)

    logger.info(f"模型已保存: {MODEL_FILE} ({MODEL_FILE.stat().st_size/1024:.1f} KB)")

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
    parser.add_argument("--data-dir", type=str, default=None, help="PlantVillage 数据目录")
    parser.add_argument("--samples", type=int, default=10000, help="合成数据样本数")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-decay", type=float, default=0.98)
    args = parser.parse_args()
    train(args)
