"""
智润智慧农业套件 - 病害特征提取 + 小样本匹配模块

使用预训练 MobileNetV2 提取图像特征，建立病害特征库，
通过余弦相似度实现跨作物病害泛化识别。

特征维度: 1280 (MobileNetV2 最后一层全局平均池化)
匹配方式: 余弦相似度，阈值 0.65
"""

import os
import json
import time
import logging
import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

FEATURE_DIM = 1280
SIMILARITY_THRESHOLD = 0.4
INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DiseaseFeatureExtractor:
    """基于 MobileNetV2 的病害特征提取器"""

    def __init__(self, model_path: str, library_path: str):
        self._model_path = model_path
        self._library_path = library_path
        self._session = None
        self._model_loaded = False
        self._library: Dict[str, List[dict]] = {}  # disease_name -> [{feature, source, timestamp}]
        self._all_features: Optional[np.ndarray] = None  # (N, FEATURE_DIM)
        self._all_labels: List[str] = []
        self._all_sources: List[str] = []

    def begin(self) -> bool:
        """加载模型和特征库"""
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self._model_path,
                providers=['CPUExecutionProvider']
            )
            self._model_loaded = True
            logger.info(f"特征提取模型已加载: {self._model_path}")
        except Exception as e:
            logger.error(f"特征提取模型加载失败: {e}")
            return False

        self._load_library()
        self._rebuild_index()
        return True

    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """从图像提取 1280 维特征向量

        Args:
            image: BGR 格式 numpy 数组 (H, W, 3)

        Returns:
            1280 维 float32 特征向量，失败返回 None
        """
        if not self._model_loaded:
            return None

        # 预处理: BGR -> RGB, resize, normalize
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)  # add batch dim

        # 推理
        result = self._session.run(None, {'input': img})
        features = result[0][0]  # (1280,)

        # L2 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features.astype(np.float32)

    def match(self, image: np.ndarray, top_k: int = 3) -> List[dict]:
        """匹配图像到已知病害

        Args:
            image: BGR 格式 numpy 数组
            top_k: 返回前 k 个匹配结果

        Returns:
            [{disease, similarity, source, treatment}, ...]
        """
        features = self.extract_features(image)
        if features is None:
            return []

        if self._all_features is None or len(self._all_features) == 0:
            return []

        # 余弦相似度 (特征已 L2 归一化，点积即余弦相似度)
        similarities = self._all_features @ features

        # 取 top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            disease = self._all_labels[idx]
            logger.info(f"Top match: {disease} similarity={sim:.4f} (threshold={SIMILARITY_THRESHOLD})")
            if sim < SIMILARITY_THRESHOLD:
                continue
            # 从特征库获取治疗建议
            treatment = ""
            if disease in self._library and self._library[disease]:
                treatment = self._library[disease][0].get("treatment", "")

            results.append({
                "disease": disease,
                "similarity": round(sim, 4),
                "source": self._all_sources[idx],
                "treatment": treatment,
            })

        return results

    def register_disease(self, disease_name: str, images: List[np.ndarray],
                         treatment: str = "", source: str = "manual") -> int:
        """注册新病害到特征库

        Args:
            disease_name: 病害名称
            images: 图片列表 (BGR numpy 数组)
            treatment: 治疗建议
            source: 来源标识

        Returns:
            成功注册的图片数
        """
        if disease_name not in self._library:
            self._library[disease_name] = []

        count = 0
        for img in images:
            features = self.extract_features(img)
            if features is None:
                continue
            self._library[disease_name].append({
                "feature": features.tolist(),
                "treatment": treatment,
                "source": source,
                "timestamp": time.time(),
            })
            count += 1

        if count > 0:
            self._save_library()
            self._rebuild_index()

        return count

    def get_library_stats(self) -> dict:
        """获取特征库统计"""
        stats = {}
        for disease, entries in self._library.items():
            stats[disease] = len(entries)
        return {
            "diseases": stats,
            "total_diseases": len(self._library),
            "total_features": sum(len(v) for v in self._library.values()),
        }

    def _rebuild_index(self):
        """重建特征索引（合并所有特征为矩阵，加速匹配）"""
        all_features = []
        all_labels = []
        all_sources = []
        for disease, entries in self._library.items():
            for entry in entries:
                feat = np.array(entry["feature"], dtype=np.float32)
                all_features.append(feat)
                all_labels.append(disease)
                all_sources.append(entry.get("source", ""))

        if all_features:
            self._all_features = np.stack(all_features)
            # L2 归一化
            norms = np.linalg.norm(self._all_features, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self._all_features = self._all_features / norms
        else:
            self._all_features = None
        self._all_labels = all_labels
        self._all_sources = all_sources

    def _save_library(self):
        """保存特征库到文件"""
        dirname = os.path.dirname(self._library_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        try:
            with open(self._library_path, 'w', encoding='utf-8') as f:
                json.dump(self._library, f, ensure_ascii=False, indent=2)
            logger.info(f"特征库已保存: {len(self._library)} 种病害")
        except Exception as e:
            logger.warning(f"特征库保存失败: {e}")

    def _load_library(self):
        """从文件加载特征库"""
        if not os.path.exists(self._library_path):
            logger.info("特征库文件不存在，将创建新库")
            return
        try:
            with open(self._library_path, 'r', encoding='utf-8') as f:
                self._library = json.load(f)
            total = sum(len(v) for v in self._library.values())
            logger.info(f"特征库已加载: {len(self._library)} 种病害, {total} 条特征")
        except Exception as e:
            logger.warning(f"特征库加载失败: {e}")

    def to_dict(self) -> dict:
        """API 序列化"""
        stats = self.get_library_stats()
        return {
            "modelLoaded": self._model_loaded,
            "featureDim": FEATURE_DIM,
            "threshold": SIMILARITY_THRESHOLD,
            "library": stats,
        }


def build_feature_library(model_path: str, dataset_dir: str, output_path: str,
                          max_per_class: int = 10):
    """从数据集目录构建特征库

    数据集目录结构:
        dataset_dir/
            Tomato___Bacterial_spot/
                image1.jpg
                ...
            Tomato___Early_blight/
                ...

    Args:
        model_path: MobileNetV2 ONNX 模型路径
        dataset_dir: 数据集根目录
        output_path: 输出特征库 JSON 路径
        max_per_class: 每类最多取多少张图片
    """
    extractor = DiseaseFeatureExtractor(model_path, output_path)
    if not extractor.begin():
        print("模型加载失败")
        return

    # 病害中文名映射
    disease_cn = {
        "Tomato___Bacterial_spot": "番茄细菌性斑点病",
        "Tomato___Early_blight": "番茄早疫病",
        "Tomato___Late_blight": "番茄晚疫病",
        "Tomato___Leaf_Mold": "番茄叶霉病",
        "Tomato___Septoria_leaf_spot": "番茄叶斑病",
        "Tomato___Spider_mites": "番茄蜘蛛螨",
        "Tomato___Target_Spot": "番茄靶斑病",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "番茄黄化曲叶病毒",
        "Tomato___Tomato_mosaic_virus": "番茄花叶病毒",
        "Tomato___healthy": "番茄健康",
        "Potato___Early_blight": "马铃薯早疫病",
        "Potato___Late_blight": "马铃薯晚疫病",
        "Potato___healthy": "马铃薯健康",
        "Corn___Cercospora_leaf_spot": "玉米叶斑病",
        "Corn___Common_rust": "玉米普通锈病",
        "Corn___Northern_Leaf_Blight": "玉米北方叶枯病",
        "Corn___healthy": "玉米健康",
        "Grape___Black_rot": "葡萄黑腐病",
        "Grape___Esca": "葡萄埃斯卡病",
        "Grape___Leaf_blight": "葡萄叶枯病",
        "Grape___healthy": "葡萄健康",
        "Apple___Apple_scab": "苹果黑星病",
        "Apple___Black_rot": "苹果黑腐病",
        "Apple___Cedar_apple_rust": "苹果锈病",
        "Apple___healthy": "苹果健康",
        "Peach___Bacterial_spot": "桃细菌性斑点病",
        "Peach___healthy": "桃健康",
        "Pepper_bell___Bacterial_spot": "辣椒细菌性斑点病",
        "Pepper_bell___healthy": "辣椒健康",
        "Pepper___Bacterial_spot": "辣椒细菌性斑点病",
        "Pepper___healthy": "辣椒健康",
        "Strawberry___Leaf_scorch": "草莓叶灼病",
        "Strawberry___healthy": "草莓健康",
    }

    treatments = {
        "番茄细菌性斑点病": "清除病残体，铜制剂喷施，轮作",
        "番茄早疫病": "清除病叶，代森锰锌/百菌清喷施",
        "番茄晚疫病": "甲霜灵/霜脲氰喷施，降低湿度",
        "番茄叶霉病": "改善通风，嘧霉胺喷施",
        "番茄叶斑病": "清除病叶，代森联喷施",
        "番茄蜘蛛螨": "阿维菌素喷施，增加湿度",
        "番茄靶斑病": "苯醚甲环唑喷施",
        "番茄黄化曲叶病毒": "防治烟粉虱，拔除病株",
        "番茄花叶病毒": "防治蚜虫，拔除病株，种子消毒",
        "番茄健康": "保持通风，定期检查",
        "马铃薯早疫病": "代森锰锌喷施，合理施肥",
        "马铃薯晚疫病": "甲霜灵喷施，清除病株",
        "马铃薯健康": "保持通风，合理灌溉",
        "玉米叶斑病": "苯醚甲环唑喷施，轮作",
        "玉米普通锈病": "三唑酮喷施",
        "玉米北方叶枯病": "嘧菌酯喷施，抗病品种",
        "玉米健康": "合理密植，适时灌溉",
        "葡萄黑腐病": "清除病果，代森锰锌喷施",
        "葡萄埃斯卡病": "修剪病枝，伤口涂封",
        "葡萄叶枯病": "嘧菌酯喷施",
        "葡萄健康": "合理修剪，保持通风",
        "苹果黑星病": "代森锰锌/氟硅唑喷施",
        "苹果黑腐病": "清除病果，铜制剂喷施",
        "苹果锈病": "铲除转主寄主，三唑酮喷施",
        "苹果健康": "冬季清园，合理修剪",
        "桃细菌性斑点病": "铜制剂喷施，抗病品种",
        "桃健康": "合理修剪，冬季清园",
        "辣椒细菌性斑点病": "铜制剂喷施，种子消毒",
        "辣椒健康": "合理密植，保持通风",
        "草莓叶灼病": "清除病叶，降低叶面湿度",
        "草莓健康": "保持通风，定期检查",
    }

    total_registered = 0
    if not os.path.isdir(dataset_dir):
        print(f"数据集目录不存在: {dataset_dir}")
        return

    for class_dir in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        # 获取类名
        disease_name = disease_cn.get(class_dir, class_dir)
        treatment = treatments.get(disease_name, "")

        # 收集图片
        images = []
        for fname in os.listdir(class_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = os.path.join(class_path, fname)
            # Use np.fromfile to handle Chinese paths on Windows
            try:
                buf = np.fromfile(img_path, dtype=np.uint8)
                if buf.size == 0:
                    continue
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                continue
            if img is not None:
                images.append(img)
            if len(images) >= max_per_class:
                break

        if not images:
            continue

        # 注册
        count = extractor.register_disease(
            disease_name, images,
            treatment=treatment,
            source="PlantVillage"
        )
        total_registered += count
        print(f"  {disease_name}: {count} 张已注册")

    print(f"\n特征库构建完成: {len(extractor._library)} 种病害, {total_registered} 条特征")
    print(f"保存至: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python disease_feature_extractor.py <model_path> <dataset_dir> [output_path]")
        sys.exit(1)
    model_path = sys.argv[1]
    dataset_dir = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "disease_features.json"
    build_feature_library(model_path, dataset_dir, output_path, max_per_class=10)
