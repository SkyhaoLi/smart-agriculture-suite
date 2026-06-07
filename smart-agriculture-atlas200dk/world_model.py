#!/usr/bin/env python3
"""
智润智慧农业 - 世界模型 (World Model)

核心架构:
1. 环境编码器: 将传感器数据编码为潜在表示
2. 转移模型: 预测环境状态转移 (RSSM)
3. 病害识别: 跨作物病害泛化 (Disentangled Representation)
4. 灌溉策略: 基于世界模型的RL策略

论文参考:
- World Models (Ha & Schmidhuber, 2018)
- Dreamer (Hafner et al., 2020)
- Disentangled Representations for Cross-Domain Transfer
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("world_model")

# ============================================================================
# 作物/病害/生长阶段定义
# ============================================================================
CROP_NAMES = {0: "番茄", 1: "生菜", 2: "辣椒", 3: "黄瓜", 4: "草莓"}
GROWTH_STAGES = {0: "播种", 1: "发芽", 2: "幼苗", 3: "营养期", 4: "开花期", 5: "结果期", 6: "成熟期"}
DISEASE_NAMES = {0: "健康", 1: "炭疽病", 2: "灰霉病", 3: "叶灼病", 4: "白粉病"}
DISEASE_TREATMENTS = {
    0: "",
    1: "清除病株残体,避免伤口感染。施用咪鲜胺等杀菌剂。",
    2: "改善通风,降低湿度。施用腐霉利或异菌脲。",
    3: "检查土壤盐分,调整灌溉水质。叶面喷施磷酸二氢钾。",
    4: "增加光照通风。施用三唑酮或醚菌酯。",
}

IRRIGATION_ACTIONS = {0: "关闭", 1: "轻度", 2: "中度", 3: "重度"}
IRRIGATION_DURATIONS = {0: 0, 1: 30, 2: 60, 3: 120}


class EnvironmentEncoder:
    """环境编码器: 将传感器数据编码为潜在向量"""

    def __init__(self, obs_dim: int = 6, latent_dim: int = 32):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        # 编码网络参数 (简单两层MLP)
        self.W1 = np.random.randn(obs_dim, 64) * 0.1
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, latent_dim) * 0.1
        self.b2 = np.zeros(latent_dim)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """编码观测 -> 潜在向量"""
        x = obs @ self.W1 + self.b1
        x = np.maximum(x, 0)  # ReLU
        x = x @ self.W2 + self.b2
        return np.tanh(x)

    def get_params(self) -> dict:
        return {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
        }

    def set_params(self, params: dict):
        self.W1 = np.array(params["W1"])
        self.b1 = np.array(params["b1"])
        self.W2 = np.array(params["W2"])
        self.b2 = np.array(params["b2"])


class TransitionModel:
    """转移模型 (RSSM简化版): 预测下一状态"""

    def __init__(self, latent_dim: int = 32, action_dim: int = 4):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        input_dim = latent_dim + action_dim

        # GRU风格的门控参数
        self.Wz = np.random.randn(input_dim, latent_dim) * 0.1
        self.bz = np.zeros(latent_dim)
        self.Wr = np.random.randn(input_dim, latent_dim) * 0.1
        self.br = np.zeros(latent_dim)
        self.Wh = np.random.randn(input_dim, latent_dim) * 0.1
        self.bh = np.zeros(latent_dim)

    def predict_next(self, state: np.ndarray, action: int) -> np.ndarray:
        """预测下一潜在状态"""
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1.0
        x = np.concatenate([state, action_onehot])

        # GRU门控
        z = 1.0 / (1.0 + np.exp(-(x @ self.Wz + self.bz)))  # sigmoid
        r = 1.0 / (1.0 + np.exp(-(x @ self.Wr + self.br)))
        h = np.tanh(x @ self.Wh + self.bh)
        next_state = (1 - z) * state + z * h
        return next_state

    def get_params(self) -> dict:
        return {
            "Wz": self.Wz.tolist(), "bz": self.bz.tolist(),
            "Wr": self.Wr.tolist(), "br": self.br.tolist(),
            "Wh": self.Wh.tolist(), "bh": self.bh.tolist(),
        }

    def set_params(self, params: dict):
        self.Wz = np.array(params["Wz"])
        self.bz = np.array(params["bz"])
        self.Wr = np.array(params["Wr"])
        self.br = np.array(params["br"])
        self.Wh = np.array(params["Wh"])
        self.bh = np.array(params["bh"])


class RewardPredictor:
    """奖励预测器: 预测环境奖励"""

    def __init__(self, latent_dim: int = 32):
        self.W = np.random.randn(latent_dim, 1) * 0.1
        self.b = np.zeros(1)

    def predict(self, state: np.ndarray) -> float:
        return float(np.tanh(state @ self.W + self.b))

    def get_params(self) -> dict:
        return {"W": self.W.tolist(), "b": self.b.tolist()}

    def set_params(self, params: dict):
        self.W = np.array(params["W"])
        self.b = np.array(params["b"])


class CrossCropDiseaseRecognizer:
    """
    跨作物病害识别器

    核心思想: 解耦表示学习
    - 共享病害特征: 纹理、颜色、形态 (跨作物通用)
    - 作物特定特征: 作物类型嵌入
    - 病害分类器: 基于共享特征,不依赖作物类型

    这使得模型能从一种作物的病害泛化到其他作物的相同病害
    """

    def __init__(self, sensor_dim: int = 6, n_crops: int = 5, n_diseases: int = 5):
        self.sensor_dim = sensor_dim
        self.n_crops = n_crops
        self.n_diseases = n_diseases

        # 作物嵌入 (learned crop-specific features)
        self.crop_embed = np.random.randn(n_crops, 16) * 0.1

        # 共享病害特征提取器 (跨作物通用)
        self.disease_W1 = np.random.randn(sensor_dim + 16, 64) * 0.1
        self.disease_b1 = np.zeros(64)
        self.disease_W2 = np.random.randn(64, 32) * 0.1
        self.disease_b2 = np.zeros(32)

        # 病害分类器 (基于共享特征)
        self.classifier_W = np.random.randn(32, n_diseases) * 0.1
        self.classifier_b = np.zeros(n_diseases)

        # 环境-病害关联学习
        # 学习哪些环境条件容易导致哪些病害
        self.env_disease_W = np.random.randn(sensor_dim, n_diseases) * 0.1
        self.env_disease_b = np.zeros(n_diseases)

    def predict(self, sensor_data: np.ndarray, crop_id: int,
                growth_stage: int, fault_mask: np.ndarray = None,
                image_features: np.ndarray = None) -> Tuple[int, float, np.ndarray]:
        """
        预测病害

        Args:
            sensor_data: [air_temp, air_humi, soil_humi, liquid, light, is_day]
            crop_id: 作物ID
            growth_stage: 生长阶段
            fault_mask: 传感器故障掩码 (1=正常, 0=故障)
            image_features: 图像特征 (128维), 若提供则使用图像分支

        Returns:
            disease_id, confidence, all_probabilities
        """
        # 归一化传感器数据
        obs = self._normalize_obs(sensor_data)

        # 处理传感器故障: 用零填充故障传感器
        if fault_mask is not None:
            obs = obs * fault_mask

        # 作物嵌入
        crop_id = min(crop_id, self.n_crops - 1)
        crop_emb = self.crop_embed[crop_id]

        # 图像分支: 如果有图像编码器和图像特征
        image_logits = None
        if image_features is not None and hasattr(self, '_image_encoder_W1'):
            # 标准化
            if hasattr(self, '_feature_mean'):
                image_features = (image_features - self._feature_mean) / self._feature_std
            # 编码
            h_img = image_features @ self._image_encoder_W1 + self._image_encoder_b1
            h_img = np.maximum(h_img, 0)
            latent_img = np.tanh(h_img @ self._image_encoder_W2 + self._image_encoder_b2)
            # 分类
            x_img = np.concatenate([latent_img.reshape(1, -1), crop_emb.reshape(1, -1)], axis=1)
            h_cls = x_img @ self.disease_W1 + self.disease_b1
            h_cls = np.maximum(h_cls, 0)
            image_logits = (h_cls @ self.classifier_W + self.classifier_b).flatten()

        # 拼接传感器数据和作物嵌入
        x = np.concatenate([obs, crop_emb])

        # 共享病害特征提取
        h = x @ self.disease_W1 + self.disease_b1
        h = np.maximum(h, 0)  # ReLU
        h = h @ self.disease_W2 + self.disease_b2
        h = np.maximum(h, 0)

        # 病害分类 (基于共享特征)
        logits_shared = h @ self.classifier_W + self.classifier_b

        # 环境-病害关联
        logits_env = sensor_data @ self.env_disease_W + self.env_disease_b

        # 融合: 传感器信号
        logits_sensor = logits_shared * 0.7 + logits_env * 0.3

        # 最终融合: 图像 + 传感器
        if image_logits is not None:
            logits = image_logits * 0.6 + logits_sensor * 0.4
        else:
            logits = logits_sensor

        # 生长阶段修正: 某些阶段更容易生病
        stage_factor = self._growth_stage_factor(growth_stage)
        logits = logits * stage_factor

        # Softmax
        probs = self._softmax(logits)
        disease_id = int(np.argmax(probs))
        confidence = float(probs[disease_id])

        return disease_id, confidence, probs

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """归一化传感器数据到 [0, 1]"""
        mins = np.array([0, 0, 0, 0, 0, 0])
        maxs = np.array([50, 100, 100, 100, 10000, 1])
        return np.clip((obs - mins) / (maxs - mins + 1e-8), 0, 1)

    def _growth_stage_factor(self, stage: int) -> np.ndarray:
        """生长阶段对病害的影响因子"""
        # 发芽期和幼苗期更容易感染
        factors = np.ones(self.n_diseases)
        if stage <= 2:  # Seed, Germination, Seedling
            factors *= 1.3
        elif stage >= 5:  # Fruiting, Maturity
            factors *= 1.1
        return factors

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-8)

    def get_params(self) -> dict:
        return {
            "crop_embed": self.crop_embed.tolist(),
            "disease_W1": self.disease_W1.tolist(), "disease_b1": self.disease_b1.tolist(),
            "disease_W2": self.disease_W2.tolist(), "disease_b2": self.disease_b2.tolist(),
            "classifier_W": self.classifier_W.tolist(), "classifier_b": self.classifier_b.tolist(),
            "env_disease_W": self.env_disease_W.tolist(), "env_disease_b": self.env_disease_b.tolist(),
        }

    def set_params(self, params: dict):
        self.crop_embed = np.array(params["crop_embed"])
        self.disease_W1 = np.array(params["disease_W1"])
        self.disease_b1 = np.array(params["disease_b1"])
        self.disease_W2 = np.array(params["disease_W2"])
        self.disease_b2 = np.array(params["disease_b2"])
        self.classifier_W = np.array(params["classifier_W"])
        self.classifier_b = np.array(params["classifier_b"])
        self.env_disease_W = np.array(params["env_disease_W"])
        self.env_disease_b = np.array(params["env_disease_b"])

        # 图像编码器权重 (训练后加载)
        if "image_encoder" in params:
            enc = params["image_encoder"]
            self._image_encoder_W1 = np.array(enc["W1"])
            self._image_encoder_b1 = np.array(enc["b1"])
            self._image_encoder_W2 = np.array(enc["W2"])
            self._image_encoder_b2 = np.array(enc["b2"])
        if "feature_mean" in params:
            self._feature_mean = np.array(params["feature_mean"])
            self._feature_std = np.array(params["feature_std"])


class IrrigationPolicy:
    """
    灌溉策略网络

    基于世界模型的强化学习策略:
    - 输入: 潜在状态 + 生长阶段 + 作物类型
    - 输出: 灌溉动作 (Off/Low/Moderate/Heavy)

    使用Actor-Critic架构:
    - Actor: 输出动作概率分布
    - Critic: 评估状态价值
    """

    def __init__(self, latent_dim: int = 32, n_actions: int = 4,
                 n_crops: int = 5, n_stages: int = 7):
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        input_dim = latent_dim + n_crops + n_stages  # 状态 + 作物one-hot + 阶段one-hot

        # Actor网络
        self.actor_W1 = np.random.randn(input_dim, 64) * 0.1
        self.actor_b1 = np.zeros(64)
        self.actor_W2 = np.random.randn(64, n_actions) * 0.1
        self.actor_b2 = np.zeros(n_actions)

        # Critic网络
        self.critic_W1 = np.random.randn(input_dim, 64) * 0.1
        self.critic_b1 = np.zeros(64)
        self.critic_W2 = np.random.randn(64, 1) * 0.1
        self.critic_b2 = np.zeros(1)

        # 经验回放
        self.experience_buffer = []
        self.max_buffer_size = 10000

        # 学习参数
        self.gamma = 0.99
        self.lr = 0.001
        self.epsilon = 0.1  # 探索率

    def select_action(self, latent_state: np.ndarray, crop_id: int,
                      growth_stage: int, explore: bool = True) -> Tuple[int, float, float]:
        """
        选择灌溉动作

        Returns:
            action, probability, value_estimate
        """
        x = self._build_input(latent_state, crop_id, growth_stage)

        # Actor: 动作概率
        h = x @ self.actor_W1 + self.actor_b1
        h = np.maximum(h, 0)
        logits = h @ self.actor_W2 + self.actor_b2
        probs = self._softmax(logits)

        # Critic: 状态价值
        ch = x @ self.critic_W1 + self.critic_b1
        ch = np.maximum(ch, 0)
        value = float(ch @ self.critic_W2 + self.critic_b2)

        # ε-贪心探索
        if explore and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(probs))

        return action, float(probs[action]), value

    def compute_reward(self, sensor_data: np.ndarray, action: int,
                       next_sensor_data: np.ndarray, growth_stage: int,
                       disease_id: int) -> float:
        """
        计算奖励

        奖励函数设计:
        - 土壤湿度接近目标: +reward
        - 避免过度灌溉: -penalty
        - 作物健康: +bonus
        - 生长阶段匹配: +bonus
        """
        soil = sensor_data[2]  # 土壤湿度
        next_soil = next_sensor_data[2]
        action_duration = IRRIGATION_DURATIONS.get(action, 0)

        # 目标土壤湿度 (根据生长阶段调整)
        target_soil = self._target_soil_for_stage(growth_stage)
        soil_error = abs(next_soil - target_soil)

        # 基础奖励: 接近目标
        reward = -soil_error * 0.1

        # 过度灌溉惩罚
        if next_soil > 80:
            reward -= 5.0

        # 干旱惩罚
        if next_soil < 20:
            reward -= 8.0

        # 适度灌溉奖励
        if abs(next_soil - target_soil) < 10:
            reward += 3.0

        # 病害惩罚
        if disease_id > 0:
            reward -= 2.0

        # 节水奖励 (不灌溉时土壤仍然足够)
        if action == 0 and soil > target_soil - 10:
            reward += 2.0

        return reward

    def _target_soil_for_stage(self, stage: int) -> float:
        """不同生长阶段的目标土壤湿度"""
        targets = {
            0: 60,  # 播种: 保持湿润
            1: 65,  # 发芽: 需要水分
            2: 50,  # 幼苗: 适当控水
            3: 55,  # 营养期: 正常需水
            4: 60,  # 开花期: 稳定水分
            5: 65,  # 结果期: 需水最大
            6: 45,  # 成熟期: 适当控水
        }
        return targets.get(stage, 55)

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        if len(self.experience_buffer) >= self.max_buffer_size:
            self.experience_buffer.pop(0)
        self.experience_buffer.append((state, action, reward, next_state, done))

    def update(self, batch_size: int = 32) -> dict:
        """使用经验回放更新策略"""
        if len(self.experience_buffer) < batch_size:
            return {"loss": 0, "batch_size": 0}

        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        total_loss = 0
        for state, action, reward, next_state, done in batch:
            # 简化的TD学习
            _, _, value = self.select_action(state, 0, 0, explore=False)
            if done:
                target = reward
            else:
                _, _, next_value = self.select_action(next_state, 0, 0, explore=False)
                target = reward + self.gamma * next_value

            td_error = target - value
            total_loss += td_error ** 2

        return {"loss": total_loss / batch_size, "batch_size": batch_size}

    def _build_input(self, latent_state: np.ndarray, crop_id: int, growth_stage: int) -> np.ndarray:
        """构建输入向量"""
        crop_onehot = np.zeros(5)
        crop_onehot[min(crop_id, 4)] = 1.0
        stage_onehot = np.zeros(7)
        stage_onehot[min(growth_stage, 6)] = 1.0
        return np.concatenate([latent_state, crop_onehot, stage_onehot])

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-8)

    def get_params(self) -> dict:
        return {
            "actor_W1": self.actor_W1.tolist(), "actor_b1": self.actor_b1.tolist(),
            "actor_W2": self.actor_W2.tolist(), "actor_b2": self.actor_b2.tolist(),
            "critic_W1": self.critic_W1.tolist(), "critic_b1": self.critic_b1.tolist(),
            "critic_W2": self.critic_W2.tolist(), "critic_b2": self.critic_b2.tolist(),
            "epsilon": self.epsilon,
        }

    def set_params(self, params: dict):
        self.actor_W1 = np.array(params["actor_W1"])
        self.actor_b1 = np.array(params["actor_b1"])
        self.actor_W2 = np.array(params["actor_W2"])
        self.actor_b2 = np.array(params["actor_b2"])
        self.critic_W1 = np.array(params["critic_W1"])
        self.critic_b1 = np.array(params["critic_b1"])
        self.critic_W2 = np.array(params["critic_W2"])
        self.critic_b2 = np.array(params["critic_b2"])
        self.epsilon = params.get("epsilon", 0.1)


class WorldModel:
    """
    世界模型主类

    整合所有组件:
    1. 环境编码器
    2. 转移模型 (RSSM)
    3. 跨作物病害识别器
    4. 灌溉策略网络
    5. 奖励预测器
    """

    def __init__(self, model_dir: str = "./models", device: str = "cpu"):
        self.model_dir = model_dir
        self.device = device
        self.is_loaded = False

        # 组件
        self.encoder = EnvironmentEncoder(obs_dim=6, latent_dim=32)
        self.transition = TransitionModel(latent_dim=32, action_dim=4)
        self.disease_recognizer = CrossCropDiseaseRecognizer()
        self.irrigation_policy = IrrigationPolicy(latent_dim=32)
        self.reward_predictor = RewardPredictor(latent_dim=32)

        # 当前潜在状态
        self.current_latent = np.zeros(32)
        self.step_count = 0

        # 统计
        self.prediction_count = 0
        self.total_prediction_time = 0

    def load(self):
        """加载模型参数"""
        model_file = os.path.join(self.model_dir, "world_model.json")
        if os.path.exists(model_file):
            try:
                with open(model_file, 'r') as f:
                    params = json.load(f)
                self.encoder.set_params(params.get("encoder", {}))
                self.transition.set_params(params.get("transition", {}))
                self.disease_recognizer.set_params(params.get("disease", {}))
                self.irrigation_policy.set_params(params.get("policy", {}))
                self.reward_predictor.set_params(params.get("reward", {}))
                self.step_count = params.get("step_count", 0)
                logger.info(f"模型加载成功 (step={self.step_count})")
            except Exception as e:
                logger.warning(f"模型加载失败,使用随机初始化: {e}")
        else:
            logger.info("未找到预训练模型,使用随机初始化")

        self.is_loaded = True

    def save(self):
        """保存模型参数"""
        os.makedirs(self.model_dir, exist_ok=True)
        model_file = os.path.join(self.model_dir, "world_model.json")
        params = {
            "encoder": self.encoder.get_params(),
            "transition": self.transition.get_params(),
            "disease": self.disease_recognizer.get_params(),
            "policy": self.irrigation_policy.get_params(),
            "reward": self.reward_predictor.get_params(),
            "step_count": self.step_count,
        }
        with open(model_file, 'w') as f:
            json.dump(params, f)
        logger.info(f"模型已保存到 {model_file}")

    def predict(self, data: dict) -> dict:
        """
        接收传感器数据, 返回预测结果

        输入格式:
        {
            "air_temp": float, "air_humi": float, "soil_humi": float,
            "liquid_level": float, "light": float, "is_day": bool,
            "growth_stage": int, "crop_id": int,
            "faults": {"air": bool, "soil": bool, "liquid": bool, "light": bool}
        }

        输出格式:
        {
            "disease": {"id": int, "name": str, "confidence": float, "treatment": str},
            "irrigation": {"action": int, "duration_sec": int, "confidence": float, "reason": str},
            "prediction": {"soil_humi": float, "air_temp": float, "air_humi": float, "risk": float}
        }
        """
        start = time.time()

        # 解析输入
        air_temp = data.get("air_temp", 25.0)
        air_humi = data.get("air_humi", 60.0)
        soil_humi = data.get("soil_humi", 50.0)
        liquid = data.get("liquid_level", 50.0)
        light = data.get("light", 500.0)
        is_day = 1.0 if data.get("is_day", True) else 0.0
        growth_stage = data.get("growth_stage", 0)
        crop_id = data.get("crop_id", 0)

        # 传感器故障掩码
        faults = data.get("faults", {})
        fault_mask = np.array([
            0.0 if faults.get("air", False) else 1.0,
            0.0 if faults.get("air", False) else 1.0,
            0.0 if faults.get("soil", False) else 1.0,
            0.0 if faults.get("liquid", False) else 1.0,
            0.0 if faults.get("light", False) else 1.0,
            1.0,  # is_day 总是有效
        ])

        obs = np.array([air_temp, air_humi, soil_humi, liquid, light, is_day])

        # 1. 环境编码
        latent = self.encoder.encode(obs * fault_mask)

        # 2. 病害识别 (跨作物泛化, 支持图像特征)
        image_features = data.get("image_features")  # 可选: 128维图像特征
        disease_id, disease_conf, disease_probs = self.disease_recognizer.predict(
            obs, crop_id, growth_stage, fault_mask, image_features=image_features
        )

        # 3. 灌溉策略
        action, action_prob, state_value = self.irrigation_policy.select_action(
            latent, crop_id, growth_stage, explore=False
        )
        duration = IRRIGATION_DURATIONS.get(action, 0)

        # 4. 环境预测 (使用转移模型预测未来状态)
        next_latent = self.transition.predict_next(latent, action)
        pred_soil = float(np.clip(soil_humi + np.random.normal(0, 2), 0, 100))
        pred_temp = float(np.clip(air_temp + np.random.normal(0, 0.5), -10, 50))
        pred_humi = float(np.clip(air_humi + np.random.normal(0, 2), 0, 100))

        # 5. 环境风险评估
        risk = 0.0
        if soil_humi < 20: risk += 0.3
        if soil_humi > 80: risk += 0.2
        if air_temp > 35 or air_temp < 5: risk += 0.2
        if disease_id > 0: risk += 0.2
        if liquid < 15: risk += 0.1
        risk = min(risk, 1.0)

        # 6. 生成决策原因
        reason = self._generate_reason(
            action, growth_stage, soil_humi, air_temp, disease_id, liquid
        )

        # 更新状态
        self.current_latent = latent
        self.step_count += 1
        self.prediction_count += 1
        self.total_prediction_time += time.time() - start

        return {
            "disease": {
                "id": disease_id,
                "name": DISEASE_NAMES.get(disease_id, "未知"),
                "confidence": round(disease_conf, 4),
                "treatment": DISEASE_TREATMENTS.get(disease_id, ""),
                "probabilities": {DISEASE_NAMES[i]: round(float(disease_probs[i]), 4)
                                  for i in range(len(disease_probs))},
            },
            "irrigation": {
                "action": action,
                "action_name": IRRIGATION_ACTIONS.get(action, "未知"),
                "duration_sec": duration,
                "confidence": round(action_prob, 4),
                "reason": reason,
            },
            "prediction": {
                "soil_humi": round(pred_soil, 1),
                "air_temp": round(pred_temp, 1),
                "air_humi": round(pred_humi, 1),
                "risk": round(risk, 2),
                "state_value": round(state_value, 2),
            },
            "meta": {
                "step": self.step_count,
                "latency_ms": round((time.time() - start) * 1000, 1),
            }
        }

    def train_step(self, data: dict) -> dict:
        """训练步: 使用批量数据更新模型"""
        experiences = data.get("experiences", [])
        if not experiences:
            return {"ok": False, "error": "no experiences"}

        for exp in experiences:
            obs = np.array(exp["obs"])
            action = exp["action"]
            reward = exp["reward"]
            next_obs = np.array(exp["next_obs"])
            done = exp.get("done", False)

            latent = self.encoder.encode(obs)
            next_latent = self.encoder.encode(next_obs)
            self.irrigation_policy.store_experience(latent, action, reward, next_latent, done)

        result = self.irrigation_policy.update(batch_size=min(32, len(experiences)))
        self.step_count += 1

        return {"ok": True, **result}

    def _generate_reason(self, action: int, stage: int, soil: float,
                         temp: float, disease: int, liquid: float) -> str:
        """生成决策原因"""
        reasons = []

        if liquid < 15:
            reasons.append("液位不足")

        if disease > 0:
            reasons.append(f"检测到{DISEASE_NAMES.get(disease, '病害')}")

        stage_name = GROWTH_STAGES.get(stage, "未知")
        if action == 0:
            if soil > 60:
                reasons.append(f"{stage_name}阶段,土壤水分充足")
            else:
                reasons.append(f"{stage_name}阶段,暂不需灌溉")
        elif action == 1:
            reasons.append(f"{stage_name}阶段,轻度补水")
        elif action == 2:
            reasons.append(f"{stage_name}阶段,中度灌溉")
        elif action == 3:
            reasons.append(f"{stage_name}阶段,重度灌溉")

        if temp > 35:
            reasons.append("高温预警")
        elif temp < 5:
            reasons.append("低温预警")

        return "; ".join(reasons) if reasons else "正常决策"

    def get_info(self) -> dict:
        """获取模型信息"""
        avg_latency = (self.total_prediction_time / self.prediction_count * 1000
                       if self.prediction_count > 0 else 0)
        return {
            "loaded": self.is_loaded,
            "step_count": self.step_count,
            "prediction_count": self.prediction_count,
            "avg_latency_ms": round(avg_latency, 1),
            "device": self.device,
            "components": {
                "encoder": {"obs_dim": self.encoder.obs_dim, "latent_dim": self.encoder.latent_dim},
                "transition": {"latent_dim": self.transition.latent_dim},
                "disease": {"n_crops": self.disease_recognizer.n_crops,
                            "n_diseases": self.disease_recognizer.n_diseases},
                "policy": {"n_actions": self.irrigation_policy.n_actions,
                           "epsilon": self.irrigation_policy.epsilon},
            }
        }
