#!/usr/bin/env python3
"""
智润智慧农业 - 世界模型训练脚本

训练流程:
1. 使用模拟数据或真实传感器数据
2. 训练跨作物病害识别器
3. 训练灌溉策略网络
4. 保存模型参数

使用方法:
    python train.py --epochs 100 --data-dir ./data --model-dir ./models
    python train.py --simulate  # 使用模拟数据训练
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_model import WorldModel, CROP_NAMES, GROWTH_STAGES, DISEASE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("train")


def generate_simulated_data(n_samples: int = 5000) -> List[Dict]:
    """
    生成模拟训练数据

    模拟规则:
    - 不同作物在不同环境条件下有不同的病害概率
    - 灌溉决策基于土壤湿度和生长阶段
    - 传感器数据在合理范围内波动
    """
    logger.info(f"生成 {n_samples} 条模拟数据...")
    data = []

    for _ in range(n_samples):
        crop_id = np.random.randint(0, 5)
        growth_stage = np.random.randint(0, 7)

        # 生成传感器数据 (基于作物和阶段的合理范围)
        base_temp = 20 + np.random.normal(0, 8)
        base_humi = 60 + np.random.normal(0, 15)
        base_soil = 45 + np.random.normal(0, 20)
        base_liquid = 50 + np.random.normal(0, 25)
        base_light = 500 + np.random.normal(0, 300)

        air_temp = np.clip(base_temp, 0, 45)
        air_humi = np.clip(base_humi, 10, 95)
        soil_humi = np.clip(base_soil, 5, 95)
        liquid = np.clip(base_liquid, 5, 95)
        light = np.clip(base_light, 0, 5000)
        is_day = 1.0 if 6 <= (12 + np.random.normal(0, 4)) % 24 <= 18 else 0.0

        # 模拟病害 (基于环境条件)
        disease_probs = _simulate_disease_probs(air_temp, air_humi, soil_humi, crop_id, growth_stage)
        disease_id = np.random.choice(5, p=disease_probs)

        # 模拟灌溉决策 (基于规则)
        action = _simulate_irrigation_action(soil_humi, growth_stage, air_temp, air_humi)

        # 模拟奖励
        target_soil = _target_soil(growth_stage)
        next_soil = np.clip(soil_humi + (30 if action > 0 else 0) * action / 3, 0, 100)
        reward = -abs(next_soil - target_soil) * 0.1
        if next_soil > 80: reward -= 5
        if next_soil < 20: reward -= 8
        if abs(next_soil - target_soil) < 10: reward += 3

        data.append({
            "obs": [air_temp, air_humi, soil_humi, liquid, light, is_day],
            "crop_id": crop_id,
            "growth_stage": growth_stage,
            "disease_id": disease_id,
            "action": action,
            "reward": float(reward),
            "next_obs": [air_temp + np.random.normal(0, 0.5),
                         air_humi + np.random.normal(0, 1),
                         next_soil,
                         liquid - action * 2,
                         light + np.random.normal(0, 50),
                         is_day],
            "done": np.random.random() < 0.01,
        })

    return data


def _simulate_disease_probs(temp, humi, soil, crop_id, stage) -> np.ndarray:
    """模拟病害概率"""
    probs = np.ones(5) * 0.02  # 基础概率

    # 健康的基础概率最高
    probs[0] = 0.7

    # 高湿度容易得灰霉病
    if humi > 80:
        probs[2] += 0.2
    # 高温高湿容易得炭疽病
    if temp > 30 and humi > 70:
        probs[1] += 0.15
    # 低湿度容易得叶灼病
    if humi < 30:
        probs[3] += 0.15
    # 温湿度适中容易得白粉病
    if 20 < temp < 30 and 40 < humi < 70:
        probs[4] += 0.1

    # 幼苗期更容易感染
    if stage <= 2:
        probs[1:] *= 1.5

    # 归一化
    return probs / probs.sum()


def _simulate_irrigation_action(soil, stage, temp, humi) -> int:
    """模拟灌溉决策"""
    target = _target_soil(stage)

    if soil < target - 20:
        return 3  # 重度
    elif soil < target - 10:
        return 2  # 中度
    elif soil < target:
        return 1  # 轻度
    else:
        return 0  # 关闭


def _target_soil(stage: int) -> float:
    targets = {0: 60, 1: 65, 2: 50, 3: 55, 4: 60, 5: 65, 6: 45}
    return targets.get(stage, 55)


def train_world_model(model: WorldModel, data: List[Dict], epochs: int = 10):
    """训练世界模型"""
    logger.info(f"开始训练, 数据量: {len(data)}, 轮次: {epochs}")

    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss = 0
        n_batches = 0

        # 分批训练
        batch_size = 32
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            # 训练灌溉策略
            experiences = []
            for sample in batch:
                obs = np.array(sample["obs"])
                next_obs = np.array(sample["next_obs"])
                action = sample["action"]
                reward = sample["reward"]
                done = sample.get("done", False)

                # 编码
                latent = model.encoder.encode(obs)
                next_latent = model.encoder.encode(next_obs)

                experiences.append({
                    "obs": obs.tolist(),
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs.tolist(),
                    "done": done,
                })

            # 更新策略
            result = model.train_step({"experiences": experiences})
            total_loss += result.get("loss", 0)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

        # 衰减探索率
        model.irrigation_policy.epsilon = max(
            0.01, model.irrigation_policy.epsilon * 0.95
        )

    logger.info("训练完成")


def evaluate_model(model: WorldModel, data: List[Dict]):
    """评估模型"""
    logger.info("评估模型...")

    correct_disease = 0
    correct_action = 0
    total_reward = 0
    n = len(data)

    for sample in data:
        obs = sample["obs"]
        result = model.predict({
            "air_temp": obs[0], "air_humi": obs[1], "soil_humi": obs[2],
            "liquid_level": obs[3], "light": obs[4], "is_day": obs[5] > 0.5,
            "growth_stage": sample["growth_stage"], "crop_id": sample["crop_id"],
        })

        if result["disease"]["id"] == sample["disease_id"]:
            correct_disease += 1
        if result["irrigation"]["action"] == sample["action"]:
            correct_action += 1
        total_reward += sample["reward"]

    logger.info(f"病害识别准确率: {correct_disease / n * 100:.1f}%")
    logger.info(f"灌溉决策准确率: {correct_action / n * 100:.1f}%")
    logger.info(f"平均奖励: {total_reward / n:.3f}")


def main():
    parser = argparse.ArgumentParser(description="训练世界模型")
    parser.add_argument('--simulate', action='store_true', help='使用模拟数据')
    parser.add_argument('--data-dir', default='./data', help='数据目录')
    parser.add_argument('--model-dir', default='./models', help='模型目录')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮次')
    parser.add_argument('--samples', type=int, default=5000, help='模拟数据量')
    parser.add_argument('--eval-split', type=float, default=0.2, help='评估集比例')
    args = parser.parse_args()

    # 初始化模型
    model = WorldModel(model_dir=args.model_dir)
    model.load()

    # 加载或生成数据
    if args.simulate:
        data = generate_simulated_data(args.samples)
    else:
        data_file = os.path.join(args.data_dir, "training_data.json")
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                data = json.load(f)
            logger.info(f"加载数据: {len(data)} 条")
        else:
            logger.warning(f"未找到数据文件 {data_file}, 使用模拟数据")
            data = generate_simulated_data(args.samples)

    # 训练/评估分割
    split = int(len(data) * (1 - args.eval_split))
    train_data = data[:split]
    eval_data = data[split:]

    # 训练
    train_world_model(model, train_data, epochs=args.epochs)

    # 评估
    evaluate_model(model, eval_data)

    # 保存
    model.save()
    logger.info(f"模型已保存到 {args.model_dir}")


if __name__ == '__main__':
    main()
