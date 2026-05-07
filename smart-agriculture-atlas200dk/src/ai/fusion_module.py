"""
智润智慧农业套件 - Atlas 200I DK A2 版
传感器融合模块 - 5通道卡尔曼滤波 + 5->8->3神经网络混合决策

对应原ESP32项目的 FusionModule.h/FusionModule.cpp
使用NumPy加速矩阵运算, 算法逻辑完全一致
"""

import math
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from config.app_types import SensorSnapshot, ControlSource
from config.hardware_config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class SensorChannel:
    name: str = ""
    label: str = ""
    unit: str = ""
    raw_value: float = 0.0
    kalman_estimate: float = 0.0
    kalman_error: float = 1.0
    kalman_gain: float = 0.0
    normalized_value: float = 0.0
    reliability: float = 1.0
    weight: float = 0.2
    fault_count: int = 0
    healthy: bool = True
    min_range: float = 0.0
    max_range: float = 100.0


@dataclass
class FusionResult:
    decision: str = "none"  # "none", "moderate", "heavy"
    confidence: float = 0.0
    need_score: float = 0.0
    weighted_score: float = 0.0
    nn_score: float = 0.0
    final_score: float = 0.0


SENSOR_COUNT = 5
HIDDEN_SIZE = 8
OUTPUT_SIZE = 3

# 预训练权重 (与ESP32版一致的合成数据训练结果)
PRESET_WEIGHTS_IH = np.array([
    [-0.7618, -0.1648,  0.9222,  0.2010,  0.1822,  0.8395,  0.6211, -0.7773],
    [ 0.7061, -0.4602, -0.9656, -0.1033, -0.5191,  0.2046, -0.2469,  1.1025],
    [ 1.1686,  0.1717, -0.5193, -0.1273, -0.0344, -0.5801, -0.5640,  1.7332],
    [ 0.1232, -0.2913,  1.0482,  0.3295, -0.1933,  0.3476,  0.3208, -0.1836],
    [-0.4105,  0.1005,  1.5418, -2.6921, -0.4168,  1.2474, -2.1958, -0.7266],
], dtype=np.float32)

PRESET_BIAS_H = np.array([0.5547, -0.0495, 0.2924, 0.5884, -0.0884, 0.3348, 0.4393, 0.7546],
                           dtype=np.float32)

PRESET_WEIGHTS_HO = np.array([
    [ 1.1199,  0.6752, -2.4982],
    [ 0.6507, -0.6800,  0.1269],
    [-1.5823, -0.1323,  1.2406],
    [ 2.7740, -0.9054, -1.8639],
    [-0.6520,  0.0906, -0.5720],
    [-1.1279,  0.6628,  0.5211],
    [ 2.5613, -2.2228, -0.1250],
    [ 1.4902, -0.1788, -3.4092],
], dtype=np.float32)

PRESET_BIAS_O = np.array([-0.1035, 0.3601, -0.2272], dtype=np.float32)


class FusionModule:
    """5通道卡尔曼滤波 + 神经网络混合传感器融合"""

    FUSION_INTERVAL = 10.0  # 秒
    WEIGHTED_RATIO = 0.6
    NN_RATIO = 0.4

    def __init__(self, data_dir: str = DATA_DIR):
        self._data_dir = data_dir
        self._auto_control_enabled = True
        self._channels: List[SensorChannel] = []
        self._init_channels()

        self._weights_ih = PRESET_WEIGHTS_IH.copy()
        self._bias_h = PRESET_BIAS_H.copy()
        self._weights_ho = PRESET_WEIGHTS_HO.copy()
        self._bias_o = PRESET_BIAS_O.copy()

        self._result = FusionResult()
        self._nn_output = np.zeros(OUTPUT_SIZE)
        self._last_fusion_time = 0.0
        self._total_decisions = 0
        self._irrigation_count = 0
        self._average_confidence = 0.0

    def _init_channels(self):
        configs = [
            ("AirTemp", "Temperature", "C", 25.0, 0.20, 0.0, 40.0),
            ("AirHumi", "Humidity", "%", 60.0, 0.20, 0.0, 100.0),
            ("SoilHumi", "Soil", "%", 50.0, 0.30, 0.0, 100.0),
            ("Light", "Light", "lux", 500.0, 0.15, 0.0, 10000.0),
            ("Liquid", "Liquid", "%", 80.0, 0.15, 0.0, 100.0),
        ]
        self._channels = [
            SensorChannel(name=n, label=l, unit=u, kalman_estimate=e,
                          weight=w, min_range=mn, max_range=mx)
            for n, l, u, e, w, mn, mx in configs
        ]

    def begin(self, auto_control_enabled: bool = True):
        self._auto_control_enabled = auto_control_enabled
        self._load_network()

    def update(self, snapshot: SensorSnapshot, sample_updated: bool,
               now: float, actuator) -> Optional[FusionResult]:
        if sample_updated:
            raw_values = [
                snapshot.air_temp, snapshot.air_humi, snapshot.soil_humi,
                snapshot.light_intensity, snapshot.liquid_level
            ]
            for i, val in enumerate(raw_values):
                self._channels[i].raw_value = val
                self._apply_kalman_filter(self._channels[i], val)
                self._normalize_value(self._channels[i])
                self._update_reliability(self._channels[i])
            self._calculate_weights()

        if now - self._last_fusion_time < self.FUSION_INTERVAL:
            return None

        self._result = self._perform_fusion()
        self._last_fusion_time = now

        if (not self._auto_control_enabled or actuator.is_busy(now)):
            return self._result

        duration = 0.0
        if self._result.decision == "moderate":
            duration = 45.0
        elif self._result.decision == "heavy":
            duration = 120.0

        if duration > 0:
            actuator.start_timed_run(ControlSource.TimedRun, duration, now)

        return self._result

    # ------------------------------------------------------------------
    # 卡尔曼滤波
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_kalman_filter(channel: SensorChannel, measurement: float):
        Q = 0.01
        R = 0.1
        predicted_estimate = channel.kalman_estimate
        predicted_error = channel.kalman_error + Q
        channel.kalman_gain = predicted_error / (predicted_error + R)
        channel.kalman_estimate = predicted_estimate + channel.kalman_gain * (measurement - predicted_estimate)
        channel.kalman_error = (1.0 - channel.kalman_gain) * predicted_error

    @staticmethod
    def _normalize_value(channel: SensorChannel):
        rng = channel.max_range - channel.min_range
        if rng <= 0:
            channel.normalized_value = 0.0
            return
        channel.normalized_value = max(0.0, min(1.0,
            (channel.kalman_estimate - channel.min_range) / rng))

    @staticmethod
    def _update_reliability(channel: SensorChannel):
        reliability = 1.0
        if channel.raw_value <= 0.001 or channel.raw_value > channel.max_range * 1.5:
            reliability *= 0.1
            channel.fault_count += 1
        else:
            channel.fault_count = max(0, channel.fault_count - 1)

        if channel.kalman_gain > 0.8:
            reliability *= 0.7

        if channel.fault_count > 5:
            reliability *= 0.3
            channel.healthy = False
        else:
            channel.healthy = True

        channel.reliability = channel.reliability * 0.8 + reliability * 0.2

    def _calculate_weights(self):
        total = sum(c.reliability for c in self._channels)
        if total <= 0:
            return
        for c in self._channels:
            c.weight = c.reliability / total

    # ------------------------------------------------------------------
    # 神经网络
    # ------------------------------------------------------------------
    def _run_neural_network(self, inputs: np.ndarray) -> np.ndarray:
        hidden = self._bias_h + inputs @ self._weights_ih
        hidden = np.maximum(hidden, 0.0)  # ReLU

        raw = self._bias_o + hidden @ self._weights_ho
        # Softmax
        exp_vals = np.exp(raw - np.max(raw))
        output = exp_vals / exp_vals.sum()
        self._nn_output = output
        return output

    # ------------------------------------------------------------------
    # 融合决策
    # ------------------------------------------------------------------
    def _perform_fusion(self) -> FusionResult:
        result = FusionResult()

        # 需求因子
        need_factors = [
            self._channels[0].normalized_value,                    # 温度高 -> 需水
            1.0 - self._channels[1].normalized_value,              # 湿度低 -> 需水
            1.0 - self._channels[2].normalized_value,              # 土壤干 -> 需水
            self._channels[3].normalized_value * 0.5,              # 光照强 -> 微增
            self._channels[4].normalized_value,                    # 液位高 -> 安全
        ]

        result.weighted_score = sum(
            f * c.weight for f, c in zip(need_factors, self._channels)
        ) * 100.0

        inputs = np.array([c.normalized_value for c in self._channels], dtype=np.float32)
        outputs = self._run_neural_network(inputs)
        result.nn_score = outputs[1] * 50.0 + outputs[2] * 100.0
        result.final_score = result.weighted_score * self.WEIGHTED_RATIO + result.nn_score * self.NN_RATIO
        result.need_score = result.final_score

        if self._channels[4].raw_value < 20.0:
            result.decision = "none"
            result.confidence = 0.95
        elif result.final_score > 65.0:
            result.decision = "heavy"
            result.confidence = min(1.0, result.final_score / 100.0)
        elif result.final_score > 35.0:
            result.decision = "moderate"
            result.confidence = 0.5 + (result.final_score - 35.0) / 60.0
        else:
            result.decision = "none"
            result.confidence = 1.0 - result.final_score / 70.0

        self._total_decisions += 1
        if result.decision != "none":
            self._irrigation_count += 1
        self._average_confidence = self._average_confidence * 0.95 + result.confidence * 0.05

        return result

    # ------------------------------------------------------------------
    # 权重持久化
    # ------------------------------------------------------------------
    def _load_network(self):
        import os
        path = os.path.join(self._data_dir, "fusion_weights.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self._weights_ih = np.array(data["weights_ih"], dtype=np.float32)
                self._bias_h = np.array(data["bias_h"], dtype=np.float32)
                self._weights_ho = np.array(data["weights_ho"], dtype=np.float32)
                self._bias_o = np.array(data["bias_o"], dtype=np.float32)
                logger.info("融合网络权重已加载")
            except Exception as e:
                logger.warning(f"融合网络权重加载失败: {e}")

    def save_network(self):
        import os
        os.makedirs(self._data_dir, exist_ok=True)
        path = os.path.join(self._data_dir, "fusion_weights.json")
        try:
            with open(path, 'w') as f:
                json.dump({
                    "weights_ih": self._weights_ih.tolist(),
                    "bias_h": self._bias_h.tolist(),
                    "weights_ho": self._weights_ho.tolist(),
                    "bias_o": self._bias_o.tolist(),
                }, f)
            logger.info("融合网络权重已保存")
        except Exception as e:
            logger.warning(f"融合网络权重保存失败: {e}")

    def to_dict(self) -> dict:
        return {
            "autoControlEnabled": self._auto_control_enabled,
            "decision": self._result.decision,
            "decisionName": self._result.decision,
            "confidence": round(self._result.confidence, 3),
            "needScore": round(self._result.need_score, 2),
            "weightedScore": round(self._result.weighted_score, 2),
            "nnScore": round(self._result.nn_score, 2),
            "finalScore": round(self._result.final_score, 2),
            "totalDecisions": self._total_decisions,
            "irrigationCount": self._irrigation_count,
            "avgConfidence": round(self._average_confidence, 3),
            "nn": {
                "none": round(float(self._nn_output[0]), 4),
                "moderate": round(float(self._nn_output[1]), 4),
                "heavy": round(float(self._nn_output[2]), 4),
            },
            "sensors": [
                {
                    "name": c.name, "label": c.label, "unit": c.unit,
                    "raw": round(c.raw_value, 2),
                    "filtered": round(c.kalman_estimate, 2),
                    "normalized": round(c.normalized_value, 4),
                    "reliability": round(c.reliability, 3),
                    "weight": round(c.weight, 4),
                    "kalmanGain": round(c.kalman_gain, 4),
                    "healthy": c.healthy,
                }
                for c in self._channels
            ],
        }
