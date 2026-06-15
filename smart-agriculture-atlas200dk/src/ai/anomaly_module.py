"""
智润智慧农业套件 - Atlas 200I DK A2 版
异常检测模块 - 三层异常检测: 滑动窗口 + Z-Score + Isolation Forest

对应原ESP32项目的 AnomalyModule.h/AnomalyModule.cpp
Python版使用NumPy加速, 算法逻辑完全一致
"""

import math
import time
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

import numpy as np

from config.app_types import SensorSnapshot, AnomalyLevel, AnomalyResult

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    timestamp: float = 0.0
    level: AnomalyLevel = AnomalyLevel.Normal
    sensor: str = ""
    message: str = ""


@dataclass
class SensorStats:
    name: str = ""
    label: str = ""
    current_value: float = 0.0
    last_value: float = 0.0
    mean: float = 0.0
    stddev: float = 0.0
    last_z_score: float = 0.0
    is_anomalous: bool = False
    is_stuck: bool = False
    is_disconnected: bool = False
    stuck_count: int = 0
    anomaly_count: int = 0
    min_ever: float = float('inf')
    max_ever: float = float('-inf')
    window: deque = field(default_factory=lambda: deque(maxlen=60))


@dataclass
class IForestNode:
    split_feature: int = 0
    split_value: float = 0.0
    left: int = -1
    right: int = -1


@dataclass
class IForestTree:
    nodes: List[IForestNode] = field(default_factory=list)


class AnomalyModule:
    """三层异常检测: 滑动窗口统计 + Z-Score + Isolation Forest"""

    FEATURE_COUNT = 4
    WINDOW_SIZE = 60
    STUCK_THRESHOLD = 30
    ZSCORE_WARN = 2.5
    ZSCORE_CRITICAL = 3.5
    IFOREST_THRESHOLD = 0.65
    MAX_TREES = 10
    MAX_DEPTH = 8
    MAX_NODES_PER_TREE = 255
    TRAIN_BUFFER_SIZE = 200
    MIN_BUILD_SAMPLES = 50
    MAX_ALERTS = 20

    def __init__(self, buzzer=None):
        self._buzzer = buzzer  # ActuatorController reference

        self._sensor_names = ["AirTemp", "AirHumi", "SoilHumi", "Light"]
        self._sensor_labels = ["AirTemp", "AirHumi", "Soil", "Light"]
        self._sensors = [
            SensorStats(name=n, label=l)
            for n, l in zip(self._sensor_names, self._sensor_labels)
        ]

        self._current_level = AnomalyLevel.Normal
        self._total_samples = 0
        self._total_anomalies = 0
        self._iforest_score = 0.0
        self._iforest_trained = False
        self._last_iforest_time = 0.0

        self._train_buffer = []
        self._train_buffer_count = 0

        self._forest: List[IForestTree] = []
        self._alerts: deque = deque(maxlen=self.MAX_ALERTS)

    def update(self, snapshot: SensorSnapshot, sample_updated: bool, now: float = 0.0):
        if now == 0.0:
            now = time.time()

        if sample_updated:
            values = [
                snapshot.air_temp,
                snapshot.air_humi,
                snapshot.soil_humi,
                snapshot.light_intensity,
            ]

            for i, val in enumerate(values):
                self._check_sensor_fault(self._sensors[i], val)
                self._sensors[i].window.append(val)
                self._sensors[i].current_value = val
                self._calculate_stats(self._sensors[i])

                if val < self._sensors[i].min_ever:
                    self._sensors[i].min_ever = val
                if val > self._sensors[i].max_ever:
                    self._sensors[i].max_ever = val

            if self._train_buffer_count < self.TRAIN_BUFFER_SIZE:
                self._train_buffer.append(values[:])
                self._train_buffer_count += 1

            self._detect_z_score_anomaly()
            self._update_alert_level()
            self._total_samples += 1

        # Isolation Forest每60秒运行一次
        if now - self._last_iforest_time >= 60.0:
            if self._train_buffer_count >= self.MIN_BUILD_SAMPLES and not self._iforest_trained:
                self._build_isolation_forest()
            if self._iforest_trained:
                features = [s.current_value for s in self._sensors]
                self._iforest_score = self._run_isolation_forest(features)
                if self._iforest_score > self.IFOREST_THRESHOLD:
                    self._add_alert(AnomalyLevel.Warning, "IForest",
                                    "multi-sensor anomaly detected")
                    self._total_anomalies += 1
                self._update_alert_level()
            self._last_iforest_time = now

    @property
    def current_level(self) -> AnomalyLevel:
        return self._current_level

    @property
    def iforest_score(self) -> float:
        return self._iforest_score

    @property
    def iforest_trained(self) -> bool:
        return self._iforest_trained

    @property
    def total_anomalies(self) -> int:
        return self._total_anomalies

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def clear(self):
        self._alerts.clear()
        self._current_level = AnomalyLevel.Normal
        self._total_anomalies = 0
        for s in self._sensors:
            s.anomaly_count = 0
            s.is_anomalous = False
            s.is_disconnected = False
            s.is_stuck = False

    # ------------------------------------------------------------------
    # 滑动窗口统计
    # ------------------------------------------------------------------
    def _calculate_stats(self, stats: SensorStats):
        if len(stats.window) < 3:
            return
        arr = np.array(stats.window)
        stats.mean = float(np.mean(arr))
        stats.stddev = float(np.std(arr))
        if stats.stddev > 0.001:
            stats.last_z_score = (stats.current_value - stats.mean) / stats.stddev
        else:
            stats.last_z_score = 0.0

    def _check_sensor_fault(self, stats: SensorStats, value: float):
        is_light = stats.name == "Light"

        if not is_light and value <= 0.001 and len(stats.window) > 10:
            if not stats.is_disconnected:
                self._add_alert(AnomalyLevel.Critical, stats.name,
                                f"{stats.label} disconnected")
            stats.is_disconnected = True
            return
        stats.is_disconnected = False

        if abs(value - stats.last_value) < 0.01:
            stats.stuck_count += 1
            if stats.stuck_count >= self.STUCK_THRESHOLD and not stats.is_stuck:
                stats.is_stuck = True
                self._add_alert(AnomalyLevel.Normal, stats.name,
                                f"{stats.label} may be stuck")
        else:
            stats.stuck_count = 0
            stats.is_stuck = False
        stats.last_value = value

    def _detect_z_score_anomaly(self):
        for s in self._sensors:
            if len(s.window) < 10:
                continue
            abs_z = abs(s.last_z_score)
            if abs_z > self.ZSCORE_WARN:
                if not s.is_anomalous:
                    s.is_anomalous = True
                    s.anomaly_count += 1
                    self._total_anomalies += 1
                    level = AnomalyLevel.Critical if abs_z > self.ZSCORE_CRITICAL else AnomalyLevel.Warning
                    self._add_alert(level, s.name,
                                    f"{s.label} z-score={s.last_z_score:.2f}")
            else:
                s.is_anomalous = False

    # ------------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------------
    def _build_isolation_forest(self):
        data = np.array(self._train_buffer[:self._train_buffer_count])
        n = data.shape[0]
        if n <= 1:
            return

        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)

        self._forest = []
        for _ in range(self.MAX_TREES):
            tree = IForestTree()
            self._build_isolation_node(tree, min_vals.copy(), max_vals.copy(), 0)
            self._forest.append(tree)

        self._iforest_trained = True
        logger.info(f"Isolation Forest构建完成: {self.MAX_TREES}棵树, {n}个训练样本")

    def _build_isolation_node(self, tree: IForestTree, min_vals: np.ndarray,
                               max_vals: np.ndarray, depth: int) -> int:
        if depth >= self.MAX_DEPTH or len(tree.nodes) >= self.MAX_NODES_PER_TREE:
            return -1

        node_index = len(tree.nodes)
        feature = random.randint(0, self.FEATURE_COUNT - 1)
        split_value = min_vals[feature] + random.random() * (max_vals[feature] - min_vals[feature])

        node = IForestNode(split_feature=feature, split_value=split_value)
        tree.nodes.append(node)

        left_max = max_vals.copy()
        left_max[feature] = split_value
        right_min = min_vals.copy()
        right_min[feature] = split_value

        left_idx = self._build_isolation_node(tree, min_vals.copy(), left_max, depth + 1)
        right_idx = self._build_isolation_node(tree, right_min.copy(), max_vals.copy(), depth + 1)

        tree.nodes[node_index].left = left_idx
        tree.nodes[node_index].right = right_idx
        return node_index

    def _run_isolation_forest(self, features: list) -> float:
        if not self._iforest_trained or self._train_buffer_count <= 1:
            return 0.0

        avg_path = 0.0
        for tree in self._forest:
            avg_path += self._path_length(tree, 0, features, 0)
        avg_path /= self.MAX_TREES

        c = self._average_path_length(self._train_buffer_count)
        if c <= 0.0:
            return 0.0
        return 2.0 ** (-(avg_path / c))

    def _path_length(self, tree: IForestTree, node_idx: int,
                      features: list, depth: int) -> float:
        if depth >= self.MAX_DEPTH or node_idx < 0 or node_idx >= len(tree.nodes):
            return float(depth)

        node = tree.nodes[node_idx]
        if node.left == -1 and node.right == -1:
            return float(depth)

        if features[node.split_feature] < node.split_value:
            return self._path_length(tree, node.left, features, depth + 1)
        return self._path_length(tree, node.right, features, depth + 1)

    @staticmethod
    def _average_path_length(n: int) -> float:
        if n <= 1:
            return 1.0
        h = math.log(n - 1) + 0.5772156649
        return 2.0 * h - 2.0 * ((n - 1) / n)

    # ------------------------------------------------------------------
    # 告警管理
    # ------------------------------------------------------------------
    def _add_alert(self, level: AnomalyLevel, sensor: str, message: str):
        now = time.time()
        # 5秒内同传感器不重复告警
        for alert in self._alerts:
            if now - alert.timestamp < 5.0 and alert.sensor == sensor:
                return

        self._alerts.append(Alert(
            timestamp=now, level=level, sensor=sensor, message=message
        ))
        self._beep(level)

    def _update_alert_level(self):
        max_level = AnomalyLevel.Normal
        for s in self._sensors:
            if s.is_disconnected:
                max_level = AnomalyLevel.Critical
                break
            if s.is_anomalous and max_level.value < AnomalyLevel.Warning.value:
                max_level = AnomalyLevel.Warning
            if s.is_stuck and max_level.value < AnomalyLevel.Normal.value:
                max_level = AnomalyLevel.Normal
        if self._iforest_score > self.IFOREST_THRESHOLD and max_level.value < AnomalyLevel.Warning.value:
            max_level = AnomalyLevel.Warning
        self._current_level = max_level

    def _beep(self, level: AnomalyLevel):
        if not self._buzzer or level == AnomalyLevel.Normal:
            return
        if level == AnomalyLevel.Warning:
            self._buzzer.beep(count=2, on_ms=100, off_ms=60)
        elif level == AnomalyLevel.Critical:
            self._buzzer.beep(count=4, on_ms=50, off_ms=50)

    def to_dict(self) -> dict:
        return {
            "alertLevel": self._current_level.value,
            "alertLevelName": self._current_level.name,
            "totalSamples": self._total_samples,
            "totalAnomalies": self._total_anomalies,
            "iforestTrained": self._iforest_trained,
            "iforestScore": round(self._iforest_score, 4),
            "sensors": [
                {
                    "name": s.name,
                    "label": s.label,
                    "value": round(s.current_value, 2),
                    "mean": round(s.mean, 2),
                    "stddev": round(s.stddev, 4),
                    "zScore": round(s.last_z_score, 4),
                    "isAnomalous": s.is_anomalous,
                    "isStuck": s.is_stuck,
                    "isDisconnected": s.is_disconnected,
                    "anomalyCount": s.anomaly_count,
                    "min": round(s.min_ever, 2) if s.min_ever != float('inf') else 0,
                    "max": round(s.max_ever, 2) if s.max_ever != float('-inf') else 0,
                }
                for s in self._sensors
            ],
        }
