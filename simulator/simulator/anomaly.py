"""AnomalyModule — 3-layer anomaly detection matching firmware."""
from __future__ import annotations

import math
import random
import threading
from collections import deque

# Constants from firmware
K_ZSCORE_THRESHOLD = 2.5
K_STUCK_THRESHOLD = 30
K_WINDOW_SIZE = 60
K_MAX_ALERTS = 20
K_MAX_NODES_PER_TREE = 255


class AlertLevel:
    NONE = 0
    INFO = 1
    WARNING = 2
    CRITICAL = 3


ALERT_LEVEL_NAMES = {
    AlertLevel.NONE: "normal",
    AlertLevel.INFO: "info",
    AlertLevel.WARNING: "warning",
    AlertLevel.CRITICAL: "critical",
}


class IForestNode:
    __slots__ = ("splitFeature", "splitValue", "left", "right")

    def __init__(self):
        self.splitFeature = 0
        self.splitValue = 0.0
        self.left = -1
        self.right = -1


class SensorStats:
    def __init__(self, name: str, label: str):
        self.name = name
        self.label = label
        self.currentValue = 0.0
        self.lastValue = 0.0
        self.mean = 0.0
        self.stddev = 0.0
        self.lastZScore = 0.0
        self.isAnomalous = False
        self.isStuck = False
        self.isDisconnected = False
        self.anomalyCount = 0
        self.stuckCount = 0
        self.faultCount = 0
        self.minEver = 9999.0
        self.maxEver = -9999.0
        self.window = deque(maxlen=K_WINDOW_SIZE)
        self.windowCount = 0


class Alert:
    __slots__ = ("timestamp", "level", "sensor", "message")

    def __init__(self):
        self.timestamp = 0
        self.level = AlertLevel.NONE
        self.sensor = ""
        self.message = ""


class AnomalyModule:
    """Direct port of agri::AnomalyModule."""

    K_FEATURE_COUNT = 4
    K_TREES = 10
    K_DEPTH = 8
    K_TRAIN_BUFFER_SIZE = 200

    def __init__(self):
        self._lock = threading.Lock()
        self._sensors = [
            SensorStats("AirTemp", "AirTemp"),
            SensorStats("AirHumi", "AirHumi"),
            SensorStats("SoilHumi", "Soil"),
            SensorStats("Light", "Light"),
        ]

        # Alert ring buffer
        self._alerts = [Alert() for _ in range(K_MAX_ALERTS)]
        self._alertIndex = 0
        self._alertCount = 0
        self._currentLevel = AlertLevel.NONE
        self._totalSamples = 0
        self._totalAnomalies = 0

        # Isolation Forest
        self._iforestTrained = False
        self._iforestScore = 0.0
        self._lastIforestRunMs = 0
        self._trainBuffer = [[0.0] * self.K_FEATURE_COUNT for _ in range(self.K_TRAIN_BUFFER_SIZE)]
        self._trainBufferCount = 0

        # Forest: list of trees, each tree is a list of IForestNode
        self._forest = [[] for _ in range(self.K_TREES)]

    def update(self, snapshot, sample_updated: bool, now_ms: int):
        with self._lock:
            if sample_updated:
                values = [
                    snapshot.airTemp,
                    snapshot.airHumi,
                    snapshot.soilHumi,
                    snapshot.lightValue,
                ]

                for i in range(self.K_FEATURE_COUNT):
                    self._check_sensor_fault(self._sensors[i], values[i])
                    self._update_sliding_window(self._sensors[i], values[i])
                    self._sensors[i].currentValue = values[i]
                    self._calculate_stats(self._sensors[i])

                    if values[i] < self._sensors[i].minEver:
                        self._sensors[i].minEver = values[i]
                    if values[i] > self._sensors[i].maxEver:
                        self._sensors[i].maxEver = values[i]

                # Train buffer
                if self._trainBufferCount < self.K_TRAIN_BUFFER_SIZE:
                    for i in range(self.K_FEATURE_COUNT):
                        self._trainBuffer[self._trainBufferCount][i] = values[i]
                    self._trainBufferCount += 1

                self._detect_zscore_anomaly()
                self._update_alert_level()
                self._totalSamples += 1

            # Isolation Forest check
            if now_ms - self._lastIforestRunMs >= 60000:
                if self._trainBufferCount >= 50 and not self._iforestTrained:
                    self._build_isolation_forest()
                if self._iforestTrained:
                    features = [s.currentValue for s in self._sensors]
                    self._iforestScore = self._run_isolation_forest(features)
                    if self._iforestScore > 0.65:
                        self._add_alert(AlertLevel.WARNING, "IForest", "multi-sensor anomaly detected")
                        self._totalAnomalies += 1
                    self._update_alert_level()
                self._lastIforestRunMs = now_ms

    def clear(self):
        with self._lock:
            self._alertCount = 0
            self._alertIndex = 0
            self._currentLevel = AlertLevel.NONE
            self._totalAnomalies = 0
            for s in self._sensors:
                s.anomalyCount = 0
                s.isAnomalous = False
                s.isDisconnected = False
                s.isStuck = False

    def _update_sliding_window(self, stats: SensorStats, value: float):
        stats.window.append(value)
        stats.windowCount = min(stats.windowCount + 1, K_WINDOW_SIZE)

    def _calculate_stats(self, stats: SensorStats):
        if stats.windowCount < 3:
            return

        values = list(stats.window)
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        stddev = math.sqrt(variance)

        stats.mean = mean
        stats.stddev = stddev

        if stddev > 0.001:
            stats.lastZScore = (stats.currentValue - mean) / stddev
        else:
            stats.lastZScore = 0.0

    def _check_sensor_fault(self, stats: SensorStats, value: float):
        is_light = stats.name == "Light"

        if not is_light and value <= 0.001 and stats.windowCount > 10:
            if not stats.isDisconnected:
                self._add_alert(AlertLevel.CRITICAL, stats.name, f"{stats.label} disconnected")
            stats.isDisconnected = True
            return
        stats.isDisconnected = False

        if abs(value - stats.lastValue) < 0.01:
            stats.stuckCount += 1
            if stats.stuckCount >= K_STUCK_THRESHOLD and not stats.isStuck:
                stats.isStuck = True
                self._add_alert(AlertLevel.INFO, stats.name, f"{stats.label} may be stuck")
        else:
            stats.stuckCount = 0
            stats.isStuck = False
        stats.lastValue = value

    def _detect_zscore_anomaly(self):
        for s in self._sensors:
            if s.windowCount < 10:
                continue
            abs_z = abs(s.lastZScore)
            if abs_z > K_ZSCORE_THRESHOLD:
                if not s.isAnomalous:
                    s.isAnomalous = True
                    s.anomalyCount += 1
                    self._totalAnomalies += 1
                    level = AlertLevel.CRITICAL if abs_z > 3.5 else AlertLevel.WARNING
                    msg = f"{s.label} z-score={s.lastZScore:.2f}"
                    self._add_alert(level, s.name, msg)
            else:
                s.isAnomalous = False

    def _build_isolation_forest(self):
        n = self._trainBufferCount
        if n <= 1:
            return

        for t in range(self.K_TREES):
            self._forest[t] = []

            min_vals = [9999.0] * self.K_FEATURE_COUNT
            max_vals = [-9999.0] * self.K_FEATURE_COUNT
            for i in range(n):
                for f in range(self.K_FEATURE_COUNT):
                    if self._trainBuffer[i][f] < min_vals[f]:
                        min_vals[f] = self._trainBuffer[i][f]
                    if self._trainBuffer[i][f] > max_vals[f]:
                        max_vals[f] = self._trainBuffer[i][f]

            self._build_isolation_node(t, min_vals[:], max_vals[:], 0)

        self._iforestTrained = True

    def _build_isolation_node(self, tree_idx, min_vals, max_vals, depth):
        if depth >= self.K_DEPTH or len(self._forest[tree_idx]) >= K_MAX_NODES_PER_TREE:
            return -1

        node = IForestNode()
        node.splitFeature = random.randint(0, self.K_FEATURE_COUNT - 1)
        feature = node.splitFeature
        if max_vals[feature] > min_vals[feature]:
            node.splitValue = min_vals[feature] + random.random() * (max_vals[feature] - min_vals[feature])
        else:
            node.splitValue = min_vals[feature]

        node_index = len(self._forest[tree_idx])
        self._forest[tree_idx].append(node)

        left_min = min_vals[:]
        left_max = max_vals[:]
        left_max[feature] = node.splitValue

        right_min = min_vals[:]
        right_max = max_vals[:]
        right_min[feature] = node.splitValue

        left_idx = self._build_isolation_node(tree_idx, left_min, left_max, depth + 1)
        right_idx = self._build_isolation_node(tree_idx, right_min, right_max, depth + 1)

        self._forest[tree_idx][node_index].left = left_idx
        self._forest[tree_idx][node_index].right = right_idx

        return node_index

    def _run_isolation_forest(self, features):
        if not self._iforestTrained or self._trainBufferCount <= 1:
            return 0.0

        avg_path = 0.0
        for t in range(self.K_TREES):
            avg_path += self._path_length(t, 0, features, 0)
        avg_path /= self.K_TREES

        c = self._average_path_length(self._trainBufferCount)
        if c <= 0:
            return 0.0
        return 2.0 ** (-(avg_path / c))

    def _path_length(self, tree_idx, node_idx, features, depth):
        if depth >= self.K_DEPTH or node_idx < 0 or node_idx >= len(self._forest[tree_idx]):
            return float(depth)

        node = self._forest[tree_idx][node_idx]
        if node.left == -1 and node.right == -1:
            return float(depth)

        if features[node.splitFeature] < node.splitValue:
            return self._path_length(tree_idx, node.left, features, depth + 1)
        return self._path_length(tree_idx, node.right, features, depth + 1)

    def _average_path_length(self, n):
        if n <= 1:
            return 1.0
        h = math.log(n - 1) + 0.5772156649
        return 2.0 * h - 2.0 * ((n - 1) / n)

    def _add_alert(self, level, sensor, message):
        # Dedup: skip if same sensor already has an alert in recent slots
        for i in range(min(self._alertCount, K_MAX_ALERTS)):
            idx = (self._alertIndex - 1 - i + K_MAX_ALERTS) % K_MAX_ALERTS
            if self._alerts[idx].sensor == sensor and self._alerts[idx].level == level:
                return

        idx = self._alertIndex % K_MAX_ALERTS
        self._alerts[idx].timestamp = self._totalSamples
        self._alerts[idx].level = level
        self._alerts[idx].sensor = sensor
        self._alerts[idx].message = message
        self._alertIndex = (self._alertIndex + 1) % K_MAX_ALERTS
        self._alertCount += 1

    def _update_alert_level(self):
        max_level = AlertLevel.NONE
        for s in self._sensors:
            if s.isDisconnected:
                max_level = AlertLevel.CRITICAL
                break
            if s.isAnomalous and max_level < AlertLevel.WARNING:
                max_level = AlertLevel.WARNING
            if s.isStuck and max_level < AlertLevel.INFO:
                max_level = AlertLevel.INFO
        if self._iforestScore > 0.65 and max_level < AlertLevel.WARNING:
            max_level = AlertLevel.WARNING
        self._currentLevel = max_level

    # ── Public accessors ──

    @property
    def level(self) -> int:
        return self._currentLevel

    @property
    def iforest_score(self) -> float:
        return self._iforestScore

    @property
    def iforest_trained(self) -> bool:
        return self._iforestTrained

    @property
    def total_samples(self) -> int:
        return self._totalSamples

    @property
    def total_anomalies(self) -> int:
        return self._totalAnomalies

    def status(self) -> dict:
        with self._lock:
            return {
                "alertLevel": self._currentLevel,
                "alertLevelName": ALERT_LEVEL_NAMES.get(self._currentLevel, "normal"),
                "totalSamples": self._totalSamples,
                "totalAnomalies": self._totalAnomalies,
                "iforestTrained": self._iforestTrained,
                "iforestScore": round(self._iforestScore, 4),
                "sensors": [
                    {
                        "name": s.name,
                        "label": s.label,
                        "value": round(s.currentValue, 2),
                        "mean": round(s.mean, 2),
                        "stddev": round(s.stddev, 4),
                        "zScore": round(s.lastZScore, 4),
                        "isAnomalous": s.isAnomalous,
                        "isStuck": s.isStuck,
                        "isDisconnected": s.isDisconnected,
                        "anomalyCount": s.anomalyCount,
                        "min": round(s.minEver, 2),
                        "max": round(s.maxEver, 2),
                    }
                    for s in self._sensors
                ],
            }

    def alerts(self) -> list:
        with self._lock:
            result = []
            count = min(self._alertCount, K_MAX_ALERTS)
            for i in range(count):
                idx = (self._alertIndex - 1 - i + K_MAX_ALERTS) % K_MAX_ALERTS
                a = self._alerts[idx]
                result.append({
                    "timestamp": a.timestamp,
                    "level": a.level,
                    "levelName": ALERT_LEVEL_NAMES.get(a.level, "normal"),
                    "sensor": a.sensor,
                    "message": a.message,
                })
            return result

    def sensor_detail(self, sensor_name: str) -> dict | None:
        with self._lock:
            for s in self._sensors:
                if s.name == sensor_name:
                    return {
                        "name": s.name,
                        "label": s.label,
                        "value": round(s.currentValue, 2),
                        "mean": round(s.mean, 2),
                        "stddev": round(s.stddev, 4),
                        "zScore": round(s.lastZScore, 4),
                        "window": [round(v, 2) for v in s.window],
                    }
            return None
