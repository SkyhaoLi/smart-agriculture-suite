"""FusionModule — Kalman filter + neural network fusion matching firmware."""

import math
import threading
from .actuator import ActuatorController, ControlSource


class Decision:
    NONE = 0
    MODERATE = 1
    HEAVY = 2


DECISION_NAMES = {
    Decision.NONE: "none",
    Decision.MODERATE: "moderate",
    Decision.HEAVY: "heavy",
}

K_SENSOR_COUNT = 5
K_HIDDEN = 8
K_OUTPUTS = 3
K_FUSION_INTERVAL_MS = 10000


class SensorChannel:
    def __init__(self):
        self.name = ""
        self.label = ""
        self.unit = ""
        self.rawValue = 0.0
        self.kalmanEstimate = 0.0
        self.kalmanError = 1.0
        self.kalmanGain = 0.0
        self.normalizedValue = 0.0
        self.reliability = 1.0
        self.weight = 0.0
        self.minRange = 0.0
        self.maxRange = 100.0
        self.faultCount = 0
        self.healthy = True


class FusionResult:
    def __init__(self):
        self.decision = Decision.NONE
        self.confidence = 0.0
        self.needScore = 0.0
        self.weightedScore = 0.0
        self.nnScore = 0.0
        self.finalScore = 0.0


class FusionModule:
    """Direct port of agri::FusionModule."""

    def __init__(self):
        self._lock = threading.Lock()
        self._autoControlEnabled = False
        self._channels = [SensorChannel() for _ in range(K_SENSOR_COUNT)]
        self._result = FusionResult()
        self._lastFusionMs = 0

        # Neural network weights
        self._weightsIH = [[0.0] * K_HIDDEN for _ in range(K_SENSOR_COUNT)]
        self._biasH = [0.0] * K_HIDDEN
        self._weightsHO = [[0.0] * K_OUTPUTS for _ in range(K_HIDDEN)]
        self._biasO = [0.0] * K_OUTPUTS
        self._hiddenOutput = [0.0] * K_HIDDEN
        self._output = [0.0] * K_OUTPUTS

        # Statistics
        self._totalDecisions = 0
        self._irrigationCount = 0
        self._averageConfidence = 0.0

        self._init_channels()
        self._init_network()

    def begin(self, auto_control_enabled: bool = False):
        self._autoControlEnabled = auto_control_enabled
        self._init_channels()
        self._init_network()

    @property
    def auto_control_enabled(self) -> bool:
        return self._autoControlEnabled

    def set_auto_control_enabled(self, enabled: bool):
        self._autoControlEnabled = enabled

    def update(self, snapshot, sample_updated: bool, now_ms: int, actuator: ActuatorController):
        with self._lock:
            if sample_updated:
                raw_values = [
                    snapshot.airTemp,
                    snapshot.airHumi,
                    snapshot.soilHumi,
                    snapshot.lightValue,
                    snapshot.liquidLevel,
                ]

                for i in range(K_SENSOR_COUNT):
                    self._channels[i].rawValue = raw_values[i]
                    self._apply_kalman_filter(self._channels[i], raw_values[i])
                    self._normalize_value(self._channels[i])
                    self._update_reliability(self._channels[i])
                self._calculate_weights()

            if now_ms - self._lastFusionMs < K_FUSION_INTERVAL_MS:
                return

            self._result = self._perform_fusion()
            self._lastFusionMs = now_ms

            if not self._autoControlEnabled or not actuator.status.autoMode or actuator.is_busy(now_ms):
                return

            duration = 0
            if self._result.decision == Decision.MODERATE:
                duration = 45
            elif self._result.decision == Decision.HEAVY:
                duration = 120

            if duration > 0:
                actuator.start_timed_run(ControlSource.FUSION, duration, now_ms)

    def update_weights(self, data: dict):
        """Update NN weights from API request (matches /api/fusion/weights)."""
        with self._lock:
            if "weightsIH" in data:
                wih = data["weightsIH"]
                for i, row in enumerate(wih):
                    if i >= K_SENSOR_COUNT:
                        break
                    for h, val in enumerate(row):
                        if h >= K_HIDDEN:
                            break
                        self._weightsIH[i][h] = float(val)

            if "biasH" in data:
                for h, val in enumerate(data["biasH"]):
                    if h >= K_HIDDEN:
                        break
                    self._biasH[h] = float(val)

            if "weightsHO" in data:
                who = data["weightsHO"]
                for h, row in enumerate(who):
                    if h >= K_HIDDEN:
                        break
                    for o, val in enumerate(row):
                        if o >= K_OUTPUTS:
                            break
                        self._weightsHO[h][o] = float(val)

            if "biasO" in data:
                for o, val in enumerate(data["biasO"]):
                    if o >= K_OUTPUTS:
                        break
                    self._biasO[o] = float(val)

    def get_weights(self) -> dict:
        with self._lock:
            return {
                "weightsIH": [[self._weightsIH[i][h] for h in range(K_HIDDEN)] for i in range(K_SENSOR_COUNT)],
                "biasH": list(self._biasH),
                "weightsHO": [[self._weightsHO[h][o] for o in range(K_OUTPUTS)] for h in range(K_HIDDEN)],
                "biasO": list(self._biasO),
            }

    # ── Channel initialization (matches firmware) ──

    def _init_channels(self):
        c = self._channels
        c[0].name = "AirTemp";  c[0].label = "Temperature"; c[0].unit = "C"
        c[0].kalmanEstimate = 25.0; c[0].weight = 0.20; c[0].maxRange = 40.0

        c[1].name = "AirHumi";  c[1].label = "Humidity";    c[1].unit = "%"
        c[1].kalmanEstimate = 60.0; c[1].weight = 0.20; c[1].maxRange = 100.0

        c[2].name = "SoilHumi"; c[2].label = "Soil";        c[2].unit = "%"
        c[2].kalmanEstimate = 50.0; c[2].weight = 0.30; c[2].maxRange = 100.0

        c[3].name = "Light";    c[3].label = "Light";       c[3].unit = "lux"
        c[3].kalmanEstimate = 500.0; c[3].weight = 0.15; c[3].maxRange = 10000.0

        c[4].name = "Liquid";   c[4].label = "Liquid";      c[4].unit = "%"
        c[4].kalmanEstimate = 80.0; c[4].weight = 0.15; c[4].maxRange = 100.0

    def _init_network(self):
        """Preset weights from firmware initNetwork() — trained on synthetic data (val acc 86.4%)."""
        preset_ih = [
            [-0.7618, -0.1648, 0.9222, 0.2010, 0.1822, 0.8395, 0.6211, -0.7773],
            [0.7061, -0.4602, -0.9656, -0.1033, -0.5191, 0.2046, -0.2469, 1.1025],
            [1.1686, 0.1717, -0.5193, -0.1273, -0.0344, -0.5801, -0.5640, 1.7332],
            [0.1232, -0.2913, 1.0482, 0.3298, -0.1933, 0.3476, 0.3208, -0.1836],
            [-0.4105, 0.1005, 1.5418, -2.6921, -0.4168, 1.2474, -2.1958, -0.7266],
        ]
        preset_bh = [0.5547, -0.0495, 0.2924, 0.5884, -0.0884, 0.3348, 0.4393, 0.7546]
        preset_ho = [
            [1.1199, 0.6752, -2.4982],
            [0.6507, -0.6800, 0.1269],
            [-1.5823, -0.1323, 1.2406],
            [2.7740, -0.9054, -1.8639],
            [-0.6520, 0.0906, -0.5720],
            [-1.1279, 0.6628, 0.5219],
            [2.5613, -2.2228, -0.1253],
            [1.4902, -0.1788, -3.4092],
        ]
        preset_bo = [-0.1035, 0.3601, -0.2272]

        for i in range(K_SENSOR_COUNT):
            for h in range(K_HIDDEN):
                self._weightsIH[i][h] = preset_ih[i][h]
        for h in range(K_HIDDEN):
            self._biasH[h] = preset_bh[h]
        for h in range(K_HIDDEN):
            for o in range(K_OUTPUTS):
                self._weightsHO[h][o] = preset_ho[h][o]
        for o in range(K_OUTPUTS):
            self._biasO[o] = preset_bo[o]

    # ── Kalman filter (matches firmware) ──

    @staticmethod
    def _apply_kalman_filter(channel: SensorChannel, measurement: float):
        kQ = 0.01
        kR = 0.1

        predicted_estimate = channel.kalmanEstimate
        predicted_error = channel.kalmanError + kQ

        channel.kalmanGain = predicted_error / (predicted_error + kR)
        channel.kalmanEstimate = predicted_estimate + channel.kalmanGain * (measurement - predicted_estimate)
        channel.kalmanError = (1.0 - channel.kalmanGain) * predicted_error

    @staticmethod
    def _normalize_value(channel: SensorChannel):
        range_val = channel.maxRange - channel.minRange
        if range_val <= 0.0:
            channel.normalizedValue = 0.0
            return
        channel.normalizedValue = (channel.kalmanEstimate - channel.minRange) / range_val
        channel.normalizedValue = max(0.0, min(1.0, channel.normalizedValue))

    @staticmethod
    def _update_reliability(channel: SensorChannel):
        reliability = 1.0

        if channel.rawValue <= 0.001 or channel.rawValue > channel.maxRange * 1.5:
            reliability *= 0.1
            channel.faultCount += 1
        else:
            channel.faultCount = max(0, channel.faultCount - 1)

        if channel.kalmanGain > 0.8:
            reliability *= 0.7

        if channel.faultCount > 5:
            reliability *= 0.3
            channel.healthy = False
        else:
            channel.healthy = True

        channel.reliability = channel.reliability * 0.8 + reliability * 0.2

    def _calculate_weights(self):
        total_reliability = sum(c.reliability for c in self._channels)
        if total_reliability <= 0.0:
            return
        for c in self._channels:
            c.weight = c.reliability / total_reliability

    # ── Neural network (matches firmware) ──

    def _run_neural_network(self, inputs: list, outputs: list):
        # Hidden layer with ReLU
        for h in range(K_HIDDEN):
            s = self._biasH[h]
            for i in range(K_SENSOR_COUNT):
                s += inputs[i] * self._weightsIH[i][h]
            self._hiddenOutput[h] = max(0.0, s)  # ReLU

        # Output layer
        raw = [0.0] * K_OUTPUTS
        for o in range(K_OUTPUTS):
            s = self._biasO[o]
            for h in range(K_HIDDEN):
                s += self._hiddenOutput[h] * self._weightsHO[h][o]
            raw[o] = s

        # Softmax
        max_val = max(raw)
        exp_vals = [math.exp(r - max_val) for r in raw]
        sum_exp = sum(exp_vals)
        for o in range(K_OUTPUTS):
            outputs[o] = exp_vals[o] / sum_exp
            self._output[o] = outputs[o]

    # ── Fusion decision (matches firmware) ──

    def _perform_fusion(self) -> FusionResult:
        result = FusionResult()

        # Need factors — matches firmware
        need_factors = [
            self._channels[0].normalizedValue,              # AirTemp: higher → more need
            1.0 - self._channels[1].normalizedValue,        # AirHumi: lower humidity → more need
            1.0 - self._channels[2].normalizedValue,        # SoilHumi: lower soil → more need
            self._channels[3].normalizedValue * 0.5,        # Light: partial factor
            self._channels[4].normalizedValue,              # Liquid: higher level → more need
        ]

        result.weightedScore = sum(need_factors[i] * self._channels[i].weight for i in range(K_SENSOR_COUNT)) * 100.0

        inputs = [c.normalizedValue for c in self._channels]
        outputs = [0.0] * K_OUTPUTS
        self._run_neural_network(inputs, outputs)
        result.nnScore = outputs[1] * 50.0 + outputs[2] * 100.0
        result.finalScore = result.weightedScore * 0.6 + result.nnScore * 0.4
        result.needScore = result.finalScore

        # Decision logic — matches firmware
        if self._channels[4].rawValue < 20.0:
            result.decision = Decision.NONE
            result.confidence = 0.95
        elif result.finalScore > 65.0:
            result.decision = Decision.HEAVY
            result.confidence = min(1.0, result.finalScore / 100.0)
        elif result.finalScore > 35.0:
            result.decision = Decision.MODERATE
            result.confidence = 0.5 + (result.finalScore - 35.0) / 60.0
        else:
            result.decision = Decision.NONE
            result.confidence = 1.0 - result.finalScore / 70.0

        self._totalDecisions += 1
        if result.decision != Decision.NONE:
            self._irrigationCount += 1
        self._averageConfidence = self._averageConfidence * 0.95 + result.confidence * 0.05

        return result

    # ── Public accessors ──

    def status(self) -> dict:
        with self._lock:
            return {
                "autoControlEnabled": self._autoControlEnabled,
                "decision": self._result.decision,
                "decisionName": DECISION_NAMES.get(self._result.decision, "none"),
                "confidence": round(self._result.confidence, 4),
                "needScore": round(self._result.needScore, 2),
                "weightedScore": round(self._result.weightedScore, 2),
                "nnScore": round(self._result.nnScore, 2),
                "finalScore": round(self._result.finalScore, 2),
                "totalDecisions": self._totalDecisions,
                "irrigationCount": self._irrigationCount,
                "avgConfidence": round(self._averageConfidence, 4),
                "nn": {
                    "none": round(self._output[0], 4),
                    "moderate": round(self._output[1], 4),
                    "heavy": round(self._output[2], 4),
                },
            }

    def sensors(self) -> list:
        with self._lock:
            return [
                {
                    "name": c.name,
                    "label": c.label,
                    "unit": c.unit,
                    "raw": round(c.rawValue, 2),
                    "filtered": round(c.kalmanEstimate, 2),
                    "normalized": round(c.normalizedValue, 4),
                    "reliability": round(c.reliability, 4),
                    "weight": round(c.weight, 4),
                    "kalmanGain": round(c.kalmanGain, 4),
                    "healthy": c.healthy,
                }
                for c in self._channels
            ]
