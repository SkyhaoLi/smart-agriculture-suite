"""
智润智慧农业套件 - Atlas 200I DK A2 版
环境状态预测模块 - 纯NumPy GRU时序预测

功能:
- 5维传感器输入 (气温/气湿/土湿/光照/灌溉动作)
- 16维GRU隐藏层, 4维输出 (气温/气湿/土湿/光照预测)
- 15/30/60分钟多步预测
- 冷启动: 指数平滑 -> 混合 -> 全GRU
- 在线增量训练: 每5分钟, batch=8, seq_len=15
- 总参数: 1,220个 (~5KB)
"""

import math
import json
import time
import logging
import os
from typing import Optional, List, Dict

import numpy as np

from config.hardware_config import DATA_DIR

logger = logging.getLogger(__name__)

INPUT_DIM = 5    # temp, humi, soil, light, irrigation
HIDDEN_DIM = 16
OUTPUT_DIM = 4   # temp, humi, soil, light (no irrigation prediction)
HISTORY_LEN = 60  # 输入窗口60步(60分钟)
PRED_HORIZONS = [15, 30, 60]  # 预测步长(分钟)
BUFFER_CAPACITY = 1440  # 24小时 = 1440分钟


# ============================================================================
# NumPy GRU 实现
# ============================================================================
class GRUPredictor:
    """纯NumPy GRU: INPUT_DIM -> HIDDEN_DIM -> OUTPUT_DIM"""

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        h = hidden_dim
        scale = 1.0 / math.sqrt(h)
        # GRU门参数: [update, reset, candidate]
        self.W_z = np.random.randn(input_dim, h).astype(np.float32) * scale
        self.U_z = np.random.randn(h, h).astype(np.float32) * scale
        self.b_z = np.zeros(h, dtype=np.float32)
        self.W_r = np.random.randn(input_dim, h).astype(np.float32) * scale
        self.U_r = np.random.randn(h, h).astype(np.float32) * scale
        self.b_r = np.zeros(h, dtype=np.float32)
        self.W_h = np.random.randn(input_dim, h).astype(np.float32) * scale
        self.U_h = np.random.randn(h, h).astype(np.float32) * scale
        self.b_h = np.zeros(h, dtype=np.float32)
        # 输出层
        self.W_out = np.random.randn(h, output_dim).astype(np.float32) * scale
        self.b_out = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x_seq):
        """前向传播, x_seq: (seq_len, input_dim) -> (output_dim,)"""
        seq_len = x_seq.shape[0]
        h = self.hidden_dim
        H = np.zeros((seq_len + 1, h), dtype=np.float32)
        Z = np.zeros((seq_len, h), dtype=np.float32)
        R = np.zeros((seq_len, h), dtype=np.float32)
        Hc = np.zeros((seq_len, h), dtype=np.float32)
        for t in range(seq_len):
            x_t = x_seq[t]
            z = self._sigmoid(x_t @ self.W_z + H[t] @ self.U_z + self.b_z)
            r = self._sigmoid(x_t @ self.W_r + H[t] @ self.U_r + self.b_r)
            h_cand = np.tanh(x_t @ self.W_h + (r * H[t]) @ self.U_h + self.b_h)
            H[t + 1] = z * H[t] + (1.0 - z) * h_cand
            Z[t], R[t], Hc[t] = z, r, h_cand
        output = H[seq_len] @ self.W_out + self.b_out
        cache = (x_seq, H, Z, R, Hc)
        return output, cache

    def backward(self, output_grad, cache, lr=0.001, max_grad_norm=5.0):
        """BPTT + SGD更新"""
        x_seq, H, Z, R, Hc = cache
        seq_len = x_seq.shape[0]
        h = self.hidden_dim
        dW_out = H[seq_len].reshape(-1, 1) @ output_grad.reshape(1, -1)
        db_out = output_grad.copy()
        dh_next = output_grad @ self.W_out.T
        dW_z = np.zeros_like(self.W_z)
        dU_z = np.zeros_like(self.U_z)
        db_z = np.zeros_like(self.b_z)
        dW_r = np.zeros_like(self.W_r)
        dU_r = np.zeros_like(self.U_r)
        db_r = np.zeros_like(self.b_r)
        dW_h = np.zeros_like(self.W_h)
        dU_h = np.zeros_like(self.U_h)
        db_h = np.zeros_like(self.b_h)
        for t in range(seq_len - 1, -1, -1):
            dh_raw = dh_next * (1.0 - Z[t])
            dtanh = dh_raw * (1.0 - Hc[t] ** 2)
            dW_h += x_seq[t].reshape(-1, 1) @ dtanh.reshape(1, -1)
            dU_h += (R[t] * H[t]).reshape(-1, 1) @ dtanh.reshape(1, -1)
            db_h += dtanh
            dr_h = dtanh @ self.U_h.T
            dr_raw = R[t] * (1.0 - R[t]) * (H[t] * dr_h)
            dW_r += x_seq[t].reshape(-1, 1) @ dr_raw.reshape(1, -1)
            dU_r += H[t].reshape(-1, 1) @ dr_raw.reshape(1, -1)
            db_r += dr_raw
            dz_raw = Z[t] * (1.0 - Z[t]) * (H[t] - Hc[t]) * dh_next
            dW_z += x_seq[t].reshape(-1, 1) @ dz_raw.reshape(1, -1)
            dU_z += H[t].reshape(-1, 1) @ dz_raw.reshape(1, -1)
            db_z += dz_raw
            dh_next = dh_next * Z[t] + dz_raw @ self.U_z.T + dr_raw @ self.U_r.T
        all_grads = [dW_z, dU_z, db_z, dW_r, dU_r, db_r, dW_h, dU_h, db_h, dW_out, db_out]
        total_norm = math.sqrt(sum(float(np.sum(g ** 2)) for g in all_grads))
        if total_norm > max_grad_norm:
            scale = max_grad_norm / (total_norm + 1e-6)
            all_grads = [g * scale for g in all_grads]
        dW_z, dU_z, db_z, dW_r, dU_r, db_r, dW_h, dU_h, db_h, dW_out, db_out = all_grads
        self.W_z -= lr * dW_z
        self.U_z -= lr * dU_z
        self.b_z -= lr * db_z
        self.W_r -= lr * dW_r
        self.U_r -= lr * dU_r
        self.b_r -= lr * db_r
        self.W_h -= lr * dW_h
        self.U_h -= lr * dU_h
        self.b_h -= lr * db_h
        self.W_out -= lr * dW_out
        self.b_out -= lr * db_out

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))


# ============================================================================
# 环形缓冲区
# ============================================================================
class SensorHistoryBuffer:
    """1分钟聚合的环形缓冲区, 容量1440条(24小时)"""

    def __init__(self, capacity=BUFFER_CAPACITY):
        self.capacity = capacity
        self.data = np.zeros((capacity, INPUT_DIM), dtype=np.float32)
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self.head = 0
        self.count = 0
        # 1分钟聚合临时缓冲
        self._agg_values = np.zeros(INPUT_DIM, dtype=np.float32)
        self._agg_count = 0
        self._agg_start_time = 0.0

    def add_sample(self, values, timestamp, is_irrigating=False):
        """添加原始样本, 自动1分钟聚合"""
        vals = np.array(values, dtype=np.float32)
        if is_irrigating:
            vals = np.append(vals, 1.0)
        else:
            vals = np.append(vals, 0.0)
        if self._agg_count == 0:
            self._agg_start_time = timestamp
        self._agg_values += vals
        self._agg_count += 1
        if timestamp - self._agg_start_time >= 60.0 and self._agg_count > 0:
            self._flush_aggregate(timestamp)

    def _flush_aggregate(self, timestamp):
        avg = self._agg_values / self._agg_count
        self.data[self.head] = avg
        self.timestamps[self.head] = timestamp
        self.head = (self.head + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1
        self._agg_values[:] = 0.0
        self._agg_count = 0

    def get_recent(self, n):
        """获取最近n条聚合数据, 返回 (n_actual, input_dim)"""
        n = min(n, self.count)
        if n == 0:
            return np.zeros((0, INPUT_DIM), dtype=np.float32)
        idx = (self.head - n + self.capacity) % self.capacity
        if idx + n <= self.capacity:
            return self.data[idx:idx + n].copy()
        else:
            part1 = self.data[idx:]
            part2 = self.data[:n - len(part1)]
            return np.concatenate([part1, part2], axis=0)


# ============================================================================
# 预测结果
# ============================================================================
class PredictionResult:
    """环境预测结果"""
    __slots__ = [
        'predicted_temp', 'predicted_humi', 'predicted_soil', 'predicted_light',
        'confidence', 'mode', 'training_steps', 'avg_loss',
        'buffer_minutes', 'timestamp', 'soil_moisture_risk', 'risk_horizon',
    ]

    def __init__(self):
        self.predicted_temp: Dict[int, float] = {15: 0, 30: 0, 60: 0}
        self.predicted_humi: Dict[int, float] = {15: 0, 30: 0, 60: 0}
        self.predicted_soil: Dict[int, float] = {15: 0, 30: 0, 60: 0}
        self.predicted_light: Dict[int, float] = {15: 0, 30: 0, 60: 0}
        self.confidence: Dict[int, float] = {15: 0.0, 30: 0.0, 60: 0.0}
        self.mode: str = "cold_start"
        self.training_steps: int = 0
        self.avg_loss: float = 0.0
        self.buffer_minutes: int = 0
        self.timestamp: float = 0.0
        self.soil_moisture_risk: bool = False
        self.risk_horizon: int = 0

    @property
    def risk_score(self) -> float:
        """综合风险评分 0.0~1.0"""
        if not self.soil_moisture_risk:
            return 0.0
        if self.risk_horizon <= 15:
            return 0.9
        if self.risk_horizon <= 30:
            return 0.6
        return 0.3


# ============================================================================
# 世界模型主类
# ============================================================================
class WorldModelModule:
    """环境状态预测模块 - GRU时序预测 + 指数平滑冷启动"""

    PREDICT_INTERVAL = 300.0   # 5分钟
    TRAIN_INTERVAL = 300.0     # 5分钟
    MIN_HISTORY_FOR_GRU = 15   # GRU最少历史(分钟)
    MIN_HISTORY_FOR_BLEND = 10  # 混合模式最少历史
    ALPHA_TEMP = 0.3
    ALPHA_HUMI = 0.3
    ALPHA_SOIL = 0.25
    ALPHA_LIGHT = 0.4

    def __init__(self, data_dir=DATA_DIR):
        self._data_dir = data_dir
        self._auto_control_enabled = True
        self._buffer = SensorHistoryBuffer()
        self._gru = GRUPredictor()
        self._training_steps = 0
        self._avg_loss = 0.0
        self._last_train_time = 0.0
        self._last_predict_time = 0.0
        self._last_irrigation = False
        self._es_values = np.zeros(OUTPUT_DIM, dtype=np.float32)
        self._es_initialized = False
        self._result = PredictionResult()
        self._load()

    def begin(self, auto_control_enabled=True):
        self._auto_control_enabled = auto_control_enabled

    def update(self, snapshot, sample_updated, now, actuator):
        if sample_updated:
            values = [snapshot.air_temp, snapshot.air_humi,
                      snapshot.soil_humi, snapshot.light_intensity]
            irrigating = actuator.status.valve_on if actuator else False
            self._buffer.add_sample(values, now, irrigating)
            self._update_exponential_smoothing(values)
            self._last_irrigation = irrigating

        if now - self._last_train_time >= self.TRAIN_INTERVAL and self._buffer.count >= self.MIN_HISTORY_FOR_GRU:
            self._train_step()
            self._last_train_time = now

        if now - self._last_predict_time >= self.PREDICT_INTERVAL:
            self._generate_predictions(now)
            self._last_predict_time = now

        return self._result

    def _update_exponential_smoothing(self, values):
        if not self._es_initialized:
            self._es_values = np.array(values, dtype=np.float32)
            self._es_initialized = True
            return
        alphas = [self.ALPHA_TEMP, self.ALPHA_HUMI, self.ALPHA_SOIL, self.ALPHA_LIGHT]
        for i in range(OUTPUT_DIM):
            self._es_values[i] = alphas[i] * values[i] + (1.0 - alphas[i]) * self._es_values[i]

    def _generate_predictions(self, now):
        n = self._buffer.count
        if n < 2:
            return
        if n < self.MIN_HISTORY_FOR_GRU:
            self._generate_es_predictions(now, n)
            return
        if n < self.MIN_HISTORY_FOR_BLEND:
            self._generate_blended_predictions(now, n)
            return
        self._generate_gru_predictions(now, n)

    def _generate_gru_predictions(self, now, n):
        input_seq = self._buffer.get_recent(min(HISTORY_LEN, n))
        current = input_seq[-1, :OUTPUT_DIM]
        predictions = {}
        confidence_by_horizon = {}
        for horizon in PRED_HORIZONS:
            n_steps = min(horizon, 60)
            pred_input = input_seq.copy()
            preds = []
            for _ in range(n_steps):
                window = pred_input[-min(HISTORY_LEN, len(pred_input)):]
                output, _ = self._gru.forward(window)
                pred_input = np.vstack([pred_input, np.append(output, 0.0)])
                preds.append(output)
            predictions[horizon] = preds[-1]
            n_samples = min(n, 200)
            recent = self._buffer.get_recent(n_samples)[:, :OUTPUT_DIM]
            std = np.std(recent, axis=0) + 1e-6
            max_err = np.max(np.abs(preds[-1] - current) / std)
            confidence_by_horizon[horizon] = float(max(0.3, min(1.0, 1.0 - max_err * 0.15)))
        soil_dry = predictions[60][2] < 25.0
        self._result = PredictionResult()
        self._result.predicted_temp = {h: float(predictions[h][0]) for h in PRED_HORIZONS}
        self._result.predicted_humi = {h: float(predictions[h][1]) for h in PRED_HORIZONS}
        self._result.predicted_soil = {h: float(predictions[h][2]) for h in PRED_HORIZONS}
        self._result.predicted_light = {h: float(predictions[h][3]) for h in PRED_HORIZONS}
        self._result.confidence = confidence_by_horizon
        self._result.mode = "gru"
        self._result.training_steps = self._training_steps
        self._result.avg_loss = self._avg_loss
        self._result.buffer_minutes = self._buffer.count
        self._result.timestamp = now
        self._result.soil_moisture_risk = soil_dry
        self._result.risk_horizon = 60 if soil_dry else 0

    def _generate_es_predictions(self, now, n):
        preds = {h: self._es_values.copy() for h in PRED_HORIZONS}
        conf = {h: max(0.3, 0.5 + n * 0.02) for h in PRED_HORIZONS}
        self._result = PredictionResult()
        self._result.predicted_temp = {h: float(preds[h][0]) for h in PRED_HORIZONS}
        self._result.predicted_humi = {h: float(preds[h][1]) for h in PRED_HORIZONS}
        self._result.predicted_soil = {h: float(preds[h][2]) for h in PRED_HORIZONS}
        self._result.predicted_light = {h: float(preds[h][3]) for h in PRED_HORIZONS}
        self._result.confidence = conf
        self._result.mode = "exponential_smoothing"
        self._result.buffer_minutes = n
        self._result.timestamp = now

    def _generate_blended_predictions(self, now, n):
        es_pred = self._es_values.copy()
        input_seq = self._buffer.get_recent(n)
        output, _ = self._gru.forward(input_seq)
        gru_pred = output
        alpha = (n - self.MIN_HISTORY_FOR_BLEND) / (self.MIN_HISTORY_FOR_GRU - self.MIN_HISTORY_FOR_BLEND)
        alpha = max(0.0, min(1.0, alpha))
        blended = (1.0 - alpha) * es_pred + alpha * gru_pred
        conf = {h: max(0.3, 0.4 + alpha * 0.4) for h in PRED_HORIZONS}
        self._result = PredictionResult()
        self._result.predicted_temp = {h: float(blended[0]) for h in PRED_HORIZONS}
        self._result.predicted_humi = {h: float(blended[1]) for h in PRED_HORIZONS}
        self._result.predicted_soil = {h: float(blended[2]) for h in PRED_HORIZONS}
        self._result.predicted_light = {h: float(blended[3]) for h in PRED_HORIZONS}
        self._result.confidence = conf
        self._result.mode = "blended"
        self._result.buffer_minutes = n
        self._result.timestamp = now

    def _train_step(self):
        n = self._buffer.count
        if n < self.MIN_HISTORY_FOR_GRU + 1:
            return
        batch_size = min(8, n - self.MIN_HISTORY_FOR_GRU)
        seq_len = min(15, n - 1)
        indices = np.random.choice(n - seq_len, size=batch_size, replace=False)
        batch_loss = 0.0
        lr = max(0.0001, 0.001 * (0.999 ** (self._training_steps / 100)))
        for idx in indices:
            seq = self._buffer.get_recent(seq_len + 1)
            if len(seq) < seq_len + 1:
                continue
            x = seq[:seq_len]
            target = seq[seq_len, :OUTPUT_DIM]
            pred, cache = self._gru.forward(x)
            error = pred - target
            loss = float(np.mean(error ** 2))
            batch_loss += loss
            self._gru.backward(error, cache, lr=lr)
        self._training_steps += 1
        batch_loss /= max(1, batch_size)
        self._avg_loss = 0.95 * self._avg_loss + 0.05 * batch_loss

    def get_prediction_risk(self):
        """返回预测风险评分 0.0~1.0, 供其他模块调用"""
        return self._result.risk_score

    def get_predictions(self):
        """返回预测结果, 供API调用"""
        return self._result

    def get_history(self, minutes=60):
        """返回历史数据, 供API调用"""
        n = min(minutes, self._buffer.count)
        if n == 0:
            return []
        data = self._buffer.get_recent(n)
        ts = []
        idx = (self._buffer.head - n + self._buffer.capacity) % self._buffer.capacity
        for i in range(n):
            pos = (idx + i) % self._buffer.capacity
            ts.append(self._buffer.timestamps[pos])
        result = []
        for i in range(n):
            result.append({
                "timestamp": float(ts[i]),
                "temp": round(float(data[i, 0]), 2),
                "humi": round(float(data[i, 1]), 2),
                "soil": round(float(data[i, 2]), 2),
                "light": round(float(data[i, 3]), 2),
                "irrigation": float(data[i, 4]),
            })
        return result

    def save(self):
        os.makedirs(self._data_dir, exist_ok=True)
        path = os.path.join(self._data_dir, "world_model.json")
        try:
            data = {
                "training_steps": self._training_steps,
                "avg_loss": self._avg_loss,
                "gru": {
                    "W_z": self._gru.W_z.tolist(), "U_z": self._gru.U_z.tolist(), "b_z": self._gru.b_z.tolist(),
                    "W_r": self._gru.W_r.tolist(), "U_r": self._gru.U_r.tolist(), "b_r": self._gru.b_r.tolist(),
                    "W_h": self._gru.W_h.tolist(), "U_h": self._gru.U_h.tolist(), "b_h": self._gru.b_h.tolist(),
                    "W_out": self._gru.W_out.tolist(), "b_out": self._gru.b_out.tolist(),
                },
            }
            with open(path, 'w') as f:
                json.dump(data, f)
            logger.info("世界模型权重已保存")
        except Exception as e:
            logger.warning(f"世界模型保存失败: {e}")

    def _load(self):
        path = os.path.join(self._data_dir, "world_model.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._training_steps = data.get("training_steps", 0)
            self._avg_loss = data.get("avg_loss", 0.0)
            g = data["gru"]
            self._gru.W_z = np.array(g["W_z"], dtype=np.float32)
            self._gru.U_z = np.array(g["U_z"], dtype=np.float32)
            self._gru.b_z = np.array(g["b_z"], dtype=np.float32)
            self._gru.W_r = np.array(g["W_r"], dtype=np.float32)
            self._gru.U_r = np.array(g["U_r"], dtype=np.float32)
            self._gru.b_r = np.array(g["b_r"], dtype=np.float32)
            self._gru.W_h = np.array(g["W_h"], dtype=np.float32)
            self._gru.U_h = np.array(g["U_h"], dtype=np.float32)
            self._gru.b_h = np.array(g["b_h"], dtype=np.float32)
            self._gru.W_out = np.array(g["W_out"], dtype=np.float32)
            self._gru.b_out = np.array(g["b_out"], dtype=np.float32)
            logger.info("世界模型权重已加载")
        except Exception as e:
            logger.warning(f"世界模型加载失败: {e}")

    def to_dict(self):
        return {
            "autoControlEnabled": self._auto_control_enabled,
            "mode": self._result.mode,
            "predictions": {
                "temp": {str(h): round(self._result.predicted_temp.get(h, 0), 2) for h in PRED_HORIZONS},
                "humi": {str(h): round(self._result.predicted_humi.get(h, 0), 2) for h in PRED_HORIZONS},
                "soil": {str(h): round(self._result.predicted_soil.get(h, 0), 2) for h in PRED_HORIZONS},
                "light": {str(h): round(self._result.predicted_light.get(h, 0), 2) for h in PRED_HORIZONS},
            },
            "confidence": {str(h): round(v, 3) for h, v in self._result.confidence.items()},
            "riskScore": round(self._result.risk_score, 3),
            "soilRisk": bool(self._result.soil_moisture_risk),
            "riskHorizon": self._result.risk_horizon,
            "model": {
                "trainingSteps": self._training_steps,
                "avgLoss": round(self._avg_loss, 6),
                "bufferMinutes": self._buffer.count,
            },
        }
