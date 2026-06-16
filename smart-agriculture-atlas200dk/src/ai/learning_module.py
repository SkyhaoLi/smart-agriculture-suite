"""
智润智慧农业套件 - Atlas 200I DK A2 版
Q-Learning学习模块 - 900状态空间, 4动作, 文件持久化Q表

对应原ESP32项目的 LearningModule.h/LearningModule.cpp
使用JSON文件持久化替代NVS
"""

import math
import time
import json
import random
import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from config.app_types import SensorSnapshot, ControlSource, IrrigationAction
from config.hardware_config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    auto_control_enabled: bool = True
    decision_interval_sec: float = 300.0  # 5分钟
    epsilon: float = 0.3
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995
    alpha: float = 0.1        # 学习率
    gamma: float = 0.95       # 折扣因子
    target_soil: float = 55.0  # 目标土壤湿度
    soil_tolerance: float = 10.0

    # 离散化分档
    temp_bins: int = 5
    humi_bins: int = 4
    soil_bins: int = 5
    light_bins: int = 3
    time_bins: int = 3


ACTION_DURATIONS = [0, 30, 45, 120]  # Off, Low, Moderate, Heavy (秒)
ACTION_NAMES = ["off", "low", "medium", "high"]


class LearningModule:
    """Q-Learning灌溉决策 - 5×4×5×3×3 = 900状态, 4动作"""

    STATE_COUNT = 5 * 4 * 5 * 3 * 3  # 900

    def __init__(self, config: LearningConfig = None, data_dir: str = DATA_DIR):
        self._config = config or LearningConfig()
        self._data_dir = data_dir

        self._q_table = np.zeros((self.STATE_COUNT, 4))
        self._total_episodes = 0
        self._total_reward = 0.0
        self._average_reward = 0.0
        self._last_reward = 0.0
        self._last_state = 0
        self._last_action = IrrigationAction.Off
        self._has_pending_reward = False
        self._pending_soil_before = 0.0
        self._last_decision_time = 0.0
        self._latest_snapshot = SensorSnapshot()

        self._user_override_count = 0
        self._user_satisfaction = 50.0

        # 历史记录 (最近100条)
        self._history: List[dict] = []

    def begin(self):
        self._load_q_table()

    def update(self, snapshot: SensorSnapshot, sample_updated: bool,
               now: float, actuator, prediction_risk: float = 0.0) -> Optional[IrrigationAction]:
        if not sample_updated:
            return None

        self._latest_snapshot = snapshot
        current_state = self._discretize_state(snapshot)

        # 处理上次决策的奖励
        if (self._has_pending_reward and
                now - self._last_decision_time >= self._config.decision_interval_sec):
            reward = self._calculate_reward(
                self._pending_soil_before, snapshot.soil_humi, self._last_action, prediction_risk)
            self._last_reward = reward
            self._total_reward += reward
            self._average_reward = (self._total_reward / self._total_episodes
                                     if self._total_episodes > 0 else 0.0)
            self._update_q_value(self._last_state, self._last_action, reward, current_state)

            self._history.append({
                "state": self._last_state,
                "action": ACTION_NAMES[self._last_action.value],
                "reward": round(reward, 3),
                "soilBefore": round(self._pending_soil_before, 1),
                "soilAfter": round(snapshot.soil_humi, 1),
            })
            if len(self._history) > 100:
                self._history.pop(0)

            self._has_pending_reward = False

        # 检查是否需要做决策
        if (not self._config.auto_control_enabled or
                not actuator._auto_mode or
                actuator.is_busy(now)):
            return None

        if now - self._last_decision_time < self._config.decision_interval_sec:
            return None

        action = self._select_action(current_state)
        if action != IrrigationAction.Off:
            duration = ACTION_DURATIONS[action.value]
            actuator.start_timed_run(ControlSource.TimedRun, duration, now)

        self._pending_soil_before = snapshot.soil_humi
        self._last_state = current_state
        self._last_action = action
        self._has_pending_reward = True
        self._total_episodes += 1
        self._last_decision_time = now

        # 自适应epsilon衰减
        decay_factor = self._config.epsilon_decay
        if self._total_episodes > 100 and self._average_reward > 2.0:
            decay_factor *= 0.998
        self._config.epsilon = max(self._config.epsilon_min,
                                    self._config.epsilon * decay_factor)

        if self._total_episodes % 50 == 0:
            self._save_q_table()

        return action

    def record_user_feedback(self, positive: bool):
        feedback_reward = 5.0 if positive else -5.0
        state = self._discretize_state(self._latest_snapshot)
        self._update_q_value(state, self._last_action, feedback_reward, state)
        self._user_override_count += 1
        self._user_satisfaction = max(0.0, min(100.0,
            self._user_satisfaction + (2.0 if positive else -3.0)))

    def reset(self):
        self._q_table = np.zeros((self.STATE_COUNT, 4))
        self._total_episodes = 0
        self._total_reward = 0.0
        self._average_reward = 0.0
        self._last_reward = 0.0
        self._has_pending_reward = False
        self._history.clear()
        self._config.epsilon = 0.3
        self._save_q_table()

    # ------------------------------------------------------------------
    # 状态离散化
    # ------------------------------------------------------------------
    def _discretize_state(self, snapshot: SensorSnapshot) -> int:
        t = self._discretize_temp(snapshot.air_temp)
        h = self._discretize_humi(snapshot.air_humi)
        s = self._discretize_soil(snapshot.soil_humi)
        l = self._discretize_light(snapshot.light_intensity)
        p = self._get_time_period()

        return (t * (4 * 5 * 3 * 3) +
                h * (5 * 3 * 3) +
                s * (3 * 3) +
                l * 3 + p)

    @staticmethod
    def _discretize_temp(temp: float) -> int:
        if temp < 10: return 0
        if temp < 18: return 1
        if temp < 25: return 2
        if temp < 33: return 3
        return 4

    @staticmethod
    def _discretize_humi(humi: float) -> int:
        if humi < 30: return 0
        if humi < 50: return 1
        if humi < 70: return 2
        return 3

    @staticmethod
    def _discretize_soil(soil: float) -> int:
        if soil < 20: return 0
        if soil < 35: return 1
        if soil < 50: return 2
        if soil < 65: return 3
        return 4

    @staticmethod
    def _discretize_light(light: float) -> int:
        if light < 100: return 0
        if light < 500: return 1
        return 2

    @staticmethod
    def _get_time_period() -> int:
        hour = time.localtime().tm_hour
        if 6 <= hour < 12: return 0
        if 12 <= hour < 18: return 1
        return 2

    # ------------------------------------------------------------------
    # 动作选择与Q值更新
    # ------------------------------------------------------------------
    def _select_action(self, state: int) -> IrrigationAction:
        if random.random() < self._config.epsilon:
            return IrrigationAction(random.randint(0, 3))
        best = int(np.argmax(self._q_table[state]))
        return IrrigationAction(best)

    def _calculate_reward(self, soil_before: float, soil_after: float,
                           action: IrrigationAction, prediction_risk: float = 0.0) -> float:
        cfg = self._config
        diff_after = abs(soil_after - cfg.target_soil)
        diff_before = abs(soil_before - cfg.target_soil)

        reward = 0.0

        if diff_after <= cfg.soil_tolerance:
            reward += 10.0
        else:
            reward -= diff_after * 0.3

        if diff_after < diff_before:
            reward += 3.0

        if action == IrrigationAction.Off and soil_before > cfg.target_soil - cfg.soil_tolerance:
            reward += 2.0

        if action == IrrigationAction.Low and diff_before < cfg.soil_tolerance * 2.0:
            reward += 1.5

        if action == IrrigationAction.Heavy and diff_before < cfg.soil_tolerance:
            reward -= 1.0

        if soil_after > 80.0:
            reward -= 5.0
        if soil_after < 20.0:
            reward -= 8.0

        # 预测风险惩罚: 土壤预测偏低时, 不灌溉会受额外惩罚
        if prediction_risk > 0.3 and action == IrrigationAction.Off:
            reward -= prediction_risk * 2.0

        if soil_before < cfg.target_soil and soil_after > cfg.target_soil + cfg.soil_tolerance:
            reward -= 2.0

        return reward

    def _update_q_value(self, state: int, action: IrrigationAction,
                         reward: float, next_state: int):
        max_next_q = float(np.max(self._q_table[next_state]))
        current_q = self._q_table[state][action.value]
        new_q = current_q + self._config.alpha * (
            reward + self._config.gamma * max_next_q - current_q)
        self._q_table[state][action.value] = new_q

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _load_q_table(self):
        import os
        path = os.path.join(self._data_dir, "q_table.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self._q_table = np.array(data["q_table"])
                self._total_episodes = data.get("episodes", 0)
                self._average_reward = data.get("avg_reward", 0.0)
                self._config.epsilon = data.get("epsilon", self._config.epsilon)
                self._total_reward = self._average_reward * self._total_episodes
                logger.info(f"Q表已加载: {self._total_episodes}轮")
            except Exception as e:
                logger.warning(f"Q表加载失败: {e}")

    def _save_q_table(self):
        import os
        os.makedirs(self._data_dir, exist_ok=True)
        path = os.path.join(self._data_dir, "q_table.json")
        try:
            with open(path, 'w') as f:
                json.dump({
                    "q_table": self._q_table.tolist(),
                    "episodes": self._total_episodes,
                    "avg_reward": self._average_reward,
                    "epsilon": self._config.epsilon,
                }, f)
            logger.info(f"Q表已保存: {self._total_episodes}轮")
        except Exception as e:
            logger.warning(f"Q表保存失败: {e}")

    def to_dict(self) -> dict:
        state = self._discretize_state(self._latest_snapshot)
        best_action = int(np.argmax(self._q_table[state]))
        return {
            "autoControlEnabled": self._config.auto_control_enabled,
            "currentState": state,
            "lastAction": self._last_action.value,
            "lastActionName": ACTION_NAMES[self._last_action.value],
            "lastReward": round(self._last_reward, 3),
            "averageReward": round(self._average_reward, 3),
            "totalEpisodes": self._total_episodes,
            "epsilon": round(self._config.epsilon, 5),
            "targetSoil": self._config.target_soil,
            "userOverrides": self._user_override_count,
            "userSatisfaction": round(self._user_satisfaction, 1),
            "recommendedAction": ACTION_NAMES[best_action],
            "qValues": [
                {"action": ACTION_NAMES[a], "value": round(float(self._q_table[state][a]), 4)}
                for a in range(4)
            ],
        }
