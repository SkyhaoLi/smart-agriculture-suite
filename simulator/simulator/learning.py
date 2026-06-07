"""LearningModule — Q-Learning irrigation optimization matching firmware."""

import random
import math
import threading
import numpy as np
from .time_clock import SimClock
from .actuator import ActuatorController, ControlSource


# Action space — matches firmware
OFF = 0
LOW = 1
MEDIUM = 2
HIGH = 3
ACTION_COUNT = 4
ACTION_NAMES = ["off", "low", "medium", "high"]
ACTION_DURATIONS = [0, 30, 60, 120]  # seconds

# State space — matches firmware
K_TEMP_LEVELS = 5
K_HUMI_LEVELS = 4
K_SOIL_LEVELS = 5
K_LIGHT_LEVELS = 3
K_TIME_LEVELS = 3
K_STATE_COUNT = K_TEMP_LEVELS * K_HUMI_LEVELS * K_SOIL_LEVELS * K_LIGHT_LEVELS * K_TIME_LEVELS  # 900

K_HISTORY_SIZE = 20


class LearningConfig:
    """Matches firmware LearningConfig."""

    def __init__(self):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.3
        self.epsilonDecay = 0.999
        self.epsilonMin = 0.05
        self.targetSoil = 55.0
        self.soilTolerance = 10.0
        self.decisionIntervalMs = 300000
        self.autoControlEnabled = False

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilonDecay": self.epsilonDecay,
            "epsilonMin": self.epsilonMin,
            "targetSoil": self.targetSoil,
            "soilTolerance": self.soilTolerance,
            "decisionIntervalMs": self.decisionIntervalMs,
            "autoControlEnabled": self.autoControlEnabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LearningConfig":
        c = cls()
        for k in ["alpha", "gamma", "epsilon", "epsilonDecay", "epsilonMin",
                    "targetSoil", "soilTolerance", "decisionIntervalMs", "autoControlEnabled"]:
            if k in d:
                setattr(c, k, d[k])
        return c


class LearningModule:
    """Direct port of agri::LearningModule."""

    def __init__(self, clock: SimClock):
        self._clock = clock
        self._lock = threading.Lock()
        self._config = LearningConfig()

        # Q-Table: 900 states × 4 actions
        self._qTable = np.zeros((K_STATE_COUNT, ACTION_COUNT), dtype=np.float32)

        # State tracking
        self._latestSnapshot = None
        self._lastState = 0
        self._lastAction = OFF
        self._lastReward = 0.0
        self._hasPendingReward = False
        self._pendingSoilBefore = 0.0
        self._lastDecisionMs = 0

        # Statistics
        self._totalEpisodes = 0
        self._totalReward = 0.0
        self._averageReward = 0.0
        self._userOverrideCount = 0
        self._userSatisfaction = 50.0

        # History
        self._history = [{} for _ in range(K_HISTORY_SIZE)]
        self._historyIndex = 0
        self._historyCount = 0

    def begin(self, config: dict):
        self._config = LearningConfig.from_dict(config)

    @property
    def config(self) -> LearningConfig:
        return self._config

    @property
    def last_reward(self) -> float:
        return self._lastReward

    def set_config(self, config: LearningConfig):
        self._config = config

    def update(self, snapshot, sample_updated: bool, now_ms: int, actuator: ActuatorController):
        with self._lock:
            if not sample_updated:
                return

            self._latestSnapshot = snapshot
            current_state = self._discretize_state(snapshot)

            # Pending reward evaluation
            if self._hasPendingReward and now_ms - self._lastDecisionMs >= self._config.decisionIntervalMs:
                reward = self._calculate_reward(self._pendingSoilBefore, snapshot.soilHumi, self._lastAction)
                self._lastReward = reward
                self._totalReward += reward
                self._averageReward = self._totalReward / self._totalEpisodes if self._totalEpisodes > 0 else 0.0
                self._update_q_value(self._lastState, self._lastAction, reward, current_state)

                # Record history
                self._history[self._historyIndex % K_HISTORY_SIZE] = {
                    "state": self._lastState,
                    "action": self._lastAction,
                    "reward": reward,
                    "soilBefore": self._pendingSoilBefore,
                    "soilAfter": snapshot.soilHumi,
                }
                self._historyIndex = (self._historyIndex + 1) % K_HISTORY_SIZE
                self._historyCount = min(self._historyCount + 1, K_HISTORY_SIZE)

                self._hasPendingReward = False

            # Decision
            if not self._config.autoControlEnabled or not actuator.status.autoMode or actuator.is_busy(now_ms):
                return
            if now_ms - self._lastDecisionMs < self._config.decisionIntervalMs:
                return

            action = self._select_action(current_state)
            if action != OFF:
                actuator.start_timed_run(ControlSource.LEARNING, ACTION_DURATIONS[action], now_ms)

            self._pendingSoilBefore = snapshot.soilHumi
            self._lastState = current_state
            self._lastAction = action
            self._hasPendingReward = True
            self._totalEpisodes += 1
            self._lastDecisionMs = now_ms

            # Adaptive epsilon decay
            decay_factor = self._config.epsilonDecay
            if self._totalEpisodes > 100 and self._averageReward > 2.0:
                decay_factor = self._config.epsilonDecay * 0.998
            self._config.epsilon = max(self._config.epsilonMin, self._config.epsilon * decay_factor)

    def record_user_feedback(self, positive: bool):
        with self._lock:
            feedback_reward = 5.0 if positive else -5.0
            if self._latestSnapshot:
                state = self._discretize_state(self._latestSnapshot)
                self._update_q_value(state, self._lastAction, feedback_reward, state)
            self._userOverrideCount += 1
            self._userSatisfaction = max(0.0, min(100.0, self._userSatisfaction + (2.0 if positive else -3.0)))

    def reset(self):
        with self._lock:
            self._qTable = np.zeros((K_STATE_COUNT, ACTION_COUNT), dtype=np.float32)
            self._totalEpisodes = 0
            self._totalReward = 0.0
            self._averageReward = 0.0
            self._lastReward = 0.0
            self._lastState = 0
            self._lastAction = OFF
            self._hasPendingReward = False
            self._historyCount = 0
            self._historyIndex = 0
            self._config.epsilon = 0.3

    # ── Discretization (matches firmware) ──

    def _discretize_state(self, snapshot) -> int:
        temp = self._discretize_temp(snapshot.airTemp)
        humi = self._discretize_humi(snapshot.airHumi)
        soil = self._discretize_soil(snapshot.soilHumi)
        light = self._discretize_light(snapshot.lightValue)
        period = self._get_time_period()

        return (temp * (K_HUMI_LEVELS * K_SOIL_LEVELS * K_LIGHT_LEVELS * K_TIME_LEVELS) +
                humi * (K_SOIL_LEVELS * K_LIGHT_LEVELS * K_TIME_LEVELS) +
                soil * (K_LIGHT_LEVELS * K_TIME_LEVELS) +
                light * K_TIME_LEVELS + period)

    @staticmethod
    def _discretize_temp(temp: float) -> int:
        if temp < 10.0: return 0
        if temp < 18.0: return 1
        if temp < 25.0: return 2
        if temp < 33.0: return 3
        return 4

    @staticmethod
    def _discretize_humi(humi: float) -> int:
        if humi < 30.0: return 0
        if humi < 50.0: return 1
        if humi < 70.0: return 2
        return 3

    @staticmethod
    def _discretize_soil(soil: float) -> int:
        if soil < 20.0: return 0
        if soil < 35.0: return 1
        if soil < 50.0: return 2
        if soil < 65.0: return 3
        return 4

    @staticmethod
    def _discretize_light(light: float) -> int:
        if light < 100.0: return 0
        if light < 500.0: return 1
        return 2

    def _get_time_period(self) -> int:
        hours = (self._clock.millis() // 3600000) % 24
        if 6 <= hours < 12: return 0
        if 12 <= hours < 18: return 1
        return 2

    # ── Q-Learning core (matches firmware) ──

    def _select_action(self, state: int) -> int:
        if random.random() < self._config.epsilon:
            return random.randint(0, ACTION_COUNT - 1)

        best_action = int(np.argmax(self._qTable[state]))
        return best_action

    def _calculate_reward(self, soil_before: float, soil_after: float, action: int) -> float:
        reward = 0.0
        diff_after = abs(soil_after - self._config.targetSoil)
        diff_before = abs(soil_before - self._config.targetSoil)

        # Core reward: proximity to target
        if diff_after <= self._config.soilTolerance:
            reward += 10.0
        else:
            reward -= diff_after * 0.3

        # Improvement reward
        if diff_after < diff_before:
            reward += 3.0

        # Efficiency: prefer not watering when adequate
        if action == OFF and soil_before > self._config.targetSoil - self._config.soilTolerance:
            reward += 2.0

        # Energy efficiency
        if action == LOW and diff_before < self._config.soilTolerance * 2.0:
            reward += 1.5
        if action == HIGH and diff_before < self._config.soilTolerance:
            reward -= 1.0

        # Safety penalties
        if soil_after > 80.0:
            reward -= 5.0
        if soil_after < 20.0:
            reward -= 8.0

        # Overshooting penalty
        if soil_before < self._config.targetSoil and soil_after > self._config.targetSoil + self._config.soilTolerance:
            reward -= 2.0

        return reward

    def _update_q_value(self, state: int, action: int, reward: float, next_state: int):
        max_next_q = float(np.max(self._qTable[next_state]))
        current_q = self._qTable[state, action]
        self._qTable[state, action] = current_q + self._config.alpha * (
            reward + self._config.gamma * max_next_q - current_q
        )

    def explain(self) -> dict:
        """Explain the current Q-Learning decision — for interpretability."""
        with self._lock:
            if not self._latestSnapshot:
                return {"error": "no sensor data"}

            snap = self._latestSnapshot
            temp = self._discretize_temp(snap.airTemp)
            humi = self._discretize_humi(snap.airHumi)
            soil = self._discretize_soil(snap.soilHumi)
            light = self._discretize_light(snap.lightValue)
            period = self._get_time_period()
            state = self._discretize_state(snap)

            q_values = self._qTable[state]
            best_action = int(np.argmax(q_values))
            sorted_actions = sorted(
                enumerate(q_values), key=lambda x: x[1], reverse=True
            )

            # Human-readable state descriptions
            temp_desc = ["寒冷(<10°C)", "凉爽(10-18°C)", "适温(18-25°C)", "温暖(25-33°C)", "高温(>33°C)"]
            humi_desc = ["干燥(<30%)", "偏低(30-50%)", "适中(50-70%)", "潮湿(>70%)"]
            soil_desc = ["极干(<20%)", "偏干(20-35%)", "适中(35-50%)", "偏湿(50-65%)", "过湿(>65%)"]
            light_desc = ["暗(<100lux)", "正常(100-500lux)", "强光(>500lux)"]
            period_desc = ["上午(6-12时)", "下午(12-18时)", "夜间(18-6时)"]

            # Build reasoning
            reasons = []
            if soil <= 1:
                reasons.append(f"土壤湿度{snap.soilHumi:.0f}%处于极干状态，急需灌溉")
            elif soil == 2:
                reasons.append(f"土壤湿度{snap.soilHumi:.0f}%偏低，建议补充水分")
            elif soil >= 4:
                reasons.append(f"土壤湿度{snap.soilHumi:.0f}%已过湿，无需灌溉")
            else:
                reasons.append(f"土壤湿度{snap.soilHumi:.0f}%处于适中范围")

            if period == 2:
                reasons.append("夜间蒸发量低，灌溉需求较小")
            if temp >= 3:
                reasons.append("高温加速蒸发，可能需要更多灌溉")

            # Check if Q-table has learned enough
            coverage = np.count_nonzero(np.abs(self._qTable) > 0.001)
            total = K_STATE_COUNT * ACTION_COUNT
            if coverage / total < 0.05:
                reasons.append("Q-Table 学习尚不充分，当前为探索阶段")

            return {
                "currentState": state,
                "stateBreakdown": {
                    "temperature": {"value": temp, "description": temp_desc[temp], "raw": snap.airTemp},
                    "humidity": {"value": humi, "description": humi_desc[humi], "raw": snap.airHumi},
                    "soil": {"value": soil, "description": soil_desc[soil], "raw": snap.soilHumi},
                    "light": {"value": light, "description": light_desc[light], "raw": snap.lightValue},
                    "timePeriod": {"value": period, "description": period_desc[period]},
                },
                "qValues": [
                    {
                        "action": ACTION_NAMES[a],
                        "actionIndex": int(a),
                        "qValue": round(float(q), 4),
                        "duration": ACTION_DURATIONS[a],
                        "isRecommended": int(a) == best_action,
                    }
                    for a, q in sorted_actions
                ],
                "recommendedAction": ACTION_NAMES[best_action],
                "recommendedDuration": ACTION_DURATIONS[best_action],
                "reasoning": reasons,
                "explorationRate": round(self._config.epsilon, 4),
                "qTableCoverage": round(coverage / total * 100, 2),
            }

    # ── Public accessors ──

    def status(self) -> dict:
        with self._lock:
            state = self._discretize_state(self._latestSnapshot) if self._latestSnapshot else 0

            # Best action
            best_action = int(np.argmax(self._qTable[state]))

            return {
                "autoControlEnabled": self._config.autoControlEnabled,
                "airTemp": round(self._latestSnapshot.airTemp, 2) if self._latestSnapshot else 0,
                "airHumi": round(self._latestSnapshot.airHumi, 2) if self._latestSnapshot else 0,
                "soilHumi": round(self._latestSnapshot.soilHumi, 2) if self._latestSnapshot else 0,
                "liquidLevel": round(self._latestSnapshot.liquidLevel, 2) if self._latestSnapshot else 0,
                "lightValue": round(self._latestSnapshot.lightValue, 2) if self._latestSnapshot else 0,
                "currentState": state,
                "lastAction": self._lastAction,
                "lastActionName": ACTION_NAMES[self._lastAction],
                "lastReward": round(self._lastReward, 2),
                "averageReward": round(self._averageReward, 4),
                "totalEpisodes": self._totalEpisodes,
                "epsilon": round(self._config.epsilon, 4),
                "targetSoil": self._config.targetSoil,
                "userOverrides": self._userOverrideCount,
                "userSatisfaction": round(self._userSatisfaction, 1),
                "recommendedAction": ACTION_NAMES[best_action],
                "qValues": [
                    {"action": ACTION_NAMES[a], "value": round(float(self._qTable[state, a]), 4)}
                    for a in range(ACTION_COUNT)
                ],
            }

    def qtable_summary(self) -> dict:
        with self._lock:
            non_zero = int(np.count_nonzero(np.abs(self._qTable) > 0.001))
            total = K_STATE_COUNT * ACTION_COUNT
            if non_zero > 0:
                max_q = float(np.max(self._qTable))
                min_q = float(np.min(self._qTable[np.abs(self._qTable) > 0.001]))
            else:
                max_q = min_q = 0.0

            result = {
                "totalStates": K_STATE_COUNT,
                "totalEntries": total,
                "nonZeroEntries": non_zero,
                "coverage": round(non_zero / total * 100, 2),
                "maxQ": round(max_q, 4),
                "minQ": round(min_q, 4),
                "qTable": [[round(float(self._qTable[s, a]), 4) for a in range(ACTION_COUNT)] for s in range(K_STATE_COUNT)],
                "history": [],
            }

            for i in range(self._historyCount):
                idx = (self._historyIndex - 1 - i + K_HISTORY_SIZE) % K_HISTORY_SIZE
                h = self._history[idx]
                if h:
                    result["history"].append({
                        "state": h.get("state", 0),
                        "action": ACTION_NAMES[h.get("action", 0)],
                        "reward": round(h.get("reward", 0), 4),
                        "soilBefore": round(h.get("soilBefore", 0), 2),
                        "soilAfter": round(h.get("soilAfter", 0), 2),
                    })

            return result
