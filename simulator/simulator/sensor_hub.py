"""SensorHub — simulated sensors with Ornstein-Uhlenbeck random walk."""

import math
import random
import threading
from .time_clock import SimClock


class SensorSnapshot:
    """Mirrors firmware SensorSnapshot."""

    def __init__(self):
        self.airTemp = 25.0
        self.airHumi = 60.0
        self.soilHumi = 50.0
        self.liquidLevel = 75.0
        self.lightValue = 500.0
        self.isDay = True
        self.airValid = True
        self.soilValid = True
        self.liquidValid = True
        self.lightValid = True
        self.updatedAtMs = 0

    def to_dict(self) -> dict:
        return {
            "airTemp": round(self.airTemp, 2),
            "airHumi": round(self.airHumi, 2),
            "soilHumi": round(self.soilHumi, 2),
            "liquidLevel": round(self.liquidLevel, 2),
            "lightValue": round(self.lightValue, 2),
            "isDay": self.isDay,
            "airValid": self.airValid,
            "soilValid": self.soilValid,
            "liquidValid": self.liquidValid,
            "lightValid": self.lightValid,
            "updatedAtMs": self.updatedAtMs,
        }


class OUChannel:
    """Ornstein-Uhlenbeck random walk for one sensor channel."""

    def __init__(self, mean: float, theta: float, sigma: float, min_val: float, max_val: float):
        self.mean = mean
        self.theta = theta      # mean reversion speed
        self.sigma = sigma      # volatility
        self.min_val = min_val
        self.max_val = max_val
        self.value = mean

    def step(self) -> float:
        dx = self.theta * (self.mean - self.value) + self.sigma * random.gauss(0, 1)
        self.value += dx
        self.value = max(self.min_val, min(self.max_val, self.value))
        return self.value


class SensorHub:
    """
    Simulates all 5 sensor channels with OU processes.
    Sampling interval matches firmware: 2000 simulated ms.
    """

    kSampleIntervalMs = 2000

    def __init__(self, clock: SimClock):
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot = SensorSnapshot()
        self._lastSampleMs = 0

        # User injection overrides
        self._inject = {}  # channel name -> forced value

        # Watering feedback: when actuator is on, soil moisture increases
        self._watering_active = False

        # OU channels: mean, theta, sigma, min, max
        self._airTemp = OUChannel(mean=25.0, theta=0.05, sigma=0.3, min_val=-5.0, max_val=50.0)
        self._airHumi = OUChannel(mean=60.0, theta=0.03, sigma=0.5, min_val=10.0, max_val=99.0)
        self._soilHumi = OUChannel(mean=50.0, theta=0.04, sigma=0.4, min_val=0.0, max_val=100.0)
        self._liquidLevel = OUChannel(mean=75.0, theta=0.02, sigma=0.3, min_val=0.0, max_val=100.0)
        self._light = OUChannel(mean=5000.0, theta=0.03, sigma=100.0, min_val=0.0, max_val=12000.0)

    def set_watering_active(self, active: bool):
        """Called by ActuatorController when valve/pump turns on/off."""
        self._watering_active = active

    def inject(self, values: dict):
        """Force sensor channels to specific values. Accepts dict like {"airTemp": 30, "soilHumi": 20}."""
        with self._lock:
            self._inject.update(values)

    def clear_inject(self, channel: str = None):
        """Remove injection override. If channel is None, clear all."""
        with self._lock:
            if channel:
                self._inject.pop(channel, None)
            else:
                self._inject.clear()

    def update(self) -> bool:
        """
        Called every real 100ms. Returns True if a new sample was taken.
        Sampling rate is based on simulated time (kSampleIntervalMs = 2000).
        """
        now = self._clock.millis()

        # Adaptive sampling: at high time scales, we need more samples per real tick
        with self._lock:
            if self._lastSampleMs == 0:
                self._lastSampleMs = now

            # How many simulated samples should have occurred
            expected_samples = (now - self._lastSampleMs) // self.kSampleIntervalMs
            if expected_samples <= 0:
                return False

            # Take at most a few steps per update to avoid infinite loops
            steps = min(expected_samples, 5)
            for _ in range(steps):
                self._step(now)
            self._lastSampleMs += steps * self.kSampleIntervalMs

        return True

    def _step(self, now_ms: int):
        """Generate one sample."""
        is_day = self._clock.is_day()

        # Update light mean based on day/night
        if is_day:
            self._light.mean = 5000.0
        else:
            self._light.mean = 5.0

        # Step OU processes
        air_temp = self._airTemp.step()
        air_humi = self._airHumi.step()
        soil_humi = self._soilHumi.step()
        liquid_level = self._liquidLevel.step()
        light_val = self._light.step()

        # Watering feedback: soil moisture increases when watering
        if self._watering_active:
            soil_humi = min(100.0, soil_humi + 0.5)
            self._soilHumi.value = soil_humi

        # Apply injections
        if "airTemp" in self._inject:
            air_temp = self._inject["airTemp"]
        if "airHumi" in self._inject:
            air_humi = self._inject["airHumi"]
        if "soilHumi" in self._inject:
            soil_humi = self._inject["soilHumi"]
        if "liquidLevel" in self._inject:
            liquid_level = self._inject["liquidLevel"]
        if "lightValue" in self._inject:
            light_val = self._inject["lightValue"]

        self._snapshot.airTemp = round(air_temp, 2)
        self._snapshot.airHumi = round(air_humi, 2)
        self._snapshot.soilHumi = round(soil_humi, 2)
        self._snapshot.liquidLevel = round(liquid_level, 2)
        self._snapshot.lightValue = round(light_val, 2)
        self._snapshot.isDay = light_val >= 200.0
        self._snapshot.airValid = True
        self._snapshot.soilValid = True
        self._snapshot.liquidValid = True
        self._snapshot.lightValid = True
        self._snapshot.updatedAtMs = now_ms

    @property
    def snapshot(self) -> SensorSnapshot:
        with self._lock:
            return self._snapshot

    def light_ready(self) -> bool:
        return self._snapshot.lightValid

    def last_air_frame(self) -> str:
        return "sim-ok"
