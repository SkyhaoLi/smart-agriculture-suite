"""
智润智慧农业套件 - Atlas 200I DK A2 版
灌溉规则引擎 - 基于阈值的日夜灌溉决策

对应原ESP32项目的 IrrigationModule.h/IrrigationModule.cpp
"""

import logging
from dataclasses import dataclass

from config.app_types import SensorSnapshot, IrrigationThresholdConfig

logger = logging.getLogger(__name__)


@dataclass
class IrrigationResult:
    should_water: bool = False
    liquid_warn: bool = False
    is_day: bool = True
    reason: str = ""


class IrrigationModule:
    """规则引擎 - 根据昼夜阈值判断是否需要灌溉"""

    def __init__(self, config: IrrigationThresholdConfig = None):
        self._config = config or IrrigationThresholdConfig()
        self._enabled = True
        self._light_day_threshold = 200.0
        self._result = IrrigationResult()

    def update(self, snapshot: SensorSnapshot) -> IrrigationResult:
        self._result.is_day = snapshot.light_intensity >= self._light_day_threshold
        self._result.liquid_warn = snapshot.liquid_level < self._config.liquid_level_warn

        if not self._enabled:
            self._result.should_water = False
            self._result.reason = "rule engine disabled"
            return self._result

        if self._result.liquid_warn:
            self._result.should_water = False
            self._result.reason = "liquid tank too low"
            return self._result

        cfg = self._config
        if self._result.is_day:
            temp_pass = snapshot.air_temp >= cfg.air_temp_day_high or snapshot.air_temp <= cfg.air_temp_day_low
            humi_pass = snapshot.air_humi <= cfg.air_humi_day_low
            soil_pass = snapshot.soil_humi <= cfg.soil_humi_low
        else:
            temp_pass = snapshot.air_temp >= cfg.air_temp_night_high or snapshot.air_temp <= cfg.air_temp_night_low
            humi_pass = snapshot.air_humi <= cfg.air_humi_night_low
            soil_pass = snapshot.soil_humi <= cfg.soil_humi_low

        self._result.should_water = temp_pass and humi_pass and soil_pass
        self._result.reason = (
            f"{'day' if self._result.is_day else 'night'} thresholds matched"
            if self._result.should_water
            else "thresholds not met"
        )

        return self._result

    def update_config(self, config_dict: dict):
        # 前端发送嵌套格式: {day: {airTemp, airHumi, soilHumi}, night: {...}}
        if 'day' in config_dict and isinstance(config_dict['day'], dict):
            day = config_dict['day']
            if 'airTemp' in day:
                self._config.air_temp_day_high = float(day['airTemp'])
            if 'airHumi' in day:
                self._config.air_humi_day_low = float(day['airHumi'])
            if 'soilHumi' in day:
                self._config.soil_humi_low = float(day['soilHumi'])
        if 'night' in config_dict and isinstance(config_dict['night'], dict):
            night = config_dict['night']
            if 'airTemp' in night:
                self._config.air_temp_night_high = float(night['airTemp'])
            if 'airHumi' in night:
                self._config.air_humi_night_low = float(night['airHumi'])
            if 'soilHumi' in night:
                self._config.soil_humi_low = float(night['soilHumi'])
        # 也支持平铺格式
        for key, value in config_dict.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            elif key == "enabled":
                self._enabled = value
            elif key == "lightThreshold":
                self._light_day_threshold = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = val

    @property
    def result(self) -> IrrigationResult:
        return self._result

    @property
    def config(self) -> IrrigationThresholdConfig:
        return self._config

    def to_dict(self) -> dict:
        return {
            "enabled": self._enabled,
            "shouldWater": self._result.should_water,
            "liquidWarn": self._result.liquid_warn,
            "isDay": self._result.is_day,
            "reason": self._result.reason,
            "config": {
                "day": {
                    "airTemp": self._config.air_temp_day_high,
                    "airHumi": self._config.air_humi_day_low,
                    "soilHumi": self._config.soil_humi_low,
                },
                "night": {
                    "airTemp": self._config.air_temp_night_high,
                    "airHumi": self._config.air_humi_night_low,
                    "soilHumi": self._config.soil_humi_low,
                },
                "liquidThreshold": self._config.liquid_level_warn,
                "soilHumiHigh": self._config.soil_humi_high,
            },
        }
