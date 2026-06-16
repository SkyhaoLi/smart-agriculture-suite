"""
智润智慧农业套件 - Atlas 200I DK A2 版
作物生长跟踪模块 - GDD积累、5种作物、线性回归预测

对应原ESP32项目的 GrowthModule.h/GrowthModule.cpp
"""

import math
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List

from config.app_types import (
    SensorSnapshot, CropType, GrowthStage, GrowthState, CROP_NAMES
)

logger = logging.getLogger(__name__)


@dataclass
class CropProfile:
    name: str
    name_cn: str
    base_temp: float
    gdd_stages: List[float]  # 7个阶段的GDD阈值 [Seed, Germination, Seedling, Vegetative, Flowering, Fruiting, Maturity]
    optimal_temp: float
    optimal_humi: float
    optimal_soil: float
    daily_light_hours: float


@dataclass
class DailyRecord:
    day_index: int = 0
    avg_temp: float = 0.0
    max_temp: float = -100.0
    min_temp: float = 100.0
    avg_humi: float = 0.0
    avg_soil: float = 0.0
    total_light: float = 0.0
    daily_gdd: float = 0.0
    cumulative_gdd: float = 0.0
    stage: GrowthStage = GrowthStage.Seed


@dataclass
class LinearRegression:
    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    sample_count: int = 0


CROP_PROFILES = [
    CropProfile("Tomato", "番茄", 10.0, [0, 80, 200, 500, 800, 1100, 1400],
                25.0, 60.0, 65.0, 12.0),
    CropProfile("Lettuce", "生菜", 4.5, [0, 40, 100, 250, 400, 0, 550],
                20.0, 70.0, 70.0, 10.0),
    CropProfile("Pepper", "辣椒", 12.0, [0, 100, 250, 600, 900, 1200, 1600],
                27.0, 55.0, 60.0, 14.0),
    CropProfile("Cucumber", "黄瓜", 10.0, [0, 60, 180, 450, 700, 950, 1200],
                28.0, 65.0, 75.0, 10.0),
    CropProfile("Strawberry", "草莓", 5.0, [0, 50, 150, 400, 650, 900, 1200],
                22.0, 65.0, 70.0, 12.0),
]

STAGE_NAMES = ["Seed", "Germination", "Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"]
STAGE_NAMES_CN = ["种子期", "萌芽期", "幼苗期", "营养生长期", "开花期", "结果期", "成熟期"]

GROWTH_DAY_INTERVAL = 24 * 3600  # 1天(秒), 加速演示可调小


class GrowthModule:
    """作物生长跟踪 - GDD积累 + 线性回归预测"""

    def __init__(self):
        self._crop_index = 0
        self._cumulative_gdd = 0.0
        self._current_day = 0
        self._current_stage = GrowthStage.Seed
        self._latest_snapshot = SensorSnapshot()

        self._records: List[DailyRecord] = []
        self._record_count = 0

        # 当天累积
        self._day_temp_sum = 0.0
        self._day_temp_max = -100.0
        self._day_temp_min = 100.0
        self._day_humi_sum = 0.0
        self._day_soil_sum = 0.0
        self._day_light_sum = 0.0
        self._day_sample_count = 0
        self._day_started_at = 0.0
        self._last_update_time = 0.0

        self._lr_model = LinearRegression()
        self._predicted_flowering_day: Optional[int] = None
        self._predicted_maturity_day: Optional[int] = None
        self._yield_score = 100.0

    @property
    def current_crop(self) -> CropProfile:
        return CROP_PROFILES[self._crop_index]

    @property
    def state(self) -> GrowthState:
        return GrowthState(
            crop=CropType(self._crop_index),
            cumulative_gdd=self._cumulative_gdd,
            growth_day=self._current_day,
            current_stage=self._current_stage,
            yield_score=self._yield_score,
            predicted_flower_day=self._predicted_flowering_day,
            predicted_maturity_day=self._predicted_maturity_day,
        )

    def set_crop(self, crop_id: int):
        if 0 <= crop_id < len(CROP_PROFILES):
            self._crop_index = crop_id
            self.reset()

    def reset(self):
        self._cumulative_gdd = 0.0
        self._current_day = 0
        self._current_stage = GrowthStage.Seed
        self._records.clear()
        self._record_count = 0
        self._lr_model = LinearRegression()
        self._predicted_flowering_day = None
        self._predicted_maturity_day = None
        self._yield_score = 100.0
        self._day_temp_sum = 0.0
        self._day_temp_max = -100.0
        self._day_temp_min = 100.0
        self._day_humi_sum = 0.0
        self._day_soil_sum = 0.0
        self._day_light_sum = 0.0
        self._day_sample_count = 0
        self._day_started_at = 0.0
        self._last_update_time = 0.0

    def update(self, snapshot: SensorSnapshot, sample_updated: bool, now: float = None):
        if now is None:
            now = time.time()

        if self._day_started_at > 0 and now - self._day_started_at >= GROWTH_DAY_INTERVAL:
            self._finalize_daily_record(now)

        if not sample_updated:
            return

        self._latest_snapshot = snapshot
        crop = self.current_crop

        if self._last_update_time == 0:
            self._last_update_time = now
        delta_sec = now - self._last_update_time
        self._last_update_time = now

        daily_gdd = max(0.0, snapshot.air_temp - crop.base_temp)
        gdd_increment = daily_gdd * (delta_sec / GROWTH_DAY_INTERVAL)
        self._cumulative_gdd += gdd_increment

        self._current_stage = self._calculate_stage(self._cumulative_gdd)

        self._day_temp_sum += snapshot.air_temp
        self._day_temp_max = max(self._day_temp_max, snapshot.air_temp)
        self._day_temp_min = min(self._day_temp_min, snapshot.air_temp)
        self._day_humi_sum += snapshot.air_humi
        self._day_soil_sum += snapshot.soil_humi
        self._day_light_sum += snapshot.light_intensity * (delta_sec / 3600.0)
        self._day_sample_count += 1

        self._predict_growth()

        if self._day_started_at == 0:
            self._day_started_at = now

    def _calculate_stage(self, gdd: float) -> GrowthStage:
        crop = self.current_crop
        for i in range(len(GrowthStage) - 1, -1, -1):
            if crop.gdd_stages[i] > 0.0 and gdd >= crop.gdd_stages[i]:
                return GrowthStage(i)
        return GrowthStage.Seed

    def _finalize_daily_record(self, now: float):
        if self._day_sample_count == 0:
            self._day_started_at = now
            return

        self._current_day += 1

        record = DailyRecord(
            day_index=self._current_day,
            avg_temp=self._day_temp_sum / self._day_sample_count,
            max_temp=self._day_temp_max,
            min_temp=self._day_temp_min,
            avg_humi=self._day_humi_sum / self._day_sample_count,
            avg_soil=self._day_soil_sum / self._day_sample_count,
            total_light=self._day_light_sum,
            daily_gdd=max(0.0, ((self._day_temp_max + self._day_temp_min) * 0.5 - self.current_crop.base_temp)),
            cumulative_gdd=self._cumulative_gdd,
            stage=self._current_stage,
        )
        self._records.append(record)
        self._record_count += 1

        # 重置日累计
        self._day_temp_sum = 0.0
        self._day_temp_max = -100.0
        self._day_temp_min = 100.0
        self._day_humi_sum = 0.0
        self._day_soil_sum = 0.0
        self._day_light_sum = 0.0
        self._day_sample_count = 0
        self._day_started_at = now

        self._run_linear_regression()
        self._predict_growth()

    def _run_linear_regression(self):
        n = len(self._records)
        if n < 3:
            self._lr_model = LinearRegression()
            return

        xs = [r.day_index for r in self._records]
        ys = [r.cumulative_gdd for r in self._records]

        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_x2 = sum(x * x for x in xs)
        sum_y2 = sum(y * y for y in ys)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 0.001:
            self._lr_model = LinearRegression()
            return

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        ss_tot = sum_y2 - (sum_y * sum_y) / n
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        self._lr_model = LinearRegression(slope=slope, intercept=intercept,
                                           r_squared=r_squared, sample_count=n)

    def _predict_growth(self):
        crop = self.current_crop
        flowering_gdd = crop.gdd_stages[GrowthStage.Flowering]
        maturity_gdd = crop.gdd_stages[GrowthStage.Maturity]

        if self._lr_model.sample_count >= 3 and self._lr_model.slope > 0:
            gdd_per_day = self._lr_model.slope
            if flowering_gdd > self._cumulative_gdd:
                self._predicted_flowering_day = self._current_day + int(
                    (flowering_gdd - self._cumulative_gdd) / gdd_per_day)
            else:
                self._predicted_flowering_day = self._current_day
            if maturity_gdd > self._cumulative_gdd:
                self._predicted_maturity_day = self._current_day + int(
                    (maturity_gdd - self._cumulative_gdd) / gdd_per_day)
            else:
                self._predicted_maturity_day = self._current_day
        else:
            avg_gdd_per_day = max(0.0, self._latest_snapshot.air_temp - crop.base_temp)
            # 回退: 用历史平均GDD率 (避免温度暂时低于base_temp时预测为None)
            if avg_gdd_per_day <= 0 and self._cumulative_gdd > 0:
                if self._current_day > 0:
                    avg_gdd_per_day = self._cumulative_gdd / self._current_day
                elif self._last_update_time > 0 and self._day_started_at > 0:
                    elapsed_hours = (self._last_update_time - self._day_started_at) / 3600.0
                    if elapsed_hours > 0.1:
                        avg_gdd_per_day = self._cumulative_gdd / elapsed_hours * 24.0
            if avg_gdd_per_day > 0:
                self._predicted_flowering_day = (
                    self._current_day + int((flowering_gdd - self._cumulative_gdd) / avg_gdd_per_day)
                    if flowering_gdd > self._cumulative_gdd else self._current_day)
                self._predicted_maturity_day = (
                    self._current_day + int((maturity_gdd - self._cumulative_gdd) / avg_gdd_per_day)
                    if maturity_gdd > self._cumulative_gdd else self._current_day)

        self._yield_score = self._calculate_yield_score()

    def _calculate_yield_score(self) -> float:
        crop = self.current_crop
        score = 100.0
        score -= abs(self._latest_snapshot.air_temp - crop.optimal_temp) * 2.0
        score -= abs(self._latest_snapshot.air_humi - crop.optimal_humi) * 0.5
        score -= abs(self._latest_snapshot.soil_humi - crop.optimal_soil) * 0.8
        return max(0.0, min(100.0, score))

    def irrigation_advice(self) -> str:
        advice_map = {
            GrowthStage.Seed: "保持土壤适度湿润，少量多次浇水。",
            GrowthStage.Germination: "保持土壤适度湿润，少量多次浇水。",
            GrowthStage.Seedling: "避免过度浇水，促进根系发育。",
            GrowthStage.Vegetative: "保持土壤湿度接近作物最适值。",
            GrowthStage.Flowering: "花期避免水分胁迫，保持稳定供水。",
            GrowthStage.Fruiting: "需水量较高，保持土壤湿度稳定。",
            GrowthStage.Maturity: "采收前逐渐减少浇水量。",
        }
        return advice_map.get(self._current_stage, "正常灌溉。")

    def to_dict(self) -> dict:
        crop = self.current_crop
        total_gdd = crop.gdd_stages[GrowthStage.Maturity] or crop.gdd_stages[GrowthStage.Flowering]
        return {
            "crop": crop.name,
            "cropCn": crop.name_cn,
            "dayOfGrowth": self._current_day,
            "cumulativeGdd": round(self._cumulative_gdd, 2),
            "currentStage": self._current_stage.value,
            "stageName": STAGE_NAMES[self._current_stage.value],
            "stageNameCn": STAGE_NAMES_CN[self._current_stage.value],
            "progressPercent": round((self._cumulative_gdd / total_gdd) * 100, 1) if total_gdd > 0 else 0,
            "airTemp": round(self._latest_snapshot.air_temp, 1),
            "airHumi": round(self._latest_snapshot.air_humi, 1),
            "soilHumi": round(self._latest_snapshot.soil_humi, 1),
            "lightValue": round(self._latest_snapshot.light_intensity, 1),
            "predictedFloweringDay": self._predicted_flowering_day,
            "predictedMaturityDay": self._predicted_maturity_day,
            "yieldScore": round(self._yield_score, 1),
            "irrigationAdvice": self.irrigation_advice(),
        }
