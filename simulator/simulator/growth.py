"""GrowthModule — GDD-based growth tracking matching firmware."""

import math
import threading
from .time_clock import SimClock


# Crop profiles — directly from GrowthModule.cpp
CROP_PROFILES = [
    {
        "name": "Tomato", "nameCn": "番茄",
        "baseTemp": 10.0,
        "gddStages": [0, 80, 200, 500, 800, 1100, 1400],
        "optimalTemp": 25.0, "optimalHumi": 60.0, "optimalSoil": 65.0,
        "dailyLightHours": 12.0,
    },
    {
        "name": "Lettuce", "nameCn": "生菜",
        "baseTemp": 4.5,
        "gddStages": [0, 40, 100, 250, 400, 0, 550],
        "optimalTemp": 20.0, "optimalHumi": 70.0, "optimalSoil": 70.0,
        "dailyLightHours": 10.0,
    },
    {
        "name": "Pepper", "nameCn": "辣椒",
        "baseTemp": 12.0,
        "gddStages": [0, 100, 250, 600, 900, 1200, 1600],
        "optimalTemp": 27.0, "optimalHumi": 55.0, "optimalSoil": 60.0,
        "dailyLightHours": 14.0,
    },
    {
        "name": "Cucumber", "nameCn": "黄瓜",
        "baseTemp": 10.0,
        "gddStages": [0, 60, 180, 450, 700, 950, 1200],
        "optimalTemp": 28.0, "optimalHumi": 65.0, "optimalSoil": 75.0,
        "dailyLightHours": 10.0,
    },
    {
        "name": "Strawberry", "nameCn": "草莓",
        "baseTemp": 5.0,
        "gddStages": [0, 50, 150, 400, 650, 900, 1200],
        "optimalTemp": 22.0, "optimalHumi": 65.0, "optimalSoil": 70.0,
        "dailyLightHours": 12.0,
    },
]

STAGE_NAMES_EN = ["Seed", "Germination", "Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"]
STAGE_NAMES_CN = ["种子期", "萌芽期", "幼苗期", "营养生长期", "开花期", "结果期", "成熟期"]

IRRIGATION_ADVICE = [
    "保持土壤适度湿润，少量多次浇水。",
    "保持土壤适度湿润，少量多次浇水。",
    "避免过度浇水，促进根系发育。",
    "保持土壤湿度接近作物最适值。",
    "花期避免水分胁迫，保持稳定供水。",
    "需水量较高，保持土壤湿度稳定。",
    "采收前逐渐减少浇水量。",
]

K_GROWTH_DAY_INTERVAL_MS = 24 * 60 * 60 * 1000
K_MAX_HISTORY = 90


class DailyRecord:
    def __init__(self):
        self.dayIndex = 0
        self.avgTemp = 0.0
        self.maxTemp = -100.0
        self.minTemp = 100.0
        self.avgHumi = 0.0
        self.avgSoil = 0.0
        self.totalLight = 0.0
        self.dailyGdd = 0.0
        self.cumulativeGdd = 0.0
        self.stage = 0


class LinearRegression:
    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
        self.rSquared = 0.0
        self.sampleCount = 0


class GrowthModule:
    """Direct port of agri::GrowthModule."""

    COUNT = 7  # number of growth stages

    def __init__(self, clock: SimClock):
        self._clock = clock
        self._lock = threading.Lock()
        self._currentCropIndex = 0
        self._cumulativeGdd = 0.0
        self._currentDayOfGrowth = 0
        self._currentStage = 0  # Seed

        self._latestSnapshot = None
        self._lastUpdateMs = 0

        # Daily accumulation
        self._dayTempSum = 0.0
        self._dayTempMax = -100.0
        self._dayTempMin = 100.0
        self._dayHumiSum = 0.0
        self._daySoilSum = 0.0
        self._dayLightSum = 0.0
        self._daySampleCount = 0
        self._dayStartedAtMs = 0

        # History
        self._records = [DailyRecord() for _ in range(K_MAX_HISTORY)]
        self._recordCount = 0

        # Prediction
        self._lrModel = LinearRegression()
        self._predictedFloweringDay = -1
        self._predictedMaturityDay = -1
        self._predictedYieldScore = 0.0

    def begin(self, crop_index: int = 0):
        self._currentCropIndex = max(0, min(crop_index, len(CROP_PROFILES) - 1))
        self._dayStartedAtMs = self._clock.millis()
        self._predict_growth()

    def update(self, snapshot, sample_updated: bool, now_ms: int):
        with self._lock:
            if not sample_updated:
                if self._dayStartedAtMs > 0 and now_ms - self._dayStartedAtMs >= K_GROWTH_DAY_INTERVAL_MS:
                    self._finalize_daily_record(now_ms)
                return

            self._latestSnapshot = snapshot

            crop = CROP_PROFILES[self._currentCropIndex]
            if self._lastUpdateMs == 0:
                self._lastUpdateMs = now_ms
            delta_ms = now_ms - self._lastUpdateMs
            self._lastUpdateMs = now_ms

            daily_gdd = max(0.0, snapshot.airTemp - crop["baseTemp"])
            gdd_increment = daily_gdd * (delta_ms / K_GROWTH_DAY_INTERVAL_MS)
            self._cumulativeGdd += gdd_increment

            self._currentStage = self._calculate_stage(self._cumulativeGdd)

            self._dayTempSum += snapshot.airTemp
            self._dayTempMax = max(self._dayTempMax, snapshot.airTemp)
            self._dayTempMin = min(self._dayTempMin, snapshot.airTemp)
            self._dayHumiSum += snapshot.airHumi
            self._daySoilSum += snapshot.soilHumi
            self._dayLightSum += snapshot.lightValue * (delta_ms / 3600000.0)
            self._daySampleCount += 1

            self._predict_growth()

            if self._dayStartedAtMs == 0:
                self._dayStartedAtMs = now_ms
            if now_ms - self._dayStartedAtMs >= K_GROWTH_DAY_INTERVAL_MS:
                self._finalize_daily_record(now_ms)

    def set_crop(self, crop_id: int):
        with self._lock:
            if crop_id < 0 or crop_id >= len(CROP_PROFILES):
                return
            self._currentCropIndex = crop_id
            self._reset_unlocked()

    def reset(self):
        with self._lock:
            self._reset_unlocked()

    def _reset_unlocked(self):
        self._latestSnapshot = None
        self._cumulativeGdd = 0.0
        self._currentDayOfGrowth = 0
        self._currentStage = 0
        self._recordCount = 0
        self._predictedFloweringDay = -1
        self._predictedMaturityDay = -1
        self._predictedYieldScore = 0.0
        self._lrModel = LinearRegression()
        self._dayTempSum = 0.0
        self._dayTempMax = -100.0
        self._dayTempMin = 100.0
        self._dayHumiSum = 0.0
        self._daySoilSum = 0.0
        self._dayLightSum = 0.0
        self._daySampleCount = 0
        self._dayStartedAtMs = self._clock.millis()
        self._lastUpdateMs = 0

    def _calculate_stage(self, gdd: float) -> int:
        crop = CROP_PROFILES[self._currentCropIndex]
        for i in range(self.COUNT - 1, -1, -1):
            if crop["gddStages"][i] > 0.0 and gdd >= crop["gddStages"][i]:
                return i
        return 0

    def _finalize_daily_record(self, now_ms: int):
        if self._daySampleCount == 0:
            self._dayStartedAtMs = now_ms
            return

        self._currentDayOfGrowth += 1

        index = self._recordCount % K_MAX_HISTORY
        record = self._records[index]
        record.dayIndex = self._currentDayOfGrowth
        record.avgTemp = self._dayTempSum / self._daySampleCount
        record.maxTemp = self._dayTempMax
        record.minTemp = self._dayTempMin
        record.avgHumi = self._dayHumiSum / self._daySampleCount
        record.avgSoil = self._daySoilSum / self._daySampleCount
        record.totalLight = self._dayLightSum
        record.dailyGdd = max(0.0, ((self._dayTempMax + self._dayTempMin) * 0.5) - CROP_PROFILES[self._currentCropIndex]["baseTemp"])
        record.cumulativeGdd = self._cumulativeGdd
        record.stage = self._currentStage
        self._recordCount += 1

        # Reset daily accumulators
        self._dayTempSum = 0.0
        self._dayTempMax = -100.0
        self._dayTempMin = 100.0
        self._dayHumiSum = 0.0
        self._daySoilSum = 0.0
        self._dayLightSum = 0.0
        self._daySampleCount = 0
        self._dayStartedAtMs = now_ms

        self._run_linear_regression()
        self._predict_growth()

    def _run_linear_regression(self):
        n = min(self._recordCount, K_MAX_HISTORY)
        if n < 3:
            self._lrModel = LinearRegression()
            return

        sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0.0
        for i in range(n):
            x = float(self._records[i].dayIndex)
            y = self._records[i].cumulativeGdd
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x
            sum_y2 += y * y

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 0.001:
            self._lrModel = LinearRegression()
            return

        self._lrModel.slope = (n * sum_xy - sum_x * sum_y) / denom
        self._lrModel.intercept = (sum_y - self._lrModel.slope * sum_x) / n
        self._lrModel.sampleCount = n

        ss_tot = sum_y2 - (sum_y * sum_y) / n
        ss_res = 0.0
        for i in range(n):
            predicted = self._lrModel.slope * self._records[i].dayIndex + self._lrModel.intercept
            diff = self._records[i].cumulativeGdd - predicted
            ss_res += diff * diff
        self._lrModel.rSquared = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    def _predict_growth(self):
        crop = CROP_PROFILES[self._currentCropIndex]
        flowering_gdd = crop["gddStages"][4]  # Flowering
        maturity_gdd = crop["gddStages"][6]   # Maturity

        if self._lrModel.sampleCount >= 3 and self._lrModel.slope > 0.0:
            gdd_per_day = self._lrModel.slope
            self._predictedFloweringDay = (
                self._currentDayOfGrowth + int((flowering_gdd - self._cumulativeGdd) / gdd_per_day)
                if flowering_gdd > self._cumulativeGdd
                else self._currentDayOfGrowth
            )
            self._predictedMaturityDay = (
                self._currentDayOfGrowth + int((maturity_gdd - self._cumulativeGdd) / gdd_per_day)
                if maturity_gdd > self._cumulativeGdd
                else self._currentDayOfGrowth
            )
        else:
            if self._latestSnapshot:
                avg_gdd = max(0.0, self._latestSnapshot.airTemp - crop["baseTemp"])
                if avg_gdd > 0.0:
                    self._predictedFloweringDay = (
                        self._currentDayOfGrowth + int((flowering_gdd - self._cumulativeGdd) / avg_gdd)
                        if flowering_gdd > self._cumulativeGdd
                        else self._currentDayOfGrowth
                    )
                    self._predictedMaturityDay = (
                        self._currentDayOfGrowth + int((maturity_gdd - self._cumulativeGdd) / avg_gdd)
                        if maturity_gdd > self._cumulativeGdd
                        else self._currentDayOfGrowth
                    )

        self._predictedYieldScore = self._calculate_yield_score()

    def _calculate_yield_score(self) -> float:
        if not self._latestSnapshot:
            return 100.0
        crop = CROP_PROFILES[self._currentCropIndex]
        score = 100.0
        score -= abs(self._latestSnapshot.airTemp - crop["optimalTemp"]) * 2.0
        score -= abs(self._latestSnapshot.airHumi - crop["optimalHumi"]) * 0.5
        score -= abs(self._latestSnapshot.soilHumi - crop["optimalSoil"]) * 0.8
        return max(0.0, min(100.0, score))

    @property
    def current_crop_index(self) -> int:
        return self._currentCropIndex

    def status(self) -> dict:
        with self._lock:
            crop = CROP_PROFILES[self._currentCropIndex]
            total_gdd = crop["gddStages"][6] if crop["gddStages"][6] > 0 else crop["gddStages"][4]
            progress = (self._cumulativeGdd / total_gdd) * 100.0 if total_gdd > 0 else 0.0

            return {
                "crop": crop["name"],
                "cropCn": crop["nameCn"],
                "dayOfGrowth": self._currentDayOfGrowth,
                "cumulativeGdd": round(self._cumulativeGdd, 2),
                "currentStage": self._currentStage,
                "stageName": STAGE_NAMES_EN[self._currentStage],
                "stageNameCn": STAGE_NAMES_CN[self._currentStage],
                "progressPercent": round(progress, 1),
                "airTemp": round(self._latestSnapshot.airTemp, 2) if self._latestSnapshot else 0,
                "airHumi": round(self._latestSnapshot.airHumi, 2) if self._latestSnapshot else 0,
                "soilHumi": round(self._latestSnapshot.soilHumi, 2) if self._latestSnapshot else 0,
                "lightValue": round(self._latestSnapshot.lightValue, 2) if self._latestSnapshot else 0,
                "predictedFloweringDay": self._predictedFloweringDay,
                "predictedMaturityDay": self._predictedMaturityDay,
                "yieldScore": round(self._predictedYieldScore, 1),
                "irrigationAdvice": IRRIGATION_ADVICE[self._currentStage],
            }

    def history(self) -> list:
        with self._lock:
            result = []
            n = min(self._recordCount, K_MAX_HISTORY)
            for i in range(n):
                r = self._records[i]
                result.append({
                    "day": r.dayIndex,
                    "avgTemp": round(r.avgTemp, 2),
                    "maxTemp": round(r.maxTemp, 2),
                    "minTemp": round(r.minTemp, 2),
                    "avgHumi": round(r.avgHumi, 2),
                    "avgSoil": round(r.avgSoil, 2),
                    "totalLight": round(r.totalLight, 2),
                    "dailyGdd": round(r.dailyGdd, 2),
                    "cumulativeGdd": round(r.cumulativeGdd, 2),
                    "stage": r.stage,
                    "stageName": STAGE_NAMES_EN[r.stage],
                    "stageNameCn": STAGE_NAMES_CN[r.stage],
                })
            return result

    def prediction(self) -> dict:
        with self._lock:
            crop = CROP_PROFILES[self._currentCropIndex]
            result = {
                "crop": crop["nameCn"],
                "currentDay": self._currentDayOfGrowth,
                "currentStage": STAGE_NAMES_CN[self._currentStage],
                "cumulativeGdd": round(self._cumulativeGdd, 2),
                "predictedFloweringDay": self._predictedFloweringDay,
                "predictedMaturityDay": self._predictedMaturityDay,
                "yieldScore": round(self._predictedYieldScore, 1),
                "irrigationAdvice": IRRIGATION_ADVICE[self._currentStage],
                "stages": [],
                "availableCrops": [],
            }

            if self._lrModel.sampleCount >= 3:
                result["avgGddPerDay"] = round(self._lrModel.slope, 2)
                result["modelRSquared"] = round(self._lrModel.rSquared, 4)
                result["modelSamples"] = self._lrModel.sampleCount

            for i in range(self.COUNT):
                result["stages"].append({
                    "name": STAGE_NAMES_EN[i],
                    "nameCn": STAGE_NAMES_CN[i],
                    "requiredGdd": crop["gddStages"][i],
                    "reached": crop["gddStages"][i] > 0.0 and self._cumulativeGdd >= crop["gddStages"][i],
                })

            for i, c in enumerate(CROP_PROFILES):
                result["availableCrops"].append({"id": i, "name": c["name"], "nameCn": c["nameCn"]})

            return result
