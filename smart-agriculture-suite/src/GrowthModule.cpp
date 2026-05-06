#include "GrowthModule.h"

#include <math.h>

#include "AppConfig.h"

namespace agri {

namespace {

GrowthModule::CropProfile makeCropProfile(
    const char* name,
    const char* nameCn,
    float baseTemp,
    const float (&stages)[GrowthModule::Count],
    float optimalTemp,
    float optimalHumi,
    float optimalSoil,
    float dailyLightHours) {
    GrowthModule::CropProfile profile;
    profile.name = name;
    profile.nameCn = nameCn;
    profile.baseTemp = baseTemp;
    memcpy(profile.gddStages, stages, sizeof(profile.gddStages));
    profile.optimalTemp = optimalTemp;
    profile.optimalHumi = optimalHumi;
    profile.optimalSoil = optimalSoil;
    profile.dailyLightHours = dailyLightHours;
    return profile;
}

}  // namespace

const GrowthModule::CropProfile GrowthModule::kCropProfiles_[5] = {
    makeCropProfile("Tomato", "番茄", 10.0f, {0, 80, 200, 500, 800, 1100, 1400}, 25.0f, 60.0f, 65.0f, 12.0f),
    makeCropProfile("Lettuce", "生菜", 4.5f, {0, 40, 100, 250, 400, 0, 550}, 20.0f, 70.0f, 70.0f, 10.0f),
    makeCropProfile("Pepper", "辣椒", 12.0f, {0, 100, 250, 600, 900, 1200, 1600}, 27.0f, 55.0f, 60.0f, 14.0f),
    makeCropProfile("Cucumber", "黄瓜", 10.0f, {0, 60, 180, 450, 700, 950, 1200}, 28.0f, 65.0f, 75.0f, 10.0f),
    makeCropProfile("Strawberry", "草莓", 5.0f, {0, 50, 150, 400, 650, 900, 1200}, 22.0f, 65.0f, 70.0f, 12.0f)
};

const char* GrowthModule::stageName(GrowthStage stage) {
    static const char* kNames[] = {
        "Seed", "Germination", "Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"
    };
    return kNames[static_cast<int>(stage)];
}

const char* GrowthModule::stageNameCn(GrowthStage stage) {
    static const char* kNames[] = {
        "种子期", "萌芽期", "幼苗期", "营养生长期", "开花期", "结果期", "成熟期"
    };
    return kNames[static_cast<int>(stage)];
}

void GrowthModule::begin() {
    prefs_.begin("growth", false);
    currentCropIndex_ = constrain(prefs_.getInt("crop", 0), 0, cropCount() - 1);
    cumulativeGdd_ = prefs_.getFloat("gdd", 0.0f);
    currentDayOfGrowth_ = prefs_.getInt("day", 0);
    currentStage_ = calculateStage(cumulativeGdd_);
    dayStartedAtMs_ = millis();
    lastUpdateMs_ = 0;
    predictGrowth();
}

void GrowthModule::update(const SensorSnapshot& snapshot, bool sampleUpdated, unsigned long nowMs) {
    if (!sampleUpdated) {
        if (dayStartedAtMs_ > 0 && nowMs - dayStartedAtMs_ >= kGrowthDayIntervalMs) {
            finalizeDailyRecord(nowMs);
        }
        return;
    }

    latestSnapshot_ = snapshot;

    const CropProfile& crop = currentCrop();
    if (lastUpdateMs_ == 0) {
        lastUpdateMs_ = nowMs;
    }
    const unsigned long deltaMs = nowMs - lastUpdateMs_;
    lastUpdateMs_ = nowMs;

    const float dailyGdd = max(0.0f, snapshot.airTemp - crop.baseTemp);
    const float gddIncrement = dailyGdd * (static_cast<float>(deltaMs) / static_cast<float>(kGrowthDayIntervalMs));
    cumulativeGdd_ += gddIncrement;

    currentStage_ = calculateStage(cumulativeGdd_);

    dayTempSum_ += snapshot.airTemp;
    dayTempMax_ = max(dayTempMax_, snapshot.airTemp);
    dayTempMin_ = min(dayTempMin_, snapshot.airTemp);
    dayHumiSum_ += snapshot.airHumi;
    daySoilSum_ += snapshot.soilHumi;
    dayLightSum_ += snapshot.lightValue * (static_cast<float>(deltaMs) / 3600000.0f);
    ++daySampleCount_;

    predictGrowth();

    if (dayStartedAtMs_ == 0) {
        dayStartedAtMs_ = nowMs;
    }
    if (nowMs - dayStartedAtMs_ >= kGrowthDayIntervalMs) {
        finalizeDailyRecord(nowMs);
    }
}

void GrowthModule::setCrop(int cropId) {
    if (cropId < 0 || cropId >= cropCount()) {
        return;
    }
    currentCropIndex_ = cropId;
    reset();
    saveState();
}

void GrowthModule::reset() {
    cumulativeGdd_ = 0.0f;
    currentDayOfGrowth_ = 0;
    currentStage_ = Seed;
    recordCount_ = 0;
    predictedFloweringDay_ = -1;
    predictedMaturityDay_ = -1;
    predictedYieldScore_ = 0.0f;
    tempGddModel_ = LinearRegression{};
    dayTempSum_ = 0.0f;
    dayTempMax_ = -100.0f;
    dayTempMin_ = 100.0f;
    dayHumiSum_ = 0.0f;
    daySoilSum_ = 0.0f;
    dayLightSum_ = 0.0f;
    daySampleCount_ = 0;
    dayStartedAtMs_ = millis();
    lastUpdateMs_ = 0;
    saveState();
}

int GrowthModule::cropCount() const {
    return static_cast<int>(sizeof(kCropProfiles_) / sizeof(kCropProfiles_[0]));
}

const GrowthModule::CropProfile& GrowthModule::currentCrop() const {
    return kCropProfiles_[currentCropIndex_];
}

GrowthModule::GrowthStage GrowthModule::calculateStage(float gdd) const {
    const CropProfile& crop = currentCrop();
    for (int i = Count - 1; i >= 0; --i) {
        if (crop.gddStages[i] > 0.0f && gdd >= crop.gddStages[i]) {
            return static_cast<GrowthStage>(i);
        }
    }
    return Seed;
}

void GrowthModule::finalizeDailyRecord(unsigned long nowMs) {
    if (daySampleCount_ == 0) {
        dayStartedAtMs_ = nowMs;
        return;
    }

    ++currentDayOfGrowth_;

    const int index = recordCount_ % 90;
    DailyRecord& record = records_[index];
    record.dayIndex = currentDayOfGrowth_;
    record.avgTemp = dayTempSum_ / static_cast<float>(daySampleCount_);
    record.maxTemp = dayTempMax_;
    record.minTemp = dayTempMin_;
    record.avgHumi = dayHumiSum_ / static_cast<float>(daySampleCount_);
    record.avgSoil = daySoilSum_ / static_cast<float>(daySampleCount_);
    record.totalLight = dayLightSum_;
    record.dailyGdd = max(0.0f, ((dayTempMax_ + dayTempMin_) * 0.5f) - currentCrop().baseTemp);
    record.cumulativeGdd = cumulativeGdd_;
    record.stage = currentStage_;
    ++recordCount_;

    dayTempSum_ = 0.0f;
    dayTempMax_ = -100.0f;
    dayTempMin_ = 100.0f;
    dayHumiSum_ = 0.0f;
    daySoilSum_ = 0.0f;
    dayLightSum_ = 0.0f;
    daySampleCount_ = 0;
    dayStartedAtMs_ = nowMs;

    runLinearRegression();
    predictGrowth();
    saveState();
}

void GrowthModule::runLinearRegression() {
    const int n = min(recordCount_, 90);
    if (n < 3) {
        tempGddModel_ = LinearRegression{};
        return;
    }

    float sumX = 0.0f;
    float sumY = 0.0f;
    float sumXY = 0.0f;
    float sumX2 = 0.0f;
    float sumY2 = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float x = static_cast<float>(records_[i].dayIndex);
        const float y = records_[i].cumulativeGdd;
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
        sumY2 += y * y;
    }

    const float denom = n * sumX2 - sumX * sumX;
    if (fabsf(denom) < 0.001f) {
        tempGddModel_ = LinearRegression{};
        return;
    }

    tempGddModel_.slope = (n * sumXY - sumX * sumY) / denom;
    tempGddModel_.intercept = (sumY - tempGddModel_.slope * sumX) / static_cast<float>(n);
    tempGddModel_.sampleCount = n;

    const float ssTot = sumY2 - (sumY * sumY) / static_cast<float>(n);
    float ssRes = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float predicted = tempGddModel_.slope * records_[i].dayIndex + tempGddModel_.intercept;
        const float diff = records_[i].cumulativeGdd - predicted;
        ssRes += diff * diff;
    }
    tempGddModel_.rSquared = ssTot > 0.0f ? (1.0f - ssRes / ssTot) : 0.0f;
}

void GrowthModule::predictGrowth() {
    const CropProfile& crop = currentCrop();

    if (tempGddModel_.sampleCount >= 3 && tempGddModel_.slope > 0.0f) {
        const float gddPerDay = tempGddModel_.slope;
        const float floweringGdd = crop.gddStages[Flowering];
        const float maturityGdd = crop.gddStages[Maturity];

        predictedFloweringDay_ = floweringGdd > cumulativeGdd_
                                     ? currentDayOfGrowth_ + static_cast<int>((floweringGdd - cumulativeGdd_) / gddPerDay)
                                     : currentDayOfGrowth_;
        predictedMaturityDay_ = maturityGdd > cumulativeGdd_
                                    ? currentDayOfGrowth_ + static_cast<int>((maturityGdd - cumulativeGdd_) / gddPerDay)
                                    : currentDayOfGrowth_;
    } else {
        const float avgGddPerDay = max(0.0f, latestSnapshot_.airTemp - crop.baseTemp);
        if (avgGddPerDay > 0.0f) {
            predictedFloweringDay_ = crop.gddStages[Flowering] > cumulativeGdd_
                                         ? currentDayOfGrowth_ + static_cast<int>((crop.gddStages[Flowering] - cumulativeGdd_) / avgGddPerDay)
                                         : currentDayOfGrowth_;
            predictedMaturityDay_ = crop.gddStages[Maturity] > cumulativeGdd_
                                        ? currentDayOfGrowth_ + static_cast<int>((crop.gddStages[Maturity] - cumulativeGdd_) / avgGddPerDay)
                                        : currentDayOfGrowth_;
        }
    }

    predictedYieldScore_ = calculateYieldScore();
}

float GrowthModule::calculateYieldScore() const {
    const CropProfile& crop = currentCrop();
    float score = 100.0f;
    score -= fabsf(latestSnapshot_.airTemp - crop.optimalTemp) * 2.0f;
    score -= fabsf(latestSnapshot_.airHumi - crop.optimalHumi) * 0.5f;
    score -= fabsf(latestSnapshot_.soilHumi - crop.optimalSoil) * 0.8f;
    return constrain(score, 0.0f, 100.0f);
}

void GrowthModule::saveState() {
    prefs_.putInt("crop", currentCropIndex_);
    prefs_.putFloat("gdd", cumulativeGdd_);
    prefs_.putInt("day", currentDayOfGrowth_);
}

String GrowthModule::irrigationAdvice() const {
    switch (currentStage_) {
        case Seed:
        case Germination:
            return "保持土壤适度湿润，少量多次浇水。";
        case Seedling:
            return "避免过度浇水，促进根系发育。";
        case Vegetative:
            return "保持土壤湿度接近作物最适值。";
        case Flowering:
            return "花期避免水分胁迫，保持稳定供水。";
        case Fruiting:
            return "需水量较高，保持土壤湿度稳定。";
        case Maturity:
            return "采收前逐渐减少浇水量。";
        case Count:
        default:
            return "正常灌溉。";
    }
}

void GrowthModule::writeStatus(JsonDocument& doc) const {
    const CropProfile& crop = currentCrop();
    doc["crop"] = crop.name;
    doc["cropCn"] = crop.nameCn;
    doc["dayOfGrowth"] = currentDayOfGrowth_;
    doc["cumulativeGdd"] = cumulativeGdd_;
    doc["currentStage"] = static_cast<int>(currentStage_);
    doc["stageName"] = stageName(currentStage_);
    doc["stageNameCn"] = stageNameCn(currentStage_);

    const float totalGdd = crop.gddStages[Maturity] > 0 ? crop.gddStages[Maturity] : crop.gddStages[Flowering];
    doc["progressPercent"] = totalGdd > 0.0f ? (cumulativeGdd_ / totalGdd) * 100.0f : 0.0f;

    doc["airTemp"] = latestSnapshot_.airTemp;
    doc["airHumi"] = latestSnapshot_.airHumi;
    doc["soilHumi"] = latestSnapshot_.soilHumi;
    doc["lightValue"] = latestSnapshot_.lightValue;

    doc["predictedFloweringDay"] = predictedFloweringDay_;
    doc["predictedMaturityDay"] = predictedMaturityDay_;
    doc["yieldScore"] = predictedYieldScore_;
    doc["irrigationAdvice"] = irrigationAdvice();
}

void GrowthModule::writeHistory(JsonDocument& doc) const {
    JsonArray history = doc.to<JsonArray>();
    const int count = min(recordCount_, 90);
    for (int i = 0; i < count; ++i) {
        const DailyRecord& record = records_[i];
        JsonObject item = history.add<JsonObject>();
        item["day"] = record.dayIndex;
        item["avgTemp"] = record.avgTemp;
        item["maxTemp"] = record.maxTemp;
        item["minTemp"] = record.minTemp;
        item["avgHumi"] = record.avgHumi;
        item["avgSoil"] = record.avgSoil;
        item["totalLight"] = record.totalLight;
        item["dailyGdd"] = record.dailyGdd;
        item["cumulativeGdd"] = record.cumulativeGdd;
        item["stage"] = static_cast<int>(record.stage);
        item["stageName"] = stageName(record.stage);
        item["stageNameCn"] = stageNameCn(record.stage);
    }
}

void GrowthModule::writePrediction(JsonDocument& doc) const {
    const CropProfile& crop = currentCrop();
    doc["crop"] = crop.nameCn;
    doc["currentDay"] = currentDayOfGrowth_;
    doc["currentStage"] = stageNameCn(currentStage_);
    doc["cumulativeGdd"] = cumulativeGdd_;
    doc["predictedFloweringDay"] = predictedFloweringDay_;
    doc["predictedMaturityDay"] = predictedMaturityDay_;
    doc["yieldScore"] = predictedYieldScore_;
    doc["irrigationAdvice"] = irrigationAdvice();

    if (tempGddModel_.sampleCount >= 3) {
        doc["avgGddPerDay"] = tempGddModel_.slope;
        doc["modelRSquared"] = tempGddModel_.rSquared;
        doc["modelSamples"] = tempGddModel_.sampleCount;
    }

    JsonArray stages = doc["stages"].to<JsonArray>();
    for (int i = 0; i < Count; ++i) {
        JsonObject stage = stages.add<JsonObject>();
        stage["name"] = stageName(static_cast<GrowthStage>(i));
        stage["nameCn"] = stageNameCn(static_cast<GrowthStage>(i));
        stage["requiredGdd"] = crop.gddStages[i];
        stage["reached"] = crop.gddStages[i] > 0.0f && cumulativeGdd_ >= crop.gddStages[i];
    }

    JsonArray crops = doc["availableCrops"].to<JsonArray>();
    for (int i = 0; i < cropCount(); ++i) {
        JsonObject item = crops.add<JsonObject>();
        item["id"] = i;
        item["name"] = kCropProfiles_[i].name;
        item["nameCn"] = kCropProfiles_[i].nameCn;
    }
}

}  // namespace agri
