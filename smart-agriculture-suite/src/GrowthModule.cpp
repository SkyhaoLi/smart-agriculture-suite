#include "GrowthModule.h"
#include <ArduinoJson.h>

namespace agri {

const CropProfile GrowthModule::kCropProfiles_[5] = {
    // Tomato
    {"Tomato", "番茄", 10.0f, {0, 80, 200, 500, 800, 1100, 1400}, 25.0f, 60.0f, 60.0f},
    // Lettuce
    {"Lettuce", "生菜", 4.5f, {0, 40, 100, 250, 400, 400, 550}, 18.0f, 65.0f, 55.0f},
    // Pepper
    {"Pepper", "辣椒", 12.0f, {0, 100, 250, 600, 900, 1200, 1600}, 28.0f, 55.0f, 55.0f},
    // Cucumber
    {"Cucumber", "黄瓜", 10.0f, {0, 60, 180, 450, 700, 950, 1200}, 26.0f, 70.0f, 65.0f},
    // Strawberry
    {"Strawberry", "草莓", 5.0f, {0, 50, 150, 400, 650, 900, 1200}, 20.0f, 70.0f, 60.0f},
};

void GrowthModule::begin() {
    prefs_.begin("growth", false);
    currentCropIndex_ = prefs_.getUChar("crop", 0);
    cumulativeGdd_ = prefs_.getFloat("gdd", 0.0f);
    currentDayOfGrowth_ = prefs_.getInt("day", 0);
    currentStage_ = calculateStage(cumulativeGdd_);
    dayStartedAtMs_ = millis();
}

void GrowthModule::update(const SensorSnapshot& snapshot, bool sampleUpdated, unsigned long nowMs) {
    if (!sampleUpdated || snapshot.fault.airFault()) return;

    const CropProfile& crop = currentCrop();
    float dailyGdd = max(0.0f, snapshot.airTemp - crop.baseTemp);

    // 每24小时累计一次GDD
    if (nowMs - dayStartedAtMs_ >= 86400000UL) {
        cumulativeGdd_ += dailyGdd;
        currentDayOfGrowth_++;
        currentStage_ = calculateStage(cumulativeGdd_);
        dayStartedAtMs_ = nowMs;
        saveState();
    }

    // 产量评分
    float tempPenalty = abs(snapshot.airTemp - crop.optimalTemp) * 2.0f;
    float humiPenalty = abs(snapshot.airHumi - crop.optimalHumi) * 0.5f;
    float soilPenalty = snapshot.fault.soilFault() ? 0.0f :
                        abs(snapshot.soilHumi - crop.optimalSoil) * 0.8f;
    yieldScore_ = constrain(100.0f - tempPenalty - humiPenalty - soilPenalty, 0.0f, 100.0f);

    lastUpdateMs_ = nowMs;
}

void GrowthModule::setCrop(uint8_t cropId) {
    if (cropId >= 5) return;
    currentCropIndex_ = cropId;
    cumulativeGdd_ = 0.0f;
    currentDayOfGrowth_ = 0;
    currentStage_ = GrowthStage::Seed;
    dayStartedAtMs_ = millis();
    saveState();
}

void GrowthModule::reset() {
    cumulativeGdd_ = 0.0f;
    currentDayOfGrowth_ = 0;
    currentStage_ = GrowthStage::Seed;
    dayStartedAtMs_ = millis();
    saveState();
}

const CropProfile& GrowthModule::currentCrop() const {
    return kCropProfiles_[currentCropIndex_];
}

GrowthStage GrowthModule::calculateStage(float gdd) const {
    const CropProfile& crop = currentCrop();
    for (int i = static_cast<int>(GrowthStage::Count) - 1; i >= 0; i--) {
        if (gdd >= crop.gddStages[i]) return static_cast<GrowthStage>(i);
    }
    return GrowthStage::Seed;
}

String GrowthModule::irrigationAdvice() const {
    switch (currentStage_) {
        case GrowthStage::Seed:
        case GrowthStage::Germination:
            return "保持土壤湿润,少量多次灌溉";
        case GrowthStage::Seedling:
            return "幼苗期适当控水,促进根系生长";
        case GrowthStage::Vegetative:
            return "营养生长期需水量增加,保证充足供水";
        case GrowthStage::Flowering:
            return "开花期需稳定水分,避免干旱胁迫";
        case GrowthStage::Fruiting:
            return "结果期需水量最大,保持均匀灌溉";
        case GrowthStage::Maturity:
            return "成熟期适当控水,提高品质";
        default:
            return "监测中...";
    }
}

void GrowthModule::saveState() {
    prefs_.putUChar("crop", currentCropIndex_);
    prefs_.putFloat("gdd", cumulativeGdd_);
    prefs_.putInt("day", currentDayOfGrowth_);
}

void GrowthModule::writeStatus(JsonObject& obj) const {
    obj["crop"] = cropName(static_cast<CropType>(currentCropIndex_));
    obj["crop_cn"] = cropNameCn(static_cast<CropType>(currentCropIndex_));
    obj["stage"] = stageName(currentStage_);
    obj["stage_cn"] = stageNameCn(currentStage_);
    obj["day"] = currentDayOfGrowth_;
    obj["gdd"] = cumulativeGdd_;
    obj["yield_score"] = yieldScore_;
    obj["advice"] = irrigationAdvice();
}

}  // namespace agri
