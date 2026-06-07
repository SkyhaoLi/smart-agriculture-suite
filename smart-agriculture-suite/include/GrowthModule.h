#pragma once

#include <Arduino.h>
#include <Preferences.h>
#include <ArduinoJson.h>

#include "AppTypes.h"

namespace agri {

struct CropProfile {
    const char* name;
    const char* nameCn;
    float baseTemp;
    float gddStages[static_cast<uint8_t>(GrowthStage::Count)];
    float optimalTemp;
    float optimalHumi;
    float optimalSoil;
};

class GrowthModule {
public:
    void begin();
    void update(const SensorSnapshot& snapshot, bool sampleUpdated, unsigned long nowMs);
    void setCrop(uint8_t cropId);
    void reset();

    int cropCount() const { return 5; }
    uint8_t currentCropIndex() const { return currentCropIndex_; }
    float cumulativeGdd() const { return cumulativeGdd_; }
    int currentDayOfGrowth() const { return currentDayOfGrowth_; }
    GrowthStage currentStage() const { return currentStage_; }
    const CropProfile& currentCrop() const;
    float yieldScore() const { return yieldScore_; }
    String irrigationAdvice() const;

    void writeStatus(JsonObject& obj) const;

    static const char* stageName(GrowthStage s) { return agri::stageName(s); }
    static const char* stageNameCn(GrowthStage s) { return agri::stageNameCn(s); }

private:
    static const CropProfile kCropProfiles_[5];

    GrowthStage calculateStage(float gdd) const;
    void saveState();

    Preferences prefs_;
    float cumulativeGdd_ = 0.0f;
    GrowthStage currentStage_ = GrowthStage::Seed;
    uint8_t currentCropIndex_ = 0;
    int currentDayOfGrowth_ = 0;
    float yieldScore_ = 100.0f;
    unsigned long lastUpdateMs_ = 0;
    unsigned long dayStartedAtMs_ = 0;
};

}  // namespace agri
