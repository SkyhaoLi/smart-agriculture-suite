#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>
#include <Preferences.h>

#include "AppTypes.h"

namespace agri {

class GrowthModule {
public:
    enum GrowthStage : uint8_t {
        Seed = 0,
        Germination,
        Seedling,
        Vegetative,
        Flowering,
        Fruiting,
        Maturity,
        Count
    };

    struct CropProfile {
        const char* name = "";
        const char* nameCn = "";
        float baseTemp = 0.0f;
        float gddStages[Count] = {};
        float optimalTemp = 0.0f;
        float optimalHumi = 0.0f;
        float optimalSoil = 0.0f;
        float dailyLightHours = 0.0f;
    };

    struct DailyRecord {
        int dayIndex = 0;
        float avgTemp = 0.0f;
        float maxTemp = 0.0f;
        float minTemp = 0.0f;
        float avgHumi = 0.0f;
        float avgSoil = 0.0f;
        float totalLight = 0.0f;
        float dailyGdd = 0.0f;
        float cumulativeGdd = 0.0f;
        GrowthStage stage = Seed;
    };

    void begin();
    void update(const SensorSnapshot& snapshot, bool sampleUpdated, unsigned long nowMs);
    void setCrop(int cropId);
    void reset();

    int cropCount() const;
    int currentCropIndex() const { return currentCropIndex_; }
    float cumulativeGdd() const { return cumulativeGdd_; }
    int currentDayOfGrowth() const { return currentDayOfGrowth_; }
    GrowthStage currentStage() const { return currentStage_; }
    const CropProfile& currentCrop() const;
    String irrigationAdvice() const;

    void writeStatus(JsonDocument& doc) const;
    void writeHistory(JsonDocument& doc) const;
    void writePrediction(JsonDocument& doc) const;

    static const char* stageName(GrowthStage stage);
    static const char* stageNameCn(GrowthStage stage);

private:
    struct LinearRegression {
        float slope = 0.0f;
        float intercept = 0.0f;
        float rSquared = 0.0f;
        int sampleCount = 0;
    };

    static const CropProfile kCropProfiles_[5];

    GrowthStage calculateStage(float gdd) const;
    void finalizeDailyRecord(unsigned long nowMs);
    void runLinearRegression();
    void predictGrowth();
    float calculateYieldScore() const;
    void saveState();

    Preferences prefs_;
    SensorSnapshot latestSnapshot_{};
    float cumulativeGdd_ = 0.0f;
    GrowthStage currentStage_ = Seed;
    int currentCropIndex_ = 0;
    int currentDayOfGrowth_ = 0;
    DailyRecord records_[90];
    int recordCount_ = 0;
    float dayTempSum_ = 0.0f;
    float dayTempMax_ = -100.0f;
    float dayTempMin_ = 100.0f;
    float dayHumiSum_ = 0.0f;
    float daySoilSum_ = 0.0f;
    float dayLightSum_ = 0.0f;
    int daySampleCount_ = 0;
    LinearRegression tempGddModel_;
    int predictedFloweringDay_ = -1;
    int predictedMaturityDay_ = -1;
    float predictedYieldScore_ = 0.0f;
    unsigned long lastUpdateMs_ = 0;
    unsigned long dayStartedAtMs_ = 0;
};

}  // namespace agri
