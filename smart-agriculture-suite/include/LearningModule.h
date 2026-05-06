#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>
#include <Preferences.h>

#include "ActuatorController.h"
#include "AppTypes.h"

namespace agri {

class LearningModule {
public:
    enum Action : uint8_t {
        Off = 0,
        Low,
        Medium,
        High,
        Count
    };

    struct LearningRecord {
        int state = 0;
        Action action = Off;
        float reward = 0.0f;
        float soilBefore = 0.0f;
        float soilAfter = 0.0f;
    };

    void begin(const LearningConfig& config);
    void update(const SensorSnapshot& snapshot, bool sampleUpdated, unsigned long nowMs, ActuatorController& actuator);
    void setConfig(const LearningConfig& config) { config_ = config; }
    const LearningConfig& config() const { return config_; }
    void reset();
    void recordUserFeedback(bool positive);
    void writeStatus(JsonDocument& doc) const;
    void writeQTableSummary(JsonDocument& doc) const;

    static const char* actionName(Action action);

private:
    static constexpr int kTempLevels = 5;
    static constexpr int kHumiLevels = 4;
    static constexpr int kSoilLevels = 5;
    static constexpr int kLightLevels = 3;
    static constexpr int kTimeLevels = 3;
    static constexpr int kStateCount = kTempLevels * kHumiLevels * kSoilLevels * kLightLevels * kTimeLevels;
    static constexpr int kHistorySize = 50;

    void initQTable();
    void loadQTable();
    void saveQTable();
    int discretizeState(const SensorSnapshot& snapshot) const;
    int discretizeTemp(float temp) const;
    int discretizeHumi(float humi) const;
    int discretizeSoil(float soil) const;
    int discretizeLight(float light) const;
    int getTimePeriod(unsigned long nowMs) const;
    Action selectAction(int state) const;
    float calculateReward(float soilBefore, float soilAfter, Action action) const;
    void updateQValue(int state, Action action, float reward, int nextState);

    Preferences prefs_;
    LearningConfig config_{};
    float qTable_[kStateCount][Count] = {};
    LearningRecord history_[kHistorySize];
    int historyCount_ = 0;
    int historyIndex_ = 0;
    int totalEpisodes_ = 0;
    float totalReward_ = 0.0f;
    float averageReward_ = 0.0f;
    float lastReward_ = 0.0f;
    int lastState_ = 0;
    Action lastAction_ = Off;
    bool hasPendingReward_ = false;
    float pendingSoilBefore_ = 0.0f;
    unsigned long lastDecisionMs_ = 0;
    int userOverrideCount_ = 0;
    float userSatisfaction_ = 50.0f;
    SensorSnapshot latestSnapshot_{};
};

}  // namespace agri
