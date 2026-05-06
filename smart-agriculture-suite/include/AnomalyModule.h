#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>

#include "AppTypes.h"

namespace agri {

class AnomalyModule {
public:
    enum class AlertLevel : uint8_t {
        None = 0,
        Info,
        Warning,
        Critical
    };

    struct SensorStats {
        const char* name = "";
        const char* label = "";
        float window[60] = {};
        int windowIndex = 0;
        int windowCount = 0;
        float mean = 0.0f;
        float stddev = 0.0f;
        float currentValue = 0.0f;
        float lastZScore = 0.0f;
        int stuckCount = 0;
        float lastValue = 0.0f;
        bool isStuck = false;
        bool isDisconnected = false;
        bool isAnomalous = false;
        float minEver = 9999.0f;
        float maxEver = -9999.0f;
        int anomalyCount = 0;
    };

    struct AlertRecord {
        unsigned long timestamp = 0;
        AlertLevel level = AlertLevel::None;
        char sensor[20] = {0};
        char message[96] = {0};
    };

    void begin(int buzzerPin);
    void update(const SensorSnapshot& snapshot, bool sampleUpdated, unsigned long nowMs);
    void clear();

    AlertLevel level() const { return currentLevel_; }
    bool iforestTrained() const { return iforestTrained_; }
    float iforestScore() const { return iforestScore_; }
    int totalSamples() const { return totalSamples_; }
    int totalAnomalies() const { return totalAnomalies_; }

    void writeStatus(JsonDocument& doc) const;
    void writeAlerts(JsonDocument& doc) const;
    bool writeSensorDetail(const String& sensorName, JsonDocument& doc) const;

    static const char* alertLevelName(AlertLevel level);

private:
    struct IForestNode {
        int splitFeature = 0;
        float splitValue = 0.0f;
        int left = -1;
        int right = -1;
    };

    struct IsolationTree {
        IForestNode nodes[255];
        int nodeCount = 0;
    };

    void updateSlidingWindow(SensorStats& stats, float value);
    void calculateStats(SensorStats& stats);
    void checkSensorFault(SensorStats& stats, float value);
    void detectZScoreAnomaly();
    void buildIsolationForest();
    int buildIsolationNode(int treeIdx, float minVals[], float maxVals[], int depth);
    float runIsolationForest(float features[]) const;
    float pathLength(int treeIdx, int nodeIdx, float features[], int depth) const;
    float averagePathLength(int n) const;
    void addAlert(AlertLevel level, const char* sensor, const String& message);
    void updateAlertLevel();
    void beep(AlertLevel level) const;

    static constexpr int kFeatureCount = 4;
    static constexpr int kTrainBufferSize = 200;
    static constexpr int kTrees = 10;
    static constexpr int kDepth = 8;
    static constexpr int kMaxAlerts = 20;

    SensorStats sensors_[kFeatureCount];
    IsolationTree forest_[kTrees];
    float trainBuffer_[kTrainBufferSize][kFeatureCount] = {};
    int trainBufferCount_ = 0;
    AlertRecord alerts_[kMaxAlerts];
    int alertCount_ = 0;
    int alertIndex_ = 0;
    AlertLevel currentLevel_ = AlertLevel::None;
    bool iforestTrained_ = false;
    float iforestScore_ = 0.0f;
    int totalSamples_ = 0;
    int totalAnomalies_ = 0;
    int buzzerPin_ = -1;
    unsigned long lastIforestRunMs_ = 0;
};

}  // namespace agri
