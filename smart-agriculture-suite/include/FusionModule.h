#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>

#include "ActuatorController.h"
#include "AppTypes.h"

namespace agri {

class FusionModule {
public:
    enum class Decision : uint8_t {
        None = 0,
        Moderate,
        Heavy
    };

    struct SensorChannel {
        const char* name = "";
        const char* label = "";
        const char* unit = "";
        float rawValue = 0.0f;
        float normalizedValue = 0.0f;
        float kalmanEstimate = 0.0f;
        float kalmanError = 1.0f;
        float kalmanGain = 0.0f;
        float reliability = 1.0f;
        float weight = 0.2f;
        float minRange = 0.0f;
        float maxRange = 100.0f;
        bool healthy = true;
        int faultCount = 0;
    };

    struct FusionResult {
        Decision decision = Decision::None;
        float confidence = 0.0f;
        float needScore = 0.0f;
        float weightedScore = 0.0f;
        float nnScore = 0.0f;
        float finalScore = 0.0f;
    };

    void begin(bool autoControlEnabled);
    void update(const SensorSnapshot& snapshot, bool sampleUpdated, unsigned long nowMs, ActuatorController& actuator);

    void setAutoControlEnabled(bool enabled) { autoControlEnabled_ = enabled; }
    bool autoControlEnabled() const { return autoControlEnabled_; }
    const FusionResult& result() const { return result_; }
    void loadNetworkFromPrefs();
    void saveNetworkToPrefs();

    void writeStatus(JsonDocument& doc) const;
    void writeSensors(JsonDocument& doc) const;

    static const char* decisionName(Decision decision);

private:
    static constexpr int kSensorCount = 5;
    static constexpr int kHidden = 8;
    static constexpr int kOutputs = 3;

    void initChannels();
    void initNetwork();
    void applyKalmanFilter(SensorChannel& channel, float measurement);
    void normalizeValue(SensorChannel& channel);
    void updateReliability(SensorChannel& channel);
    void calculateWeights();
    void runNeuralNetwork(const float inputs[], float outputs[]);
    static float relu(float value);
    static void softmax(float input[], float output[], int size);
    FusionResult performFusion();

    SensorChannel channels_[kSensorCount];
    float weightsIH_[kSensorCount][kHidden] = {};
    float biasH_[kHidden] = {};
    float weightsHO_[kHidden][kOutputs] = {};
    float biasO_[kOutputs] = {};
    float hiddenOutput_[kHidden] = {};
    float output_[kOutputs] = {};
    FusionResult result_{};
    bool autoControlEnabled_ = false;
    unsigned long lastFusionMs_ = 0;
    int totalDecisions_ = 0;
    int irrigationCount_ = 0;
    float averageConfidence_ = 0.0f;
};

}  // namespace agri
