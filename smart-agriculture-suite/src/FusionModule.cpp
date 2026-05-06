#include "FusionModule.h"

#include <math.h>
#include <string.h>
#include <Preferences.h>

#include "AppConfig.h"

namespace agri {

namespace {
constexpr const char* kFusionNs = "fusion_nn";
}  // namespace

const char* FusionModule::decisionName(Decision decision) {
    switch (decision) {
        case Decision::Moderate: return "moderate";
        case Decision::Heavy: return "heavy";
        case Decision::None:
        default:
            return "none";
    }
}

void FusionModule::begin(bool autoControlEnabled) {
    autoControlEnabled_ = autoControlEnabled;
    initChannels();
    initNetwork();
    loadNetworkFromPrefs();
}

void FusionModule::update(const SensorSnapshot& snapshot, bool sampleUpdated, unsigned long nowMs, ActuatorController& actuator) {
    if (sampleUpdated) {
        const float rawValues[kSensorCount] = {
            snapshot.airTemp,
            snapshot.airHumi,
            snapshot.soilHumi,
            snapshot.lightValue,
            snapshot.liquidLevel
        };

        for (int i = 0; i < kSensorCount; ++i) {
            channels_[i].rawValue = rawValues[i];
            applyKalmanFilter(channels_[i], rawValues[i]);
            normalizeValue(channels_[i]);
            updateReliability(channels_[i]);
        }
        calculateWeights();
    }

    if (nowMs - lastFusionMs_ < kFusionIntervalMs) {
        return;
    }

    result_ = performFusion();
    lastFusionMs_ = nowMs;

    if (!autoControlEnabled_ || !actuator.status().autoMode || actuator.isBusy(nowMs)) {
        return;
    }

    unsigned long duration = 0;
    switch (result_.decision) {
        case Decision::Moderate: duration = 45UL; break;
        case Decision::Heavy: duration = 120UL; break;
        case Decision::None:
        default:
            duration = 0;
            break;
    }
    if (duration > 0) {
        actuator.startTimedRun(ControlSource::Fusion, duration, nowMs);
    }
}

void FusionModule::writeStatus(JsonDocument& doc) const {
    doc["autoControlEnabled"] = autoControlEnabled_;
    doc["decision"] = static_cast<int>(result_.decision);
    doc["decisionName"] = decisionName(result_.decision);
    doc["confidence"] = result_.confidence;
    doc["needScore"] = result_.needScore;
    doc["weightedScore"] = result_.weightedScore;
    doc["nnScore"] = result_.nnScore;
    doc["finalScore"] = result_.finalScore;
    doc["totalDecisions"] = totalDecisions_;
    doc["irrigationCount"] = irrigationCount_;
    doc["avgConfidence"] = averageConfidence_;

    doc["nn"]["none"] = output_[0];
    doc["nn"]["moderate"] = output_[1];
    doc["nn"]["heavy"] = output_[2];
}

void FusionModule::writeSensors(JsonDocument& doc) const {
    JsonArray sensors = doc.to<JsonArray>();
    for (int i = 0; i < kSensorCount; ++i) {
        JsonObject item = sensors.add<JsonObject>();
        item["name"] = channels_[i].name;
        item["label"] = channels_[i].label;
        item["unit"] = channels_[i].unit;
        item["raw"] = channels_[i].rawValue;
        item["filtered"] = channels_[i].kalmanEstimate;
        item["normalized"] = channels_[i].normalizedValue;
        item["reliability"] = channels_[i].reliability;
        item["weight"] = channels_[i].weight;
        item["kalmanGain"] = channels_[i].kalmanGain;
        item["healthy"] = channels_[i].healthy;
    }
}

void FusionModule::initChannels() {
    channels_[0].name = "AirTemp";
    channels_[0].label = "Temperature";
    channels_[0].unit = "C";
    channels_[0].kalmanEstimate = 25.0f;
    channels_[0].weight = 0.20f;
    channels_[0].maxRange = 40.0f;

    channels_[1].name = "AirHumi";
    channels_[1].label = "Humidity";
    channels_[1].unit = "%";
    channels_[1].kalmanEstimate = 60.0f;
    channels_[1].weight = 0.20f;
    channels_[1].maxRange = 100.0f;

    channels_[2].name = "SoilHumi";
    channels_[2].label = "Soil";
    channels_[2].unit = "%";
    channels_[2].kalmanEstimate = 50.0f;
    channels_[2].weight = 0.30f;
    channels_[2].maxRange = 100.0f;

    channels_[3].name = "Light";
    channels_[3].label = "Light";
    channels_[3].unit = "lux";
    channels_[3].kalmanEstimate = 500.0f;
    channels_[3].weight = 0.15f;
    channels_[3].maxRange = 10000.0f;

    channels_[4].name = "Liquid";
    channels_[4].label = "Liquid";
    channels_[4].unit = "%";
    channels_[4].kalmanEstimate = 80.0f;
    channels_[4].weight = 0.15f;
    channels_[4].maxRange = 100.0f;
}

void FusionModule::initNetwork() {
    // Trained weights from synthetic data (val acc 86.4%)
    const float presetIH[kSensorCount][kHidden] = {
        {-0.7618f, -0.1648f, 0.9222f, 0.2010f, 0.1822f, 0.8395f, 0.6211f, -0.7773f},
        {0.7061f, -0.4602f, -0.9656f, -0.1033f, -0.5191f, 0.2046f, -0.2469f, 1.1025f},
        {1.1686f, 0.1717f, -0.5193f, -0.1273f, -0.0344f, -0.5801f, -0.5640f, 1.7332f},
        {0.1232f, -0.2913f, 1.0482f, 0.3298f, -0.1933f, 0.3476f, 0.3208f, -0.1836f},
        {-0.4105f, 0.1005f, 1.5418f, -2.6921f, -0.4168f, 1.2474f, -2.1958f, -0.7266f},
    };
    const float presetBH[kHidden] = {
        0.5547f, -0.0495f, 0.2924f, 0.5884f, -0.0884f, 0.3348f, 0.4393f, 0.7546f
    };
    const float presetHO[kHidden][kOutputs] = {
        {1.1199f, 0.6752f, -2.4982f},
        {0.6507f, -0.6800f, 0.1269f},
        {-1.5823f, -0.1323f, 1.2406f},
        {2.7740f, -0.9054f, -1.8639f},
        {-0.6520f, 0.0906f, -0.5720f},
        {-1.1279f, 0.6628f, 0.5219f},
        {2.5613f, -2.2228f, -0.1253f},
        {1.4902f, -0.1788f, -3.4092f},
    };
    const float presetBO[kOutputs] = {
        -0.1035f, 0.3601f, -0.2272f
    };

    memcpy(weightsIH_, presetIH, sizeof(presetIH));
    memcpy(biasH_, presetBH, sizeof(presetBH));
    memcpy(weightsHO_, presetHO, sizeof(presetHO));
    memcpy(biasO_, presetBO, sizeof(presetBO));
}

void FusionModule::applyKalmanFilter(SensorChannel& channel, float measurement) {
    constexpr float kQ = 0.01f;
    constexpr float kR = 0.1f;

    const float predictedEstimate = channel.kalmanEstimate;
    const float predictedError = channel.kalmanError + kQ;

    channel.kalmanGain = predictedError / (predictedError + kR);
    channel.kalmanEstimate = predictedEstimate + channel.kalmanGain * (measurement - predictedEstimate);
    channel.kalmanError = (1.0f - channel.kalmanGain) * predictedError;
}

void FusionModule::normalizeValue(SensorChannel& channel) {
    const float range = channel.maxRange - channel.minRange;
    if (range <= 0.0f) {
        channel.normalizedValue = 0.0f;
        return;
    }

    channel.normalizedValue = (channel.kalmanEstimate - channel.minRange) / range;
    channel.normalizedValue = constrain(channel.normalizedValue, 0.0f, 1.0f);
}

void FusionModule::updateReliability(SensorChannel& channel) {
    float reliability = 1.0f;

    if (channel.rawValue <= 0.001f || channel.rawValue > channel.maxRange * 1.5f) {
        reliability *= 0.1f;
        ++channel.faultCount;
    } else {
        channel.faultCount = max(0, channel.faultCount - 1);
    }

    if (channel.kalmanGain > 0.8f) {
        reliability *= 0.7f;
    }

    if (channel.faultCount > 5) {
        reliability *= 0.3f;
        channel.healthy = false;
    } else {
        channel.healthy = true;
    }

    channel.reliability = channel.reliability * 0.8f + reliability * 0.2f;
}

void FusionModule::calculateWeights() {
    float totalReliability = 0.0f;
    for (int i = 0; i < kSensorCount; ++i) {
        totalReliability += channels_[i].reliability;
    }
    if (totalReliability <= 0.0f) {
        return;
    }
    for (int i = 0; i < kSensorCount; ++i) {
        channels_[i].weight = channels_[i].reliability / totalReliability;
    }
}

void FusionModule::runNeuralNetwork(const float inputs[], float outputs[]) {
    for (int h = 0; h < kHidden; ++h) {
        float sum = biasH_[h];
        for (int i = 0; i < kSensorCount; ++i) {
            sum += inputs[i] * weightsIH_[i][h];
        }
        hiddenOutput_[h] = relu(sum);
    }

    float raw[kOutputs];
    for (int o = 0; o < kOutputs; ++o) {
        float sum = biasO_[o];
        for (int h = 0; h < kHidden; ++h) {
            sum += hiddenOutput_[h] * weightsHO_[h][o];
        }
        raw[o] = sum;
    }

    softmax(raw, outputs, kOutputs);
    memcpy(output_, outputs, sizeof(output_));
}

float FusionModule::relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

void FusionModule::softmax(float input[], float output[], int size) {
    float maxValue = input[0];
    for (int i = 1; i < size; ++i) {
        maxValue = max(maxValue, input[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = expf(input[i] - maxValue);
        sum += output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

FusionModule::FusionResult FusionModule::performFusion() {
    FusionResult result;

    float needFactors[kSensorCount];
    needFactors[0] = channels_[0].normalizedValue;
    needFactors[1] = 1.0f - channels_[1].normalizedValue;
    needFactors[2] = 1.0f - channels_[2].normalizedValue;
    needFactors[3] = channels_[3].normalizedValue * 0.5f;
    needFactors[4] = channels_[4].normalizedValue;

    result.weightedScore = 0.0f;
    for (int i = 0; i < kSensorCount; ++i) {
        result.weightedScore += needFactors[i] * channels_[i].weight;
    }
    result.weightedScore *= 100.0f;

    float inputs[kSensorCount];
    for (int i = 0; i < kSensorCount; ++i) {
        inputs[i] = channels_[i].normalizedValue;
    }

    float outputs[kOutputs];
    runNeuralNetwork(inputs, outputs);
    result.nnScore = outputs[1] * 50.0f + outputs[2] * 100.0f;
    result.finalScore = result.weightedScore * 0.6f + result.nnScore * 0.4f;
    result.needScore = result.finalScore;

    if (channels_[4].rawValue < 20.0f) {
        result.decision = Decision::None;
        result.confidence = 0.95f;
    } else if (result.finalScore > 65.0f) {
        result.decision = Decision::Heavy;
        result.confidence = min(1.0f, result.finalScore / 100.0f);
    } else if (result.finalScore > 35.0f) {
        result.decision = Decision::Moderate;
        result.confidence = 0.5f + (result.finalScore - 35.0f) / 60.0f;
    } else {
        result.decision = Decision::None;
        result.confidence = 1.0f - result.finalScore / 70.0f;
    }

    ++totalDecisions_;
    if (result.decision != Decision::None) {
        ++irrigationCount_;
    }
    averageConfidence_ = averageConfidence_ * 0.95f + result.confidence * 0.05f;
    return result;
}

void FusionModule::loadNetworkFromPrefs() {
    Preferences prefs;
    if (!prefs.begin(kFusionNs, true)) {
        return;
    }

    // Check if persisted weights exist (key "ver" acts as marker)
    if (prefs.getUInt("ver", 0) == 0) {
        prefs.end();
        return;
    }

    // weightsIH_: 5×8 = 40 floats stored as "wi<row><col>" compact keys
    for (int i = 0; i < kSensorCount; ++i) {
        char key[8];
        for (int h = 0; h < kHidden; ++h) {
            snprintf(key, sizeof(key), "w%d%d", i, h);
            weightsIH_[i][h] = prefs.getFloat(key, weightsIH_[i][h]);
        }
    }

    // biasH_: 8 floats
    for (int h = 0; h < kHidden; ++h) {
        char key[8];
        snprintf(key, sizeof(key), "bh%d", h);
        biasH_[h] = prefs.getFloat(key, biasH_[h]);
    }

    // weightsHO_: 8×3 = 24 floats
    for (int h = 0; h < kHidden; ++h) {
        char key[8];
        for (int o = 0; o < kOutputs; ++o) {
            snprintf(key, sizeof(key), "v%d%d", h, o);
            weightsHO_[h][o] = prefs.getFloat(key, weightsHO_[h][o]);
        }
    }

    // biasO_: 3 floats
    for (int o = 0; o < kOutputs; ++o) {
        char key[8];
        snprintf(key, sizeof(key), "bo%d", o);
        biasO_[o] = prefs.getFloat(key, biasO_[o]);
    }

    prefs.end();
    Serial.println("[Fusion] loaded trained weights from NVS");
}

void FusionModule::saveNetworkToPrefs() {
    Preferences prefs;
    if (!prefs.begin(kFusionNs, false)) {
        Serial.println("[Fusion] failed to open NVS for writing");
        return;
    }

    prefs.putUInt("ver", 1);  // marker that weights exist

    for (int i = 0; i < kSensorCount; ++i) {
        char key[8];
        for (int h = 0; h < kHidden; ++h) {
            snprintf(key, sizeof(key), "w%d%d", i, h);
            prefs.putFloat(key, weightsIH_[i][h]);
        }
    }

    for (int h = 0; h < kHidden; ++h) {
        char key[8];
        snprintf(key, sizeof(key), "bh%d", h);
        prefs.putFloat(key, biasH_[h]);
    }

    for (int h = 0; h < kHidden; ++h) {
        char key[8];
        for (int o = 0; o < kOutputs; ++o) {
            snprintf(key, sizeof(key), "v%d%d", h, o);
            prefs.putFloat(key, weightsHO_[h][o]);
        }
    }

    for (int o = 0; o < kOutputs; ++o) {
        char key[8];
        snprintf(key, sizeof(key), "bo%d", o);
        prefs.putFloat(key, biasO_[o]);
    }

    prefs.end();
    Serial.println("[Fusion] saved trained weights to NVS");
}

}  // namespace agri
