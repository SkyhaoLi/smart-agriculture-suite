#include "AnomalyModule.h"

#include <math.h>
#include <string.h>

#include "AppConfig.h"

namespace agri {

namespace {
constexpr float kZScoreThreshold = 2.5f;
constexpr int kStuckThreshold = 30;
constexpr int kWindowSize = 60;
constexpr int kMaxNodesPerTree = 255;
}  // namespace

const char* AnomalyModule::alertLevelName(AlertLevel level) {
    switch (level) {
        case AlertLevel::Info: return "info";
        case AlertLevel::Warning: return "warning";
        case AlertLevel::Critical: return "critical";
        case AlertLevel::None:
        default:
            return "normal";
    }
}

void AnomalyModule::begin(int buzzerPin) {
    buzzerPin_ = buzzerPin;
    if (buzzerPin_ >= 0) {
        pinMode(buzzerPin_, OUTPUT);
        digitalWrite(buzzerPin_, LOW);
    }

    const char* names[kFeatureCount] = {"AirTemp", "AirHumi", "SoilHumi", "Light"};
    const char* labels[kFeatureCount] = {"AirTemp", "AirHumi", "Soil", "Light"};

    for (int i = 0; i < kFeatureCount; ++i) {
        sensors_[i].name = names[i];
        sensors_[i].label = labels[i];
    }
}

void AnomalyModule::update(const SensorSnapshot& snapshot, bool sampleUpdated, unsigned long nowMs) {
    if (sampleUpdated) {
        float values[kFeatureCount] = {
            snapshot.airTemp,
            snapshot.airHumi,
            snapshot.soilHumi,
            snapshot.lightValue
        };

        for (int i = 0; i < kFeatureCount; ++i) {
            checkSensorFault(sensors_[i], values[i]);
            updateSlidingWindow(sensors_[i], values[i]);
            sensors_[i].currentValue = values[i];
            calculateStats(sensors_[i]);

            if (values[i] < sensors_[i].minEver) {
                sensors_[i].minEver = values[i];
            }
            if (values[i] > sensors_[i].maxEver) {
                sensors_[i].maxEver = values[i];
            }
        }

        if (trainBufferCount_ < kTrainBufferSize) {
            for (int i = 0; i < kFeatureCount; ++i) {
                trainBuffer_[trainBufferCount_][i] = values[i];
            }
            ++trainBufferCount_;
        }

        detectZScoreAnomaly();
        updateAlertLevel();
        ++totalSamples_;
    }

    if (nowMs - lastIforestRunMs_ >= kAnomalyIsolationIntervalMs) {
        if (trainBufferCount_ >= 50 && !iforestTrained_) {
            buildIsolationForest();
        }
        if (iforestTrained_) {
            float features[kFeatureCount] = {
                sensors_[0].currentValue,
                sensors_[1].currentValue,
                sensors_[2].currentValue,
                sensors_[3].currentValue
            };
            iforestScore_ = runIsolationForest(features);
            if (iforestScore_ > 0.65f) {
                addAlert(AlertLevel::Warning, "IForest", "multi-sensor anomaly detected");
                ++totalAnomalies_;
            }
            updateAlertLevel();
        }
        lastIforestRunMs_ = nowMs;
    }
}

void AnomalyModule::clear() {
    alertCount_ = 0;
    alertIndex_ = 0;
    currentLevel_ = AlertLevel::None;
    totalAnomalies_ = 0;
    for (int i = 0; i < kFeatureCount; ++i) {
        sensors_[i].anomalyCount = 0;
        sensors_[i].isAnomalous = false;
        sensors_[i].isDisconnected = false;
        sensors_[i].isStuck = false;
    }
}

void AnomalyModule::updateSlidingWindow(SensorStats& stats, float value) {
    stats.window[stats.windowIndex] = value;
    stats.windowIndex = (stats.windowIndex + 1) % kWindowSize;
    if (stats.windowCount < kWindowSize) {
        ++stats.windowCount;
    }
}

void AnomalyModule::calculateStats(SensorStats& stats) {
    if (stats.windowCount < 3) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < stats.windowCount; ++i) {
        sum += stats.window[i];
    }
    stats.mean = sum / static_cast<float>(stats.windowCount);

    float sumSq = 0.0f;
    for (int i = 0; i < stats.windowCount; ++i) {
        const float diff = stats.window[i] - stats.mean;
        sumSq += diff * diff;
    }
    stats.stddev = sqrtf(sumSq / static_cast<float>(stats.windowCount));

    if (stats.stddev > 0.001f) {
        stats.lastZScore = (stats.currentValue - stats.mean) / stats.stddev;
    } else {
        stats.lastZScore = 0.0f;
    }
}

void AnomalyModule::checkSensorFault(SensorStats& stats, float value) {
    const bool isLight = strcmp(stats.name, "Light") == 0;

    if (!isLight && value <= 0.001f && stats.windowCount > 10) {
        if (!stats.isDisconnected) {
            addAlert(AlertLevel::Critical, stats.name, String(stats.label) + " disconnected");
        }
        stats.isDisconnected = true;
        return;
    }
    stats.isDisconnected = false;

    if (fabsf(value - stats.lastValue) < 0.01f) {
        ++stats.stuckCount;
        if (stats.stuckCount >= kStuckThreshold && !stats.isStuck) {
            stats.isStuck = true;
            addAlert(AlertLevel::Info, stats.name, String(stats.label) + " may be stuck");
        }
    } else {
        stats.stuckCount = 0;
        stats.isStuck = false;
    }
    stats.lastValue = value;
}

void AnomalyModule::detectZScoreAnomaly() {
    for (int i = 0; i < kFeatureCount; ++i) {
        if (sensors_[i].windowCount < 10) {
            continue;
        }

        const float absZ = fabsf(sensors_[i].lastZScore);
        if (absZ > kZScoreThreshold) {
            if (!sensors_[i].isAnomalous) {
                sensors_[i].isAnomalous = true;
                ++sensors_[i].anomalyCount;
                ++totalAnomalies_;

                String message = String(sensors_[i].label) + " z-score=" + String(sensors_[i].lastZScore, 2);
                addAlert(absZ > 3.5f ? AlertLevel::Critical : AlertLevel::Warning, sensors_[i].name, message);
            }
        } else {
            sensors_[i].isAnomalous = false;
        }
    }
}

void AnomalyModule::buildIsolationForest() {
    const int n = trainBufferCount_;
    if (n <= 1) {
        return;
    }

    for (int t = 0; t < kTrees; ++t) {
        forest_[t].nodeCount = 0;

        float minVals[kFeatureCount];
        float maxVals[kFeatureCount];
        for (int f = 0; f < kFeatureCount; ++f) {
            minVals[f] = 9999.0f;
            maxVals[f] = -9999.0f;
        }

        for (int i = 0; i < n; ++i) {
            for (int f = 0; f < kFeatureCount; ++f) {
                if (trainBuffer_[i][f] < minVals[f]) {
                    minVals[f] = trainBuffer_[i][f];
                }
                if (trainBuffer_[i][f] > maxVals[f]) {
                    maxVals[f] = trainBuffer_[i][f];
                }
            }
        }

        // Build tree recursively to full depth kDepth
        buildIsolationNode(t, minVals, maxVals, 0);
    }

    iforestTrained_ = true;
}

int AnomalyModule::buildIsolationNode(int treeIdx, float minVals[], float maxVals[], int depth) {
    if (depth >= kDepth || forest_[treeIdx].nodeCount >= kMaxNodesPerTree) {
        return -1;
    }

    const int nodeIndex = forest_[treeIdx].nodeCount++;

    int feature = random(kFeatureCount);
    float splitValue = minVals[feature] + random(1000) / 1000.0f * (maxVals[feature] - minVals[feature]);
    forest_[treeIdx].nodes[nodeIndex].splitFeature = feature;
    forest_[treeIdx].nodes[nodeIndex].splitValue = splitValue;
    forest_[treeIdx].nodes[nodeIndex].left = -1;
    forest_[treeIdx].nodes[nodeIndex].right = -1;

    float leftMin[kFeatureCount], leftMax[kFeatureCount];
    float rightMin[kFeatureCount], rightMax[kFeatureCount];
    memcpy(leftMin, minVals, sizeof(leftMin));
    memcpy(leftMax, maxVals, sizeof(leftMax));
    memcpy(rightMin, minVals, sizeof(rightMin));
    memcpy(rightMax, maxVals, sizeof(rightMax));

    // Left child: samples where value < splitValue, so max of split feature = splitValue
    leftMax[feature] = splitValue;
    // Right child: samples where value >= splitValue, so min of split feature = splitValue
    rightMin[feature] = splitValue;

    int leftIdx = buildIsolationNode(treeIdx, leftMin, leftMax, depth + 1);
    int rightIdx = buildIsolationNode(treeIdx, rightMin, rightMax, depth + 1);

    forest_[treeIdx].nodes[nodeIndex].left = leftIdx;
    forest_[treeIdx].nodes[nodeIndex].right = rightIdx;

    return nodeIndex;
}

float AnomalyModule::runIsolationForest(float features[]) const {
    if (!iforestTrained_ || trainBufferCount_ <= 1) {
        return 0.0f;
    }

    float avgPath = 0.0f;
    for (int t = 0; t < kTrees; ++t) {
        avgPath += pathLength(t, 0, features, 0);
    }
    avgPath /= static_cast<float>(kTrees);

    const float c = averagePathLength(trainBufferCount_);
    if (c <= 0.0f) {
        return 0.0f;
    }
    return powf(2.0f, -(avgPath / c));
}

float AnomalyModule::pathLength(int treeIdx, int nodeIdx, float features[], int depth) const {
    if (depth >= kDepth || nodeIdx < 0 || nodeIdx >= forest_[treeIdx].nodeCount) {
        return static_cast<float>(depth);
    }

    const IForestNode& node = forest_[treeIdx].nodes[nodeIdx];
    if (node.left == -1 && node.right == -1) {
        return static_cast<float>(depth);
    }

    if (features[node.splitFeature] < node.splitValue) {
        return pathLength(treeIdx, node.left, features, depth + 1);
    }
    return pathLength(treeIdx, node.right, features, depth + 1);
}

float AnomalyModule::averagePathLength(int n) const {
    if (n <= 1) {
        return 1.0f;
    }
    const float h = logf(static_cast<float>(n - 1)) + 0.5772156649f;
    return 2.0f * h - 2.0f * (static_cast<float>(n - 1) / static_cast<float>(n));
}

void AnomalyModule::addAlert(AlertLevel level, const char* sensor, const String& message) {
    for (int i = 0; i < min(alertCount_, kMaxAlerts); ++i) {
        const int idx = (alertIndex_ - 1 - i + kMaxAlerts) % kMaxAlerts;
        if (millis() - alerts_[idx].timestamp < 5000UL && strcmp(alerts_[idx].sensor, sensor) == 0) {
            return;
        }
    }

    const int idx = alertIndex_ % kMaxAlerts;
    alerts_[idx].timestamp = millis();
    alerts_[idx].level = level;
    strncpy(alerts_[idx].sensor, sensor, sizeof(alerts_[idx].sensor) - 1);
    alerts_[idx].sensor[sizeof(alerts_[idx].sensor) - 1] = '\0';
    strncpy(alerts_[idx].message, message.c_str(), sizeof(alerts_[idx].message) - 1);
    alerts_[idx].message[sizeof(alerts_[idx].message) - 1] = '\0';

    alertIndex_ = (alertIndex_ + 1) % kMaxAlerts;
    ++alertCount_;

    beep(level);
}

void AnomalyModule::updateAlertLevel() {
    AlertLevel maxLevel = AlertLevel::None;
    for (int i = 0; i < kFeatureCount; ++i) {
        if (sensors_[i].isDisconnected) {
            maxLevel = AlertLevel::Critical;
            break;
        }
        if (sensors_[i].isAnomalous && maxLevel < AlertLevel::Warning) {
            maxLevel = AlertLevel::Warning;
        }
        if (sensors_[i].isStuck && maxLevel < AlertLevel::Info) {
            maxLevel = AlertLevel::Info;
        }
    }
    if (iforestScore_ > 0.65f && maxLevel < AlertLevel::Warning) {
        maxLevel = AlertLevel::Warning;
    }
    currentLevel_ = maxLevel;
}

void AnomalyModule::beep(AlertLevel level) const {
    if (buzzerPin_ < 0 || level == AlertLevel::None) {
        return;
    }

    int count = 1;
    int onMs = 60;
    int offMs = 60;

    if (level == AlertLevel::Warning) {
        count = 2;
        onMs = 100;
    } else if (level == AlertLevel::Critical) {
        count = 4;
        onMs = 50;
        offMs = 50;
    }

    for (int i = 0; i < count; ++i) {
        digitalWrite(buzzerPin_, HIGH);
        delay(onMs);
        digitalWrite(buzzerPin_, LOW);
        delay(offMs);
    }
}

void AnomalyModule::writeStatus(JsonDocument& doc) const {
    doc["alertLevel"] = static_cast<int>(currentLevel_);
    doc["alertLevelName"] = alertLevelName(currentLevel_);
    doc["totalSamples"] = totalSamples_;
    doc["totalAnomalies"] = totalAnomalies_;
    doc["iforestTrained"] = iforestTrained_;
    doc["iforestScore"] = iforestScore_;

    JsonArray sensors = doc["sensors"].to<JsonArray>();
    for (int i = 0; i < kFeatureCount; ++i) {
        JsonObject sensor = sensors.add<JsonObject>();
        sensor["name"] = sensors_[i].name;
        sensor["label"] = sensors_[i].label;
        sensor["value"] = sensors_[i].currentValue;
        sensor["mean"] = sensors_[i].mean;
        sensor["stddev"] = sensors_[i].stddev;
        sensor["zScore"] = sensors_[i].lastZScore;
        sensor["isAnomalous"] = sensors_[i].isAnomalous;
        sensor["isStuck"] = sensors_[i].isStuck;
        sensor["isDisconnected"] = sensors_[i].isDisconnected;
        sensor["anomalyCount"] = sensors_[i].anomalyCount;
        sensor["min"] = sensors_[i].minEver;
        sensor["max"] = sensors_[i].maxEver;
    }
}

void AnomalyModule::writeAlerts(JsonDocument& doc) const {
    JsonArray alerts = doc.to<JsonArray>();
    const int count = min(alertCount_, kMaxAlerts);
    for (int i = 0; i < count; ++i) {
        const int idx = (alertIndex_ - 1 - i + kMaxAlerts) % kMaxAlerts;
        JsonObject alert = alerts.add<JsonObject>();
        alert["timestamp"] = alerts_[idx].timestamp;
        alert["level"] = static_cast<int>(alerts_[idx].level);
        alert["levelName"] = alertLevelName(alerts_[idx].level);
        alert["sensor"] = alerts_[idx].sensor;
        alert["message"] = alerts_[idx].message;
    }
}

bool AnomalyModule::writeSensorDetail(const String& sensorName, JsonDocument& doc) const {
    for (int i = 0; i < kFeatureCount; ++i) {
        if (sensorName != sensors_[i].name) {
            continue;
        }

        doc["name"] = sensors_[i].name;
        doc["label"] = sensors_[i].label;
        doc["value"] = sensors_[i].currentValue;
        doc["mean"] = sensors_[i].mean;
        doc["stddev"] = sensors_[i].stddev;
        doc["zScore"] = sensors_[i].lastZScore;

        JsonArray window = doc["window"].to<JsonArray>();
        for (int w = 0; w < sensors_[i].windowCount; ++w) {
            const int index = (sensors_[i].windowIndex - sensors_[i].windowCount + w + kWindowSize) % kWindowSize;
            window.add(sensors_[i].window[index]);
        }
        return true;
    }
    return false;
}

}  // namespace agri
