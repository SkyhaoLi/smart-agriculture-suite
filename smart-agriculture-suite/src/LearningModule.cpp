#include "LearningModule.h"

#include <math.h>

namespace agri {

namespace {
const int kActionDurations[LearningModule::Count] = {0, 30, 60, 120};
}  // namespace

const char* LearningModule::actionName(Action action) {
    static const char* kNames[] = {"off", "low", "medium", "high"};
    return kNames[static_cast<int>(action)];
}

void LearningModule::begin(const LearningConfig& config) {
    config_ = config;
    loadQTable();
}

void LearningModule::update(const SensorSnapshot& snapshot, bool sampleUpdated, unsigned long nowMs, ActuatorController& actuator) {
    if (!sampleUpdated) {
        return;
    }

    latestSnapshot_ = snapshot;
    const int currentState = discretizeState(snapshot);

    if (hasPendingReward_ && nowMs - lastDecisionMs_ >= config_.decisionIntervalMs) {
        const float reward = calculateReward(pendingSoilBefore_, snapshot.soilHumi, lastAction_);
        lastReward_ = reward;
        totalReward_ += reward;
        averageReward_ = totalEpisodes_ > 0 ? totalReward_ / static_cast<float>(totalEpisodes_) : 0.0f;
        updateQValue(lastState_, lastAction_, reward, currentState);

        LearningRecord& record = history_[historyIndex_ % kHistorySize];
        record.state = lastState_;
        record.action = lastAction_;
        record.reward = reward;
        record.soilBefore = pendingSoilBefore_;
        record.soilAfter = snapshot.soilHumi;
        historyIndex_ = (historyIndex_ + 1) % kHistorySize;
        historyCount_ = min(historyCount_ + 1, kHistorySize);

        hasPendingReward_ = false;
    }

    if (!config_.autoControlEnabled || !actuator.status().autoMode || actuator.isBusy(nowMs)) {
        return;
    }
    if (nowMs - lastDecisionMs_ < config_.decisionIntervalMs) {
        return;
    }

    const Action action = selectAction(currentState);
    if (action != Off) {
        actuator.startTimedRun(ControlSource::Learning, static_cast<unsigned long>(kActionDurations[action]), nowMs);
    }

    pendingSoilBefore_ = snapshot.soilHumi;
    lastState_ = currentState;
    lastAction_ = action;
    hasPendingReward_ = true;
    ++totalEpisodes_;
    lastDecisionMs_ = nowMs;
    // Adaptive epsilon decay: faster decay when average reward is improving
    float decayFactor = config_.epsilonDecay;
    if (totalEpisodes_ > 100 && averageReward_ > 2.0f) {
        decayFactor = config_.epsilonDecay * 0.998f;  // Decay faster when learning well
    }
    config_.epsilon = max(config_.epsilonMin, config_.epsilon * decayFactor);

    if (totalEpisodes_ % 50 == 0) {
        saveQTable();
    }
}

void LearningModule::reset() {
    initQTable();
    totalEpisodes_ = 0;
    totalReward_ = 0.0f;
    averageReward_ = 0.0f;
    lastReward_ = 0.0f;
    lastState_ = 0;
    lastAction_ = Off;
    hasPendingReward_ = false;
    historyCount_ = 0;
    historyIndex_ = 0;
    config_.epsilon = 0.3f;
    prefs_.begin("qlearn", false);
    prefs_.clear();
    prefs_.end();
}

void LearningModule::recordUserFeedback(bool positive) {
    const float feedbackReward = positive ? 5.0f : -5.0f;
    const int state = discretizeState(latestSnapshot_);
    updateQValue(state, lastAction_, feedbackReward, state);
    ++userOverrideCount_;
    userSatisfaction_ = constrain(userSatisfaction_ + (positive ? 2.0f : -3.0f), 0.0f, 100.0f);
}

void LearningModule::writeStatus(JsonDocument& doc) const {
    const int state = discretizeState(latestSnapshot_);

    doc["autoControlEnabled"] = config_.autoControlEnabled;
    doc["airTemp"] = latestSnapshot_.airTemp;
    doc["airHumi"] = latestSnapshot_.airHumi;
    doc["soilHumi"] = latestSnapshot_.soilHumi;
    doc["liquidLevel"] = latestSnapshot_.liquidLevel;
    doc["lightValue"] = latestSnapshot_.lightValue;
    doc["currentState"] = state;
    doc["lastAction"] = static_cast<int>(lastAction_);
    doc["lastActionName"] = actionName(lastAction_);
    doc["lastReward"] = lastReward_;
    doc["averageReward"] = averageReward_;
    doc["totalEpisodes"] = totalEpisodes_;
    doc["epsilon"] = config_.epsilon;
    doc["targetSoil"] = config_.targetSoil;
    doc["userOverrides"] = userOverrideCount_;
    doc["userSatisfaction"] = userSatisfaction_;

    int bestAction = 0;
    for (int a = 1; a < Count; ++a) {
        if (qTable_[state][a] > qTable_[state][bestAction]) {
            bestAction = a;
        }
    }
    doc["recommendedAction"] = actionName(static_cast<Action>(bestAction));

    JsonArray qValues = doc["qValues"].to<JsonArray>();
    for (int a = 0; a < Count; ++a) {
        JsonObject item = qValues.add<JsonObject>();
        item["action"] = actionName(static_cast<Action>(a));
        item["value"] = qTable_[state][a];
    }
}

void LearningModule::writeQTableSummary(JsonDocument& doc) const {
    int nonZero = 0;
    float maxQ = -9999.0f;
    float minQ = 9999.0f;

    for (int s = 0; s < kStateCount; ++s) {
        for (int a = 0; a < Count; ++a) {
            if (fabsf(qTable_[s][a]) > 0.001f) {
                ++nonZero;
                maxQ = max(maxQ, qTable_[s][a]);
                minQ = min(minQ, qTable_[s][a]);
            }
        }
    }

    doc["totalStates"] = kStateCount;
    doc["totalEntries"] = kStateCount * Count;
    doc["nonZeroEntries"] = nonZero;
    doc["coverage"] = (static_cast<float>(nonZero) / static_cast<float>(kStateCount * Count)) * 100.0f;
    doc["maxQ"] = nonZero > 0 ? maxQ : 0.0f;
    doc["minQ"] = nonZero > 0 ? minQ : 0.0f;

    JsonArray history = doc["history"].to<JsonArray>();
    for (int i = 0; i < historyCount_; ++i) {
        const int idx = (historyIndex_ - 1 - i + kHistorySize) % kHistorySize;
        JsonObject item = history.add<JsonObject>();
        item["state"] = history_[idx].state;
        item["action"] = actionName(history_[idx].action);
        item["reward"] = history_[idx].reward;
        item["soilBefore"] = history_[idx].soilBefore;
        item["soilAfter"] = history_[idx].soilAfter;
    }
}

void LearningModule::initQTable() {
    memset(qTable_, 0, sizeof(qTable_));
}

void LearningModule::loadQTable() {
    prefs_.begin("qlearn", true);
    totalEpisodes_ = prefs_.getInt("episodes", 0);
    averageReward_ = prefs_.getFloat("avgReward", 0.0f);
    totalReward_ = averageReward_ * static_cast<float>(totalEpisodes_);
    config_.epsilon = prefs_.getFloat("epsilon", config_.epsilon);

    if (totalEpisodes_ > 0) {
        for (int s = 0; s < kStateCount; ++s) {
            char key[12];
            snprintf(key, sizeof(key), "q%d", s);
            prefs_.getBytes(key, qTable_[s], sizeof(float) * Count);
        }
    } else {
        initQTable();
    }
    prefs_.end();
}

void LearningModule::saveQTable() {
    prefs_.begin("qlearn", false);
    for (int s = 0; s < kStateCount; ++s) {
        char key[12];
        snprintf(key, sizeof(key), "q%d", s);
        prefs_.putBytes(key, qTable_[s], sizeof(float) * Count);
    }
    prefs_.putInt("episodes", totalEpisodes_);
    prefs_.putFloat("avgReward", averageReward_);
    prefs_.putFloat("epsilon", config_.epsilon);
    prefs_.end();
}

int LearningModule::discretizeState(const SensorSnapshot& snapshot) const {
    const int temp = discretizeTemp(snapshot.airTemp);
    const int humi = discretizeHumi(snapshot.airHumi);
    const int soil = discretizeSoil(snapshot.soilHumi);
    const int light = discretizeLight(snapshot.lightValue);
    const int period = getTimePeriod(millis());

    return temp * (kHumiLevels * kSoilLevels * kLightLevels * kTimeLevels) +
           humi * (kSoilLevels * kLightLevels * kTimeLevels) +
           soil * (kLightLevels * kTimeLevels) +
           light * kTimeLevels +
           period;
}

int LearningModule::discretizeTemp(float temp) const {
    if (temp < 10.0f) return 0;
    if (temp < 18.0f) return 1;
    if (temp < 25.0f) return 2;
    if (temp < 33.0f) return 3;
    return 4;
}

int LearningModule::discretizeHumi(float humi) const {
    if (humi < 30.0f) return 0;
    if (humi < 50.0f) return 1;
    if (humi < 70.0f) return 2;
    return 3;
}

int LearningModule::discretizeSoil(float soil) const {
    if (soil < 20.0f) return 0;
    if (soil < 35.0f) return 1;
    if (soil < 50.0f) return 2;
    if (soil < 65.0f) return 3;
    return 4;
}

int LearningModule::discretizeLight(float light) const {
    if (light < 100.0f) return 0;
    if (light < 500.0f) return 1;
    return 2;
}

int LearningModule::getTimePeriod(unsigned long nowMs) const {
    const unsigned long hours = (nowMs / 3600000UL) % 24UL;
    if (hours >= 6UL && hours < 12UL) {
        return 0;
    }
    if (hours >= 12UL && hours < 18UL) {
        return 1;
    }
    return 2;
}

LearningModule::Action LearningModule::selectAction(int state) const {
    if (random(1000) < config_.epsilon * 1000.0f) {
        return static_cast<Action>(random(Count));
    }

    int bestAction = 0;
    float bestValue = qTable_[state][0];
    for (int a = 1; a < Count; ++a) {
        if (qTable_[state][a] > bestValue) {
            bestValue = qTable_[state][a];
            bestAction = a;
        }
    }
    return static_cast<Action>(bestAction);
}

float LearningModule::calculateReward(float soilBefore, float soilAfter, Action action) const {
    float reward = 0.0f;
    const float diffAfter = fabsf(soilAfter - config_.targetSoil);
    const float diffBefore = fabsf(soilBefore - config_.targetSoil);

    // Core reward: proximity to target soil moisture
    if (diffAfter <= config_.soilTolerance) {
        reward += 10.0f;
    } else {
        reward -= diffAfter * 0.3f;
    }

    // Improvement reward: getting closer to target
    if (diffAfter < diffBefore) {
        reward += 3.0f;
    }

    // Efficiency reward: prefer not watering when soil is already adequate
    if (action == Off && soilBefore > config_.targetSoil - config_.soilTolerance) {
        reward += 2.0f;
    }

    // Energy efficiency: prefer lower intensity when possible
    if (action == Low && diffBefore < config_.soilTolerance * 2.0f) {
        reward += 1.5f;  // Reward conservative watering
    }
    if (action == High && diffBefore < config_.soilTolerance) {
        reward -= 1.0f;  // Penalize overwatering with high intensity
    }

    // Safety penalties
    if (soilAfter > 80.0f) {
        reward -= 5.0f;
    }
    if (soilAfter < 20.0f) {
        reward -= 8.0f;
    }

    // Overshooting penalty: went past target
    if (soilBefore < config_.targetSoil && soilAfter > config_.targetSoil + config_.soilTolerance) {
        reward -= 2.0f;
    }

    return reward;
}

void LearningModule::updateQValue(int state, Action action, float reward, int nextState) {
    float maxNextQ = qTable_[nextState][0];
    for (int a = 1; a < Count; ++a) {
        maxNextQ = max(maxNextQ, qTable_[nextState][a]);
    }

    const float currentQ = qTable_[state][action];
    const float newQ = currentQ + config_.alpha * (reward + config_.gamma * maxNextQ - currentQ);
    qTable_[state][action] = newQ;
}

}  // namespace agri
