#include "IrrigationModule.h"

namespace agri {

void IrrigationModule::update(const SensorSnapshot& snapshot) {
    isDay_ = snapshot.lightValue >= config_.lightDayThreshold;
    liquidWarn_ = snapshot.liquidValid && snapshot.liquidLevel < config_.liquidLevelThreshold;

    if (!enabled_) {
        shouldWater_ = false;
        reason_ = "rule engine disabled";
        return;
    }

    if (liquidWarn_) {
        shouldWater_ = false;
        reason_ = "liquid tank too low";
        return;
    }

    const bool tempPass = snapshot.airTemp >= (isDay_ ? config_.dayAirTempThreshold : config_.nightAirTempThreshold);
    const bool humiPass = snapshot.airHumi <= (isDay_ ? config_.dayAirHumiThreshold : config_.nightAirHumiThreshold);
    const bool soilPass = snapshot.soilHumi <= (isDay_ ? config_.daySoilHumiThreshold : config_.nightSoilHumiThreshold);

    shouldWater_ = tempPass && humiPass && soilPass;
    if (shouldWater_) {
        reason_ = isDay_ ? "day thresholds matched" : "night thresholds matched";
    } else {
        reason_ = "thresholds not met";
    }
}

bool IrrigationModule::updateConfigFromJson(const JsonVariantConst& value) {
    bool updated = false;

    if (value["enabled"].is<bool>()) {
        enabled_ = value["enabled"].as<bool>();
        updated = true;
    }

    JsonVariantConst day = value["day"];
    if (!day.isNull()) {
        if (day["airTemp"].is<float>() || day["airTemp"].is<double>() || day["airTemp"].is<int>()) {
            config_.dayAirTempThreshold = day["airTemp"].as<float>();
            updated = true;
        }
        if (day["airHumi"].is<float>() || day["airHumi"].is<double>() || day["airHumi"].is<int>()) {
            config_.dayAirHumiThreshold = day["airHumi"].as<float>();
            updated = true;
        }
        if (day["soilHumi"].is<float>() || day["soilHumi"].is<double>() || day["soilHumi"].is<int>()) {
            config_.daySoilHumiThreshold = day["soilHumi"].as<float>();
            updated = true;
        }
    }

    JsonVariantConst night = value["night"];
    if (!night.isNull()) {
        if (night["airTemp"].is<float>() || night["airTemp"].is<double>() || night["airTemp"].is<int>()) {
            config_.nightAirTempThreshold = night["airTemp"].as<float>();
            updated = true;
        }
        if (night["airHumi"].is<float>() || night["airHumi"].is<double>() || night["airHumi"].is<int>()) {
            config_.nightAirHumiThreshold = night["airHumi"].as<float>();
            updated = true;
        }
        if (night["soilHumi"].is<float>() || night["soilHumi"].is<double>() || night["soilHumi"].is<int>()) {
            config_.nightSoilHumiThreshold = night["soilHumi"].as<float>();
            updated = true;
        }
    }

    if (value["liquidThreshold"].is<float>() || value["liquidThreshold"].is<double>() || value["liquidThreshold"].is<int>()) {
        config_.liquidLevelThreshold = value["liquidThreshold"].as<float>();
        updated = true;
    }

    if (value["lightThreshold"].is<float>() || value["lightThreshold"].is<double>() || value["lightThreshold"].is<int>()) {
        config_.lightDayThreshold = value["lightThreshold"].as<float>();
        updated = true;
    }

    return updated;
}

void IrrigationModule::writeStatus(JsonDocument& doc) const {
    doc["enabled"] = enabled_;
    doc["shouldWater"] = shouldWater_;
    doc["liquidWarn"] = liquidWarn_;
    doc["isDay"] = isDay_;
    doc["reason"] = reason_;

    JsonObject day = doc["config"]["day"].to<JsonObject>();
    day["airTemp"] = config_.dayAirTempThreshold;
    day["airHumi"] = config_.dayAirHumiThreshold;
    day["soilHumi"] = config_.daySoilHumiThreshold;

    JsonObject night = doc["config"]["night"].to<JsonObject>();
    night["airTemp"] = config_.nightAirTempThreshold;
    night["airHumi"] = config_.nightAirHumiThreshold;
    night["soilHumi"] = config_.nightSoilHumiThreshold;

    doc["config"]["liquidThreshold"] = config_.liquidLevelThreshold;
    doc["config"]["lightThreshold"] = config_.lightDayThreshold;
}

}  // namespace agri
