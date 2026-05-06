#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>

#include "AppTypes.h"

namespace agri {

class IrrigationModule {
public:
    IrrigationModule() = default;

    void setConfig(const IrrigationThresholdConfig& config) { config_ = config; }
    const IrrigationThresholdConfig& config() const { return config_; }
    const IrrigationThresholdConfig& thresholdConfig() const { return config_; }

    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }

    void update(const SensorSnapshot& snapshot);
    bool shouldWater() const { return shouldWater_; }
    bool liquidWarn() const { return liquidWarn_; }
    bool isDay() const { return isDay_; }
    const String& reason() const { return reason_; }
    bool updateConfigFromJson(const JsonVariantConst& value);
    void writeStatus(JsonDocument& doc) const;

private:
    IrrigationThresholdConfig config_{};
    bool enabled_ = true;
    bool shouldWater_ = false;
    bool liquidWarn_ = false;
    bool isDay_ = true;
    String reason_;
};

}  // namespace agri
