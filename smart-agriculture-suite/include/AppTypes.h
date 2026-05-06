#pragma once

#include <Arduino.h>

namespace agri {

enum class ControlSource : uint8_t {
    None = 0,
    SafetyLock,
    RuleEngine,
    Learning,
    Fusion,
    Manual
};

inline const char* controlSourceName(ControlSource source) {
    switch (source) {
        case ControlSource::SafetyLock: return "safety";
        case ControlSource::RuleEngine: return "rule";
        case ControlSource::Learning: return "learning";
        case ControlSource::Fusion: return "fusion";
        case ControlSource::Manual: return "manual";
        case ControlSource::None:
        default:
            return "idle";
    }
}

struct SensorSnapshot {
    float airTemp = 0.0f;
    float airHumi = 0.0f;
    float soilHumi = 0.0f;
    float liquidLevel = 0.0f;
    float lightValue = 0.0f;
    bool isDay = true;
    bool airValid = false;
    bool soilValid = false;
    bool liquidValid = false;
    bool lightValid = false;
    unsigned long updatedAtMs = 0;
};

struct IrrigationThresholdConfig {
    float liquidLevelThreshold = 30.0f;
    float lightDayThreshold = 200.0f;
    float dayAirTempThreshold = 20.0f;
    float dayAirHumiThreshold = 60.0f;
    float daySoilHumiThreshold = 50.0f;
    float nightAirTempThreshold = 15.0f;
    float nightAirHumiThreshold = 70.0f;
    float nightSoilHumiThreshold = 45.0f;
};

struct LearningConfig {
    float alpha = 0.1f;
    float gamma = 0.9f;
    float epsilon = 0.3f;
    float epsilonDecay = 0.999f;
    float epsilonMin = 0.05f;
    float targetSoil = 55.0f;
    float soilTolerance = 10.0f;
    unsigned long decisionIntervalMs = 300000UL;
    bool autoControlEnabled = false;
};

struct PlantDoctorConfig {
    bool enabled = false;
    bool autoDetect = true;
    int detectIntervalSec = 30;
    float confidenceThreshold = 0.70f;
    bool buzzerEnabled = true;
};

struct SystemConfig {
    IrrigationThresholdConfig irrigation;
    LearningConfig learning;
    PlantDoctorConfig plantDoctor;
    bool ruleEngineEnabled = true;
    bool fusionAutoEnabled = false;
};

struct ActuatorStatus {
    bool valveOn = false;
    bool pumpOn = false;
    bool autoMode = true;
    bool manualValve = false;
    bool manualPump = false;
    bool lowLiquidLock = false;
    bool timedRunActive = false;
    unsigned long activeUntilMs = 0;
    ControlSource source = ControlSource::None;
};

inline unsigned long secondsRemaining(unsigned long nowMs, unsigned long untilMs) {
    if (untilMs <= nowMs) {
        return 0;
    }
    return (untilMs - nowMs + 999UL) / 1000UL;
}

}  // namespace agri
