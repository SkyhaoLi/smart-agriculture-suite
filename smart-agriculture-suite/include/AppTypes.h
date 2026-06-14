#pragma once

#include <Arduino.h>

namespace agri {

// ============================================================================
// 控制源优先级 (数值越小优先级越高)
// ============================================================================
enum class ControlSource : uint8_t {
    SafetyLock = 0,
    WorldModel = 1,
    Manual = 2,
    Fallback = 3,
    None = 99
};

inline const char* controlSourceName(ControlSource s) {
    switch (s) {
        case ControlSource::SafetyLock: return "safety";
        case ControlSource::WorldModel: return "world_model";
        case ControlSource::Manual:     return "manual";
        case ControlSource::Fallback:   return "fallback";
        case ControlSource::None:       return "idle";
    }
    return "unknown";
}

// ============================================================================
// 灌溉动作
// ============================================================================
enum class IrrigationAction : uint8_t {
    Off = 0,
    Low = 1,       // 30s
    Moderate = 2,  // 60s
    Heavy = 3      // 120s
};

// ============================================================================
// 生长阶段
// ============================================================================
enum class GrowthStage : uint8_t {
    Seed = 0,
    Germination,
    Seedling,
    Vegetative,
    Flowering,
    Fruiting,
    Maturity,
    Count
};

inline const char* stageName(GrowthStage s) {
    static const char* names[] = {
        "Seed", "Germination", "Seedling", "Vegetative",
        "Flowering", "Fruiting", "Maturity"
    };
    return names[static_cast<uint8_t>(s)];
}

inline const char* stageNameCn(GrowthStage s) {
    static const char* names[] = {
        "播种", "发芽", "幼苗", "营养期", "开花期", "结果期", "成熟期"
    };
    return names[static_cast<uint8_t>(s)];
}

// ============================================================================
// 作物类型
// ============================================================================
enum class CropType : uint8_t {
    Tomato = 0,
    Lettuce,
    Pepper,
    Cucumber,
    Strawberry,
    Count
};

inline const char* cropName(CropType c) {
    static const char* names[] = {"Tomato", "Lettuce", "Pepper", "Cucumber", "Strawberry"};
    return names[static_cast<uint8_t>(c)];
}

inline const char* cropNameCn(CropType c) {
    static const char* names[] = {"番茄", "生菜", "辣椒", "黄瓜", "草莓"};
    return names[static_cast<uint8_t>(c)];
}

// ============================================================================
// 传感器故障信息
// ============================================================================
struct SensorFault {
    bool airTimeout = true;     // 温湿度传感器(SHT40)超时
    bool soilTimeout = true;    // 土壤传感器超时
    bool lightTimeout = true;   // 光照传感器(BH1750)超时
    bool airRange = false;      // 温湿度传感器数据异常
    bool soilRange = false;     // 土壤传感器数据异常
    bool lightRange = false;    // 光照传感器数据异常

    bool anyFault() const {
        return airTimeout || soilTimeout || lightTimeout ||
               airRange || soilRange || lightRange;
    }
    bool airFault() const { return airTimeout || airRange; }
    bool soilFault() const { return soilTimeout || soilRange; }
    bool lightFault() const { return lightTimeout || lightRange; }
};

// ============================================================================
// 传感器快照
// ============================================================================
struct SensorSnapshot {
    float airTemp = 0.0f;
    float airHumi = 0.0f;
    float soilHumi = 0.0f;
    float lightValue = 0.0f;
    bool isDay = true;
    unsigned long updatedAtMs = 0;
    SensorFault fault;
};

// ============================================================================
// 世界模型响应 (来自Atlas 200I)
// ============================================================================
struct WorldModelResponse {
    // 病害诊断
    int diseaseId = 0;           // 0=健康, 1-4=病害
    float diseaseConfidence = 0.0f;
    char diseaseName[32] = "";
    char treatment[128] = "";

    // 灌溉决策
    IrrigationAction action = IrrigationAction::Off;
    int actionDurationSec = 0;
    float actionConfidence = 0.0f;
    char actionReason[64] = "";

    // 世界模型状态
    float predictedSoilHumi = 0.0f;
    float predictedAirTemp = 0.0f;
    float predictedAirHumi = 0.0f;
    float environmentRisk = 0.0f;  // 0-1

    bool valid = false;
    unsigned long timestampMs = 0;
};

// ============================================================================
// 执行器状态
// ============================================================================
struct ActuatorStatus {
    bool valveOn = false;
    bool pumpOn = false;
    bool autoMode = true;
    bool manualValve = false;
    bool manualPump = false;
    bool timedRunActive = false;
    unsigned long activeUntilMs = 0;
    ControlSource source = ControlSource::None;
};

inline unsigned long secondsRemaining(unsigned long nowMs, unsigned long untilMs) {
    if (nowMs >= untilMs) return 0;
    return (untilMs - nowMs + 999) / 1000;
}

// ============================================================================
// 灌溉阈值配置
// ============================================================================
struct IrrigationThresholdConfig {
    float lightDayThreshold = 200.0f;
    float dayAirTempThreshold = 20.0f;
    float dayAirHumiThreshold = 60.0f;
    float daySoilHumiThreshold = 50.0f;
    float nightAirTempThreshold = 15.0f;
    float nightAirHumiThreshold = 70.0f;
    float nightSoilHumiThreshold = 45.0f;
};

// ============================================================================
// 系统配置
// ============================================================================
struct SystemConfig {
    IrrigationThresholdConfig irrigation;
    bool worldModelEnabled = true;
    unsigned long worldModelIntervalMs = 30000;  // 30s
};

}  // namespace agri
