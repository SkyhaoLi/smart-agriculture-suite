#pragma once

#include "AppTypes.h"

namespace agri {

#ifndef AGRI_HW_PROFILE
#define AGRI_HW_PROFILE 1
#endif

enum class HardwareProfile : uint8_t {
    ControllerKit = 1,
    HybridDevKit = 2,
    CameraEyeStandalone = 3
};

struct PinConfig {
    HardwareProfile profile = HardwareProfile::ControllerKit;
    int i2cSda = 8;
    int i2cScl = 9;
    int airRx = 17;
    int airTx = 18;
    int soilPin = 1;
    int liquidPin = 2;
    int valvePin = 10;
    int pumpPin = 11;
    int buzzerPin = 5;
};

inline HardwareProfile defaultProfile() {
    return static_cast<HardwareProfile>(AGRI_HW_PROFILE);
}

inline const char* hardwareProfileName(HardwareProfile p) {
    switch (p) {
        case HardwareProfile::HybridDevKit: return "hybrid-devkit";
        case HardwareProfile::CameraEyeStandalone: return "camera-eye-standalone";
        case HardwareProfile::ControllerKit:
        default: return "controller-kit";
    }
}

inline PinConfig defaultPins() {
    PinConfig pins;
    switch (defaultProfile()) {
        case HardwareProfile::CameraEyeStandalone:
            pins.profile = HardwareProfile::CameraEyeStandalone;
            pins.i2cSda = 3;
            pins.i2cScl = 4;
            pins.valvePin = -1;
            pins.pumpPin = -1;
            break;
        case HardwareProfile::HybridDevKit:
            pins.profile = HardwareProfile::HybridDevKit;
            break;
        case HardwareProfile::ControllerKit:
        default:
            pins.profile = HardwareProfile::ControllerKit;
            break;
    }
    return pins;
}

inline SystemConfig defaultSystemConfig() {
    SystemConfig config;
    return config;
}

constexpr unsigned long kSerialBaud = 115200UL;
constexpr unsigned long kAirSensorBaud = 9600UL;
constexpr unsigned long kSensorSampleIntervalMs = 2000UL;
constexpr unsigned long kDisplayPageIntervalMs = 4000UL;
constexpr unsigned long kDisplayRefreshIntervalMs = 700UL;
constexpr unsigned long kSensorFaultTimeoutMs = 10000UL;  // 10s无数据判定故障
constexpr const char* kDefaultAtlasHost = "192.168.1.100";
constexpr uint16_t kDefaultAtlasPort = 8080;

}  // namespace agri
