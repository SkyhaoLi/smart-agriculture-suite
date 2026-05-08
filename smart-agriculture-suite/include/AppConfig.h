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

// ESP32-S3 使用DVP接口摄像头 (硬件限制, 不支持USB摄像头)
// Atlas 200I DK A2 版使用USB摄像头 (通过OpenCV/VideoCapture读取)
struct CameraPinConfig {
    bool enabled = false;
    int pwdn = -1;
    int reset = -1;
    int xclk = -1;
    int sccbSda = -1;
    int sccbScl = -1;
    int d0 = -1;
    int d1 = -1;
    int d2 = -1;
    int d3 = -1;
    int d4 = -1;
    int d5 = -1;
    int d6 = -1;
    int d7 = -1;
    int vsync = -1;
    int href = -1;
    int pclk = -1;
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
    CameraPinConfig camera;
};

inline HardwareProfile defaultProfile() {
    return static_cast<HardwareProfile>(AGRI_HW_PROFILE);
}

inline const char* hardwareProfileName(HardwareProfile profile) {
    switch (profile) {
        case HardwareProfile::HybridDevKit: return "hybrid-devkit";
        case HardwareProfile::CameraEyeStandalone: return "camera-eye-standalone";
        case HardwareProfile::ControllerKit:
        default:
            return "controller-kit";
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
	            pins.camera.enabled = true;
	            pins.camera.pwdn = -1;
	            pins.camera.reset = -1;
	            pins.camera.xclk = 10;
	            pins.camera.sccbSda = 40;
	            pins.camera.sccbScl = 39;
	            pins.camera.d7 = 48;
	            pins.camera.d6 = 11;
	            pins.camera.d5 = 12;
	            pins.camera.d4 = 14;
	            pins.camera.d3 = 16;
	            pins.camera.d2 = 18;
	            pins.camera.d1 = 17;
	            pins.camera.d0 = 15;
	            pins.camera.vsync = 38;
	            pins.camera.href = 47;
	            pins.camera.pclk = 13;
	            break;
	        case HardwareProfile::HybridDevKit:
	            pins.profile = HardwareProfile::HybridDevKit;
	            pins.camera.enabled = true;
	            // Hybrid: irrigation + camera coexist
	            // Camera pin remapping to avoid conflicts with irrigation pins:
	            // D6=11 conflicts with pumpPin=11, remap D6 to GPIO 21
	            // D1=17 conflicts with airRx=17, remap D1 to GPIO 7
	            pins.camera.pwdn = -1;
	            pins.camera.reset = -1;
	            pins.camera.xclk = 10;
	            pins.camera.sccbSda = 40;
	            pins.camera.sccbScl = 39;
	            pins.camera.d7 = 48;
	            pins.camera.d6 = 21;   // remapped from 11 (conflicts with pumpPin)
	            pins.camera.d5 = 12;
	            pins.camera.d4 = 14;
	            pins.camera.d3 = 16;
	            pins.camera.d2 = 18;
	            pins.camera.d1 = 7;    // remapped from 17 (conflicts with airRx)
	            pins.camera.d0 = 15;
	            pins.camera.vsync = 38;
	            pins.camera.href = 47;
	            pins.camera.pclk = 13;
	            break;
	        case HardwareProfile::ControllerKit:
	        default:
	            pins.profile = HardwareProfile::ControllerKit;
	            pins.camera.enabled = false;
	            pins.camera.pwdn = -1;
	            pins.camera.reset = -1;
	            pins.camera.xclk = 10;
	            pins.camera.sccbSda = 40;
	            pins.camera.sccbScl = 39;
	            pins.camera.d7 = 48;
	            pins.camera.d6 = 11;
	            pins.camera.d5 = 12;
	            pins.camera.d4 = 14;
	            pins.camera.d3 = 16;
	            pins.camera.d2 = 18;
	            pins.camera.d1 = 17;
	            pins.camera.d0 = 15;
	            pins.camera.vsync = 38;
	            pins.camera.href = 47;
	            pins.camera.pclk = 13;
	            break;
	    }

	    return pins;
	}

inline SystemConfig defaultSystemConfig() {
    SystemConfig config;
    if (defaultProfile() == HardwareProfile::CameraEyeStandalone) {
        config.ruleEngineEnabled = false;
        config.fusionAutoEnabled = false;
        config.learning.autoControlEnabled = false;
        config.plantDoctor.enabled = true;
    } else if (defaultProfile() == HardwareProfile::HybridDevKit) {
        config.plantDoctor.enabled = true;
    }
    return config;
}

constexpr unsigned long kSerialBaud = 115200UL;
constexpr unsigned long kAirSensorBaud = 9600UL;
constexpr unsigned long kSensorSampleIntervalMs = 2000UL;
constexpr unsigned long kDisplayPageIntervalMs = 4000UL;
constexpr unsigned long kDisplayRefreshIntervalMs = 700UL;
constexpr unsigned long kGrowthDayIntervalMs = 24UL * 60UL * 60UL * 1000UL;
constexpr unsigned long kFusionIntervalMs = 10000UL;
constexpr unsigned long kAnomalyIsolationIntervalMs = 60000UL;
constexpr const char* kWifiSsid = "";
constexpr const char* kWifiPassword = "";

}  // namespace agri
