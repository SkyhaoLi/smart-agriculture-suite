#include "Sensors.h"

namespace agri {

void SensorHub::begin(const PinConfig& pins) {
    pins_ = pins;

    if (pins_.soilPin >= 0) pinMode(pins_.soilPin, INPUT);

    // 初始化BH1750光照传感器
    lightReady_ = lightMeter_.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, 0x23, &Wire);

    // 初始化SHT40温湿度传感器
    shtReady_ = sht40_.begin(&Wire);
    if (shtReady_) {
        sht40_.setPrecision(SHT4X_HIGH_PRECISION);
        sht40_.setHeater(SHT4X_NO_HEATER);
    }
}

bool SensorHub::update(unsigned long nowMs) {
    if (nowMs - lastSampleMs_ < kSensorSampleIntervalMs) return false;
    lastSampleMs_ = nowMs;

    // SHT40温湿度传感器
    if (shtReady_) {
        sensors_event_t humidity, temp;
        if (sht40_.getEvent(&humidity, &temp)) {
            if (temp.temperature > -40.0f && temp.temperature < 80.0f &&
                humidity.relative_humidity >= 0.0f && humidity.relative_humidity <= 100.0f) {
                snapshot_.airTemp = temp.temperature;
                snapshot_.airHumi = humidity.relative_humidity;
                lastAirOkMs_ = nowMs;
            }
        }
    }

    // 土壤湿度
    if (pins_.soilPin >= 0) {
        int raw = readAnalogAverage(pins_.soilPin);
        float val = map(raw, 3500, 1500, 0, 100);
        val = constrain(val, 0.0f, 100.0f);
        if (val > 0.5f) {  // 基本有效性检查
            snapshot_.soilHumi = val;
            lastSoilOkMs_ = nowMs;
        }
    }

    // 光照
    if (lightReady_) {
        float lux = lightMeter_.readLightLevel();
        if (lux >= 0) {
            snapshot_.lightValue = lux;
            snapshot_.isDay = lux >= 200.0f;
            lastLightOkMs_ = nowMs;
        }
    }

    snapshot_.updatedAtMs = nowMs;
    checkFaults(nowMs);
    return true;
}

void SensorHub::checkFaults(unsigned long nowMs) {
    snapshot_.fault.airTimeout = !shtReady_ || (nowMs - lastAirOkMs_ > kSensorFaultTimeoutMs);
    snapshot_.fault.soilTimeout = (pins_.soilPin < 0) || (nowMs - lastSoilOkMs_ > kSensorFaultTimeoutMs);
    snapshot_.fault.lightTimeout = !lightReady_ || (nowMs - lastLightOkMs_ > kSensorFaultTimeoutMs);

    // 范围检查 (仅在传感器有数据时检查)
    if (!snapshot_.fault.airTimeout) {
        snapshot_.fault.airRange = (snapshot_.airTemp < -20.0f || snapshot_.airTemp > 60.0f);
    }
    if (!snapshot_.fault.soilTimeout) {
        snapshot_.fault.soilRange = (snapshot_.soilHumi < 0.0f || snapshot_.soilHumi > 100.0f);
    }
    if (!snapshot_.fault.lightTimeout) {
        snapshot_.fault.lightRange = (snapshot_.lightValue < 0.0f || snapshot_.lightValue > 200000.0f);
    }
}

int SensorHub::readAnalogAverage(int pin, int samples) {
    long sum = 0;
    for (int i = 0; i < samples; i++) {
        sum += analogRead(pin);
        delayMicroseconds(500);
    }
    return sum / samples;
}

}  // namespace agri
