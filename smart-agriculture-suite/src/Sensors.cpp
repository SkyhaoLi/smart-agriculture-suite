#include "Sensors.h"

namespace agri {

void SensorHub::begin(const PinConfig& pins) {
    pins_ = pins;

    Serial1.begin(kAirSensorBaud, SERIAL_8N1, pins_.airRx, pins_.airTx);

    if (pins_.soilPin >= 0) pinMode(pins_.soilPin, INPUT);
    if (pins_.liquidPin >= 0) pinMode(pins_.liquidPin, INPUT);

    lightReady_ = lightMeter_.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, 0x23, &Wire);
}

bool SensorHub::update(unsigned long nowMs) {
    if (nowMs - lastSampleMs_ < kSensorSampleIntervalMs) return false;
    lastSampleMs_ = nowMs;

    // 空气传感器
    if (Serial1.available()) {
        lastAirFrame_ = Serial1.readStringUntil('\n');
        lastAirFrame_.trim();
        if (lastAirFrame_.length() > 0) {
            parseAirFrame(lastAirFrame_);
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

    // 液位
    if (pins_.liquidPin >= 0) {
        int raw = readAnalogAverage(pins_.liquidPin);
        float val = map(raw, 500, 3500, 0, 100);
        val = constrain(val, 0.0f, 100.0f);
        snapshot_.liquidLevel = val;
        lastLiquidOkMs_ = nowMs;
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

void SensorHub::parseAirFrame(const String& frame) {
    // 格式: "Temp:X,Humi:Y"
    int tempIdx = frame.indexOf("Temp:");
    int humiIdx = frame.indexOf("Humi:");
    if (tempIdx < 0 || humiIdx < 0) return;

    float temp = frame.substring(tempIdx + 5, frame.indexOf(',', tempIdx)).toFloat();
    float humi = frame.substring(humiIdx + 5).toFloat();

    if (temp > -40.0f && temp < 80.0f && humi >= 0.0f && humi <= 100.0f) {
        snapshot_.airTemp = temp;
        snapshot_.airHumi = humi;
        lastAirOkMs_ = millis();
    }
}

void SensorHub::checkFaults(unsigned long nowMs) {
    snapshot_.fault.airTimeout = (nowMs - lastAirOkMs_ > kSensorFaultTimeoutMs);
    snapshot_.fault.soilTimeout = (pins_.soilPin < 0) || (nowMs - lastSoilOkMs_ > kSensorFaultTimeoutMs);
    snapshot_.fault.liquidTimeout = (pins_.liquidPin < 0) || (nowMs - lastLiquidOkMs_ > kSensorFaultTimeoutMs);
    snapshot_.fault.lightTimeout = !lightReady_ || (nowMs - lastLightOkMs_ > kSensorFaultTimeoutMs);

    // 范围检查 (仅在传感器有数据时检查)
    if (!snapshot_.fault.airTimeout) {
        snapshot_.fault.airRange = (snapshot_.airTemp < -20.0f || snapshot_.airTemp > 60.0f);
    }
    if (!snapshot_.fault.soilTimeout) {
        snapshot_.fault.soilRange = (snapshot_.soilHumi < 0.0f || snapshot_.soilHumi > 100.0f);
    }
    if (!snapshot_.fault.liquidTimeout) {
        snapshot_.fault.liquidRange = (snapshot_.liquidLevel < 0.0f || snapshot_.liquidLevel > 100.0f);
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
