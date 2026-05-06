#include "Sensors.h"

#include <Wire.h>

namespace agri {

void SensorHub::begin(const PinConfig& pins) {
    pins_ = pins;
    lastSampleMs_ = 0;
    snapshot_ = SensorSnapshot{};

    Serial1.begin(kAirSensorBaud, SERIAL_8N1, pins_.airRx, pins_.airTx);

    if (pins_.soilPin >= 0) {
        pinMode(pins_.soilPin, INPUT);
    }
    if (pins_.liquidPin >= 0) {
        pinMode(pins_.liquidPin, INPUT);
    }

    lightReady_ = lightMeter_.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);
}

bool SensorHub::update(unsigned long nowMs) {
    if (nowMs - lastSampleMs_ < kSensorSampleIntervalMs) {
        return false;
    }
    lastSampleMs_ = nowMs;

    if (Serial1.available() > 0) {
        String frame = Serial1.readStringUntil('\n');
        frame.trim();
        if (!frame.isEmpty()) {
            parseAirFrame(frame);
            lastAirFrame_ = frame;
        }
    }

    if (pins_.soilPin >= 0) {
        const int raw = readAnalogAverage(pins_.soilPin);
        snapshot_.soilHumi = constrain(static_cast<float>(map(raw, 3500, 1500, 0, 100)), 0.0f, 100.0f);
        snapshot_.soilValid = true;
    }

    if (pins_.liquidPin >= 0) {
        const int raw = readAnalogAverage(pins_.liquidPin);
        snapshot_.liquidLevel = constrain(static_cast<float>(map(raw, 500, 3500, 0, 100)), 0.0f, 100.0f);
        snapshot_.liquidValid = true;
    }

    if (lightReady_) {
        const float lux = lightMeter_.readLightLevel();
        if (lux >= 0.0f) {
            snapshot_.lightValue = lux;
            snapshot_.lightValid = true;
        }
    }

    snapshot_.isDay = snapshot_.lightValue >= 200.0f;
    snapshot_.updatedAtMs = nowMs;
    return true;
}

int SensorHub::readAnalogAverage(int pin) const {
    if (pin < 0) {
        return 0;
    }

    int sum = 0;
    for (int i = 0; i < 5; ++i) {
        sum += analogRead(pin);
        delay(2);
    }
    return sum / 5;
}

void SensorHub::parseAirFrame(const String& frame) {
    const int tempIndex = frame.indexOf("Temp:");
    const int humiIndex = frame.indexOf("Humi:");
    if (tempIndex < 0 || humiIndex < 0) {
        return;
    }

    const int commaIndex = frame.indexOf(',', tempIndex);
    String tempPart = (commaIndex > tempIndex) ? frame.substring(tempIndex + 5, commaIndex)
                                               : frame.substring(tempIndex + 5);
    String humiPart = frame.substring(humiIndex + 5);
    tempPart.trim();
    humiPart.trim();

    snapshot_.airTemp = tempPart.toFloat();
    snapshot_.airHumi = humiPart.toFloat();
    snapshot_.airValid = true;
}

}  // namespace agri
