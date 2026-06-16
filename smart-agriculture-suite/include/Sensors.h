#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <BH1750.h>
#include <Adafruit_SHT4x.h>

#include "AppConfig.h"
#include "AppTypes.h"

namespace agri {

class SensorHub {
public:
    void begin(const PinConfig& pins);
    bool update(unsigned long nowMs);
    const SensorSnapshot& snapshot() const { return snapshot_; }
    bool lightReady() const { return lightReady_; }
    bool shtReady() const { return shtReady_; }

private:
    int readAnalogAverage(int pin, int samples = 5);
    void checkFaults(unsigned long nowMs);

    PinConfig pins_{};
    SensorSnapshot snapshot_{};
    BH1750 lightMeter_;
    Adafruit_SHT4x sht40_;
    bool lightReady_ = false;
    bool shtReady_ = false;
    unsigned long lastSampleMs_ = 0;
    unsigned long lastAirOkMs_ = 0;
    unsigned long lastSoilOkMs_ = 0;
    unsigned long lastLightOkMs_ = 0;
};

}  // namespace agri
