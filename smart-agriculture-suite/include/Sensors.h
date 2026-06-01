#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <BH1750.h>

#include "AppConfig.h"
#include "AppTypes.h"

namespace agri {

class SensorHub {
public:
    void begin(const PinConfig& pins);
    bool update(unsigned long nowMs);
    const SensorSnapshot& snapshot() const { return snapshot_; }
    bool lightReady() const { return lightReady_; }

private:
    int readAnalogAverage(int pin, int samples = 5);
    void parseAirFrame(const String& frame);
    void checkFaults(unsigned long nowMs);

    PinConfig pins_{};
    SensorSnapshot snapshot_{};
    BH1750 lightMeter_;
    bool lightReady_ = false;
    unsigned long lastSampleMs_ = 0;
    unsigned long lastAirOkMs_ = 0;
    unsigned long lastSoilOkMs_ = 0;
    unsigned long lastLiquidOkMs_ = 0;
    unsigned long lastLightOkMs_ = 0;
    String lastAirFrame_;
};

}  // namespace agri
