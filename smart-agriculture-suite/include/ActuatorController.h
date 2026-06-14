#pragma once

#include <Arduino.h>

#include "AppConfig.h"
#include "AppTypes.h"

namespace agri {

class ActuatorController {
public:
    void begin(const PinConfig& pins);
    void setAutoMode(bool enabled) { status_.autoMode = enabled; }
    void setManualValve(bool on);
    void setManualPump(bool on);
    void setManualCombined(bool on);
    bool startTimedRun(ControlSource source, int durationSec, unsigned long nowMs);
    void stopTimedRun();
    void update(bool baseAutoRequest, unsigned long nowMs);
    bool isBusy(unsigned long nowMs) const;
    ActuatorStatus& status() { return status_; }
    const ActuatorStatus& status() const { return status_; }

private:
    void initBts7960Pin(int rpwM, int lpwM, int rEn, int lEn);
    void driveBts7960(int rpwM, int lpwM, int rEn, int lEn, bool on, int speed = 255);
    void applyOutputs(bool valveOn, bool pumpOn, ControlSource source, unsigned long untilMs);

    PinConfig pins_{};
    ActuatorStatus status_{};
    bool initialized_ = false;
};

}  // namespace agri
