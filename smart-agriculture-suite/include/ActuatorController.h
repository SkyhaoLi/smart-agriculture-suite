#pragma once

#include <Arduino.h>

#include "AppConfig.h"
#include "AppTypes.h"

namespace agri {

class ActuatorController {
public:
    void begin(const PinConfig& pins);
    void setAutoMode(bool enabled);
    void setManualCombined(bool enabled);
    void setManualValve(bool enabled);
    void setManualPump(bool enabled);
    bool startTimedRun(ControlSource source, unsigned long durationSec, unsigned long nowMs);
    void stopTimedRun();
    void update(bool lowLiquidLock, bool baseAutoRequest, unsigned long nowMs);
    bool isBusy(unsigned long nowMs) const;

    const ActuatorStatus& status() const { return status_; }

private:
    void writePin(int pin, bool high);
    void applyOutputs(bool valveOn, bool pumpOn, ControlSource source, unsigned long untilMs);
    void clearTimedRun();

    PinConfig pins_{};
    ActuatorStatus status_{};
};

}  // namespace agri
