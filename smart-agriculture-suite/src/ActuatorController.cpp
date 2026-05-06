#include "ActuatorController.h"

namespace agri {

void ActuatorController::begin(const PinConfig& pins) {
    pins_ = pins;
    status_ = ActuatorStatus{};

    if (pins_.valvePin >= 0) {
        pinMode(pins_.valvePin, OUTPUT);
        digitalWrite(pins_.valvePin, LOW);
    }
    if (pins_.pumpPin >= 0) {
        pinMode(pins_.pumpPin, OUTPUT);
        digitalWrite(pins_.pumpPin, LOW);
    }
}

void ActuatorController::setAutoMode(bool enabled) {
    status_.autoMode = enabled;
    if (!enabled) {
        clearTimedRun();
        applyOutputs(status_.manualValve, status_.manualPump, ControlSource::Manual, 0);
    }
}

void ActuatorController::setManualCombined(bool enabled) {
    status_.manualValve = enabled;
    status_.manualPump = enabled;
}

void ActuatorController::setManualValve(bool enabled) {
    status_.manualValve = enabled;
}

void ActuatorController::setManualPump(bool enabled) {
    status_.manualPump = enabled;
}

bool ActuatorController::startTimedRun(ControlSource source, unsigned long durationSec, unsigned long nowMs) {
    if (!status_.autoMode || status_.lowLiquidLock || durationSec == 0) {
        return false;
    }
    if (status_.timedRunActive && status_.activeUntilMs > nowMs) {
        return false;
    }

    status_.timedRunActive = true;
    status_.activeUntilMs = nowMs + durationSec * 1000UL;
    applyOutputs(true, true, source, status_.activeUntilMs);
    return true;
}

void ActuatorController::stopTimedRun() {
    clearTimedRun();
}

void ActuatorController::update(bool lowLiquidLock, bool baseAutoRequest, unsigned long nowMs) {
    status_.lowLiquidLock = lowLiquidLock;

    if (status_.timedRunActive && nowMs >= status_.activeUntilMs) {
        clearTimedRun();
    }

    if (status_.lowLiquidLock) {
        clearTimedRun();
        applyOutputs(false, false, ControlSource::SafetyLock, 0);
        return;
    }

    if (!status_.autoMode) {
        applyOutputs(status_.manualValve, status_.manualPump, ControlSource::Manual, 0);
        return;
    }

    if (status_.timedRunActive) {
        applyOutputs(true, true, status_.source, status_.activeUntilMs);
        return;
    }

    applyOutputs(baseAutoRequest, baseAutoRequest,
                 baseAutoRequest ? ControlSource::RuleEngine : ControlSource::None,
                 0);
}

bool ActuatorController::isBusy(unsigned long nowMs) const {
    return status_.timedRunActive && status_.activeUntilMs > nowMs;
}

void ActuatorController::writePin(int pin, bool high) {
    if (pin >= 0) {
        digitalWrite(pin, high ? HIGH : LOW);
    }
}

void ActuatorController::applyOutputs(bool valveOn, bool pumpOn, ControlSource source, unsigned long untilMs) {
    writePin(pins_.valvePin, valveOn);
    writePin(pins_.pumpPin, pumpOn);

    status_.valveOn = valveOn;
    status_.pumpOn = pumpOn;
    status_.source = source;
    if (status_.timedRunActive) {
        status_.activeUntilMs = untilMs;
    } else {
        status_.activeUntilMs = 0;
    }
}

void ActuatorController::clearTimedRun() {
    status_.timedRunActive = false;
    status_.activeUntilMs = 0;
    if (status_.source != ControlSource::Manual) {
        status_.source = ControlSource::None;
    }
}

}  // namespace agri
