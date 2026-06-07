#include "ActuatorController.h"

namespace agri {

void ActuatorController::begin(const PinConfig& pins) {
    pins_ = pins;
    if (pins_.valvePin >= 0) {
        pinMode(pins_.valvePin, OUTPUT);
        digitalWrite(pins_.valvePin, LOW);
    }
    if (pins_.pumpPin >= 0) {
        pinMode(pins_.pumpPin, OUTPUT);
        digitalWrite(pins_.pumpPin, LOW);
    }
    initialized_ = true;
}

void ActuatorController::setManualValve(bool on) {
    status_.manualValve = on;
    if (!status_.autoMode) {
        applyOutputs(on, status_.manualPump, ControlSource::Manual, 0);
    }
}

void ActuatorController::setManualPump(bool on) {
    status_.manualPump = on;
    if (!status_.autoMode) {
        applyOutputs(status_.manualValve, on, ControlSource::Manual, 0);
    }
}

void ActuatorController::setManualCombined(bool on) {
    status_.manualValve = on;
    status_.manualPump = on;
    if (!status_.autoMode) {
        applyOutputs(on, on, ControlSource::Manual, 0);
    }
}

bool ActuatorController::startTimedRun(ControlSource source, int durationSec, unsigned long nowMs) {
    if (status_.lowLiquidLock) return false;
    if (durationSec <= 0) return false;

    unsigned long untilMs = nowMs + (unsigned long)durationSec * 1000UL;
    applyOutputs(true, true, source, untilMs);
    status_.timedRunActive = true;
    status_.activeUntilMs = untilMs;
    return true;
}

void ActuatorController::stopTimedRun() {
    status_.timedRunActive = false;
    status_.activeUntilMs = 0;
    applyOutputs(false, false, ControlSource::None, 0);
}

void ActuatorController::update(bool lowLiquidLock, bool baseAutoRequest, unsigned long nowMs) {
    status_.lowLiquidLock = lowLiquidLock;

    // 安全锁定 - 最高优先级
    if (lowLiquidLock) {
        applyOutputs(false, false, ControlSource::SafetyLock, 0);
        status_.timedRunActive = false;
        return;
    }

    // 定时运行中
    if (status_.timedRunActive) {
        if (nowMs >= status_.activeUntilMs) {
            status_.timedRunActive = false;
            status_.activeUntilMs = 0;
            applyOutputs(false, false, ControlSource::None, 0);
        }
        return;  // 定时运行期间不响应其他请求
    }

    // 手动模式
    if (!status_.autoMode) {
        applyOutputs(status_.manualValve, status_.manualPump, ControlSource::Manual, 0);
        return;
    }

    // 自动模式 - 规则引擎
    if (baseAutoRequest) {
        applyOutputs(true, true, ControlSource::Fallback, 0);
    } else {
        applyOutputs(false, false, ControlSource::None, 0);
    }
}

bool ActuatorController::isBusy(unsigned long nowMs) const {
    return status_.timedRunActive && nowMs < status_.activeUntilMs;
}

void ActuatorController::writePin(int pin, bool high) {
    if (pin >= 0) digitalWrite(pin, high ? HIGH : LOW);
}

void ActuatorController::applyOutputs(bool valveOn, bool pumpOn, ControlSource source, unsigned long untilMs) {
    writePin(pins_.valvePin, valveOn);
    writePin(pins_.pumpPin, pumpOn);
    status_.valveOn = valveOn;
    status_.pumpOn = pumpOn;
    status_.source = source;
    if (untilMs > 0) status_.activeUntilMs = untilMs;
}

}  // namespace agri
