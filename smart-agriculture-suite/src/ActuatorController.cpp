#include "ActuatorController.h"

namespace agri {

void ActuatorController::begin(const PinConfig& pins) {
    pins_ = pins;

    // 初始化BTS7960引脚
    initBts7960Pin(pins_.valveRpwM, pins_.valveLpwM, pins_.valveREn, pins_.valveLEn);
    initBts7960Pin(pins_.pumpRpwM, pins_.pumpLpwM, pins_.pumpREn, pins_.pumpLEn);

    initialized_ = true;
}

void ActuatorController::initBts7960Pin(int rpwM, int lpwM, int rEn, int lEn) {
    if (rpwM >= 0) {
        pinMode(rpwM, OUTPUT);
        digitalWrite(rpwM, LOW);
    }
    if (lpwM >= 0) {
        pinMode(lpwM, OUTPUT);
        digitalWrite(lpwM, LOW);
    }
    if (rEn >= 0) {
        pinMode(rEn, OUTPUT);
        digitalWrite(rEn, HIGH);  // 使能正转
    }
    if (lEn >= 0) {
        pinMode(lEn, OUTPUT);
        digitalWrite(lEn, HIGH);  // 使能反转
    }
}

void ActuatorController::driveBts7960(int rpwM, int lpwM, int rEn, int lEn, bool on, int speed) {
    if (on) {
        // 正转: RPWM输出PWM, LPWM为低
        if (rpwM >= 0) analogWrite(rpwM, speed);
        if (lpwM >= 0) digitalWrite(lpwM, LOW);
        if (rEn >= 0) digitalWrite(rEn, HIGH);
        if (lEn >= 0) digitalWrite(lEn, HIGH);
    } else {
        // 停止: 两个PWM都为低
        if (rpwM >= 0) digitalWrite(rpwM, LOW);
        if (lpwM >= 0) digitalWrite(lpwM, LOW);
    }
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

void ActuatorController::update(bool baseAutoRequest, unsigned long nowMs) {
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

void ActuatorController::applyOutputs(bool valveOn, bool pumpOn, ControlSource source, unsigned long untilMs) {
    // 驱动BTS7960
    driveBts7960(pins_.valveRpwM, pins_.valveLpwM, pins_.valveREn, pins_.valveLEn, valveOn);
    driveBts7960(pins_.pumpRpwM, pins_.pumpLpwM, pins_.pumpREn, pins_.pumpLEn, pumpOn);

    status_.valveOn = valveOn;
    status_.pumpOn = pumpOn;
    status_.source = source;
    if (untilMs > 0) status_.activeUntilMs = untilMs;
}

}  // namespace agri
