"""ActuatorController — priority arbitration matching firmware logic."""

import threading


class ControlSource:
    NONE = 0
    SAFETY_LOCK = 1
    RULE_ENGINE = 2
    LEARNING = 3
    FUSION = 4
    MANUAL = 5


CONTROL_SOURCE_NAMES = {
    ControlSource.NONE: "idle",
    ControlSource.SAFETY_LOCK: "safety",
    ControlSource.RULE_ENGINE: "rule",
    ControlSource.LEARNING: "learning",
    ControlSource.FUSION: "fusion",
    ControlSource.MANUAL: "manual",
}


class ActuatorStatus:
    """Mirrors firmware ActuatorStatus."""

    def __init__(self):
        self.valveOn = False
        self.pumpOn = False
        self.autoMode = True
        self.manualValve = False
        self.manualPump = False
        self.lowLiquidLock = False
        self.timedRunActive = False
        self.activeUntilMs = 0
        self.source = ControlSource.NONE

    def seconds_remaining(self, now_ms: int) -> int:
        if self.activeUntilMs <= now_ms:
            return 0
        return (self.activeUntilMs - now_ms + 999) // 1000

    def to_dict(self, now_ms: int) -> dict:
        return {
            "valveOn": self.valveOn,
            "pumpOn": self.pumpOn,
            "autoMode": self.autoMode,
            "manualValve": self.manualValve,
            "manualPump": self.manualPump,
            "lowLiquidLock": self.lowLiquidLock,
            "timedRunActive": self.timedRunActive,
            "source": CONTROL_SOURCE_NAMES.get(self.source, "idle"),
            "secondsRemaining": self.seconds_remaining(now_ms),
        }


class ActuatorController:
    """Direct port of agri::ActuatorController with priority arbitration."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status = ActuatorStatus()

    @property
    def status(self) -> ActuatorStatus:
        return self._status

    def set_auto_mode(self, enabled: bool):
        with self._lock:
            self._status.autoMode = enabled
            if not enabled:
                self._clear_timed_run()
                self._apply_outputs(
                    self._status.manualValve,
                    self._status.manualPump,
                    ControlSource.MANUAL,
                    0,
                )

    def set_manual_combined(self, enabled: bool):
        with self._lock:
            self._status.manualValve = enabled
            self._status.manualPump = enabled

    def set_manual_valve(self, enabled: bool):
        with self._lock:
            self._status.manualValve = enabled

    def set_manual_pump(self, enabled: bool):
        with self._lock:
            self._status.manualPump = enabled

    def start_timed_run(self, source: int, duration_sec: int, now_ms: int) -> bool:
        """Matches ActuatorController::startTimedRun()."""
        with self._lock:
            if not self._status.autoMode or self._status.lowLiquidLock or duration_sec == 0:
                return False
            if self._status.timedRunActive and self._status.activeUntilMs > now_ms:
                return False

            self._status.timedRunActive = True
            self._status.activeUntilMs = now_ms + duration_sec * 1000
            self._apply_outputs(True, True, source, self._status.activeUntilMs)
            return True

    def stop_timed_run(self):
        with self._lock:
            self._clear_timed_run()

    def update(self, low_liquid_lock: bool, base_auto_request: bool, now_ms: int):
        """
        Priority arbitration — matches ActuatorController::update().
        Called twice per loop iteration (same as main.cpp).
        """
        with self._lock:
            self._status.lowLiquidLock = low_liquid_lock

            # Check timed run expiry
            if self._status.timedRunActive and now_ms >= self._status.activeUntilMs:
                self._clear_timed_run()

            # Priority 1: Safety lock
            if self._status.lowLiquidLock:
                self._clear_timed_run()
                self._apply_outputs(False, False, ControlSource.SAFETY_LOCK, 0)
                return

            # Priority 2: Manual mode
            if not self._status.autoMode:
                any_manual = self._status.manualValve or self._status.manualPump
                self._apply_outputs(
                    self._status.manualValve,
                    self._status.manualPump,
                    ControlSource.MANUAL if any_manual else ControlSource.NONE,
                    0,
                )
                return

            # Priority 3: Timed run (Learning/Fusion)
            if self._status.timedRunActive:
                self._apply_outputs(
                    True, True, self._status.source, self._status.activeUntilMs
                )
                return

            # Priority 4: Rule engine
            self._apply_outputs(
                base_auto_request,
                base_auto_request,
                ControlSource.RULE_ENGINE if base_auto_request else ControlSource.NONE,
                0,
            )

    def is_busy(self, now_ms: int) -> bool:
        return self._status.timedRunActive and self._status.activeUntilMs > now_ms

    def _apply_outputs(self, valve_on: bool, pump_on: bool, source: int, until_ms: int):
        self._status.valveOn = valve_on
        self._status.pumpOn = pump_on
        self._status.source = source
        if self._status.timedRunActive:
            self._status.activeUntilMs = until_ms
        else:
            self._status.activeUntilMs = 0

    def _clear_timed_run(self):
        self._status.timedRunActive = False
        self._status.activeUntilMs = 0
        if self._status.source != ControlSource.MANUAL:
            self._status.source = ControlSource.NONE
