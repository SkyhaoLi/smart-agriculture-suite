"""
智润智慧农业套件 - Atlas 200I DK A2 版
执行器控制器 - 使用Linux gpiod控制GPIO

对应原ESP32项目的 ActuatorController.h/ActuatorController.cpp
优先级: SafetyLock > Manual > TimedRun(Learning/Fusion) > RuleEngine
"""

import gpiod
import time
import logging
from typing import Optional

from config.app_types import ControlSource, ActuatorStatus

logger = logging.getLogger(__name__)


class ActuatorController:
    """执行器仲裁控制器 - 管理阀门/水泵/蜂鸣器的优先级控制"""

    def __init__(self, pin_config):
        self._pins = pin_config
        self._status = ActuatorStatus()

        self._chip: Optional[gpiod.Chip] = None
        self._valve_line: Optional[gpiod.Line] = None
        self._pump_line: Optional[gpiod.Line] = None
        self._buzzer_line: Optional[gpiod.Line] = None

        self._timed_run_active = False
        self._active_until = 0.0
        self._timed_source = ControlSource.None_
        self._auto_mode = True
        self._manual_valve = False
        self._manual_pump = False

    def begin(self) -> bool:
        """初始化GPIO引脚"""
        try:
            self._chip = gpiod.Chip(self._pins.gpio_chip)

            # 电磁阀
            self._valve_line = self._chip.get_line(self._pins.valve_gpio)
            self._valve_line.request(consumer="agri_valve", type=gpiod.LINE_REQ_DIR_OUT)
            self._valve_line.set_value(0)

            # 水泵
            self._pump_line = self._chip.get_line(self._pins.pump_gpio)
            self._pump_line.request(consumer="agri_pump", type=gpiod.LINE_REQ_DIR_OUT)
            self._pump_line.set_value(0)

            # 蜂鸣器
            self._buzzer_line = self._chip.get_line(self._pins.buzzer_gpio)
            self._buzzer_line.request(consumer="agri_buzzer", type=gpiod.LINE_REQ_DIR_OUT)
            self._buzzer_line.set_value(0)

            logger.info("执行器GPIO初始化完成")
            return True
        except Exception as e:
            logger.error(f"执行器GPIO初始化失败: {e}")
            return False

    def set_auto_mode(self, enabled: bool):
        self._auto_mode = enabled
        if not enabled:
            self._clear_timed_run()
            self._apply_outputs(self._manual_valve, self._manual_pump,
                                ControlSource.Manual, 0)

    def set_manual_combined(self, enabled: bool):
        self._manual_valve = enabled
        self._manual_pump = enabled

    def set_manual_valve(self, enabled: bool):
        self._manual_valve = enabled

    def set_manual_pump(self, enabled: bool):
        self._manual_pump = enabled

    def start_timed_run(self, source: ControlSource, duration_sec: float,
                        now: float = 0.0) -> bool:
        """启动定时灌溉运行"""
        if now == 0.0:
            now = time.time()

        if not self._auto_mode or self._status.safety_lock or duration_sec <= 0:
            return False
        if self._timed_run_active and self._active_until > now:
            return False

        self._timed_run_active = True
        self._active_until = now + duration_sec
        self._timed_source = source
        self._apply_outputs(True, True, source, self._active_until)
        return True

    def stop_timed_run(self):
        self._clear_timed_run()

    def update(self, base_auto_request: bool, now: float = 0.0):
        """主循环更新 - 仲裁执行器状态"""
        if now == 0.0:
            now = time.time()

        self._status.safety_lock = False

        # 检查定时运行是否到期
        if self._timed_run_active and now >= self._active_until:
            self._clear_timed_run()

        # 安全锁定: 最高优先级
        if self._status.safety_lock:
            self._clear_timed_run()
            self._apply_outputs(False, False, ControlSource.SafetyLock, 0)
            return

        # 手动模式
        if not self._auto_mode:
            self._apply_outputs(self._manual_valve, self._manual_pump,
                                ControlSource.Manual, 0)
            return

        # 定时运行中
        if self._timed_run_active:
            self._apply_outputs(True, True, self._timed_source, self._active_until)
            return

        # 规则引擎
        self._apply_outputs(base_auto_request, base_auto_request,
                            ControlSource.RuleEngine if base_auto_request else ControlSource.None_,
                            0)

    def is_busy(self, now: float = 0.0) -> bool:
        if now == 0.0:
            now = time.time()
        return self._timed_run_active and self._active_until > now

    @property
    def status(self) -> ActuatorStatus:
        return self._status

    # ------------------------------------------------------------------
    # 蜂鸣器
    # ------------------------------------------------------------------
    def beep(self, count: int = 1, on_ms: int = 60, off_ms: int = 60):
        """蜂鸣器鸣叫"""
        if not self._buzzer_line:
            return
        for _ in range(count):
            self._buzzer_line.set_value(1)
            time.sleep(on_ms / 1000.0)
            self._buzzer_line.set_value(0)
            if count > 1:
                time.sleep(off_ms / 1000.0)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _write_pin(self, line, high: bool):
        if line:
            line.set_value(1 if high else 0)

    def _apply_outputs(self, valve_on: bool, pump_on: bool,
                        source: ControlSource, until: float):
        self._write_pin(self._valve_line, valve_on)
        self._write_pin(self._pump_line, pump_on)

        self._status.valve_on = valve_on
        self._status.pump_on = pump_on
        self._status.active_source = source

        if self._timed_run_active:
            self._status.timed_run_remaining_ms = max(0, int((until - time.time()) * 1000))
        else:
            self._status.timed_run_remaining_ms = 0

    def _clear_timed_run(self):
        self._timed_run_active = False
        self._active_until = 0.0
        if self._timed_source != ControlSource.Manual:
            self._timed_source = ControlSource.None_

    def close(self):
        if self._valve_line:
            self._valve_line.set_value(0)
            self._valve_line.release()
        if self._pump_line:
            self._pump_line.set_value(0)
            self._pump_line.release()
        if self._buzzer_line:
            self._buzzer_line.set_value(0)
            self._buzzer_line.release()
        if self._chip:
            self._chip.close()
