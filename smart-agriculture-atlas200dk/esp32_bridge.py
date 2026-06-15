"""
ESP32串口桥接模块
从ESP32读取传感器JSON数据，发送灌溉指令
"""

import serial
import json
import time
import threading
import logging

logger = logging.getLogger(__name__)


class ESP32Bridge:
    """ESP32串口通信桥接"""

    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self._port = port
        self._baud = baud
        self._serial = None
        self._lock = threading.Lock()
        self._latest_data = {}
        self._running = False
        self._reader_thread = None
        self._callbacks = []

    def begin(self) -> bool:
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=1.0
            )
            self._running = True
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            logger.info(f"ESP32串口已连接: {self._port}")
            return True
        except serial.SerialException as e:
            logger.error(f"ESP32串口连接失败: {e}")
            return False

    def _reader_loop(self):
        while self._running:
            try:
                if self._serial and self._serial.is_open:
                    line = self._serial.readline()
                    if line:
                        text = line.decode("utf-8", errors="ignore").strip()
                        if text.startswith("{"):
                            try:
                                data = json.loads(text)
                                if data.get("type") == "sensor":
                                    with self._lock:
                                        self._latest_data = data
                                    for cb in self._callbacks:
                                        try:
                                            cb(data)
                                        except Exception as e:
                                            logger.debug(f"回调异常: {e}")
                            except json.JSONDecodeError:
                                pass
                else:
                    time.sleep(0.5)
            except Exception as e:
                logger.debug(f"串口读取异常: {e}")
                time.sleep(0.1)

    def get_sensor_data(self) -> dict:
        with self._lock:
            return dict(self._latest_data)

    def send_command(self, action: int = 0, duration: int = 0,
                     pump: bool = None, auto_mode: bool = None) -> bool:
        cmd = {"type": "cmd"}
        if action > 0 and duration > 0:
            cmd["action"] = action
            cmd["duration"] = duration
        elif action == 0:
            cmd["action"] = 0
        if pump is not None:
            cmd["pump"] = pump
        if auto_mode is not None:
            cmd["auto"] = auto_mode

        return self._send_json(cmd)

    def send_irrigation(self, duration_sec: int) -> bool:
        if duration_sec <= 0:
            return self.send_command(action=0)
        if duration_sec <= 30:
            return self.send_command(action=1, duration=duration_sec)
        elif duration_sec <= 60:
            return self.send_command(action=2, duration=duration_sec)
        else:
            return self.send_command(action=3, duration=duration_sec)

    def _send_json(self, data: dict) -> bool:
        with self._lock:
            try:
                if self._serial and self._serial.is_open:
                    msg = json.dumps(data) + "\n"
                    self._serial.write(msg.encode())
                    self._serial.flush()
                    return True
            except Exception as e:
                logger.error(f"串口发送失败: {e}")
        return False

    def on_sensor_update(self, callback):
        self._callbacks.append(callback)

    def close(self):
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
        if self._serial and self._serial.is_open:
            self._serial.close()
