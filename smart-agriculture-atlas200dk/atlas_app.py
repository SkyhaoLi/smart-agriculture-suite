#!/usr/bin/env python3
"""
智润智慧农业套件 - Atlas 200I DK A2 + ESP32串口版
通过串口从ESP32获取传感器数据，运行AI模块，发送灌溉指令
"""

import os
import sys
import time
import signal
import logging
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from esp32_bridge import ESP32Bridge
from config.hardware_config import SystemConfig
from config.app_types import SensorSnapshot, ActuatorStatus, ControlSource
from src.ai import IrrigationModule, AnomalyModule, GrowthModule, LearningModule, FusionModule, WorldModelModule
from src.ai.plant_doctor_module import PlantDoctorModule
from src.actuators import ActuatorController
from src.web import WebDashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("agri_atlas")


class AtlasESP32App:
    """Atlas + ESP32 串口协同应用"""

    def __init__(self, web_port=8080, esp32_port="/dev/ttyUSB0"):
        self._web_port = web_port
        self._esp32_port = esp32_port
        self._running = False
        self._config = SystemConfig()

        self._bridge = ESP32Bridge(port=esp32_port)
        self._snapshot = SensorSnapshot()
        self._snapshot_lock = threading.Lock()

        self._actuator = None
        self._irrigation = None
        self._anomaly = None
        self._growth = None
        self._learning = None
        self._fusion = None
        self._world_model = None
        self._plant_doctor = None
        self._web = None

    def setup(self):
        print()
        print("=" * 40)
        print("  Smart Agriculture Suite")
        print("  Atlas + ESP32 Serial Bridge")
        print("=" * 40)

        # ESP32串口桥接
        if not self._bridge.begin():
            logger.error("ESP32串口连接失败, 退出")
            sys.exit(1)

        # 等待第一条传感器数据
        logger.info("等待ESP32传感器数据...")
        for i in range(30):
            data = self._bridge.get_sensor_data()
            if data.get("type") == "sensor":
                logger.info(f"ESP32传感器数据已就绪: temp={data.get('air_temp',0):.1f}C")
                break
            time.sleep(1)
        else:
            logger.warning("未收到ESP32数据, 继续运行...")

        # 执行器 (通过ESP32串口控制)
        self._actuator = AtlasActuatorProxy(self._bridge)

        # AI模块
        self._irrigation = IrrigationModule()
        self._anomaly = AnomalyModule(buzzer=self._actuator)
        self._growth = GrowthModule()
        self._learning = LearningModule()
        self._learning.begin()
        self._fusion = FusionModule()
        self._fusion.begin(self._config.fusion_enabled)
        self._world_model = WorldModelModule()
        self._world_model.begin(self._config.fusion_enabled)
        self._plant_doctor = PlantDoctorModule(buzzer=self._actuator)
        self._plant_doctor.begin(camera_id="http://admin:@192.168.7.102/videostream.cgi", enabled=True)

        # Web仪表盘
        self._web = WebDashboard(
            self, self._actuator, self._irrigation,
            self._anomaly, self._growth, self._learning,
            self._fusion, self._plant_doctor, self._world_model,
        )

        logger.info("所有模块初始化完成")

    def loop(self):
        self._running = True
        logger.info("主循环启动")

        while self._running:
            now = time.time()

            # 1. 从ESP32获取传感器数据
            esp_data = self._bridge.get_sensor_data()
            if esp_data.get("type") == "sensor":
                with self._snapshot_lock:
                    self._snapshot.air_temp = esp_data.get("air_temp", 0)
                    self._snapshot.air_humi = esp_data.get("air_humi", 0)
                    self._snapshot.soil_humi = esp_data.get("soil_humi", 0)
                    self._snapshot.light_intensity = esp_data.get("light", 0)
                    self._snapshot.is_day = esp_data.get("is_day", True)
                    self._snapshot.timestamp = now
                    self._sensor_faults = {
                        "air": esp_data.get("fault_air", False),
                        "soil": esp_data.get("fault_soil", False),
                        "light": esp_data.get("fault_light", False),
                    }

            sample_updated = True  # ESP32每2秒更新一次

            # 2. 灌溉规则引擎
            if sample_updated:
                self._irrigation.update(self._snapshot)

            # 3. 执行器仲裁
            irrigation_result = self._irrigation.result
            base_auto = self._irrigation.enabled and irrigation_result.should_water
            self._actuator.update(base_auto, now)

            # 4. 异常检测
            self._anomaly.update(self._snapshot, sample_updated, now)

            # 5. 生长跟踪
            self._growth.update(self._snapshot, sample_updated, now)

            # 6. Q-Learning
            pred_risk = self._world_model.get_prediction_risk() if self._world_model else 0.0
            self._learning.update(self._snapshot, sample_updated, now, self._actuator, pred_risk)

            # 7. 传感器融合
            self._fusion._prediction_boost = pred_risk * 15.0  # 最高+15分
            self._fusion.update(self._snapshot, sample_updated, now, self._actuator)

            # 8. 环境状态预测
            self._world_model.update(self._snapshot, sample_updated, now, self._actuator)

            # 9. 植物病害检测
            self._plant_doctor.update(now)

            time.sleep(0.5)

    @property
    def snapshot(self) -> SensorSnapshot:
        with self._snapshot_lock:
            return self._snapshot

    @property
    def bridge(self) -> ESP32Bridge:
        return self._bridge

    def run(self):
        self.setup()

        web_thread = threading.Thread(
            target=self._web.run,
            kwargs={'host': '0.0.0.0', 'port': self._web_port, 'debug': False},
            daemon=True,
        )
        web_thread.start()
        logger.info(f"Web仪表盘: http://0.0.0.0:{self._web_port}")

        def signal_handler(sig, frame):
            logger.info("收到停止信号, 关闭中...")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.loop()
        self._cleanup()

    def _cleanup(self):
        logger.info("清理资源...")
        self._bridge.close()
        if self._learning:
            self._learning._save_q_table()
        if self._fusion:
            self._fusion.save_network()
        if self._world_model:
            self._world_model.save()
        logger.info("清理完成")


class AtlasActuatorProxy:
    """通过ESP32串口控制执行器的代理"""

    def __init__(self, bridge: ESP32Bridge):
        self._bridge = bridge
        self._valve_on = False
        self._pump_on = False
        self._auto_mode = True
        self._timed_until = 0
        self._status = ActuatorStatus()

    @property
    def status(self) -> ActuatorStatus:
        self._status.valve_on = self._valve_on
        self._status.pump_on = self._pump_on
        return self._status

    def is_busy(self, now) -> bool:
        return self._timed_until > 0 and now < self._timed_until

    def update(self, base_auto, now):
        if self._timed_until > 0 and now >= self._timed_until:
            self._timed_until = 0
            self._valve_on = False
            self._bridge.send_command(action=0)

    def start_timed_run(self, source, duration_sec, now):
        self._timed_until = now + duration_sec
        self._valve_on = True
        self._bridge.send_irrigation(duration_sec)

    def set_valve(self, on):
        self._valve_on = on
        if on:
            self._bridge.send_command(action=2, duration=60)
        else:
            self._bridge.send_command(action=0)

    def set_pump(self, on):
        self._pump_on = on
        self._bridge.send_command(pump=on)

    def set_auto_mode(self, enabled):
        self._auto_mode = enabled
        self._bridge.send_command(auto_mode=enabled)

    def beep(self, count=1, on_ms=100, off_ms=100):
        pass  # 蜂鸣器在ESP32端，暂不控制

    def close(self):
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Atlas + ESP32 智慧农业")
    parser.add_argument('--port', type=int, default=8080, help='Web端口')
    parser.add_argument('--esp32', type=str, default='/dev/ttyUSB0', help='ESP32串口')
    args = parser.parse_args()

    app = AtlasESP32App(web_port=args.port, esp32_port=args.esp32)
    app.run()


if __name__ == '__main__':
    main()
