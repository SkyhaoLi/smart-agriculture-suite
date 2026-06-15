#!/usr/bin/env python3
"""
智润智慧农业套件 - Atlas 200I DK A2 版
主入口 - 初始化所有模块, 启动主循环和Web服务

对应原ESP32项目的 main.cpp
Atlas 200I DK A2: Ascend 310B, 4核A55@1GHz, 4GB LPDDR4X, Ubuntu 22.04
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading

# 将项目根目录加入path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.hardware_config import PinConfig, ADCConfig, SystemConfig, Timing
from src.sensors import SensorHub
from src.actuators import ActuatorController
from src.ai import IrrigationModule, AnomalyModule, GrowthModule, LearningModule, FusionModule
from src.ai.plant_doctor_module import PlantDoctorModule
from src.display import DisplayModule
from src.web import WebDashboard

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("agri_main")


class SmartAgricultureApp:
    """智慧农业主应用"""

    def __init__(self, config: SystemConfig = None, pin_config: PinConfig = None,
                 adc_config: ADCConfig = None, web_port: int = 8080,
                 hw_profile: int = 1, camera_id=0, demo=False):
        self._config = config or SystemConfig()
        self._pins = pin_config or PinConfig()
        self._adc_cfg = adc_config or ADCConfig()
        self._web_port = web_port
        self._hw_profile = hw_profile
        self._camera_id = camera_id
        self._demo = demo

        self._running = False
        self._sensor_hub = None
        self._actuator = None
        self._irrigation = None
        self._anomaly = None
        self._growth = None
        self._learning = None
        self._fusion = None
        self._plant_doctor = None
        self._display = None
        self._web = None

    def setup(self):
        """初始化所有模块 (对应ESP32的 setup())"""
        print()
        print("=" * 40)
        print("  Smart Agriculture Suite")
        print("  Atlas 200I DK A2 Edition")
        print("=" * 40)
        print(f"Hardware profile: {['', 'ControllerKit', 'HybridDevKit', 'CameraEye'][self._hw_profile]}")

        # 传感器
        self._sensor_hub = SensorHub(self._pins, self._adc_cfg, demo=self._demo)
        self._sensor_hub.begin()

        # 执行器
        self._actuator = ActuatorController(self._pins)
        self._actuator.begin()

        # 灌溉规则引擎
        self._irrigation = IrrigationModule()

        # 异常检测
        self._anomaly = AnomalyModule(buzzer=self._actuator)

        # 生长跟踪
        self._growth = GrowthModule()

        # Q-Learning
        self._learning = LearningModule()
        self._learning.begin()

        # 传感器融合
        self._fusion = FusionModule()
        self._fusion.begin(self._config.fusion_enabled)

        # 植物病害检测
        enable_camera = self._hw_profile in (2, 3)
        self._plant_doctor = PlantDoctorModule(buzzer=self._actuator)
        if enable_camera:
            self._plant_doctor.begin(camera_id=self._camera_id, enabled=self._config.plant_doctor_enabled)

        # 显示
        self._display = DisplayModule(
            i2c_bus=self._pins.i2c_bus,
            oled_addr=self._pins.oled_addr,
            use_dsi=False,
        )
        self._display.begin()
        self._display.set_modules(
            self._sensor_hub, self._actuator, self._irrigation,
            self._anomaly, self._growth, self._learning,
            self._fusion, self._plant_doctor,
        )

        # Web仪表盘
        self._web = WebDashboard(
            self._sensor_hub, self._actuator, self._irrigation,
            self._anomaly, self._growth, self._learning,
            self._fusion, self._plant_doctor,
        )

        logger.info("所有模块初始化完成")

    def loop(self):
        """主循环 (对应ESP32的 loop())"""
        self._running = True
        logger.info("主循环启动")

        while self._running:
            now = time.time()

            # 1. 传感器采样
            sample_updated = self._sensor_hub.update()

            # 2. 灌溉规则引擎
            if sample_updated:
                self._irrigation.update(self._sensor_hub.snapshot)

            # 3. 执行器仲裁 (第一轮)
            irrigation_result = self._irrigation.result
            base_auto = self._irrigation.enabled and irrigation_result.should_water
            self._actuator.update(base_auto, now)

            # 4. 异常检测
            self._anomaly.update(self._sensor_hub.snapshot, sample_updated, now)

            # 5. 生长跟踪
            self._growth.update(self._sensor_hub.snapshot, sample_updated, now)

            # 6. Q-Learning
            self._learning.update(self._sensor_hub.snapshot, sample_updated, now, self._actuator)

            # 7. 传感器融合
            self._fusion.update(self._sensor_hub.snapshot, sample_updated, now, self._actuator)

            # 8. 植物病害检测
            self._plant_doctor.update(now, self._sensor_hub.snapshot.light_intensity)

            # 9. 执行器仲裁 (第二轮: 处理定时运行完成)
            self._actuator.update(base_auto, now)

            # 10. 显示刷新
            self._display.update(now)

            time.sleep(0.01)  # 10ms主循环间隔

    def run(self):
        """启动应用: Web服务在子线程, 主循环在主线程"""
        self.setup()

        # Web服务在子线程运行
        web_thread = threading.Thread(
            target=self._web.run,
            kwargs={'host': '0.0.0.0', 'port': self._web_port, 'debug': False},
            daemon=True,
        )
        web_thread.start()
        logger.info(f"Web仪表盘启动: http://0.0.0.0:{self._web_port}")

        # 信号处理
        def signal_handler(sig, frame):
            logger.info("收到停止信号, 正在关闭...")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 主循环
        self.loop()

        # 清理
        self._cleanup()

    def _cleanup(self):
        logger.info("正在清理资源...")
        if self._sensor_hub:
            self._sensor_hub.close()
        if self._actuator:
            self._actuator.close()
        if self._plant_doctor:
            self._plant_doctor.close()
        if self._fusion:
            self._fusion.save_network()
        if self._learning:
            self._learning._save_q_table()
        logger.info("清理完成, 程序退出")


def main():
    parser = argparse.ArgumentParser(description="智润智慧农业套件 - Atlas 200I DK A2")
    parser.add_argument('--profile', type=int, default=1, choices=[1, 2, 3],
                        help='硬件配置档案 (1=ControllerKit, 2=HybridDevKit, 3=CameraEye)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Web仪表盘端口')
    parser.add_argument('--camera', type=str, default=None,
                        help='摄像头ID(数字) 或 RTSP地址 (如rtsp://admin:pass@ip:554/cam/realmonitor?channel=1&subtype=0)')
    parser.add_argument('--rtsp', type=str, default=None,
                        help='大华RTSP摄像头地址 (快捷参数, 覆盖--camera)')
    parser.add_argument('--no-oled', action='store_true',
                        help='禁用OLED显示')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    parser.add_argument('--demo', action='store_true',
                        help='Demo模式 (模拟传感器数据, 无需真实硬件)')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 确定摄像头参数: --rtsp > --camera > pin_config默认
    if args.rtsp:
        camera_id = args.rtsp
    elif args.camera is not None:
        # --camera 可以是数字(ID)或RTSP URL
        try:
            camera_id = int(args.camera)
        except ValueError:
            camera_id = args.camera
    else:
        camera_id = 0

    app = SmartAgricultureApp(
        hw_profile=args.profile,
        web_port=args.port,
        camera_id=camera_id,
        demo=args.demo,
    )
    app.run()


if __name__ == '__main__':
    main()
