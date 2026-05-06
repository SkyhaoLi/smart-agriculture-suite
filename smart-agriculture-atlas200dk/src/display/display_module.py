"""
智润智慧农业套件 - Atlas 200I DK A2 版
OLED显示模块 - I2C SSD1306 128x64 / MIPI-DSI 屏

对应原ESP32项目的 OLED 6页轮播显示
Atlas 200I DK A2 可选:
1. SSD1306 I2C OLED (Pin3/Pin5 I2C7, 与BH1750共用)
2. MIPI-DSI 屏幕 (原生HDMI输出, 更高端)
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DisplayModule:
    """OLED/DSI显示模块 - 6页轮播"""

    PAGE_COUNT = 6
    PAGE_NAMES = ["Dashboard", "Irrigation", "Anomaly", "Growth", "Learn/Fusion", "PlantDoctor"]
    PAGE_INTERVAL = 4.0  # 秒
    REFRESH_INTERVAL = 0.7

    def __init__(self, i2c_bus=None, oled_addr=0x3C, use_dsi=False):
        self._i2c_bus = i2c_bus
        self._oled_addr = oled_addr
        self._use_dsi = use_dsi
        self._ready = False
        self._current_page = 0
        self._last_page_time = 0.0
        self._last_refresh_time = 0.0

        # 显示缓冲
        self._oled = None
        self._dsi_display = None

        # 状态引用 (由main.py设置)
        self._sensor_hub = None
        self._actuator = None
        self._irrigation = None
        self._anomaly = None
        self._growth = None
        self._learning = None
        self._fusion = None
        self._plant_doctor = None

    def begin(self) -> bool:
        if self._use_dsi:
            return self._setup_dsi()
        return self._setup_oled()

    def _setup_oled(self) -> bool:
        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306

            serial = i2c(port=self._i2c_bus or 7, address=self._oled_addr)
            self._oled = ssd1306(serial)
            self._ready = True
            logger.info("SSD1306 OLED初始化成功")
            return True
        except ImportError:
            logger.info("luma.oled未安装, OLED不可用 (pip install luma.oled)")
        except Exception as e:
            logger.warning(f"OLED初始化失败: {e}")

        # 退回: 使用终端输出模拟显示
        self._ready = True
        logger.info("使用终端模拟显示")
        return True

    def _setup_dsi(self) -> bool:
        logger.info("MIPI-DSI显示暂不支持, 使用终端模拟")
        self._ready = True
        return True

    def set_modules(self, sensor_hub, actuator, irrigation,
                     anomaly, growth, learning, fusion, plant_doctor):
        self._sensor_hub = sensor_hub
        self._actuator = actuator
        self._irrigation = irrigation
        self._anomaly = anomaly
        self._growth = growth
        self._learning = learning
        self._fusion = fusion
        self._plant_doctor = plant_doctor

    def update(self, now: float = 0.0):
        if not self._ready:
            return
        if now == 0.0:
            now = time.time()

        # 切换页面
        if now - self._last_page_time >= self.PAGE_INTERVAL:
            self._current_page = (self._current_page + 1) % self.PAGE_COUNT
            self._last_page_time = now

        # 刷新显示
        if now - self._last_refresh_time < self.REFRESH_INTERVAL:
            return
        self._last_refresh_time = now

        self._render_page(self._current_page)

    def _render_page(self, page: int):
        if self._oled:
            self._render_oled(page)
        else:
            self._render_terminal(page)

    def _render_terminal(self, page: int):
        """终端模拟显示 - 调试用"""
        if not self._sensor_hub:
            return

        snap = self._sensor_hub.snapshot
        title = self.PAGE_NAMES[page]

        lines = [f"=== {title} ==="]

        if page == 0:  # Dashboard
            lines.append(f"T {snap.air_temp:.1f}C H {snap.air_humi:.0f}%")
            lines.append(f"Soil {snap.soil_humi:.0f}% Lvl {snap.liquid_level:.0f}%")
            lines.append(f"Light {snap.light_intensity:.0f} {'Day' if snap.is_day else 'Night'}")
            if self._actuator:
                s = self._actuator.status
                lines.append(f"Pump {'ON' if s.pump_on else 'OFF'} Src:{s.active_source.name}")
        elif page == 1 and self._irrigation:  # Irrigation
            r = self._irrigation.result
            lines.append(f"Need {'YES' if r.should_water else 'NO'}")
            lines.append(f"Warn {'LOW LIQ' if r.liquid_warn else 'OK'}")
            lines.append(f"Reason: {r.reason}")
        elif page == 2 and self._anomaly:  # Anomaly
            lines.append(f"Level: {self._anomaly.current_level.name}")
            lines.append(f"IForest: {self._anomaly.iforest_score:.2f}")
            lines.append(f"Trained: {'YES' if self._anomaly.iforest_trained else 'NO'}")
            lines.append(f"Alerts: {self._anomaly.total_anomalies}")
        elif page == 3 and self._growth:  # Growth
            d = self._growth.to_dict()
            lines.append(f"{d['cropCn']} {d['stageNameCn']}")
            lines.append(f"Day {d['dayOfGrowth']} GDD {d['cumulativeGdd']:.0f}")
            lines.append(f"Yield {d['yieldScore']:.0f}")
        elif page == 4 and self._learning and self._fusion:  # Learn/Fusion
            ld = self._learning.to_dict()
            fd = self._fusion.to_dict()
            lines.append(f"Learn eps={ld['epsilon']:.3f} ep={ld['totalEpisodes']}")
            lines.append(f"Fusion: {fd['decision']} conf={fd['confidence']:.0%}")
        elif page == 5 and self._plant_doctor:  # Plant Doctor
            d = self._plant_doctor.to_dict()
            lines.append(f"Camera: {'READY' if d['cameraReady'] else 'NA'}")
            lines.append(f"Model: {'READY' if d['modelLoaded'] else 'NA'}")
            lines.append(f"{d['lastDiseaseNameCn']} {d['lastConfidence']:.0%}")

        # 清屏 + 输出 (终端模式直接打印)
        print(f"\033[2J\033[H" + "\n".join(lines), end="", flush=True)

    def _render_oled(self, page: int):
        """SSD1306 OLED渲染 (使用luma.oled)"""
        if not self._sensor_hub:
            return

        from luma.core.render import canvas

        snap = self._sensor_hub.snapshot
        title = self.PAGE_NAMES[page]

        with canvas(self._oled) as draw:
            # 标题栏
            draw.text((0, 0), title, fill="white")

            if page == 0:  # Dashboard
                draw.text((0, 12), f"T:{snap.air_temp:.1f}C H:{snap.air_humi:.0f}%", fill="white")
                draw.text((0, 22), f"Soil:{snap.soil_humi:.0f}% Lvl:{snap.liquid_level:.0f}%", fill="white")
                draw.text((0, 32), f"Light:{snap.light_intensity:.0f}", fill="white")
                if self._actuator:
                    s = self._actuator.status
                    draw.text((0, 44), f"Pump:{'ON' if s.pump_on else 'OFF'} {s.active_source.name}", fill="white")
            elif page == 1 and self._irrigation:  # Irrigation
                r = self._irrigation.result
                draw.text((0, 12), f"Need:{'YES' if r.should_water else 'NO'}", fill="white")
                draw.text((0, 22), f"Warn:{'LOW' if r.liquid_warn else 'OK'}", fill="white")
                draw.text((0, 32), f"Reason:", fill="white")
                draw.text((0, 42), r.reason[:21], fill="white")
            elif page == 2 and self._anomaly:  # Anomaly
                draw.text((0, 12), f"Level:{self._anomaly.current_level.name}", fill="white")
                draw.text((0, 22), f"IForest:{self._anomaly.iforest_score:.2f}", fill="white")
                draw.text((0, 32), f"Trained:{'Y' if self._anomaly.iforest_trained else 'N'}", fill="white")
                draw.text((0, 44), f"Alerts:{self._anomaly.total_anomalies}", fill="white")
            elif page == 3 and self._growth:  # Growth
                d = self._growth.to_dict()
                draw.text((0, 12), f"{d['cropCn']} {d['stageNameCn']}", fill="white")
                draw.text((0, 22), f"Day:{d['dayOfGrowth']} GDD:{d['cumulativeGdd']:.0f}", fill="white")
                draw.text((0, 32), f"Yield:{d['yieldScore']:.0f}", fill="white")
            elif page == 4 and self._learning and self._fusion:  # Learn/Fusion
                ld = self._learning.to_dict()
                fd = self._fusion.to_dict()
                draw.text((0, 12), f"eps:{ld['epsilon']:.3f} ep:{ld['totalEpisodes']}", fill="white")
                draw.text((0, 22), f"Fusion:{fd['decision']}", fill="white")
                draw.text((0, 32), f"conf:{fd['confidence']:.0%}", fill="white")
            elif page == 5 and self._plant_doctor:  # Plant Doctor
                d = self._plant_doctor.to_dict()
                draw.text((0, 12), f"Cam:{'OK' if d['cameraReady'] else 'NA'}", fill="white")
                draw.text((0, 22), f"Model:{'OK' if d['modelLoaded'] else 'NA'}", fill="white")
                draw.text((0, 32), d['lastDiseaseNameCn'][:10], fill="white")
                draw.text((0, 42), f"{d['lastConfidence']:.0%}", fill="white")

    @property
    def ready(self) -> bool:
        return self._ready
