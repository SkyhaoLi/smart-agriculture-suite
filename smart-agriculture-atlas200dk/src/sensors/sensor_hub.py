"""
智润智慧农业套件 - Atlas 200I DK A2 版
传感器中枢 - 管理所有传感器通道

对应原ESP32项目的 Sensors.h/Sensors.cpp
使用Linux I2C/UART/ADC接口替代ESP32的硬件抽象
"""

import serial
import smbus2
import time
import logging
from typing import Optional

from config.app_types import SensorSnapshot

logger = logging.getLogger(__name__)


class SensorHub:
    """五路传感器管理"""

    def __init__(self, pin_config, adc_config=None):
        self._pins = pin_config
        self._adc_cfg = adc_config
        self._snapshot = SensorSnapshot()
        self._last_sample_time = 0.0
        self._sample_interval = 2.0  # 秒

        self._air_serial: Optional[serial.Serial] = None
        self._i2c_bus: Optional[smbus2.SMBus] = None
        self._adc_bus: Optional[smbus2.SMBus] = None
        self._last_air_frame = ""

        self._initialized = False

    def begin(self) -> bool:
        """初始化所有传感器接口"""
        success = True

        # UART空气温湿度传感器
        try:
            self._air_serial = serial.Serial(
                port=self._pins.air_sensor_uart,
                baudrate=self._pins.air_sensor_baud,
                timeout=1.0
            )
            logger.info(f"空气传感器UART已连接: {self._pins.air_sensor_uart}")
        except serial.SerialException as e:
            logger.warning(f"空气传感器UART连接失败: {e}")
            success = False

        # I2C总线 (BH1750 + OLED)
        try:
            self._i2c_bus = smbus2.SMBus(self._pins.i2c_bus)
            self._init_bh1750()
            logger.info(f"I2C总线{self._pins.i2c_bus}已初始化")
        except Exception as e:
            logger.warning(f"I2C总线初始化失败: {e}")
            success = False

        # ADC (ADS1115 via I2C6)
        if self._adc_cfg:
            try:
                self._adc_bus = smbus2.SMBus(self._adc_cfg.adc_i2c_bus)
                logger.info(f"ADC I2C总线{self._adc_cfg.adc_i2c_bus}已初始化")
            except Exception as e:
                logger.warning(f"ADC初始化失败: {e}")
                success = False

        self._initialized = success
        return success

    def update(self) -> bool:
        """采样所有传感器, 返回是否有新数据"""
        now = time.time()
        if now - self._last_sample_time < self._sample_interval:
            return False

        self._last_sample_time = now

        # 空气温湿度 (UART)
        self._read_air_sensor()

        # 土壤湿度 (ADC)
        if self._adc_bus and self._adc_cfg:
            raw = self._read_ads1115_average(self._adc_cfg.soil_channel)
            if raw is not None:
                self._snapshot.soil_humi = self._map_constrain(
                    raw, self._adc_cfg.soil_adc_dry, self._adc_cfg.soil_adc_wet, 0.0, 100.0
                )

        # 液位 (ADC)
        if self._adc_bus and self._adc_cfg:
            raw = self._read_ads1115_average(self._adc_cfg.liquid_channel)
            if raw is not None:
                self._snapshot.liquid_level = self._map_constrain(
                    raw, self._adc_cfg.liquid_adc_empty, self._adc_cfg.liquid_adc_full, 0.0, 100.0
                )

        # 光照 (I2C BH1750)
        lux = self._read_bh1750()
        if lux is not None and lux >= 0:
            self._snapshot.light_intensity = lux

        # 日夜判断
        self._snapshot.is_day = self._snapshot.light_intensity >= 200.0
        self._snapshot.timestamp = now

        return True

    @property
    def snapshot(self) -> SensorSnapshot:
        return self._snapshot

    @property
    def last_air_frame(self) -> str:
        return self._last_air_frame

    # ------------------------------------------------------------------
    # UART 空气温湿度传感器
    # ------------------------------------------------------------------
    def _read_air_sensor(self):
        if not self._air_serial or not self._air_serial.is_open:
            return

        try:
            if self._air_serial.in_waiting > 0:
                line = self._air_serial.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    self._parse_air_frame(line)
                    self._last_air_frame = line
        except Exception as e:
            logger.debug(f"空气传感器读取异常: {e}")

    def _parse_air_frame(self, frame: str):
        try:
            temp_idx = frame.find("Temp:")
            humi_idx = frame.find("Humi:")
            if temp_idx < 0 or humi_idx < 0:
                return

            comma_idx = frame.find(',', temp_idx)
            if comma_idx > temp_idx:
                temp_str = frame[temp_idx + 5:comma_idx].strip()
            else:
                temp_str = frame[temp_idx + 5:].strip()
            humi_str = frame[humi_idx + 5:].strip()

            self._snapshot.air_temp = float(temp_str)
            self._snapshot.air_humi = float(humi_str)
        except (ValueError, IndexError):
            pass

    # ------------------------------------------------------------------
    # ADS1115 ADC (I2C)
    # ------------------------------------------------------------------
    ADS1115_ADDR = 0x48
    ADS1115_REG_CONFIG = 0x01
    ADS1115_REG_CONVERSION = 0x00

    # Config register bits for single-shot
    _ADS1115_OS_SINGLE = 0x8000
    _ADS1115_MUX_OFFSET = 12
    _ADS1115_PGA_OFFSET = 9
    _ADS1115_MODE_SINGLE = 0x0000
    _ADS1115_DR_OFFSET = 5
    _ADS1115_COMP_QUE_DISABLE = 0x0003

    def _read_ads1115(self, channel: int) -> Optional[int]:
        """读取ADS1115单通道原始值"""
        if not self._adc_bus:
            return None

        addr = self._adc_cfg.adc_addr if self._adc_cfg else self.ADS1115_ADDR
        pga = self._adc_cfg.adc_gain if self._adc_cfg else 4096

        # MUX配置: A0=0x4000, A1=0x5000, A2=0x6000, A3=0x7000
        mux_map = {0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000}
        mux = mux_map.get(channel, 0x4000)

        # PGA配置
        pga_map = {6144: 0x0000, 4096: 0x0200, 2048: 0x0400, 1024: 0x0600, 512: 0x0800, 256: 0x0A00}
        pga_bits = pga_map.get(pga, 0x0200)

        config = (self._ADS1115_OS_SINGLE | mux | pga_bits |
                  self._ADS1115_MODE_SINGLE | (0x0040) |  # DR=128SPS
                  self._ADS1115_COMP_QUE_DISABLE)

        try:
            self._adc_bus.write_i2c_block_data(addr, self.ADS1115_REG_CONFIG,
                                                [(config >> 8) & 0xFF, config & 0xFF])
            time.sleep(0.01)  # 等待转换

            data = self._adc_bus.read_i2c_block_data(addr, self.ADS1115_REG_CONVERSION, 2)
            raw = (data[0] << 8) | data[1]
            if raw >= 0x8000:
                raw -= 0x10000  # 补码转有符号
            return raw
        except Exception as e:
            logger.debug(f"ADS1115读取失败 ch{channel}: {e}")
            return None

    def _read_ads1115_average(self, channel: int, count: int = 5) -> Optional[float]:
        """多次采样取平均"""
        values = []
        for _ in range(count):
            v = self._read_ads1115(channel)
            if v is not None:
                values.append(v)
            time.sleep(0.002)
        return sum(values) / len(values) if values else None

    # ------------------------------------------------------------------
    # BH1750 光照传感器 (I2C)
    # ------------------------------------------------------------------
    BH1750_CONT_HIGH_RES = 0x10  # 连续高分辨率模式

    def _init_bh1750(self):
        if not self._i2c_bus:
            return
        try:
            self._i2c_bus.write_byte(self._pins.bh1750_addr, self.BH1750_CONT_HIGH_RES)
        except Exception as e:
            logger.warning(f"BH1750初始化失败: {e}")

    def _read_bh1750(self) -> Optional[float]:
        if not self._i2c_bus:
            return None
        try:
            data = self._i2c_bus.read_i2c_block_data(self._pins.bh1750_addr, 0x00, 2)
            lux = (data[0] << 8 | data[1]) / 1.2
            return lux
        except Exception as e:
            logger.debug(f"BH1750读取失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _map_constrain(value, in_min, in_max, out_min, out_max):
        mapped = out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
        return max(out_min, min(out_max, mapped))

    def close(self):
        if self._air_serial and self._air_serial.is_open:
            self._air_serial.close()
        if self._i2c_bus:
            self._i2c_bus.close()
        if self._adc_bus:
            self._adc_bus.close()
