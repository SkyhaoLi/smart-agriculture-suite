"""
智润智慧农业套件 - Atlas 200I DK A2 版
硬件引脚配置与系统常量

Atlas 200I DK A2 40Pin 接口映射:
- I2C7:  Pin3(SDA), Pin5(SCL)   -> BH1750光照传感器 + SSD1306 OLED
- I2C6:  Pin27(SDA), Pin28(SCL) -> 预留
- UART0: Pin8(TX), Pin10(RX)    -> 调试串口(默认console)
- UART2: Pin26(TX), Pin31(RX)   -> 空气温湿度传感器
- SPI:   Pin19(MOSI), Pin21(MISO), Pin23(SCLK), Pin24(CS0)
- GPIO:  Pin11(GPIO17), Pin13(GPIO27), Pin15(GPIO22), Pin16(GPIO23),
         Pin18(GPIO24), Pin22(GPIO25), Pin33(GPIO13), Pin36(GPIO16), Pin37(GPIO26)
- PWM0:  Pin32

注意: Atlas 200I DK A2 GPIO为3.3V电平，输出需外置1K~10K上拉电阻增强驱动能力
      驱动阀门/水泵等大电流负载必须通过MOS管或继电器模块
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ============================================================================
# 硬件配置档案 (对应原ESP32项目的 AGRI_HW_PROFILE)
# ============================================================================
class HWProfile(IntEnum):
    ControllerKit = 1      # 纯灌溉控制，无摄像头
    HybridDevKit = 2      # 灌溉 + 摄像头共存
    CameraEyeStandalone = 3  # 摄像头优先，无灌溉执行器


# ============================================================================
# Atlas 200I DK A2 40Pin 引脚定义
# ============================================================================
class Pin40:
    """Atlas 200I DK A2 40Pin 扩展接口引脚编号"""
    # 电源
    V3_3_1 = 1
    V5_0_1 = 2
    V5_0_2 = 4
    V3_3_2 = 17
    # GND
    GND_6 = 6
    GND_9 = 9
    GND_14 = 14
    GND_20 = 20
    GND_25 = 25
    GND_30 = 30
    GND_34 = 34
    GND_39 = 39
    # I2C7 (默认功能，无需复用配置)
    I2C7_SDA = 3    # GPIO2
    I2C7_SCL = 5    # GPIO3
    # I2C6 (需复用配置 mux=0x2)
    I2C6_SDA = 27   # GPIO14, 需 mux=0x2
    I2C6_SCL = 28   # GPIO15, 需 mux=0x2
    # UART0 (默认调试串口)
    UART0_TX = 8    # GPIO14
    UART0_RX = 10   # GPIO15
    # UART2 (需复用配置)
    UART2_TX = 26   # GPIO7
    UART2_RX = 31   # GPIO6
    # SPI
    SPI_MOSI = 19   # GPIO10
    SPI_MISO = 21   # GPIO9
    SPI_SCLK = 23   # GPIO11
    SPI_CS0 = 24    # GPIO8
    # 可用GPIO
    GPIO17 = 11
    GPIO27 = 13
    GPIO22 = 15
    GPIO23 = 16
    GPIO24 = 18
    GPIO25 = 22
    GPIO13 = 33
    GPIO16 = 36
    GPIO26 = 37
    # PWM
    PWM0 = 32
    # GPCLK
    GPCLK0 = 7
    GPCLK1 = 29


# ============================================================================
# Atlas 200I DK A2 GPIO芯片偏移映射
# 40Pin GPIO编号 -> Linux gpiochip line offset
# ============================================================================
class GPIOLine:
    """Linux gpiod 线号映射 (需根据实际 /dev/gpiochip 验证)"""
    GPIO2 = 2     # Pin3,  I2C7_SDA
    GPIO3 = 3     # Pin5,  I2C7_SCL
    GPIO6 = 6     # Pin31, UART2_RX / CAN_TX2
    GPIO7 = 7     # Pin26, UART2_TX / CAN_RX2
    GPIO8 = 8     # Pin24, SPI_CS0 / I2C12_SCL
    GPIO9 = 9     # Pin21, SPI_MISO / I2C11_SCL
    GPIO10 = 10   # Pin19, SPI_MOSI / I2C12_SDA
    GPIO11 = 11   # Pin23, SPI_SCLK / I2C11_SDA
    GPIO13 = 13   # Pin33
    GPIO14 = 14   # Pin8,  UART0_TX / I2C6_SDA
    GPIO15 = 15   # Pin10, UART0_RX / I2C6_SCL
    GPIO16 = 16   # Pin36, UART2_TX / CAN_TX3
    GPIO17 = 17   # Pin11
    GPIO22 = 22   # Pin15
    GPIO23 = 23   # Pin16
    GPIO24 = 24   # Pin18
    GPIO25 = 25   # Pin22
    GPIO26 = 26   # Pin37
    GPIO27 = 27   # Pin13


# ============================================================================
# 传感器/执行器引脚分配
# ============================================================================
@dataclass
class PinConfig:
    """业务引脚分配 - 对应原ESP32项目的 defaultPins()"""

    # 传感器接口
    air_sensor_uart: str = "/dev/ttyUART2"   # 空气温湿度 UART (UART2: Pin26-TX, Pin31-RX)
    air_sensor_baud: int = 9600

    soil_adc_chip: str = "/dev/iio:device0"  # 土壤湿度 ADC (需外接ADC模块, Atlas无原生ADC)
    soil_adc_channel: int = 0

    i2c_bus: int = 7           # I2C7 (Pin3/Pin5), 用于BH1750和OLED
    bh1750_addr: int = 0x23   # BH1750 I2C地址
    oled_addr: int = 0x3C     # SSD1306 I2C地址

    # 执行器 GPIO (通过 gpiod 控制)
    valve_gpio: int = GPIOLine.GPIO17   # 电磁阀 -> Pin11 (GPIO17)
    pump_gpio: int = GPIOLine.GPIO27    # 水泵   -> Pin13 (GPIO27)
    buzzer_gpio: int = GPIOLine.GPIO22  # 蜂鸣器 -> Pin15 (GPIO22)

    # 摄像头 (USB摄像头, 通过OpenCV/VideoCapture读取)
    camera_type: str = "usb"   # USB摄像头, 通过OpenCV/VideoCapture读取
    camera_id: int = 0         # USB设备ID (如 /dev/video0)

    # GPIO芯片名
    gpio_chip: str = "gpiochip0"


# ============================================================================
# ADC映射参数 (Atlas 200I DK A2 无原生ADC, 使用外接ADS1115等I2C ADC)
# ============================================================================
@dataclass
class ADCConfig:
    """外接ADC配置 (推荐使用 ADS1115 通过 I2C6 连接)"""
    adc_i2c_bus: int = 6       # I2C6 (Pin27/Pin28)
    adc_addr: int = 0x48       # ADS1115 默认地址
    soil_channel: int = 0      # ADS1115 A0 -> 土壤湿度
    adc_gain: int = 4096       # ADS1115 PGA = ±4.096V
    soil_adc_dry: int = 32000  # 土壤干时ADC值 (16-bit)
    soil_adc_wet: int = 12000  # 土壤湿时ADC值


# ============================================================================
# 系统配置 (对应原ESP32项目的 SystemConfig)
# ============================================================================
@dataclass
class SystemConfig:
    # 灌溉阈值
    air_temp_day_high: float = 35.0
    air_temp_day_low: float = 15.0
    air_temp_night_high: float = 25.0
    air_temp_night_low: float = 10.0
    air_humi_day_low: float = 40.0
    air_humi_night_low: float = 50.0
    soil_humi_low: float = 30.0
    soil_humi_high: float = 70.0

    # 学习模块
    learning_enabled: bool = True
    decision_interval_ms: int = 300000   # 5分钟
    epsilon_start: float = 0.3
    epsilon_min: float = 0.05

    # 融合模块
    fusion_enabled: bool = True
    fusion_interval_ms: int = 10000      # 10秒
    weighted_ratio: float = 0.6          # 加权评分占比
    nn_ratio: float = 0.4                # NN输出占比

    # 异常检测
    anomaly_sample_window: int = 60
    zscore_warn_threshold: float = 2.5
    zscore_critical_threshold: float = 3.5
    iso_forest_threshold: float = 0.65
    iso_forest_n_trees: int = 10
    iso_forest_max_depth: int = 8
    iso_forest_build_samples: int = 200
    iso_forest_min_samples: int = 50

    # 植物医生
    plant_doctor_enabled: bool = True
    plant_doctor_interval_ms: int = 60000   # 60秒
    plant_doctor_confidence_threshold: float = 0.70

    # 生长跟踪
    growth_enabled: bool = True

    # 传感器采样
    sensor_sample_interval_ms: int = 2000   # 2秒
    sensor_adc_average_count: int = 5

    # 执行器
    valve_timed_moderate_ms: int = 45000    # 中度灌溉45秒
    valve_timed_heavy_ms: int = 120000      # 重度灌溉120秒

    # 网络
    wifi_ssid: str = ""
    wifi_password: str = ""
    web_port: int = 8080

    # OLED
    oled_enabled: bool = True
    oled_page_interval_ms: int = 4000


# ============================================================================
# 时间常量
# ============================================================================
class Timing:
    MAIN_LOOP_INTERVAL = 0.01       # 主循环 10ms
    SENSOR_UPDATE = 2.0             # 传感器 2s
    ANOMALY_ISO_FOREST = 60.0       # 孤立森林 60s
    OLED_REFRESH = 0.7              # OLED刷新 700ms
    LEARNING_DECISION = 300.0       # 学习决策 5min
    FUSION_DECISION = 10.0          # 融合决策 10s
    PLANT_DOCTOR_DETECT = 60.0      # 病害检测 60s


# ============================================================================
# 运行时配置 (从JSON文件加载/保存)
# ============================================================================
CONFIG_FILE = "/etc/agri-atlas/config.json"
DATA_DIR = "/var/lib/agri-atlas"
MODEL_DIR = "/opt/agri-atlas/models"
LOG_DIR = "/var/log/agri-atlas"
