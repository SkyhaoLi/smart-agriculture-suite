"""
智润智慧农业套件 - Atlas 200I DK A2 版
跨模块共享数据类型

对应原ESP32项目的 AppTypes.h
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict
import time


# ============================================================================
# 控制源优先级 (高 -> 低)
# ============================================================================
class ControlSource(IntEnum):
    SafetyLock = 0    # 安全锁定 (液位过低等)
    Manual = 1        # 手动控制
    TimedRun = 2      # 定时运行 (学习/融合模块)
    RuleEngine = 3    # 规则引擎
    None_ = 99        # 无控制


# ============================================================================
# 灌溉动作 (Q-Learning动作空间)
# ============================================================================
class IrrigationAction(IntEnum):
    Off = 0           # 关闭
    Low = 1           # 轻度 30s
    Moderate = 2      # 中度 45s
    Heavy = 3         # 重度 120s


# ============================================================================
# 异常等级
# ============================================================================
class AnomalyLevel(IntEnum):
    Normal = 0
    Warning = 1
    Critical = 2


# ============================================================================
# 病害类别
# ============================================================================
class DiseaseClass(IntEnum):
    Healthy = 0
    Anthracnose = 1       # 炭疽病
    GrayMold = 2          # 灰霉病
    LeafScorch = 3        # 叶灼病
    PowderyMildew = 4     # 白粉病


DISEASE_NAMES = {
    DiseaseClass.Healthy: "健康",
    DiseaseClass.Anthracnose: "炭疽病",
    DiseaseClass.GrayMold: "灰霉病",
    DiseaseClass.LeafScorch: "叶灼病",
    DiseaseClass.PowderyMildew: "白粉病",
}


# ============================================================================
# 作物类型
# ============================================================================
class CropType(IntEnum):
    Tomato = 0       # 番茄
    Lettuce = 1      # 生菜
    Pepper = 2       # 辣椒
    Cucumber = 3     # 黄瓜
    Strawberry = 4   # 草莓


CROP_NAMES = {
    CropType.Tomato: "番茄",
    CropType.Lettuce: "生菜",
    CropType.Pepper: "辣椒",
    CropType.Cucumber: "黄瓜",
    CropType.Strawberry: "草莓",
}


# ============================================================================
# 生长阶段
# ============================================================================
class GrowthStage(IntEnum):
    Seed = 0          # 播种
    Germination = 1   # 发芽
    Seedling = 2      # 幼苗
    Vegetative = 3    # 营养生长期
    Flowering = 4     # 开花期
    Fruiting = 5      # 结果期
    Maturity = 6      # 成熟期


# ============================================================================
# 传感器快照
# ============================================================================
@dataclass
class SensorSnapshot:
    """五路传感器数据快照"""
    air_temp: float = 0.0          # 空气温度 (°C)
    air_humi: float = 0.0          # 空气湿度 (%)
    soil_humi: float = 0.0         # 土壤湿度 (%)
    light_intensity: float = 0.0   # 光照强度 (lux)
    is_day: bool = True            # 白天/黑夜
    timestamp: float = 0.0         # 时间戳

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# ============================================================================
# 灌溉阈值配置
# ============================================================================
@dataclass
class IrrigationThresholdConfig:
    air_temp_day_high: float = 35.0
    air_temp_day_low: float = 15.0
    air_temp_night_high: float = 25.0
    air_temp_night_low: float = 10.0
    air_humi_day_low: float = 40.0
    air_humi_night_low: float = 50.0
    soil_humi_low: float = 30.0
    soil_humi_high: float = 70.0


# ============================================================================
# 学习模块配置
# ============================================================================
@dataclass
class LearningConfig:
    enabled: bool = True
    decision_interval_ms: int = 300000
    epsilon_start: float = 0.3
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    state_bins: Dict = field(default_factory=lambda: {
        'temp': 5, 'humi': 4, 'soil': 5, 'light': 3, 'time': 3
    })


# ============================================================================
# 植物医生配置
# ============================================================================
@dataclass
class PlantDoctorConfig:
    enabled: bool = True
    interval_ms: int = 60000
    confidence_threshold: float = 0.70
    buzzer_on_detect: bool = True


# ============================================================================
# 执行器状态
# ============================================================================
@dataclass
class ActuatorStatus:
    valve_on: bool = False
    pump_on: bool = False
    buzzer_on: bool = False
    active_source: ControlSource = ControlSource.None_
    timed_run_remaining_ms: int = 0
    safety_lock: bool = False


# ============================================================================
# 异常检测结果
# ============================================================================
@dataclass
class AnomalyResult:
    level: AnomalyLevel = AnomalyLevel.Normal
    zscore_flags: List[str] = field(default_factory=list)
    iso_forest_score: float = 0.0
    sensor_faults: List[str] = field(default_factory=list)
    timestamp: float = 0.0


# ============================================================================
# 病害检测结果
# ============================================================================
@dataclass
class DiseaseResult:
    disease_class: DiseaseClass = DiseaseClass.Healthy
    confidence: float = 0.0
    all_probs: List[float] = field(default_factory=list)
    timestamp: float = 0.0


# ============================================================================
# 生长状态
# ============================================================================
@dataclass
class GrowthState:
    crop: CropType = CropType.Tomato
    cumulative_gdd: float = 0.0
    growth_day: int = 0
    current_stage: GrowthStage = GrowthStage.Seed
    yield_score: float = 100.0
    predicted_flower_day: Optional[int] = None
    predicted_maturity_day: Optional[int] = None
