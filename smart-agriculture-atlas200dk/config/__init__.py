"""智润智慧农业套件 - Atlas 200I DK A2 版 - 配置包"""
from .hardware_config import (
    PinConfig, ADCConfig, SystemConfig, Timing,
    HWProfile, Pin40, GPIOLine, CONFIG_FILE, DATA_DIR, MODEL_DIR, LOG_DIR,
)
from .app_types import (
    ControlSource, IrrigationAction, AnomalyLevel, DiseaseClass, CropType,
    GrowthStage, SensorSnapshot, IrrigationThresholdConfig, LearningConfig,
    PlantDoctorConfig, ActuatorStatus, AnomalyResult, DiseaseResult, GrowthState,
    DISEASE_NAMES, CROP_NAMES,
)

__all__ = [
    "PinConfig", "ADCConfig", "SystemConfig", "Timing",
    "HWProfile", "Pin40", "GPIOLine", "CONFIG_FILE", "DATA_DIR", "MODEL_DIR", "LOG_DIR",
    "ControlSource", "IrrigationAction", "AnomalyLevel", "DiseaseClass", "CropType",
    "GrowthStage", "SensorSnapshot", "IrrigationThresholdConfig", "LearningConfig",
    "PlantDoctorConfig", "ActuatorStatus", "AnomalyResult", "DiseaseResult", "GrowthState",
    "DISEASE_NAMES", "CROP_NAMES",
]
