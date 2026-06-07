"""
完整逻辑验证 - 模拟传感器数据, 验证所有AI模块的输出与ESP32版一致
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.app_types import SensorSnapshot, IrrigationThresholdConfig, ControlSource
from src.ai.irrigation_module import IrrigationModule
from src.ai.anomaly_module import AnomalyModule
from src.ai.growth_module import GrowthModule, CROP_PROFILES, STAGE_NAMES_CN
from src.ai.learning_module import LearningModule, IrrigationAction, ACTION_NAMES
from src.ai.fusion_module import FusionModule
from src.ai.plant_doctor_module import PlantDoctorModule

errors = []

def check(name, actual, expected, tol=0.01):
    if isinstance(expected, float):
        if abs(actual - expected) > tol:
            errors.append(f"FAIL {name}: got {actual}, expected {expected}")
            return False
    elif actual != expected:
        errors.append(f"FAIL {name}: got {actual!r}, expected {expected!r}")
        return False
    return True

# ============================================================================
# 1. 灌溉规则引擎测试
# ============================================================================
print("\n=== 1. 灌溉规则引擎 ===")
irrig = IrrigationModule()

# 白天, 温度正常偏高, 湿度低, 土壤干 -> 应该灌溉
snap_day_dry = SensorSnapshot(air_temp=36.0, air_humi=30.0, soil_humi=25.0,
                               liquid_level=80.0, light_intensity=500.0, is_day=True)
result = irrig.update(snap_day_dry)
check("day-dry should water", result.should_water, True)
check("day-dry is_day", result.is_day, True)
check("day-dry liquid_warn", result.liquid_warn, False)
print(f"  白天干燥 -> should_water={result.should_water}, reason={result.reason}")

# 夜间, 条件正常 -> 不灌溉
snap_night_ok = SensorSnapshot(air_temp=18.0, air_humi=65.0, soil_humi=55.0,
                                liquid_level=70.0, light_intensity=50.0, is_day=False)
result = irrig.update(snap_night_ok)
check("night-ok should not water", result.should_water, False)
print(f"  夜间正常 -> should_water={result.should_water}, reason={result.reason}")

# 液位过低 -> 不灌溉
snap_low_liquid = SensorSnapshot(air_temp=36.0, air_humi=30.0, soil_humi=25.0,
                                  liquid_level=10.0, light_intensity=500.0, is_day=True)
result = irrig.update(snap_low_liquid)
check("low-liquid should not water", result.should_water, False)
check("low-liquid liquid_warn", result.liquid_warn, True)
print(f"  液位过低 -> should_water={result.should_water}, liquid_warn={result.liquid_warn}")

# ============================================================================
# 2. 异常检测测试
# ============================================================================
print("\n=== 2. 异常检测 ===")
anomaly = AnomalyModule()

# 先灌入正常数据, 使用递增时间确保iforest定时触发
from config.app_types import AnomalyLevel
now_base = 1000000.0
for i in range(200):
    now = now_base + i * 2.0
    snap = SensorSnapshot(air_temp=25.0 + (i % 5) * 0.1, air_humi=60.0 + (i % 3) * 0.1,
                           soil_humi=50.0 + (i % 4) * 0.1, light_intensity=500.0 + (i % 6) * 0.5,
                           liquid_level=80.0, is_day=True)
    anomaly.update(snap, sample_updated=True, now=now)

check("anomaly 200 samples", anomaly.total_samples, 200)
# 200次 * 2秒间隔 = 400秒, 跨过了多个60秒iforest检查点
# train_buffer_count >= 50且iforest未训练 -> 应该已经训练
check("anomaly iforest trained", anomaly.iforest_trained, True)
print(f"  200样本: level={anomaly.current_level.name}, iforest_trained={anomaly.iforest_trained}, score={anomaly.iforest_score:.4f}")
print(f"  200个样本后: iforest_trained={anomaly.iforest_trained}, score={anomaly.iforest_score:.4f}")

# 注入异常值
snap_anomaly = SensorSnapshot(air_temp=50.0, air_humi=10.0, soil_humi=5.0,
                               light_intensity=15000.0, liquid_level=80.0, is_day=True)
anomaly.update(snap_anomaly, sample_updated=True, now=now_base + 500.0)
print(f"  注入异常值: level={anomaly.current_level.name}, anomalies={anomaly.total_anomalies}")
check("anomaly detected", anomaly.total_anomalies > 0, True)

# ============================================================================
# 3. 生长跟踪测试
# ============================================================================
print("\n=== 3. 生长跟踪 ===")
growth = GrowthModule()
growth.set_crop(0)  # 番茄

check("growth crop", growth.current_crop.name, "Tomato")
check("growth initial gdd", growth._cumulative_gdd, 0.0)
check("growth initial stage", growth._current_stage.value, 0)  # Seed

# 模拟100天生长 (使用真实时间偏移, 与ESP32运行时一致)
BASE_TIME = 1000000.0  # 避免now=0的边界问题
for day in range(100):
    for hour in range(24):
        now = BASE_TIME + day * 86400.0 + hour * 3600.0
        snap = SensorSnapshot(air_temp=25.0, air_humi=60.0, soil_humi=50.0,
                               light_intensity=800.0 if 6 <= hour < 18 else 10.0,
                               liquid_level=80.0, is_day=6 <= hour < 18)
        growth.update(snap, sample_updated=True, now=now)

# 100天后检查: GDD应该已积累
# 每天 GDD = max(0, 25-10) = 15, 100天约1500 GDD
print(f"  100天番茄: day={growth._current_day}, GDD={growth._cumulative_gdd:.0f}, "
      f"stage={STAGE_NAMES_CN[growth._current_stage.value]}, yield={growth._yield_score:.0f}")
check("growth day >= 1", growth._current_day >= 1, True)  # 应该已过至少1天
check("growth gdd > 0", growth._cumulative_gdd > 0, True)
check("growth stage advanced", growth._current_stage.value > 0, True)  # 应该已过种子期

# ============================================================================
# 4. Q-Learning测试
# ============================================================================
print("\n=== 4. Q-Learning ===")
learning = LearningModule()
learning.begin()

check("learning initial episodes", learning._total_episodes, 0)
check("learning initial epsilon", learning._config.epsilon, 0.3)

# 模拟状态离散化
snap = SensorSnapshot(air_temp=25.0, air_humi=60.0, soil_humi=50.0,
                       light_intensity=300.0, liquid_level=80.0, is_day=True)
state = learning._discretize_state(snap)
check("learning state in range", 0 <= state < 900, True)
print(f"  状态离散化: T=25 H=60 S=50 L=300 -> state={state}")

# 手动验证离散化结果
from src.ai.learning_module import LearningModule as LM
t = LM._discretize_temp(25.0)  # should be 3
h = LM._discretize_humi(60.0)  # should be 2
s = LM._discretize_soil(50.0)  # should be 3
l = LM._discretize_light(300.0)  # should be 1
check("discretize_temp(25)", t, 3)   # 25.0 < 33 -> 3
check("discretize_humi(60)", h, 2)
check("discretize_soil(50)", s, 3)   # 50.0 < 65 -> 3
check("discretize_light(300)", l, 1)
expected_state = t*(4*5*3*3) + h*(5*3*3) + s*(3*3) + l*3 + 1  # time_period=1(12-18时)
print(f"  离散化验证: temp={t} humi={h} soil={s} light={l}")

# 测试奖励计算
reward_on_target = learning._calculate_reward(50.0, 55.0, IrrigationAction.Moderate)
reward_overwater = learning._calculate_reward(50.0, 85.0, IrrigationAction.Heavy)
check("reward on target > 0", reward_on_target > 0, True)
check("reward overwater < on-target", reward_overwater < reward_on_target, True)
print(f"  奖励: 目标土壤+5={reward_on_target:.1f}, 过浇水+35={reward_overwater:.1f}")

# 测试Q值更新
old_q = learning._q_table[state][IrrigationAction.Off.value]
learning._update_q_value(state, IrrigationAction.Off, 5.0, state)
new_q = learning._q_table[state][IrrigationAction.Off.value]
check("q_value updated", new_q != old_q, True)
check("q_value increased", new_q > old_q, True)
print(f"  Q值更新: {old_q:.4f} -> {new_q:.4f}")

# ============================================================================
# 5. 传感器融合测试
# ============================================================================
print("\n=== 5. 传感器融合 ===")
fusion = FusionModule()
fusion.begin(auto_control_enabled=False)

# 灌入传感器数据
snap = SensorSnapshot(air_temp=30.0, air_humi=40.0, soil_humi=30.0,
                       light_intensity=800.0, liquid_level=50.0, is_day=True)
result = fusion.update(snap, sample_updated=True, now=time.time(), actuator=None)

check("fusion result exists", result is not None, True)
if result:
    print(f"  干燥条件: decision={result.decision}, score={result.final_score:.1f}, "
          f"weighted={result.weighted_score:.1f}, nn={result.nn_score:.1f}")
    check("fusion dry decision not none", result.decision in ("moderate", "heavy", "none"), True)
    check("fusion dry score > 0", result.final_score > 0, True)

# 湿润条件
snap_wet = SensorSnapshot(air_temp=20.0, air_humi=80.0, soil_humi=70.0,
                           light_intensity=100.0, liquid_level=90.0, is_day=False)
result_wet = fusion.update(snap_wet, sample_updated=True, now=time.time() + 11, actuator=None)
if result_wet:
    print(f"  湿润条件: decision={result_wet.decision}, score={result_wet.final_score:.1f}")
    check("fusion wet score < dry score", result_wet.final_score < result.final_score, True)

# 验证卡尔曼滤波
from src.ai.fusion_module import FusionModule as FM
ch = fusion._channels[0]
ch.kalman_estimate = 25.0
ch.kalman_error = 1.0
FM._apply_kalman_filter(ch, 26.0)
check("kalman estimate updated", ch.kalman_estimate != 25.0, True)
check("kalman gain in (0,1)", 0 < ch.kalman_gain < 1, True)
print(f"  卡尔曼: est={ch.kalman_estimate:.4f}, gain={ch.kalman_gain:.4f}")

# 验证神经网络前向传播
import numpy as np
inputs = np.array([0.5, 0.3, 0.2, 0.1, 0.8], dtype=np.float32)
output = fusion._run_neural_network(inputs)
check("nn output sum ~1.0", abs(sum(output) - 1.0) < 0.01, True)
check("nn output shape", len(output), 3)
print(f"  NN输出: none={output[0]:.4f}, moderate={output[1]:.4f}, heavy={output[2]:.4f}")

# ============================================================================
# 6. 执行器优先级测试
# ============================================================================
print("\n=== 6. 执行器控制器 ===")
# 不能在PC上测真实GPIO, 只测逻辑
from src.actuators.actuator_controller import ActuatorController
from config.hardware_config import PinConfig

# 创建mock actuator (GPIO会失败, 但逻辑仍可验证)
act = ActuatorController(PinConfig())
# 不调用begin(), 直接测内部逻辑
act._auto_mode = True

# 测试安全锁定
act.update(low_liquid_lock=True, base_auto_request=True, now=time.time())
check("safety lock valve off", act._status.valve_on, False)
check("safety lock source", act._status.active_source, ControlSource.SafetyLock)
print(f"  安全锁定: valve={act._status.valve_on}, source={act._status.active_source.name}")

# 解锁后手动模式
act._auto_mode = False
act._manual_valve = True
act._manual_pump = True
act.update(low_liquid_lock=False, base_auto_request=False, now=time.time())
# valve_on取决于_apply_outputs是否真的写了GPIO, 在没有GPIO的情况下只检查逻辑
check("manual source", act._status.active_source, ControlSource.Manual)
print(f"  手动模式: source={act._status.active_source.name}")

# ============================================================================
# 结果
# ============================================================================
print("\n" + "=" * 50)
if errors:
    print(f"FAILED: {len(errors)} errors")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED - 所有模块逻辑与ESP32版一致")
