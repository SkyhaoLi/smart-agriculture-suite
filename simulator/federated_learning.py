#!/usr/bin/env python3
"""
智润联邦学习模拟框架 — 多设备 Q-Table 联邦平均。

模拟多个边缘设备独立学习灌溉策略，定期通过 FedAvg 聚合 Q-Table，
展示联邦学习相比单设备独立学习的提升效果。

用法:
    python federated_learning.py [--devices N] [--rounds R] [--steps S] [--output DIR]

核心流程:
    1. N 个虚拟设备各自在不同环境条件下独立运行 Q-Learning
    2. 每 R 轮决策后，上传 Q-Table 到聚合服务器
    3. 服务器执行 FedAvg: Q_avg = mean(Q_i)
    4. 下发聚合后的 Q-Table 给所有设备
    5. 对比联邦学习 vs 独立学习的效果差异
"""

import argparse
import os
import sys
import json
import math
import random
import numpy as np
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator.time_clock import SimClock
from simulator.sensor_hub import SensorHub
from simulator.irrigation import IrrigationModule
from simulator.actuator import ActuatorController, ControlSource
from simulator.learning import (
    LearningModule, LearningConfig,
    K_STATE_COUNT, ACTION_COUNT, ACTION_NAMES, ACTION_DURATIONS,
    OFF, LOW, MEDIUM, HIGH,
)


# ── Virtual Device ────────────────────────────────────────────────────

class VirtualDevice:
    """Simulates one edge device with its own environment and Q-Learning."""

    def __init__(self, device_id: int, env_seed: int, config: LearningConfig = None):
        self.device_id = device_id
        self.clock = SimClock()
        self.clock.set_time_scale(60)
        # Seed before creating SensorHub so OU processes get different noise
        random.seed(env_seed)
        np.random.seed(env_seed)
        self.sensor_hub = SensorHub(self.clock)
        self.irrigation = IrrigationModule()
        self.actuator = ActuatorController()
        self.learning = LearningModule(self.clock)

        cfg = config or LearningConfig()
        cfg.autoControlEnabled = True
        self.learning.begin(cfg.to_dict())

        self.metrics = DeviceMetrics(device_id)

    def step(self):
        """Run one simulation step."""
        now_ms = self.clock.millis()
        sample_updated = self.sensor_hub.update()
        snap = self.sensor_hub.snapshot

        if sample_updated:
            self.irrigation.update(snap)

        self.actuator.update(
            self.irrigation.liquid_warn,
            self.irrigation.enabled and self.irrigation.should_water,
            now_ms,
        )
        self.learning.update(snap, sample_updated, now_ms, self.actuator)

        water_on = self.actuator.status.valveOn or self.actuator.status.pumpOn
        self.metrics.record(snap.soilHumi, water_on, self.learning.config.epsilon)
        self.clock.tick(2000)

    def get_qtable(self) -> np.ndarray:
        """Get a copy of the Q-Table."""
        return self.learning._qTable.copy()

    def set_qtable(self, qtable: np.ndarray):
        """Set the Q-Table (from aggregation)."""
        self.learning._qTable = qtable.copy()


# ── Device Metrics ────────────────────────────────────────────────────

class DeviceMetrics:
    """Collects per-device metrics for comparison."""

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.target_soil = 55.0
        self.tolerance = 10.0
        self.soil_history = []
        self.water_steps = 0
        self.total_steps = 0
        self.reward_history = []

    def record(self, soil: float, water_on: bool, epsilon: float = 0):
        self.total_steps += 1
        self.soil_history.append(soil)
        if water_on:
            self.water_steps += 1

    def in_range_pct(self) -> float:
        if not self.soil_history:
            return 0
        arr = np.array(self.soil_history)
        in_range = np.sum(
            (arr >= self.target_soil - self.tolerance) &
            (arr <= self.target_soil + self.tolerance)
        )
        return round(in_range / len(arr) * 100, 1)

    def soil_std(self) -> float:
        if not self.soil_history:
            return 0
        return round(float(np.std(self.soil_history)), 3)

    def soil_mean(self) -> float:
        if not self.soil_history:
            return 0
        return round(float(np.mean(self.soil_history)), 1)

    def report(self) -> dict:
        return {
            "device_id": self.device_id,
            "total_steps": self.total_steps,
            "water_seconds": self.water_steps * 2,
            "soil_mean": self.soil_mean(),
            "soil_std": self.soil_std(),
            "in_range_pct": self.in_range_pct(),
        }


# ── FedAvg Aggregator ────────────────────────────────────────────────

def fedavg(qtables: list) -> np.ndarray:
    """Federated Averaging: element-wise mean of Q-Tables."""
    stacked = np.stack(qtables, axis=0)
    return np.mean(stacked, axis=0).astype(np.float32)


# ── Simulation Runner ────────────────────────────────────────────────

def run_federated_simulation(
    num_devices: int = 3,
    total_rounds: int = 10,
    steps_per_round: int = 100,
    seed: int = 42,
) -> dict:
    """Run federated learning simulation.

    Returns dict with per-round metrics for federated and independent groups.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create federated devices (different environment seeds)
    fed_devices = [
        VirtualDevice(i, env_seed=seed + i * 1000)
        for i in range(num_devices)
    ]
    # Create independent devices (same env seeds, no aggregation)
    ind_devices = [
        VirtualDevice(i + num_devices, env_seed=seed + i * 1000)
        for i in range(num_devices)
    ]

    round_metrics = []

    for r in range(total_rounds):
        # Run steps_per_round for all devices
        for _ in range(steps_per_round):
            for d in fed_devices + ind_devices:
                d.step()

        # FedAvg aggregation for federated group
        qtables = [d.get_qtable() for d in fed_devices]
        avg_qtable = fedavg(qtables)
        for d in fed_devices:
            d.set_qtable(avg_qtable)

        # Collect round metrics
        fed_reports = [d.metrics.report() for d in fed_devices]
        ind_reports = [d.metrics.report() for d in ind_devices]

        round_metrics.append({
            "round": r + 1,
            "fed_avg_std": round(np.mean([r["soil_std"] for r in fed_reports]), 3),
            "ind_avg_std": round(np.mean([r["soil_std"] for r in ind_reports]), 3),
            "fed_avg_in_range": round(np.mean([r["in_range_pct"] for r in fed_reports]), 1),
            "ind_avg_in_range": round(np.mean([r["in_range_pct"] for r in ind_reports]), 1),
            "fed_avg_water": round(np.mean([r["water_seconds"] for r in fed_reports]), 0),
            "ind_avg_water": round(np.mean([r["water_seconds"] for r in ind_reports]), 0),
            "fed_devices": fed_reports,
            "ind_devices": ind_reports,
        })

        print(f"  Round {r+1}/{total_rounds} — "
              f"Fed σ={round_metrics[-1]['fed_avg_std']:.2f} "
              f"Ind σ={round_metrics[-1]['ind_avg_std']:.2f} "
              f"Fed InRange={round_metrics[-1]['fed_avg_in_range']:.0f}% "
              f"Ind InRange={round_metrics[-1]['ind_avg_in_range']:.0f}%")

    return {
        "num_devices": num_devices,
        "total_rounds": total_rounds,
        "steps_per_round": steps_per_round,
        "seed": seed,
        "rounds": round_metrics,
    }


# ── Report Generation ─────────────────────────────────────────────────

def generate_report(result: dict, output_dir: str):
    """Generate HTML comparison report."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(output_dir, "federated_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {json_path}")

    rounds = result["rounds"]
    round_nums = [r["round"] for r in rounds]

    html_path = os.path.join(output_dir, "federated_report.html")
    _generate_html(result, html_path)
    print(f"  报告已生成: {html_path}")


def _generate_html(result: dict, path: str):
    rounds = result["rounds"]
    round_nums = [r["round"] for r in rounds]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>智润 - 联邦学习对比实验报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
body{{font-family:-apple-system,sans-serif;background:#0f1923;color:#e0e6ed;padding:20px;max-width:1000px;margin:0 auto}}
h1{{color:#4fc3f7;text-align:center}}
h2{{color:#4fc3f7;margin-top:30px}}
.card{{background:#1a2736;border:1px solid #2d3f52;border-radius:8px;padding:16px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;margin:10px 0}}
th,td{{padding:8px 12px;text-align:center;border-bottom:1px solid #2d3f52}}
th{{color:#7a8b9a}}
.winner{{color:#66bb6a;font-weight:700}}
canvas{{max-height:300px}}
.summary{{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;margin:20px 0}}
.stat-box{{background:#1a2736;border:1px solid #2d3f52;border-radius:8px;padding:16px;min-width:150px;text-align:center}}
.stat-box .val{{font-size:28px;font-weight:700;color:#4fc3f7}}
.stat-box .lbl{{font-size:12px;color:#7a8b9a;margin-top:4px}}
</style>
</head>
<body>
<h1>智润 - 联邦学习对比实验报告</h1>
<p style="text-align:center;color:#7a8b9a">
{result['num_devices']} 个虚拟设备 | {result['total_rounds']} 轮聚合 | 每轮 {result['steps_per_round']} 步 | 种子 {result['seed']}
</p>

<div class="summary">
<div class="stat-box"><div class="val">{rounds[-1]['fed_avg_in_range']}%</div><div class="lbl">联邦学习 目标范围占比</div></div>
<div class="stat-box"><div class="val">{rounds[-1]['ind_avg_in_range']}%</div><div class="lbl">独立学习 目标范围占比</div></div>
<div class="stat-box"><div class="val">{rounds[-1]['fed_avg_std']:.3f}</div><div class="lbl">联邦学习 土壤标准差</div></div>
<div class="stat-box"><div class="val">{rounds[-1]['ind_avg_std']:.3f}</div><div class="lbl">独立学习 土壤标准差</div></div>
</div>

<div class="card">
<h3 style="color:#4fc3f7">土壤湿度标准差收敛曲线</h3>
<canvas id="stdChart"></canvas>
<div style="font-size:11px;color:#7a8b9a;margin-top:4px">标准差越低表示灌溉越稳定</div>
</div>

<div class="card">
<h3 style="color:#4fc3f7">目标范围时间占比</h3>
<canvas id="rangeChart"></canvas>
</div>

<div class="card">
<h3 style="color:#4fc3f7">用水量对比</h3>
<canvas id="waterChart"></canvas>
</div>

<div class="card">
<h3 style="color:#4fc3f7">逐轮详细数据</h3>
<table>
<tr><th>轮次</th><th>Fed σ</th><th>Ind σ</th><th>Fed InRange</th><th>Ind InRange</th><th>Fed 水量</th><th>Ind 水量</th></tr>
{"".join(f'<tr><td>{r["round"]}</td><td>{r["fed_avg_std"]:.3f}</td><td>{r["ind_avg_std"]:.3f}</td><td>{r["fed_avg_in_range"]}%</td><td>{r["ind_avg_in_range"]}%</td><td>{r["fed_avg_water"]:.0f}s</td><td>{r["ind_avg_water"]:.0f}s</td></tr>' for r in rounds)}
</table>
</div>

<script>
const rounds = {json.dumps(round_nums)};
const fedStd = {json.dumps([r['fed_avg_std'] for r in rounds])};
const indStd = {json.dumps([r['ind_avg_std'] for r in rounds])};
const fedRange = {json.dumps([r['fed_avg_in_range'] for r in rounds])};
const indRange = {json.dumps([r['ind_avg_in_range'] for r in rounds])};
const fedWater = {json.dumps([r['fed_avg_water'] for r in rounds])};
const indWater = {json.dumps([r['ind_avg_water'] for r in rounds])};

const gridColor = '#2d3f52';
const tickColor = '#7a8b9a';
const commonOpts = {{
  responsive: true,
  animation: false,
  scales: {{
    x: {{ticks:{{color:tickColor}}, grid:{{color:gridColor}}, title:{{display:true,text:'聚合轮次',color:tickColor}}}},
    y: {{ticks:{{color:tickColor}}, grid:{{color:gridColor}}}},
  }},
  plugins: {{legend: {{labels: {{color: tickColor}}}}}}
}};

new Chart(document.getElementById('stdChart'), {{
  type: 'line',
  data: {{
    labels: rounds,
    datasets: [
      {{label:'联邦学习(FedAvg)', data:fedStd, borderColor:'#42a5f5', borderWidth:2, pointRadius:3, tension:0.3}},
      {{label:'独立学习', data:indStd, borderColor:'#ef5350', borderWidth:2, pointRadius:3, tension:0.3}},
    ]
  }},
  options: {{...commonOpts, scales:{{...commonOpts.scales, y:{{...commonOpts.scales.y, title:{{display:true,text:'土壤湿度标准差',color:tickColor}}}}}}}}
}});

new Chart(document.getElementById('rangeChart'), {{
  type: 'line',
  data: {{
    labels: rounds,
    datasets: [
      {{label:'联邦学习(FedAvg)', data:fedRange, borderColor:'#66bb6a', borderWidth:2, pointRadius:3, tension:0.3, fill:true, backgroundColor:'rgba(102,187,106,0.1)'}},
      {{label:'独立学习', data:indRange, borderColor:'#ffa726', borderWidth:2, pointRadius:3, tension:0.3}},
    ]
  }},
  options: {{...commonOpts, scales:{{...commonOpts.scales, y:{{...commonOpts.scales.y, min:0, max:100, title:{{display:true,text:'目标范围占比(%)',color:tickColor}}}}}}}}
}});

new Chart(document.getElementById('waterChart'), {{
  type: 'bar',
  data: {{
    labels: rounds,
    datasets: [
      {{label:'联邦学习', data:fedWater, backgroundColor:'rgba(66,165,245,0.6)'}},
      {{label:'独立学习', data:indWater, backgroundColor:'rgba(239,83,80,0.6)'}},
    ]
  }},
  options: {{...commonOpts, scales:{{...commonOpts.scales, y:{{...commonOpts.scales.y, title:{{display:true,text:'用水量(秒)',color:tickColor}}}}}}}}
}});
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="智润联邦学习模拟框架")
    parser.add_argument("--devices", type=int, default=3, help="虚拟设备数量 (默认 3)")
    parser.add_argument("--rounds", type=int, default=10, help="聚合轮数 (默认 10)")
    parser.add_argument("--steps", type=int, default=100, help="每轮步数 (默认 100)")
    parser.add_argument("--output", default="federated_output", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=" * 50)
    print("  智润联邦学习模拟框架")
    print(f"  设备: {args.devices} | 轮数: {args.rounds} | 每轮步数: {args.steps}")
    print("=" * 50)

    result = run_federated_simulation(
        num_devices=args.devices,
        total_rounds=args.rounds,
        steps_per_round=args.steps,
        seed=args.seed,
    )

    generate_report(result, args.output)
    print("\n  实验完成!")


if __name__ == "__main__":
    main()
