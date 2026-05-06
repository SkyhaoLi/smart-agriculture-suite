#!/usr/bin/env python3
"""
智润对比实验框架 — 量化评估三种灌溉策略的效果。

用法:
    python benchmark.py [--steps N] [--time-scale N] [--output DIR]

三种模式:
    1. Rule Only      — 纯规则引擎
    2. Rule + Q-Learn — 规则 + Q-Learning 在线学习
    3. Rule + Fusion  — 规则 + 传感器融合 NN 决策

输出指标:
    - 总用水量 (灌溉时长秒数)
    - 土壤湿度标准差 (稳定性)
    - 土壤湿度在目标范围的时间占比
    - 平均响应延迟 (从土壤低于阈值到开始灌溉)
    - 累计奖励 (Q-Learning)
"""

import argparse
import os
import sys
import json
import math
import time
import random
import numpy as np
from pathlib import Path

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator.time_clock import SimClock
from simulator.sensor_hub import SensorHub, SensorSnapshot
from simulator.irrigation import IrrigationModule
from simulator.actuator import ActuatorController, ControlSource
from simulator.learning import LearningModule
from simulator.fusion import FusionModule


# ── Metrics Collector ────────────────────────────────────────────────

class MetricsCollector:
    """Collects simulation metrics for comparison."""

    def __init__(self, target_soil: float = 55.0, tolerance: float = 10.0):
        self.target_soil = target_soil
        self.tolerance = tolerance
        self.reset()

    def reset(self):
        self.soil_history = []
        self.water_on_steps = 0
        self.total_steps = 0
        self.response_delays = []
        self._below_since = None
        self._total_reward = 0.0

    def record(self, soil: float, water_on: bool, reward: float = 0.0):
        self.total_steps += 1
        self.soil_history.append(soil)
        if water_on:
            self.water_on_steps += 1
        self._total_reward += reward

        # Track response delay
        low_threshold = self.target_soil - self.tolerance
        if soil < low_threshold:
            if self._below_since is None:
                self._below_since = self.total_steps
            if water_on and self._below_since is not None:
                delay = self.total_steps - self._below_since
                self.response_delays.append(delay)
                self._below_since = None
        else:
            self._below_since = None

    def report(self) -> dict:
        soil = np.array(self.soil_history) if self.soil_history else np.array([0])
        in_range = np.sum(
            (soil >= self.target_soil - self.tolerance) &
            (soil <= self.target_soil + self.tolerance)
        )
        in_range_pct = in_range / len(soil) * 100 if len(soil) > 0 else 0

        return {
            "total_water_seconds": self.water_on_steps * 2,  # 2s per step
            "soil_mean": float(np.mean(soil)),
            "soil_std": float(np.std(soil)),
            "soil_min": float(np.min(soil)),
            "soil_max": float(np.max(soil)),
            "in_range_pct": round(in_range_pct, 1),
            "avg_response_delay": (
                round(np.mean(self.response_delays), 1)
                if self.response_delays else 0
            ),
            "max_response_delay": (
                max(self.response_delays) if self.response_delays else 0
            ),
            "total_reward": round(self._total_reward, 2),
            "total_steps": self.total_steps,
        }


# ── Simulation Runner ────────────────────────────────────────────────

def run_simulation(mode: str, steps: int, time_scale: float,
                   seed: int = 42) -> dict:
    """Run a single simulation with the given mode.

    Args:
        mode: "rule_only", "rule_learn", or "rule_fusion"
        steps: number of simulation steps
        time_scale: simulation speed multiplier
        seed: random seed for reproducibility

    Returns:
        dict with metrics and time series data
    """
    random.seed(seed)
    np.random.seed(seed)

    clock = SimClock()
    clock.set_time_scale(time_scale)

    sensor_hub = SensorHub(clock)
    irrigation = IrrigationModule()
    actuator = ActuatorController()
    learning = LearningModule(clock)
    fusion = FusionModule()

    # Configure based on mode
    learning_cfg = learning.config
    learning_cfg.autoControlEnabled = (mode in ("rule_learn",))
    fusion_auto = (mode in ("rule_fusion",))

    learning.begin(learning_cfg.to_dict())
    fusion.begin(fusion_auto)

    metrics = MetricsCollector(
        target_soil=learning_cfg.targetSoil,
        tolerance=learning_cfg.soilTolerance,
    )

    # Time series for plotting
    soil_ts = []
    water_ts = []
    reward_ts = []

    print(f"  Running {mode} ({steps} steps, {time_scale}x speed)...")

    for step in range(steps):
        now_ms = clock.millis()

        # Update sensors
        sample_updated = sensor_hub.update()
        snap = sensor_hub.snapshot

        # Irrigation rule engine
        if sample_updated:
            irrigation.update(snap)

        # Actuator update
        actuator.update(
            irrigation.liquid_warn,
            irrigation.enabled and irrigation.should_water,
            now_ms,
        )

        # Learning & Fusion
        learning.update(snap, sample_updated, now_ms, actuator)
        fusion.update(snap, sample_updated, now_ms, actuator)

        # Determine if water is on
        water_on = actuator.status.valveOn or actuator.status.pumpOn

        # Record metrics
        reward = learning.last_reward if hasattr(learning, 'last_reward') else 0
        metrics.record(snap.soilHumi, water_on, reward)

        # Time series (sample every 10 steps to keep data manageable)
        if step % 10 == 0:
            soil_ts.append(round(snap.soilHumi, 2))
            water_ts.append(1 if water_on else 0)
            reward_ts.append(round(reward, 4))

        # Tick clock
        clock.tick(2000)  # 2s per step

    report = metrics.report()
    report["mode"] = mode
    report["time_series"] = {
        "soil": soil_ts,
        "water": water_ts,
        "reward": reward_ts,
    }
    return report


# ── Report Generation ─────────────────────────────────────────────────

def generate_comparison(results: list, output_dir: str):
    """Generate comparison table and charts."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Console table ──
    print("\n" + "=" * 80)
    print("  智润灌溉策略对比实验结果")
    print("=" * 80)

    headers = [
        "模式", "用水(s)", "土壤均值", "土壤标准差",
        "目标范围%", "平均响应延迟", "累计奖励"
    ]
    print(f"  {'  '.join(f'{h:>10}' for h in headers)}")
    print("  " + "-" * 76)

    mode_names = {
        "rule_only": "纯规则",
        "rule_learn": "规则+Q-Learn",
        "rule_fusion": "规则+融合",
    }

    for r in results:
        name = mode_names.get(r["mode"], r["mode"])
        print(f"  {name:>10}  {r['total_water_seconds']:>8}  "
              f"{r['soil_mean']:>8.1f}  {r['soil_std']:>10.2f}  "
              f"{r['in_range_pct']:>8.1f}%  {r['avg_response_delay']:>10.1f}  "
              f"{r['total_reward']:>10.2f}")

    print("=" * 80)

    # ── Save JSON ──
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {json_path}")

    # ── Generate HTML chart ──
    html_path = os.path.join(output_dir, "benchmark_report.html")
    _generate_html_report(results, html_path)
    print(f"  报告已生成: {html_path}")


def _generate_html_report(results: list, path: str):
    """Generate an HTML report with Chart.js visualizations."""
    mode_names = {
        "rule_only": "纯规则引擎",
        "rule_learn": "规则 + Q-Learning",
        "rule_fusion": "规则 + 融合决策",
    }
    colors = {
        "rule_only": "#ef5350",
        "rule_learn": "#42a5f5",
        "rule_fusion": "#66bb6a",
    }

    # Prepare chart data
    soil_datasets = []
    water_datasets = []
    for r in results:
        ts = r["time_series"]
        name = mode_names.get(r["mode"], r["mode"])
        color = colors.get(r["mode"], "#999")
        soil_datasets.append({
            "label": name,
            "data": ts["soil"],
            "borderColor": color,
            "fill": False,
            "pointRadius": 0,
            "borderWidth": 2,
        })
        water_datasets.append({
            "label": name,
            "data": ts["water"],
            "borderColor": color,
            "fill": False,
            "pointRadius": 0,
            "borderWidth": 1.5,
            "borderDash": [5, 3],
        })

    # Bar chart data
    bar_labels = [mode_names.get(r["mode"], r["mode"]) for r in results]
    bar_colors = [colors.get(r["mode"], "#999") for r in results]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>智润 - 灌溉策略对比实验报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
body{{font-family:-apple-system,sans-serif;background:#0f1923;color:#e0e6ed;padding:20px;max-width:1000px;margin:0 auto}}
h1{{color:#4fc3f7;text-align:center}}
.card{{background:#1a2736;border:1px solid #2d3f52;border-radius:8px;padding:16px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;margin:10px 0}}
th,td{{padding:8px 12px;text-align:center;border-bottom:1px solid #2d3f52}}
th{{color:#7a8b9a}}
.winner{{color:#66bb6a;font-weight:700}}
canvas{{max-height:300px}}
</style>
</head>
<body>
<h1>智润 - 灌溉策略对比实验报告</h1>

<div class="card">
<h3 style="color:#4fc3f7">定量指标对比</h3>
<table>
<tr><th>指标</th>{"".join(f"<th>{mode_names.get(r['mode'], r['mode'])}</th>" for r in results)}</tr>
<tr><td>总用水量 (秒)</td>{"".join(f"<td>{r['total_water_seconds']}</td>" for r in results)}</tr>
<tr><td>土壤湿度均值 (%)</td>{"".join(f"<td>{r['soil_mean']:.1f}</td>" for r in results)}</tr>
<tr><td>土壤湿度标准差</td>{"".join(f'<td class="winner">{r["soil_std"]:.2f}</td>' if r == min(results, key=lambda x: x['soil_std']) else f'<td>{r["soil_std"]:.2f}</td>' for r in results)}</tr>
<tr><td>目标范围时间占比</td>{"".join(f'<td class="winner">{r["in_range_pct"]}%</td>' if r == max(results, key=lambda x: x['in_range_pct']) else f'<td>{r["in_range_pct"]}%</td>' for r in results)}</tr>
<tr><td>平均响应延迟 (步)</td>{"".join(f"<td>{r['avg_response_delay']}</td>" for r in results)}</tr>
<tr><td>累计奖励</td>{"".join(f"<td>{r['total_reward']}</td>" for r in results)}</tr>
</table>
</div>

<div class="card">
<h3 style="color:#4fc3f7">土壤湿度变化曲线</h3>
<canvas id="soilChart"></canvas>
</div>

<div class="card">
<h3 style="color:#4fc3f7">灌溉动作时间轴</h3>
<canvas id="waterChart"></canvas>
</div>

<div class="card">
<h3 style="color:#4fc3f7">用水量 & 土壤稳定性对比</h3>
<canvas id="barChart"></canvas>
</div>

<script>
const soilData = {json.dumps(soil_datasets)};
const waterData = {json.dumps(water_datasets)};
const barLabels = {json.dumps(bar_labels)};
const barColors = {json.dumps(bar_colors)};

new Chart(document.getElementById('soilChart'), {{
  type: 'line',
  data: {{ labels: Array.from({{length: soilData[0].data.length}}, (_, i) => i * 2 + 's'), datasets: soilData }},
  options: {{ scales: {{ y: {{ title: {{ display: true, text: '土壤湿度 (%)' }} }} }}, plugins: {{ annotation: {{ annotations: {{ target: {{ type: 'line', yMin: 55, yMax: 55, borderColor: '#ffa726', borderDash: [6,3], label: {{ content: '目标 55%', display: true }} }} }} }} }} }} }}
}});

new Chart(document.getElementById('waterChart'), {{
  type: 'line',
  data: {{ labels: Array.from({{length: waterData[0].data.length}}, (_, i) => i * 2 + 's'), datasets: waterData }},
  options: {{ scales: {{ y: {{ min: -0.1, max: 1.1, ticks: {{ callback: v => v > 0.5 ? 'ON' : 'OFF' }} }} }} }} }}
}});

new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: barLabels,
    datasets: [
      {{ label: '用水量 (秒)', data: {json.dumps([r['total_water_seconds'] for r in results])}, backgroundColor: barColors.map(c => c + '80') }},
      {{ label: '土壤标准差 ×10', data: {json.dumps([round(r['soil_std'] * 10, 1) for r in results])}, backgroundColor: barColors.map(c => c + '40') }},
    ]
  }}
}});
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="智润灌溉策略对比实验")
    parser.add_argument("--steps", type=int, default=500,
                        help="每种模式的模拟步数 (默认 500)")
    parser.add_argument("--time-scale", type=float, default=60,
                        help="模拟加速倍率 (默认 60)")
    parser.add_argument("--output", default="benchmark_output",
                        help="输出目录 (默认 benchmark_output)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    args = parser.parse_args()

    print("=" * 50)
    print("  智润灌溉策略对比实验")
    print(f"  步数: {args.steps} | 加速: {args.time_scale}x | 种子: {args.seed}")
    print("=" * 50)

    results = []
    for mode in ["rule_only", "rule_learn", "rule_fusion"]:
        result = run_simulation(mode, args.steps, args.time_scale, args.seed)
        results.append(result)

    generate_comparison(results, args.output)

    print("\n  实验完成!")


if __name__ == "__main__":
    main()
