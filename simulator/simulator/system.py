"""System — main orchestrator replicating firmware loop() timing."""

import time
import threading
import queue
from .time_clock import SimClock
from .sensor_hub import SensorHub
from .irrigation import IrrigationModule
from .actuator import ActuatorController
from .anomaly import AnomalyModule
from .growth import GrowthModule
from .learning import LearningModule
from .fusion import FusionModule
from .plant_doctor import PlantDoctorModule
from .config_manager import ConfigManager


class AgriSystem:
    """Orchestrates all modules, replicating main.cpp loop() timing."""

    def __init__(self, model_path: str = "", data_dir: str = ""):
        self._clock = SimClock()
        self._sensorHub = SensorHub(self._clock)
        self._irrigation = IrrigationModule()
        self._actuator = ActuatorController()
        self._anomaly = AnomalyModule()
        self._growth = GrowthModule(self._clock)
        self._learning = LearningModule(self._clock)
        self._fusion = FusionModule()
        self._plantDoctor = PlantDoctorModule(model_path)
        self._configManager = ConfigManager(data_dir)

        self._running = False
        self._thread = None
        self._cmdQueue = queue.Queue()

        # Expose for API routes
        self.clock = self._clock
        self.sensor_hub = self._sensorHub
        self.irrigation = self._irrigation
        self.actuator = self._actuator
        self.anomaly = self._anomaly
        self.growth = self._growth
        self.learning = self._learning
        self.fusion = self._fusion
        self.plant_doctor = self._plantDoctor
        self.config_manager = self._configManager

    def begin(self, time_scale: float = 1.0):
        """Initialize all modules — matches setup()."""
        self._clock.set_time_scale(time_scale)

        # Load persisted config
        self._configManager.load()
        config = self._configManager.config

        # Irrigation — persisted config is flat-key, convert to nested for update_config
        if "irrigation" in config:
            ic = config["irrigation"]
            nested = {
                "day": {
                    "airTemp": ic.get("dayAirTempThreshold"),
                    "airHumi": ic.get("dayAirHumiThreshold"),
                    "soilHumi": ic.get("daySoilHumiThreshold"),
                },
                "night": {
                    "airTemp": ic.get("nightAirTempThreshold"),
                    "airHumi": ic.get("nightAirHumiThreshold"),
                    "soilHumi": ic.get("nightSoilHumiThreshold"),
                },
            }
            self._irrigation.update_config(nested)
        if "system" in config:
            sys_cfg = config["system"]
            if "ruleEngineEnabled" in sys_cfg:
                self._irrigation.enabled = sys_cfg["ruleEngineEnabled"]

        # Learning
        if "learning" in config:
            self._learning.begin(config["learning"])

        # Fusion
        if "system" in config:
            fusion_auto = config["system"].get("fusionAutoEnabled", False)
            self._fusion.begin(fusion_auto)
        if "fusion" in config:
            self._fusion.update_weights(config["fusion"])

        # Growth — default crop index 0
        self._growth.begin(0)

        # Plant doctor
        pd_config = config.get("plantDoctor", {})
        self._plantDoctor.begin(pd_config)

        # Sensor hub — initializes in __init__
        print("[System] all modules initialized")

    def start(self):
        """Start the simulation loop in a daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[System] simulation loop started")

    def stop(self):
        """Stop the simulation loop and save state."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._save_state()
        print("[System] simulation stopped")

    def _loop(self):
        """Main simulation loop — replicates loop() from main.cpp."""
        last_real_time = time.monotonic()

        while self._running:
            # Process any pending commands from API thread
            self._process_commands()

            now_ms = self._clock.millis()

            # Sensor sampling
            sample_updated = self._sensorHub.update()

            # Irrigation — only on new sample
            if sample_updated:
                self._irrigation.update(self._sensorHub.snapshot)

            # Actuator — first update (matches firmware: called twice per loop)
            self._actuator.update(
                self._irrigation.liquid_warn,
                self._irrigation.enabled and self._irrigation.should_water,
                now_ms,
            )

            # Anomaly, Growth, Learning, Fusion, PlantDoctor
            self._anomaly.update(self._sensorHub.snapshot, sample_updated, now_ms)
            self._growth.update(self._sensorHub.snapshot, sample_updated, now_ms)
            self._learning.update(self._sensorHub.snapshot, sample_updated, now_ms, self._actuator)
            self._fusion.update(self._sensorHub.snapshot, sample_updated, now_ms, self._actuator)
            self._plantDoctor.update(now_ms, self._sensorHub.snapshot.lightValue)

            # Actuator — second update (matches firmware)
            self._actuator.update(
                self._irrigation.liquid_warn,
                self._irrigation.enabled and self._irrigation.should_water,
                now_ms,
            )

            # Feedback: tell sensor hub whether watering is active
            self._sensorHub.set_watering_active(self._actuator.status.valveOn)

            # Real-time pacing — 100ms interval in wall-clock time
            now_real = time.monotonic()
            elapsed = now_real - last_real_time
            last_real_time = now_real

            # Target ~10 iterations/second (matches firmware delay(10))
            sleep_time = max(0.0, 0.1 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_commands(self):
        """Process commands from the API thread queue."""
        while not self._cmdQueue.empty():
            try:
                cmd = self._cmdQueue.get_nowait()
                self._dispatch_command(cmd)
            except queue.Empty:
                break

    def _dispatch_command(self, cmd: dict):
        """Execute a command from the API thread."""
        action = cmd.get("action", "")
        try:
            if action == "set_auto_mode":
                self._actuator.set_auto_mode(cmd["enabled"])
            elif action == "set_manual_combined":
                self._actuator.set_manual_combined(cmd["enabled"])
            elif action == "set_manual_valve":
                self._actuator.set_manual_valve(cmd["enabled"])
            elif action == "set_manual_pump":
                self._actuator.set_manual_pump(cmd["enabled"])
            elif action == "stop_timed_run":
                self._actuator.stop_timed_run()
            elif action == "set_irrigation_enabled":
                self._irrigation.enabled = cmd["enabled"]
            elif action == "update_irrigation_config":
                self._irrigation.update_config(cmd["config"])
            elif action == "set_crop":
                self._growth.set_crop(cmd["cropId"])
            elif action == "reset_growth":
                self._growth.reset()
            elif action == "set_learning_config":
                self._learning.set_config(cmd["config"])
            elif action == "record_feedback":
                self._learning.record_user_feedback(cmd["positive"])
            elif action == "reset_learning":
                self._learning.reset()
            elif action == "set_fusion_auto":
                self._fusion.set_auto_control_enabled(cmd["enabled"])
            elif action == "update_fusion_weights":
                self._fusion.update_weights(cmd["weights"])
            elif action == "set_plant_doctor_config":
                self._plantDoctor.set_config(cmd["config"])
            elif action == "clear_anomaly":
                self._anomaly.clear()
            elif action == "inject_sensor":
                self._sensorHub.inject(cmd["values"])
            elif action == "clear_inject":
                self._sensorHub.clear_inject()
            elif action == "set_time_scale":
                self._clock.set_time_scale(cmd["scale"])
            elif action == "factory_reset":
                self._configManager.factory_reset()
                # Re-apply defaults
                from simulator.learning import LearningConfig
                from simulator.plant_doctor import PlantDoctorConfig
                self._learning.begin(LearningConfig().to_dict())
                self._growth.reset()
                self._anomaly.clear()
                self._fusion.begin(False)
                self._plantDoctor.set_config(PlantDoctorConfig())
                self._irrigation = IrrigationModule()
                self.irrigation = self._irrigation
        except Exception as e:
            print(f"[System] command error: {action}: {e}")

    # ── Public API — thread-safe command submission ──

    def submit(self, cmd: dict):
        """Submit a command from the API thread."""
        self._cmdQueue.put(cmd)

    def _save_state(self):
        """Persist current state to JSON."""
        try:
            config = self._configManager.config
            try:
                config["irrigation"] = self._irrigation.config
            except Exception as e:
                print(f"[System] save irrigation config error: {e}")
            try:
                config["learning"] = self._learning.config.to_dict()
            except Exception as e:
                print(f"[System] save learning config error: {e}")
            try:
                config["plantDoctor"] = self._plantDoctor.config.to_dict()
            except Exception as e:
                print(f"[System] save plantDoctor config error: {e}")
            try:
                config["system"] = {
                    "ruleEngineEnabled": self._irrigation.enabled,
                    "fusionAutoEnabled": self._fusion.auto_control_enabled,
                }
            except Exception as e:
                print(f"[System] save system config error: {e}")
            try:
                config["fusion"] = self._fusion.get_weights()
            except Exception as e:
                print(f"[System] save fusion config error: {e}")
            self._configManager.config = config
            self._configManager.save()
        except Exception as e:
            print(f"[System] save state error: {e}")

    def overall_status(self) -> dict:
        """Build combined /api/status response — matches buildOverallStatus()."""
        snapshot = self._sensorHub.snapshot
        now_ms = self._clock.millis()

        return {
            "project": "smart-agriculture-simulator",
            "hardwareProfile": "PC-Simulator",
            "simTimeScale": self._clock.time_scale,
            "simUptime": round(self._clock.uptime_seconds(), 1),
            "sensors": {
                "airTemp": round(snapshot.airTemp, 2),
                "airHumi": round(snapshot.airHumi, 2),
                "soilHumi": round(snapshot.soilHumi, 2),
                "liquidLevel": round(snapshot.liquidLevel, 2),
                "lightValue": round(snapshot.lightValue, 2),
                "updatedAtMs": snapshot.updatedAtMs,
            },
            "actuator": self._actuator.status.to_dict(now_ms),
            "modules": {
                "irrigation": self._irrigation.status(),
                "anomaly": self._anomaly.status(),
                "growth": self._growth.status(),
                "learning": self._learning.status(),
                "fusion": self._fusion.status(),
                "plantDoctor": self._plantDoctor.status(),
            },
        }
