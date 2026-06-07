"""ConfigManager — JSON file persistence replacing ESP32 NVS Preferences."""

import json
import os
import threading
import copy

DEFAULT_CONFIG = {
    "irrigation": {
        "liquidLevelThreshold": 30.0,
        "lightDayThreshold": 200.0,
        "dayAirTempThreshold": 20.0,
        "dayAirHumiThreshold": 60.0,
        "daySoilHumiThreshold": 50.0,
        "nightAirTempThreshold": 15.0,
        "nightAirHumiThreshold": 70.0,
        "nightSoilHumiThreshold": 45.0,
    },
    "learning": {
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon": 0.3,
        "epsilonDecay": 0.999,
        "epsilonMin": 0.05,
        "targetSoil": 55.0,
        "soilTolerance": 10.0,
        "decisionIntervalMs": 300000,
        "autoControlEnabled": False,
    },
    "plantDoctor": {
        "enabled": True,
        "autoDetect": True,
        "detectIntervalSec": 30,
        "confidenceThreshold": 0.70,
        "buzzerEnabled": True,
    },
    "system": {
        "ruleEngineEnabled": True,
        "fusionAutoEnabled": False,
    },
    "wifi": {
        "ssid": "",
        "password": "",
    },
}


class ConfigManager:
    """Thread-safe JSON file persistence."""

    def __init__(self, data_dir: str):
        self._path = os.path.join(data_dir, "config.json")
        self._lock = threading.Lock()
        self._config = copy.deepcopy(DEFAULT_CONFIG)
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    saved = json.load(f)
                self._deep_merge(self._config, saved)
            except Exception:
                pass

    def _deep_merge(self, base, override):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def get(self, *keys, default=None):
        with self._lock:
            obj = self._config
            for k in keys:
                if isinstance(obj, dict) and k in obj:
                    obj = obj[k]
                else:
                    return default
            return obj

    def set(self, keys: list, value):
        with self._lock:
            obj = self._config
            for k in keys[:-1]:
                if k not in obj or not isinstance(obj[k], dict):
                    obj[k] = {}
                obj = obj[k]
            obj[keys[-1]] = value
            self._save()

    def get_irrigation(self) -> dict:
        return copy.deepcopy(self.get("irrigation"))

    def save_irrigation(self, config: dict):
        with self._lock:
            self._config["irrigation"] = config
            self._save()

    def get_learning(self) -> dict:
        return copy.deepcopy(self.get("learning"))

    def save_learning(self, config: dict):
        with self._lock:
            self._config["learning"] = config
            self._save()

    def get_plant_doctor(self) -> dict:
        return copy.deepcopy(self.get("plantDoctor"))

    def save_plant_doctor(self, config: dict):
        with self._lock:
            self._config["plantDoctor"] = config
            self._save()

    def get_system(self) -> dict:
        return copy.deepcopy(self.get("system"))

    def save_system_flags(self, rule_engine: bool, fusion_auto: bool):
        with self._lock:
            self._config["system"]["ruleEngineEnabled"] = rule_engine
            self._config["system"]["fusionAutoEnabled"] = fusion_auto
            self._save()

    def get_wifi(self) -> dict:
        return copy.deepcopy(self.get("wifi"))

    def save_wifi(self, ssid: str, password: str):
        with self._lock:
            self._config["wifi"]["ssid"] = ssid
            self._config["wifi"]["password"] = password
            self._save()

    def factory_reset(self):
        with self._lock:
            self._config = copy.deepcopy(DEFAULT_CONFIG)
            self._save()

    def full_config(self) -> dict:
        return copy.deepcopy(self._config)

    # ── Public API used by system.py ──

    def load(self):
        """Public reload from disk."""
        with self._lock:
            self._load()

    def save(self):
        """Public save to disk."""
        with self._lock:
            self._save()

    @property
    def config(self) -> dict:
        with self._lock:
            return dict(self._config)

    @config.setter
    def config(self, value: dict):
        with self._lock:
            self._config = value
            self._save()
