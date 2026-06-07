"""IrrigationModule — rule-based irrigation matching firmware logic."""


class IrrigationModule:
    """Direct port of agri::IrrigationModule."""

    def __init__(self):
        self.enabled_ = True
        self.shouldWater_ = False
        self.liquidWarn_ = False
        self.isDay_ = True
        self.reason_ = "not started"

        # Thresholds — matches AppTypes.h IrrigationThresholdConfig defaults
        self.config = {
            "liquidLevelThreshold": 30.0,
            "lightDayThreshold": 200.0,
            "dayAirTempThreshold": 20.0,
            "dayAirHumiThreshold": 60.0,
            "daySoilHumiThreshold": 50.0,
            "nightAirTempThreshold": 15.0,
            "nightAirHumiThreshold": 70.0,
            "nightSoilHumiThreshold": 45.0,
        }

    def update(self, snapshot):
        """Matches IrrigationModule::update()."""
        self.isDay_ = snapshot.lightValue >= self.config["lightDayThreshold"]
        self.liquidWarn_ = snapshot.liquidValid and snapshot.liquidLevel < self.config["liquidLevelThreshold"]

        if not self.enabled_:
            self.shouldWater_ = False
            self.reason_ = "rule engine disabled"
            return

        if self.liquidWarn_:
            self.shouldWater_ = False
            self.reason_ = "liquid tank too low"
            return

        if self.isDay_:
            temp_pass = snapshot.airTemp >= self.config["dayAirTempThreshold"]
            humi_pass = snapshot.airHumi <= self.config["dayAirHumiThreshold"]
            soil_pass = snapshot.soilHumi <= self.config["daySoilHumiThreshold"]
        else:
            temp_pass = snapshot.airTemp >= self.config["nightAirTempThreshold"]
            humi_pass = snapshot.airHumi <= self.config["nightAirHumiThreshold"]
            soil_pass = snapshot.soilHumi <= self.config["nightSoilHumiThreshold"]

        self.shouldWater_ = temp_pass and humi_pass and soil_pass
        if self.shouldWater_:
            self.reason_ = "day thresholds matched" if self.isDay_ else "night thresholds matched"
        else:
            self.reason_ = "thresholds not met"

    def update_config(self, data: dict) -> bool:
        """Matches IrrigationModule::updateConfigFromJson()."""
        updated = False

        if "enabled" in data and isinstance(data["enabled"], bool):
            self.enabled_ = data["enabled"]
            updated = True

        day = data.get("day", {})
        if isinstance(day, dict):
            for key, cfg_key in [("airTemp", "dayAirTempThreshold"),
                                  ("airHumi", "dayAirHumiThreshold"),
                                  ("soilHumi", "daySoilHumiThreshold")]:
                if key in day and isinstance(day[key], (int, float)):
                    self.config[cfg_key] = float(day[key])
                    updated = True

        night = data.get("night", {})
        if isinstance(night, dict):
            for key, cfg_key in [("airTemp", "nightAirTempThreshold"),
                                  ("airHumi", "nightAirHumiThreshold"),
                                  ("soilHumi", "nightSoilHumiThreshold")]:
                if key in night and isinstance(night[key], (int, float)):
                    self.config[cfg_key] = float(night[key])
                    updated = True

        if "liquidThreshold" in data and isinstance(data["liquidThreshold"], (int, float)):
            self.config["liquidLevelThreshold"] = float(data["liquidThreshold"])
            updated = True

        if "lightThreshold" in data and isinstance(data["lightThreshold"], (int, float)):
            self.config["lightDayThreshold"] = float(data["lightThreshold"])
            updated = True

        return updated

    @property
    def enabled(self) -> bool:
        return self.enabled_

    @enabled.setter
    def enabled(self, val: bool):
        self.enabled_ = val

    @property
    def should_water(self) -> bool:
        return self.shouldWater_

    @property
    def liquid_warn(self) -> bool:
        return self.liquidWarn_

    @property
    def is_day(self) -> bool:
        return self.isDay_

    @property
    def reason(self) -> str:
        return self.reason_

    def status(self) -> dict:
        return {
            "enabled": self.enabled_,
            "shouldWater": self.shouldWater_,
            "liquidWarn": self.liquidWarn_,
            "isDay": self.isDay_,
            "reason": self.reason_,
            "config": {
                "day": {
                    "airTemp": self.config["dayAirTempThreshold"],
                    "airHumi": self.config["dayAirHumiThreshold"],
                    "soilHumi": self.config["daySoilHumiThreshold"],
                },
                "night": {
                    "airTemp": self.config["nightAirTempThreshold"],
                    "airHumi": self.config["nightAirHumiThreshold"],
                    "soilHumi": self.config["nightSoilHumiThreshold"],
                },
                "liquidThreshold": self.config["liquidLevelThreshold"],
                "lightThreshold": self.config["lightDayThreshold"],
            },
        }
