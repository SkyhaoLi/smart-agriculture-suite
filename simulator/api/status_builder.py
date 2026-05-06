"""Status builder — composite JSON for /api/status endpoint."""

from simulator.system import AgriSystem


def build_overall_status(system: AgriSystem) -> dict:
    """Build /api/status response — matches buildOverallStatus() from main.cpp."""
    return system.overall_status()


def build_irrigation_status(system: AgriSystem) -> dict:
    """Build /api/irrigation/status with sensors + actuator + module."""
    snapshot = system.sensor_hub.snapshot
    now_ms = system.clock.millis()
    return {
        "sensors": {
            "airTemp": round(snapshot.airTemp, 2),
            "airHumi": round(snapshot.airHumi, 2),
            "soilHumi": round(snapshot.soilHumi, 2),
            "liquidLevel": round(snapshot.liquidLevel, 2),
            "lightValue": round(snapshot.lightValue, 2),
        },
        "actuator": system.actuator.status.to_dict(now_ms),
        "modules": {"irrigation": system.irrigation.status()},
    }
