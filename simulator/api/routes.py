"""Flask API routes — 30+ endpoints matching firmware WebServer routes."""

import os
import json
from flask import Blueprint, request, jsonify, send_from_directory, send_file
from api.status_builder import build_overall_status, build_irrigation_status

api_bp = Blueprint("api", __name__)

# Will be set by run.py
_system = None


def init_routes(system):
    global _system
    _system = system


def _json(data: dict, status: int = 200):
    return jsonify(data), status


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _success(**kwargs):
    result = {"success": True}
    result.update(kwargs)
    return jsonify(result)


# ── Frontend ──────────────────────────────────────────────────────────

@api_bp.route("/")
def index():
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    return send_from_directory(frontend_dir, "index.html")


@api_bp.route("/dashboard")
def dashboard():
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    return send_from_directory(frontend_dir, "dashboard.html")


# ── System ────────────────────────────────────────────────────────────

@api_bp.route("/api/status")
def api_status():
    return _json(build_overall_status(_system))


@api_bp.route("/api/system/modules", methods=["GET"])
def system_modules_get():
    return _json({
        "ruleEngineEnabled": _system.irrigation.enabled,
        "learningAutoEnabled": _system.learning.config.autoControlEnabled,
        "fusionAutoEnabled": _system.fusion.auto_control_enabled,
        "plantDoctorEnabled": _system.plant_doctor.config.enabled,
    })


@api_bp.route("/api/system/modules", methods=["POST"])
def system_modules_post():
    data = request.get_json(silent=True)
    if not data:
        return _error("No body")

    if "ruleEngineEnabled" in data:
        _system.submit({"action": "set_irrigation_enabled", "enabled": bool(data["ruleEngineEnabled"])})

    if "learningAutoEnabled" in data:
        from simulator.learning import LearningConfig
        config = LearningConfig.from_dict(_system.learning.config.to_dict())
        config.autoControlEnabled = bool(data["learningAutoEnabled"])
        _system.submit({"action": "set_learning_config", "config": config})

    if "fusionAutoEnabled" in data:
        _system.submit({"action": "set_fusion_auto", "enabled": bool(data["fusionAutoEnabled"])})

    if "plantDoctorEnabled" in data:
        from simulator.plant_doctor import PlantDoctorConfig
        config = PlantDoctorConfig.from_dict(_system.plant_doctor.config.to_dict())
        config.enabled = bool(data["plantDoctorEnabled"])
        _system.submit({"action": "set_plant_doctor_config", "config": config})

    return _success()


@api_bp.route("/api/system/wifi", methods=["GET"])
def wifi_get():
    config = _system.config_manager.config.get("wifi", {})
    return _json({
        "ssid": config.get("ssid", "Simulator"),
        "hasPassword": True,
    })


@api_bp.route("/api/system/wifi", methods=["POST"])
def wifi_post():
    data = request.get_json(silent=True)
    if not data or "ssid" not in data:
        return _error("ssid required")
    return _success(message="WiFi credentials saved (simulator stub)")


@api_bp.route("/api/system/factory-reset", methods=["POST"])
def factory_reset():
    _system.submit({"action": "factory_reset"})
    return _success(message="Factory reset complete")


# ── OTA (stub) ────────────────────────────────────────────────────────

@api_bp.route("/api/ota/status")
def ota_status():
    return _json({"status": "idle", "progress": 0, "lastError": ""})


@api_bp.route("/api/ota/update", methods=["POST"])
def ota_update():
    return _error("OTA not supported in simulator", 501)


# ── Irrigation ────────────────────────────────────────────────────────

@api_bp.route("/api/irrigation/status")
def irrigation_status():
    return _json(build_irrigation_status(_system))


@api_bp.route("/api/irrigation/mode", methods=["POST"])
def irrigation_mode():
    data = request.get_json(silent=True)
    if not data or "auto" not in data:
        return _error("auto required")
    _system.submit({"action": "set_auto_mode", "enabled": bool(data["auto"])})
    return _success(autoMode=data["auto"])


@api_bp.route("/api/irrigation/pump", methods=["POST"])
def irrigation_pump():
    data = request.get_json(silent=True)
    if not data or "state" not in data:
        return _error("state required")
    _system.submit({"action": "set_manual_combined", "enabled": bool(data["state"])})
    return _success(manualValve=data["state"], manualPump=data["state"])


@api_bp.route("/api/irrigation/valve", methods=["POST"])
def irrigation_valve():
    data = request.get_json(silent=True)
    if not data or "state" not in data:
        return _error("state required")
    _system.submit({"action": "set_manual_valve", "enabled": bool(data["state"])})
    return _success(manualValve=data["state"])


@api_bp.route("/api/irrigation/pump_only", methods=["POST"])
def irrigation_pump_only():
    data = request.get_json(silent=True)
    if not data or "state" not in data:
        return _error("state required")
    _system.submit({"action": "set_manual_pump", "enabled": bool(data["state"])})
    return _success(manualPump=data["state"])


@api_bp.route("/api/irrigation/config")
def irrigation_config_get():
    return _json(_system.irrigation.status())


@api_bp.route("/api/irrigation/config", methods=["POST"])
def irrigation_config_post():
    data = request.get_json(silent=True)
    if not data:
        return _error("No body")
    _system.submit({"action": "update_irrigation_config", "config": dict(data)})
    return _success()


# ── Anomaly ───────────────────────────────────────────────────────────

@api_bp.route("/api/anomaly/status")
def anomaly_status():
    return _json(_system.anomaly.status())


@api_bp.route("/api/anomaly/alerts")
def anomaly_alerts():
    return _json(_system.anomaly.alerts())


@api_bp.route("/api/anomaly/sensor")
def anomaly_sensor():
    name = request.args.get("name", "")
    result = _system.anomaly.sensor_detail(name)
    if result is None:
        return _error("Sensor not found", 404)
    return _json(result)


@api_bp.route("/api/anomaly/clear", methods=["POST"])
def anomaly_clear():
    _system.submit({"action": "clear_anomaly"})
    return _success()


# ── Growth ────────────────────────────────────────────────────────────

@api_bp.route("/api/growth/status")
def growth_status():
    return _json(_system.growth.status())


@api_bp.route("/api/growth/history")
def growth_history():
    return _json(_system.growth.history())


@api_bp.route("/api/growth/prediction")
def growth_prediction():
    return _json(_system.growth.prediction())


@api_bp.route("/api/growth/crop", methods=["POST"])
def growth_crop():
    data = request.get_json(silent=True)
    if not data or "cropId" not in data:
        return _error("cropId required")
    _system.submit({"action": "set_crop", "cropId": int(data["cropId"])})
    return _success()


@api_bp.route("/api/growth/reset", methods=["POST"])
def growth_reset():
    _system.submit({"action": "reset_growth"})
    return _success()


# ── Learning ──────────────────────────────────────────────────────────

@api_bp.route("/api/learning/status")
def learning_status():
    return _json(_system.learning.status())


@api_bp.route("/api/learning/qtable")
def learning_qtable():
    return _json(_system.learning.qtable_summary())


@api_bp.route("/api/learning/explain")
def learning_explain():
    return _json(_system.learning.explain())


@api_bp.route("/api/learning/params", methods=["POST"])
def learning_params():
    data = request.get_json(silent=True)
    if not data:
        return _error("No body")

    from simulator.learning import LearningConfig
    config = LearningConfig.from_dict(_system.learning.config.to_dict())

    if "alpha" in data:
        config.alpha = float(data["alpha"])
    if "gamma" in data:
        config.gamma = float(data["gamma"])
    if "epsilon" in data:
        config.epsilon = float(data["epsilon"])
    if "targetSoil" in data:
        config.targetSoil = float(data["targetSoil"])
    if "soilTolerance" in data:
        config.soilTolerance = float(data["soilTolerance"])
    if "decisionInterval" in data:
        config.decisionIntervalMs = int(data["decisionInterval"]) * 1000
    if "autoControlEnabled" in data:
        config.autoControlEnabled = bool(data["autoControlEnabled"])

    _system.submit({"action": "set_learning_config", "config": config})
    return _success()


@api_bp.route("/api/learning/feedback", methods=["POST"])
def learning_feedback():
    data = request.get_json(silent=True) or {}
    positive = data.get("positive", True)
    _system.submit({"action": "record_feedback", "positive": bool(positive)})
    return _success()


@api_bp.route("/api/learning/reset", methods=["POST"])
def learning_reset():
    _system.submit({"action": "reset_learning"})
    return _success()


# ── Fusion ────────────────────────────────────────────────────────────

@api_bp.route("/api/fusion/status")
def fusion_status():
    return _json(_system.fusion.status())


@api_bp.route("/api/fusion/sensors")
def fusion_sensors():
    return _json(_system.fusion.sensors())


@api_bp.route("/api/fusion/config", methods=["POST"])
def fusion_config():
    data = request.get_json(silent=True)
    if not data:
        return _error("No body")
    if "autoControlEnabled" in data:
        _system.submit({"action": "set_fusion_auto", "enabled": bool(data["autoControlEnabled"])})
    return _success()


@api_bp.route("/api/fusion/weights", methods=["POST"])
def fusion_weights():
    data = request.get_json(silent=True)
    if not data:
        return _error("No body")
    _system.submit({"action": "update_fusion_weights", "weights": data})
    return _success(message="Fusion NN weights updated")


# ── Plant Doctor ──────────────────────────────────────────────────────

@api_bp.route("/api/plant/status")
def plant_status():
    return _json(_system.plant_doctor.status())


@api_bp.route("/api/plant/history")
def plant_history():
    return _json(_system.plant_doctor.history())


@api_bp.route("/api/plant/detect", methods=["POST"])
def plant_detect():
    """Upload image for disease detection — matches firmware POST detect."""
    if not _system.plant_doctor.config.enabled:
        return _error("Plant doctor not enabled", 503)

    image_bytes = None

    # Check for multipart file upload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename:
            image_bytes = file.read()

    # Check for raw image bytes
    elif request.content_type and "image" in request.content_type:
        image_bytes = request.get_data()

    # Check for base64 encoded image
    elif request.is_json:
        data = request.get_json(silent=True)
        if data and "image" in data:
            import base64
            try:
                image_bytes = base64.b64decode(data["image"])
            except Exception:
                return _error("Invalid base64 image data")

    if not image_bytes:
        return _error("No image provided. Use multipart 'file' field or raw image body")

    result = _system.plant_doctor.detect_from_image(image_bytes)
    if "error" in result:
        return _error(result["error"], 503)
    return _json(result)


@api_bp.route("/api/plant/detect", methods=["GET"])
def plant_detect_stub():
    """GET detect stub — in simulator, requires POST with image."""
    return _error("POST an image to /api/plant/detect for detection", 405)


@api_bp.route("/api/plant/detect_gradcam", methods=["POST"])
def plant_detect_gradcam():
    """Upload image for disease detection with Grad-CAM heatmap overlay."""
    if not _system.plant_doctor.config.enabled:
        return _error("Plant doctor not enabled", 503)

    image_bytes = None

    if "file" in request.files:
        file = request.files["file"]
        if file.filename:
            image_bytes = file.read()
    elif request.content_type and "image" in request.content_type:
        image_bytes = request.get_data()
    elif request.is_json:
        data = request.get_json(silent=True)
        if data and "image" in data:
            import base64
            try:
                image_bytes = base64.b64decode(data["image"])
            except Exception:
                return _error("Invalid base64 image data")

    if not image_bytes:
        return _error("No image provided")

    result = _system.plant_doctor.detect_with_gradcam(image_bytes)
    if "error" in result:
        return _error(result["error"], 503)
    return _json(result)


@api_bp.route("/api/plant/capture")
def plant_capture():
    """Capture stub — simulator doesn't have a camera."""
    return _error("No camera in simulator. Upload images via POST /api/plant/detect", 503)


@api_bp.route("/api/plant/config", methods=["POST"])
def plant_config():
    data = request.get_json(silent=True)
    if not data:
        return _error("No body")

    from simulator.plant_doctor import PlantDoctorConfig
    config = PlantDoctorConfig.from_dict(_system.plant_doctor.config.to_dict())
    if "enabled" in data:
        config.enabled = bool(data["enabled"])
    if "autoDetect" in data:
        config.autoDetect = bool(data["autoDetect"])
    if "detectInterval" in data:
        config.detectIntervalSec = int(data["detectInterval"])
    if "confidenceThreshold" in data:
        config.confidenceThreshold = float(data["confidenceThreshold"])
    if "buzzerEnabled" in data:
        config.buzzerEnabled = bool(data["buzzerEnabled"])

    _system.submit({"action": "set_plant_doctor_config", "config": config})
    return _success()


# ── Simulator-specific extensions ─────────────────────────────────────

@api_bp.route("/api/sensor/inject", methods=["POST"])
def sensor_inject():
    """Force sensor values for testing."""
    data = request.get_json(silent=True)
    if not data:
        return _error("No body")
    _system.submit({"action": "inject_sensor", "values": data})
    return _success()


@api_bp.route("/api/sensor/inject", methods=["DELETE"])
def sensor_inject_clear():
    """Restore automatic sensor simulation."""
    _system.submit({"action": "clear_inject"})
    return _success()


@api_bp.route("/api/simulator/status")
def simulator_status():
    return _json({
        "timeScale": _system.clock.time_scale,
        "uptime": round(_system.clock.uptime_seconds(), 1),
        "simHours": round(_system.clock.sim_elapsed_hours(), 2),
    })


@api_bp.route("/api/simulator/time-scale", methods=["POST"])
def simulator_time_scale():
    data = request.get_json(silent=True)
    if not data or "scale" not in data:
        return _error("scale required")
    scale = float(data["scale"])
    if scale < 1 or scale > 3600:
        return _error("scale must be between 1 and 3600")
    _system.submit({"action": "set_time_scale", "scale": scale})
    return _success(scale=scale)
