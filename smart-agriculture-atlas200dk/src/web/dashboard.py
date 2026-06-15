"""
智润智慧农业套件 - Atlas 200I DK A2 版
Web仪表盘 - Flask REST API + 单页前端

对应原ESP32项目的 WebFrontend.h + main.cpp中的路由注册
API接口完全兼容原ESP32版本
"""

import os
import json
import time
import base64
import logging
import cv2
import numpy as np

from flask import Flask, request, jsonify, Response, send_from_directory

logger = logging.getLogger(__name__)


class WebDashboard:
    """Flask Web仪表盘 - REST API + 单页前端"""

    def __init__(self, sensor_hub, actuator, irrigation, anomaly,
                 growth, learning, fusion, plant_doctor, world_model=None, static_dir=None):
        self._sensor_hub = sensor_hub
        self._actuator = actuator
        self._irrigation = irrigation
        self._anomaly = anomaly
        self._growth = growth
        self._learning = learning
        self._fusion = fusion
        self._plant_doctor = plant_doctor
        self._world_model = world_model

        self._app = Flask(__name__,
                          static_folder=static_dir or os.path.join(os.path.dirname(__file__), 'static'),
                          static_url_path='/static')
        self._app.config['JSON_SORT_KEYS'] = False
        self._register_routes()

    @property
    def app(self):
        return self._app

    # ------------------------------------------------------------------
    # 路由注册
    # ------------------------------------------------------------------
    def _register_routes(self):
        app = self._app

        @app.route('/')
        def index():
            return send_from_directory(
                os.path.join(os.path.dirname(__file__), 'static'), 'index.html')

        # ── 总状态 ──
        @app.route('/api/status')
        def api_status():
            return jsonify(self._build_overall_status())

        # ── 模块开关 ──
        @app.route('/api/system/modules', methods=['GET'])
        def api_modules_get():
            return jsonify({
                "ruleEngineEnabled": self._irrigation.enabled,
                "learningAutoEnabled": self._learning._config.auto_control_enabled,
                "fusionAutoEnabled": self._fusion._auto_control_enabled,
                "plantDoctorEnabled": self._plant_doctor._enabled,
            })

        @app.route('/api/system/modules', methods=['POST'])
        def api_modules_post():
            body = request.get_json(silent=True) or {}
            if 'ruleEngineEnabled' in body:
                self._irrigation.enabled = body['ruleEngineEnabled']
            if 'learningAutoEnabled' in body:
                self._learning._config.auto_control_enabled = body['learningAutoEnabled']
            if 'fusionAutoEnabled' in body:
                self._fusion._auto_control_enabled = body['fusionAutoEnabled']
            if 'plantDoctorEnabled' in body:
                self._plant_doctor._enabled = body['plantDoctorEnabled']
            return jsonify({"success": True})

        # ── 灌溉 ──
        @app.route('/api/irrigation/status')
        def api_irrigation_status():
            return jsonify({
                "sensors": self._sensor_hub.snapshot.__dict__,
                "actuator": self._actuator.status.__dict__,
                "module": self._irrigation.to_dict(),
            })

        @app.route('/api/irrigation/mode', methods=['POST'])
        def api_irrigation_mode():
            body = request.get_json(silent=True) or {}
            if 'auto' in body:
                self._actuator.set_auto_mode(body['auto'])
            return jsonify({
                "success": True,
                "autoMode": self._actuator._auto_mode,
            })

        @app.route('/api/irrigation/pump', methods=['POST'])
        def api_irrigation_pump():
            body = request.get_json(silent=True) or {}
            if 'state' in body:
                self._actuator.set_manual_combined(body['state'])
            return jsonify({
                "success": True,
                "manualValve": self._actuator._manual_valve,
                "manualPump": self._actuator._manual_pump,
            })

        @app.route('/api/irrigation/valve', methods=['POST'])
        def api_irrigation_valve():
            body = request.get_json(silent=True) or {}
            if 'state' in body:
                self._actuator.set_manual_valve(body['state'])
            return jsonify({
                "success": True,
                "manualValve": self._actuator._manual_valve,
            })

        @app.route('/api/irrigation/pump_only', methods=['POST'])
        def api_irrigation_pump_only():
            body = request.get_json(silent=True) or {}
            if 'state' in body:
                self._actuator.set_manual_pump(body['state'])
            return jsonify({
                "success": True,
                "manualPump": self._actuator._manual_pump,
            })

        @app.route('/api/irrigation/config')
        def api_irrigation_config_get():
            return jsonify(self._irrigation.to_dict())

        @app.route('/api/irrigation/config', methods=['POST'])
        def api_irrigation_config_post():
            body = request.get_json(silent=True) or {}
            self._irrigation.update_config(body)
            return jsonify({"success": True})

        # ── 异常检测 ──
        @app.route('/api/anomaly/status')
        def api_anomaly_status():
            return jsonify(self._anomaly.to_dict())

        @app.route('/api/anomaly/clear', methods=['POST'])
        def api_anomaly_clear():
            self._anomaly.clear()
            return jsonify({"success": True})

        # ── 生长跟踪 ──
        @app.route('/api/growth/status')
        def api_growth_status():
            return jsonify(self._growth.to_dict())

        @app.route('/api/growth/history')
        def api_growth_history():
            records = self._growth._records[-90:]
            return jsonify([{
                "day": r.day_index, "avgTemp": round(r.avg_temp, 1),
                "maxTemp": round(r.max_temp, 1), "minTemp": round(r.min_temp, 1),
                "avgHumi": round(r.avg_humi, 1), "avgSoil": round(r.avg_soil, 1),
                "totalLight": round(r.total_light, 1),
                "dailyGdd": round(r.daily_gdd, 2), "cumulativeGdd": round(r.cumulative_gdd, 2),
                "stage": r.stage.value,
            } for r in records])

        @app.route('/api/growth/crop', methods=['POST'])
        def api_growth_crop():
            body = request.get_json(silent=True) or {}
            crop_id = body.get('cropId')
            if crop_id is None:
                return jsonify({"error": "cropId required"}), 400
            self._growth.set_crop(crop_id)
            return jsonify({"success": True})

        @app.route('/api/growth/reset', methods=['POST'])
        def api_growth_reset():
            self._growth.reset()
            return jsonify({"success": True})

        # ── Q-Learning ──
        @app.route('/api/learning/status')
        def api_learning_status():
            return jsonify(self._learning.to_dict())

        @app.route('/api/learning/qtable')
        def api_learning_qtable():
            return jsonify(self._learning.to_dict())

        @app.route('/api/learning/params', methods=['POST'])
        def api_learning_params():
            body = request.get_json(silent=True) or {}
            cfg = self._learning._config
            for key in ['alpha', 'gamma', 'epsilon', 'targetSoil', 'soilTolerance']:
                if key in body:
                    setattr(cfg, key, float(body[key]))
            if 'decisionInterval' in body:
                cfg.decision_interval_sec = float(body['decisionInterval'])
            if 'autoControlEnabled' in body:
                cfg.auto_control_enabled = body['autoControlEnabled']
            return jsonify({"success": True})

        @app.route('/api/learning/feedback', methods=['POST'])
        def api_learning_feedback():
            body = request.get_json(silent=True) or {}
            positive = body.get('positive', True)
            self._learning.record_user_feedback(positive)
            return jsonify({"success": True})

        @app.route('/api/learning/reset', methods=['POST'])
        def api_learning_reset():
            self._learning.reset()
            return jsonify({"success": True})

        # ── 传感器融合 ──
        @app.route('/api/fusion/status')
        def api_fusion_status():
            return jsonify(self._fusion.to_dict())

        @app.route('/api/fusion/sensors')
        def api_fusion_sensors():
            return jsonify(self._fusion.to_dict().get('sensors', []))

        @app.route('/api/fusion/config', methods=['POST'])
        def api_fusion_config():
            body = request.get_json(silent=True) or {}
            if 'autoControlEnabled' in body:
                self._fusion._auto_control_enabled = body['autoControlEnabled']
            return jsonify({"success": True})

        @app.route('/api/fusion/weights', methods=['POST'])
        def api_fusion_weights():
            body = request.get_json(silent=True) or {}
            if 'weightsIH' in body:
                self._fusion._weights_ih = np.array(body['weightsIH'], dtype=np.float32)
            if 'biasH' in body:
                self._fusion._bias_h = np.array(body['biasH'], dtype=np.float32)
            if 'weightsHO' in body:
                self._fusion._weights_ho = np.array(body['weightsHO'], dtype=np.float32)
            if 'biasO' in body:
                self._fusion._bias_o = np.array(body['biasO'], dtype=np.float32)
            self._fusion.save_network()
            return jsonify({"success": True, "message": "Fusion NN weights updated and persisted"})

        # ── 植物病害检测 ──
        @app.route('/api/plant/status')
        def api_plant_status():
            return jsonify(self._plant_doctor.to_dict())

        @app.route('/api/plant/history')
        def api_plant_history():
            return jsonify([
                {"diseaseId": r.disease_id, "confidence": round(r.confidence, 4),
                 "timestamp": r.timestamp}
                for r in self._plant_doctor._history
            ])

        @app.route('/api/plant/detect')
        def api_plant_detect():
            result = self._plant_doctor.perform_detection()
            if result is None:
                return jsonify({"error": "Plant doctor unavailable"}), 503
            return jsonify(self._plant_doctor.to_dict())

        @app.route('/api/plant/upload', methods=['POST'])
        def api_plant_upload():
            """上传图片进行病害检测"""
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
            f = request.files['file']
            if not f.filename:
                return jsonify({"error": "Empty filename"}), 400
            image_bytes = f.read()
            if len(image_bytes) == 0:
                return jsonify({"error": "Empty file"}), 400
            result = self._plant_doctor.detect_from_image(image_bytes)
            if result is None:
                return jsonify({"error": "Detection failed. Model may not be loaded."}), 503
            # 返回检测结果 + 缩略图(base64)
            thumbnail = None
            jpg_bytes = self._plant_doctor.get_last_uploaded_jpeg()
            if jpg_bytes:
                thumbnail = base64.b64encode(jpg_bytes).decode('ascii')
            from config.app_types import DISEASE_NAMES
            from src.ai.plant_doctor_module import DISEASE_LABELS_CN, TREATMENTS as _TREATMENTS
            disease_id = result.disease_class.value
            return jsonify({
                "success": True,
                "diseaseId": disease_id,
                "diseaseNameCn": DISEASE_LABELS_CN[disease_id],
                "confidence": round(result.confidence, 4),
                "allProbs": result.all_probs,
                "treatment": _TREATMENTS[disease_id],
                "thumbnail": thumbnail,
            })

        @app.route('/api/plant/capture')
        def api_plant_capture():
            frame = self._plant_doctor.capture_image()
            if frame is None:
                return jsonify({"error": "Capture failed"}), 503
            _, jpg = cv2.imencode('.jpg', frame)
            return Response(jpg.tobytes(), mimetype='image/jpeg')

        @app.route('/api/plant/stream')
        def api_plant_stream():
            def generate():
                while True:
                    frame = self._plant_doctor.capture_image()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    h, w = frame.shape[:2]
                    if w > 320:
                        frame = cv2.resize(frame, (320, int(h * 320 / w)))
                    ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
                    if not ok:
                        continue
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
                    time.sleep(0.1)
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/api/plant/config', methods=['POST'])
        def api_plant_config():
            body = request.get_json(silent=True) or {}
            if 'enabled' in body:
                self._plant_doctor._enabled = body['enabled']
            if 'autoDetect' in body:
                self._plant_doctor._auto_detect = body['autoDetect']
            if 'detectInterval' in body:
                self._plant_doctor._detect_interval = float(body['detectInterval'])
            if 'confidenceThreshold' in body:
                self._plant_doctor._confidence_threshold = float(body['confidenceThreshold'])
            if 'buzzerEnabled' in body:
                self._plant_doctor._buzzer_enabled = body['buzzerEnabled']
            return jsonify({"success": True})

        # ── 环境状态预测 (世界模型) ──
        @app.route('/api/worldmodel/status')
        def api_worldmodel_status():
            if self._world_model is None:
                return jsonify({"error": "World model not available"}), 503
            return jsonify(self._world_model.to_dict())

        @app.route('/api/worldmodel/predict')
        def api_worldmodel_predict():
            if self._world_model is None:
                return jsonify({"error": "World model not available"}), 503
            pred = self._world_model.get_predictions()
            return jsonify({
                "mode": pred.mode,
                "temp": {str(h): round(pred.predicted_temp.get(h, 0), 2) for h in [15, 30, 60]},
                "humi": {str(h): round(pred.predicted_humi.get(h, 0), 2) for h in [15, 30, 60]},
                "soil": {str(h): round(pred.predicted_soil.get(h, 0), 2) for h in [15, 30, 60]},
                "light": {str(h): round(pred.predicted_light.get(h, 0), 2) for h in [15, 30, 60]},
                "confidence": {str(h): round(v, 3) for h, v in pred.confidence.items()},
                "riskScore": round(pred.risk_score, 3),
                "soilRisk": pred.soil_moisture_risk,
            })

        @app.route('/api/worldmodel/history')
        def api_worldmodel_history():
            if self._world_model is None:
                return jsonify({"error": "World model not available"}), 503
            minutes = request.args.get('minutes', 60, type=int)
            return jsonify(self._world_model.get_history(minutes))

        @app.route('/api/worldmodel/config', methods=['POST'])
        def api_worldmodel_config():
            if self._world_model is None:
                return jsonify({"error": "World model not available"}), 503
            body = request.get_json(silent=True) or {}
            if 'autoControlEnabled' in body:
                self._world_model._auto_control_enabled = body['autoControlEnabled']
            return jsonify({"success": True})

    def _build_overall_status(self) -> dict:
        snap = self._sensor_hub.snapshot
        status = self._actuator.status
        return {
            "project": "smart-agriculture-atlas200dk",
            "hardwareProfile": "Atlas200IDKA2",
            "sensors": {
                "airTemp": round(snap.air_temp, 1),
                "airHumi": round(snap.air_humi, 1),
                "soilHumi": round(snap.soil_humi, 1),
                "lightValue": round(snap.light_intensity, 1),
            },
            "actuator": {
                "valveOn": status.valve_on,
                "pumpOn": status.pump_on,
                "autoMode": self._actuator._auto_mode,
                "lowLiquidLock": False,
            },
            "modules": {
                "irrigation": self._irrigation.to_dict(),
                "anomaly": self._anomaly.to_dict(),
                "growth": self._growth.to_dict(),
                "learning": self._learning.to_dict(),
                "fusion": self._fusion.to_dict(),
                "plantDoctor": self._plant_doctor.to_dict(),
                "worldModel": self._world_model.to_dict() if self._world_model else {},
            },
        }

    def run(self, host='0.0.0.0', port=8080, debug=False):
        self._app.run(host=host, port=port, debug=debug, threaded=True)
