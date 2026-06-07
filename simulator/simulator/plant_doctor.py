"""PlantDoctorModule — TFLite plant disease detection matching firmware."""

import os
import math
import time
import threading
from datetime import datetime

# Same labels as PlantDoctorModule.cpp
CLASS_LABELS_EN = [
    "Healthy",
    "Strawberry_Anthracnose",
    "Strawberry_Gray_Mold",
    "Strawberry_Leaf_Scorch",
    "Strawberry_Powdery_Mildew",
]

CLASS_LABELS_CN = [
    "健康",
    "草莓炭疽病",
    "草莓灰霉病",
    "草莓叶焦病",
    "草莓白粉病",
]

TREATMENTS = [
    "叶片健康。保持通风和定期检查。",
    "清除病株残体，避免伤口感染。施用咪鲜胺等杀菌剂。",
    "降低湿度，增加通风。及时摘除病果，施用嘧霉胺等杀菌剂。",
    "移除感染叶片，降低叶面湿度。必要时施用杀菌剂。",
    "移除病叶，改善通风。喷施硫制剂或三唑类杀菌剂。",
]

K_NUM_CLASSES = 5
K_INPUT_WIDTH = 96
K_INPUT_HEIGHT = 96
K_INPUT_CHANNELS = 3
K_HISTORY_SIZE = 10


class PlantDoctorConfig:
    def __init__(self):
        self.enabled = True
        self.autoDetect = False
        self.detectIntervalSec = 300
        self.confidenceThreshold = 0.70
        self.buzzerEnabled = True

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "autoDetect": self.autoDetect,
            "detectIntervalSec": self.detectIntervalSec,
            "confidenceThreshold": self.confidenceThreshold,
            "buzzerEnabled": self.buzzerEnabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlantDoctorConfig":
        c = cls()
        for k in ["enabled", "autoDetect", "detectIntervalSec", "confidenceThreshold", "buzzerEnabled"]:
            if k in d:
                setattr(c, k, d[k])
        return c


class DetectionRecord:
    def __init__(self):
        self.diseaseId = 0
        self.confidence = 0.0
        self.timestamp = ""


class PlantDoctorModule:
    """Direct port of agri::PlantDoctorModule for PC simulation."""

    def __init__(self, model_path: str = ""):
        self._lock = threading.Lock()
        self._config = PlantDoctorConfig()
        self._modelPath = model_path
        self._modelLoaded = False
        self._interpreter = None
        self._inputDetails = None
        self._outputDetails = None
        self._lightValue = 0.0

        # Detection state
        self._lastDiseaseId = 0
        self._lastConfidence = 0.0
        self._lastDetectionTime = ""
        self._totalDetections = 0
        self._diseaseDetections = 0
        self._lastDetectMs = 0

        # History ring buffer
        self._history = [DetectionRecord() for _ in range(K_HISTORY_SIZE)]
        self._historyIndex = 0

    def begin(self, config: dict = None, model_path: str = ""):
        if config:
            self._config = PlantDoctorConfig.from_dict(config)
        if model_path:
            self._modelPath = model_path
        if self._config.enabled and self._modelPath:
            self._setup_tflite()

    @property
    def config(self) -> PlantDoctorConfig:
        return self._config

    def set_config(self, config: PlantDoctorConfig):
        self._config = config

    def _setup_tflite(self):
        """Load TFLite model for PC inference."""
        if self._modelLoaded:
            return

        if not os.path.exists(self._modelPath):
            print(f"[PlantDoctor] model file not found: {self._modelPath}")
            return

        try:
            import numpy as np

            try:
                import tflite_runtime.interpreter as tflite
                self._interpreter = tflite.Interpreter(model_path=self._modelPath)
            except ImportError:
                import tensorflow as tf
                self._interpreter = tf.lite.Interpreter(model_path=self._modelPath)

            self._interpreter.allocate_tensors()
            self._inputDetails = self._interpreter.get_input_details()[0]
            self._outputDetails = self._interpreter.get_output_details()[0]
            self._modelLoaded = True
            print(f"[PlantDoctor] model loaded: {self._modelPath}")

        except Exception as e:
            print(f"[PlantDoctor] model load failed: {e}")
            self._modelLoaded = False

    def update(self, now_ms: int, light_value: float):
        """Auto-detect on interval (matches firmware update loop)."""
        self._lightValue = light_value
        if not self._config.enabled or not self._config.autoDetect or not self._modelLoaded:
            return
        if now_ms - self._lastDetectMs < self._config.detectIntervalSec * 1000:
            return
        # Auto-detect would need camera input — in simulator, only manual detect via API
        self._lastDetectMs = now_ms

    def detect_from_image(self, image_bytes: bytes) -> dict:
        """
        Detect disease from uploaded image bytes.
        Matches firmware performDetection() pipeline:
          1. Resize to 96x96
          2. Normalize [-1, 1]
          3. INT8 quantize
          4. TFLite inference
          5. Dequantize + softmax
        """
        with self._lock:
            if not self._config.enabled or not self._modelLoaded:
                return {"error": "Plant doctor not available"}

            try:
                import numpy as np
                from PIL import Image
                import io
            except ImportError as e:
                return {"error": f"Missing dependency: {e}"}

            try:
                # Load and preprocess image
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = img.resize((K_INPUT_WIDTH, K_INPUT_HEIGHT), Image.BILINEAR)
                pixels = np.array(img, dtype=np.float32)

                # Normalize to [-1, 1] — same as firmware
                normalized = (pixels / 127.5) - 1.0

                # Quantize to int8 — same as firmware
                input_scale = self._inputDetails["quantization_parameters"]["scales"][0]
                input_zp = self._inputDetails["quantization_parameters"]["zero_points"][0]
                quantized = np.round(normalized / input_scale) + input_zp
                quantized = np.clip(quantized, -128, 127).astype(np.int8)
                quantized = np.expand_dims(quantized, axis=0)

                # Run inference
                self._interpreter.set_tensor(self._inputDetails["index"], quantized)
                self._interpreter.invoke()

                output_data = self._interpreter.get_tensor(self._outputDetails["index"])[0]

                # Dequantize — same as firmware
                output_scale = self._outputDetails["quantization_parameters"]["scales"][0]
                output_zp = self._outputDetails["quantization_parameters"]["zero_points"][0]
                confidences = (output_data.astype(np.float32) - output_zp) * output_scale

                # Softmax — same as firmware
                max_val = np.max(confidences)
                exp_vals = np.exp(confidences - max_val)
                softmax_output = exp_vals / np.sum(exp_vals)

                disease_id = int(np.argmax(softmax_output))
                confidence = float(softmax_output[disease_id])

                # Update state
                self._lastDiseaseId = disease_id
                self._lastConfidence = confidence
                self._lastDetectionTime = datetime.now().strftime("%H:%M:%S")
                self._totalDetections += 1

                # Add to history
                self._history[self._historyIndex].diseaseId = disease_id
                self._history[self._historyIndex].confidence = confidence
                self._history[self._historyIndex].timestamp = self._lastDetectionTime
                self._historyIndex = (self._historyIndex + 1) % K_HISTORY_SIZE

                if disease_id > 0 and confidence >= self._config.confidenceThreshold:
                    self._diseaseDetections += 1

                return {
                    "diseaseId": disease_id,
                    "diseaseName": CLASS_LABELS_EN[disease_id],
                    "diseaseNameCn": CLASS_LABELS_CN[disease_id],
                    "confidence": round(confidence, 4),
                    "treatment": TREATMENTS[disease_id],
                    "isDisease": disease_id > 0 and confidence >= self._config.confidenceThreshold,
                    "classProbabilities": {
                        CLASS_LABELS_EN[i]: round(float(softmax_output[i]), 4)
                        for i in range(K_NUM_CLASSES)
                    },
                }

            except Exception as e:
                return {"error": f"Detection failed: {e}"}

    def detect_with_gradcam(self, image_bytes: bytes) -> dict:
        """Detect disease and generate Grad-CAM-style heatmap via occlusion sensitivity."""
        import numpy as np
        from PIL import Image
        import io
        import base64

        # First run normal detection
        result = self.detect_from_image(image_bytes)
        if "error" in result:
            return result

        disease_id = result["diseaseId"]

        with self._lock:
            if not self._modelLoaded:
                return result

            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img_resized = img.resize((K_INPUT_WIDTH, K_INPUT_HEIGHT), Image.BILINEAR)
                pixels = np.array(img_resized, dtype=np.float32)
                normalized = (pixels / 127.5) - 1.0

                input_scale = self._inputDetails["quantization_parameters"]["scales"][0]
                input_zp = self._inputDetails["quantization_parameters"]["zero_points"][0]

                # Get baseline confidence for target class
                def _infer(norm_img):
                    q = np.round(norm_img / input_scale) + input_zp
                    q = np.clip(q, -128, 127).astype(np.int8)
                    q = np.expand_dims(q, axis=0)
                    self._interpreter.set_tensor(self._inputDetails["index"], q)
                    self._interpreter.invoke()
                    out = self._interpreter.get_tensor(self._outputDetails["index"])[0]
                    os_ = self._outputDetails["quantization_parameters"]["scales"][0]
                    oz = self._outputDetails["quantization_parameters"]["zero_points"][0]
                    conf = (out.astype(np.float32) - oz) * os_
                    exp_v = np.exp(conf - np.max(conf))
                    return exp_v / np.sum(exp_v)

                baseline_probs = _infer(normalized)
                baseline_score = baseline_probs[disease_id]

                # Occlusion sensitivity — slide a patch across the image
                patch_size = 12  # 12x12 patch on 96x96 = 8x8 heatmap
                stride = 12
                heatmap_h = (K_INPUT_HEIGHT - patch_size) // stride + 1
                heatmap_w = (K_INPUT_WIDTH - patch_size) // stride + 1
                heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

                mean_pixel = np.mean(normalized, axis=(0, 1))  # mean RGB for occlusion

                for i in range(heatmap_h):
                    for j in range(heatmap_w):
                        occluded = normalized.copy()
                        y_start = i * stride
                        x_start = j * stride
                        occluded[y_start:y_start+patch_size, x_start:x_start+patch_size] = mean_pixel
                        probs = _infer(occluded)
                        # Drop in confidence = importance of this region
                        heatmap[i, j] = baseline_score - probs[disease_id]

                # Normalize heatmap to [0, 1]
                hmin, hmax = heatmap.min(), heatmap.max()
                if hmax - hmin > 1e-6:
                    heatmap = (heatmap - hmin) / (hmax - hmin)
                else:
                    heatmap = np.zeros_like(heatmap)

                # Upscale heatmap to image size
                heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8), mode='L')
                heatmap_img = heatmap_img.resize((K_INPUT_WIDTH, K_INPUT_HEIGHT), Image.BILINEAR)
                heatmap_arr = np.array(heatmap_img, dtype=np.float32) / 255.0

                # Create colored overlay
                overlay = np.array(img_resized, dtype=np.float32)
                # Apply jet-like colormap: blue (cold) -> red (hot)
                for c in range(3):
                    if c == 0:  # Red channel
                        overlay[:, :, c] = np.clip(heatmap_arr * 255, 0, 255)
                    elif c == 1:  # Green channel
                        overlay[:, :, c] = np.clip((1 - np.abs(heatmap_arr - 0.5) * 2) * 255, 0, 255)
                    else:  # Blue channel
                        overlay[:, :, c] = np.clip((1 - heatmap_arr) * 255, 0, 255)

                # Blend: 60% original + 40% heatmap
                blended = (np.array(img_resized, dtype=np.float32) * 0.6 + overlay * 0.4).astype(np.uint8)
                blended_img = Image.fromarray(blended)

                # Encode to base64 PNG
                buf = io.BytesIO()
                blended_img.save(buf, format='PNG')
                heatmap_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

                result["gradcam"] = {
                    "heatmapBase64": heatmap_b64,
                    "imageWidth": K_INPUT_WIDTH,
                    "imageHeight": K_INPUT_HEIGHT,
                    "patchSize": patch_size,
                    "targetClass": CLASS_LABELS_EN[disease_id],
                    "baselineConfidence": round(float(baseline_score), 4),
                }
                return result

            except Exception as e:
                result["gradcam_error"] = str(e)
                return result

    # ── Public accessors ──

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": self._config.enabled,
                "cameraReady": True,  # Simulator always "ready" (uses uploaded images)
                "modelLoaded": self._modelLoaded,
                "lightValue": round(self._lightValue, 2),
                "lastDiseaseId": self._lastDiseaseId,
                "lastDiseaseName": CLASS_LABELS_EN[self._lastDiseaseId],
                "lastDiseaseNameCn": CLASS_LABELS_CN[self._lastDiseaseId],
                "lastConfidence": round(self._lastConfidence, 4),
                "lastDetectionTime": self._lastDetectionTime,
                "treatment": TREATMENTS[self._lastDiseaseId],
                "totalDetections": self._totalDetections,
                "diseaseDetections": self._diseaseDetections,
                "autoDetect": self._config.autoDetect,
                "detectInterval": self._config.detectIntervalSec,
                "confidenceThreshold": self._config.confidenceThreshold,
                "buzzerEnabled": self._config.buzzerEnabled,
            }

    def history(self) -> list:
        with self._lock:
            result = []
            for i in range(K_HISTORY_SIZE):
                idx = (self._historyIndex - 1 - i + K_HISTORY_SIZE) % K_HISTORY_SIZE
                h = self._history[idx]
                if not h.timestamp:
                    continue
                result.append({
                    "diseaseId": h.diseaseId,
                    "diseaseName": CLASS_LABELS_EN[h.diseaseId],
                    "diseaseNameCn": CLASS_LABELS_CN[h.diseaseId],
                    "confidence": round(h.confidence, 4),
                    "timestamp": h.timestamp,
                })
            return result
