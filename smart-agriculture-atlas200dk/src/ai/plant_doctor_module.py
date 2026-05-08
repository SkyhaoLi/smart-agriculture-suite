"""
智润智慧农业套件 - Atlas 200I DK A2 版
植物病害检测模块 - 使用NPU(Ascend 310B)进行AI推理

对应原ESP32项目的 PlantDoctorModule.h/PlantDoctorModule.cpp
ESP32使用TFLite Micro + DVP摄像头, Atlas 200I DK A2使用:
- USB摄像头 (通过OpenCV/VideoCapture读取)
- ACL Python SDK 或 ONNX Runtime 加载病害检测模型
- NPU 8 TOPS INT8 推理, 性能远超ESP32

5类草莓病害: 健康/炭疽病/灰霉病/叶灼病/白粉病
"""

import cv2
import time
import json
import logging
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

from config.app_types import DiseaseClass, DiseaseResult, DISEASE_NAMES
from config.hardware_config import MODEL_DIR

logger = logging.getLogger(__name__)

DISEASE_LABELS = ["Healthy", "Strawberry_Anthracnose", "Strawberry_Gray_Mold",
                  "Strawberry_Leaf_Scorch", "Strawberry_Powdery_Mildew"]
DISEASE_LABELS_CN = ["健康", "草莓炭疽病", "草莓灰霉病", "草莓叶焦病", "草莓白粉病"]
TREATMENTS = [
    "叶片健康。保持通风和定期检查。",
    "清除病株残体，避免伤口感染。施用咪鲜胺等杀菌剂。",
    "降低湿度，增加通风。及时摘除病果，施用嘧霉胺等杀菌剂。",
    "移除感染叶片，降低叶面湿度。必要时施用杀菌剂。",
    "移除病叶，改善通风。喷施硫制剂或三唑类杀菌剂。",
]


@dataclass
class DetectionRecord:
    disease_id: int = 0
    confidence: float = 0.0
    timestamp: str = ""


class PlantDoctorModule:
    """植物病害检测 - Atlas 200I DK A2 NPU推理版"""

    INPUT_SIZE = 96  # 模型输入96x96
    NUM_CLASSES = 5
    HISTORY_SIZE = 10

    def __init__(self, model_dir: str = MODEL_DIR, buzzer=None):
        self._model_dir = model_dir
        self._buzzer = buzzer

        self._enabled = True
        self._auto_detect = True
        self._detect_interval = 60.0  # 秒
        self._confidence_threshold = 0.70
        self._buzzer_enabled = True

        self._camera_ready = False
        self._model_loaded = False
        self._camera_id = 0
        self._capture = None

        # ACL/ONNX推理引擎 (根据实际环境选择)
        self._inference_engine = None
        self._engine_type = "none"  # "acl", "onnx", "opencv_dnn"

        self._last_disease_id = 0
        self._last_confidence = 0.0
        self._last_detection_time = 0.0
        self._total_detections = 0
        self._disease_detections = 0
        self._last_detect_time = 0.0

        self._history: List[DetectionRecord] = []
        self._history_index = 0

    def begin(self, camera_id: int = 0, enabled: bool = True) -> bool:
        """初始化摄像头和推理引擎"""
        self._enabled = enabled
        self._camera_id = camera_id

        if not self._enabled:
            return True

        success = True
        if not self._setup_camera():
            success = False
        if not self._setup_inference():
            logger.warning("推理引擎未加载, 病害检测不可用")
            success = False

        return success

    def _setup_camera(self) -> bool:
        """初始化USB摄像头 (通过OpenCV/VideoCapture读取)"""
        try:
            self._capture = cv2.VideoCapture(self._camera_id)
            if self._capture.isOpened():
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._camera_ready = True
                logger.info(f"摄像头{self._camera_id}初始化成功")
                return True
            else:
                logger.warning(f"摄像头{self._camera_id}打开失败")
                self._camera_ready = False
                return False
        except Exception as e:
            logger.warning(f"摄像头初始化异常: {e}")
            self._camera_ready = False
            return False

    def _setup_inference(self) -> bool:
        """初始化推理引擎 - 优先使用ACL(NPU), 退回ONNX/OpenCV DNN"""

        # 1. 尝试ACL (Atlas NPU推理)
        try:
            import acl
            self._init_acl_engine()
            self._engine_type = "acl"
            self._model_loaded = True
            logger.info("ACL NPU推理引擎初始化成功")
            return True
        except ImportError:
            logger.info("ACL SDK不可用, 尝试ONNX Runtime")
        except Exception as e:
            logger.warning(f"ACL引擎初始化失败: {e}")

        # 2. 尝试ONNX Runtime
        try:
            import onnxruntime as ort
            model_path = f"{self._model_dir}/plant_disease_model.onnx"
            self._inference_engine = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self._engine_type = "onnx"
            self._model_loaded = True
            logger.info("ONNX Runtime推理引擎初始化成功")
            return True
        except ImportError:
            logger.info("ONNX Runtime不可用, 尝试OpenCV DNN")
        except Exception as e:
            logger.warning(f"ONNX引擎初始化失败: {e}")

        # 3. 尝试OpenCV DNN
        try:
            model_path = f"{self._model_dir}/plant_disease_model.onnx"
            self._inference_engine = cv2.dnn.readNetFromONNX(model_path)
            self._engine_type = "opencv_dnn"
            self._model_loaded = True
            logger.info("OpenCV DNN推理引擎初始化成功")
            return True
        except Exception as e:
            logger.warning(f"OpenCV DNN引擎初始化失败: {e}")

        logger.error("所有推理引擎均不可用")
        return False

    def _init_acl_engine(self):
        """初始化ACL (Ascend Computing Language) NPU推理引擎

        使用ACL Python SDK加载.om模型到NPU:
        1. acl.init() -> 初始化ACL
        2. acl.rt.set_device(0) -> 设置NPU设备
        3. 加载.om离线模型 (需用ATC工具从ONNX转换)
        4. 创建输入/输出dataset
        """
        import acl

        RET = acl.init()
        assert RET == 0, f"ACL初始化失败: {RET}"

        RET = acl.rt.set_device(0)
        assert RET == 0, f"设置设备失败: {RET}"

        self._acl_context = acl.rt.create_context(0)
        self._acl_stream = acl.rt.create_stream()

        # 加载.om模型 (需预先用ATC工具转换)
        model_path = f"{self._model_dir}/plant_disease_model.om"
        self._acl_model_id = acl.mdl.load_from_file(model_path)

        # 获取模型描述
        self._acl_model_desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self._acl_model_desc, self._acl_model_id)

        logger.info(f"ACL模型加载完成, model_id={self._acl_model_id}")

    def update(self, now: float = 0.0, light_value: float = 0.0):
        """自动检测入口"""
        if now == 0.0:
            now = time.time()

        if not self._enabled or not self._auto_detect:
            return
        if not self._camera_ready or not self._model_loaded:
            return
        if now - self._last_detect_time < self._detect_interval:
            return

        self.perform_detection()
        self._last_detect_time = now

    def perform_detection(self) -> Optional[DiseaseResult]:
        if not self._enabled or not self._camera_ready or not self._model_loaded:
            return None

        start = time.time()

        # 1. 拍照 + 预处理
        image = self._capture_and_preprocess()
        if image is None:
            return None

        # 2. 推理
        confidences = self._run_inference(image)
        if confidences is None:
            return None

        # 3. 取最大置信度类别
        disease_id = int(np.argmax(confidences))
        confidence = float(confidences[disease_id])

        self._last_disease_id = disease_id
        self._last_confidence = confidence
        self._last_detection_time = time.time()
        self._total_detections += 1

        # 4. 记录历史
        self._add_to_history(disease_id, confidence)

        # 5. 病害告警
        if disease_id > 0 and confidence >= self._confidence_threshold:
            self._disease_detections += 1
            self._trigger_alarm()

        elapsed = (time.time() - start) * 1000
        logger.info(f"病害检测完成 {elapsed:.0f}ms -> "
                     f"{DISEASE_LABELS_CN[disease_id]} ({confidence * 100:.1f}%)")

        return DiseaseResult(
            disease_class=DiseaseClass(disease_id),
            confidence=confidence,
            all_probs=confidences.tolist(),
            timestamp=time.time(),
        )

    def capture_image(self) -> Optional[np.ndarray]:
        """捕获一帧原始图像 (供Web API返回JPEG)"""
        if not self._capture or not self._capture.isOpened():
            return None
        ret, frame = self._capture.read()
        return frame if ret else None

    # ------------------------------------------------------------------
    # 摄像头捕获与预处理
    # ------------------------------------------------------------------
    def _capture_and_preprocess(self) -> Optional[np.ndarray]:
        """拍照并预处理为模型输入格式 (96x96 RGB float32)"""
        if not self._capture or not self._capture.isOpened():
            return None

        ret, frame = self._capture.read()
        if not ret or frame is None:
            logger.debug("摄像头拍照失败")
            return None

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 缩放到96x96
        frame = cv2.resize(frame, (self.INPUT_SIZE, self.INPUT_SIZE))
        # 归一化到[-1, 1]
        frame = frame.astype(np.float32) / 127.5 - 1.0
        # 转为NCHW格式: [1, 3, 96, 96]
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)

        return frame

    # ------------------------------------------------------------------
    # 推理引擎
    # ------------------------------------------------------------------
    def _run_inference(self, image: np.ndarray) -> Optional[np.ndarray]:
        """运行推理, 返回5类概率"""
        if self._engine_type == "onnx":
            return self._run_onnx_inference(image)
        elif self._engine_type == "opencv_dnn":
            return self._run_opencv_dnn_inference(image)
        elif self._engine_type == "acl":
            return self._run_acl_inference(image)
        return None

    def _run_onnx_inference(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            input_name = self._inference_engine.get_inputs()[0].name
            output = self._inference_engine.run(None, {input_name: image})
            probs = output[0][0]
            # Softmax (如果模型输出不是概率)
            exp_vals = np.exp(probs - np.max(probs))
            probs = exp_vals / exp_vals.sum()
            return probs
        except Exception as e:
            logger.error(f"ONNX推理失败: {e}")
            return None

    def _run_opencv_dnn_inference(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            blob = cv2.dnn.blobFromImage(
                image[0].transpose(1, 2, 0),  # NCHW -> HWC
                scalefactor=1.0 / 127.5,
                size=(self.INPUT_SIZE, self.INPUT_SIZE),
                mean=(127.5, 127.5, 127.5),
            )
            self._inference_engine.setInput(blob)
            output = self._inference_engine.forward()
            probs = output.flatten()
            exp_vals = np.exp(probs - np.max(probs))
            probs = exp_vals / exp_vals.sum()
            return probs
        except Exception as e:
            logger.error(f"OpenCV DNN推理失败: {e}")
            return None

    def _run_acl_inference(self, image: np.ndarray) -> Optional[np.ndarray]:
        """ACL NPU推理 - 将numpy数组传入NPU执行"""
        try:
            import acl

            # 创建输入dataset
            input_dataset = acl.mdl.create_dataset()
            input_buffer = image.astype(np.float32).tobytes()
            input_data = acl.util.bytes_to_ptr(input_buffer)
            input_size = len(input_buffer)

            dev_buffer = acl.rt.malloc_host(input_size)
            acl.rt.memcpy(dev_buffer, input_size, input_data, input_size,
                          acl.rt.MEMCPY_HOST_TO_DEVICE)

            dataset_buffer = acl.create_data_buffer(dev_buffer, input_size)
            acl.mdl.add_dataset_buffer(input_dataset, dataset_buffer)

            # 创建输出dataset
            output_dataset = acl.mdl.create_dataset()
            output_size = self._NUM_CLASSES * 4  # 5 * float32
            dev_output = acl.rt.malloc_host(output_size)
            dataset_output = acl.create_data_buffer(dev_output, output_size)
            acl.mdl.add_dataset_buffer(output_dataset, dataset_output)

            # 执行推理
            acl.mdl.execute(self._acl_model_id, input_dataset, output_dataset)

            # 获取输出
            output_buffer = acl.util.ptr_to_bytes(dev_output, output_size)
            probs = np.frombuffer(output_buffer, dtype=np.float32)

            # Softmax
            exp_vals = np.exp(probs - np.max(probs))
            probs = exp_vals / exp_vals.sum()

            # 释放资源
            acl.rt.free_host(dev_buffer)
            acl.rt.free_host(dev_output)

            return probs
        except Exception as e:
            logger.error(f"ACL NPU推理失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 告警
    # ------------------------------------------------------------------
    def _trigger_alarm(self):
        if self._buzzer_enabled and self._buzzer:
            self._buzzer.beep(count=3, on_ms=100, off_ms=100)

    def _add_to_history(self, disease_id: int, confidence: float):
        now = time.strftime("%H:%M:%S")
        record = DetectionRecord(disease_id=disease_id, confidence=confidence, timestamp=now)
        if len(self._history) < self.HISTORY_SIZE:
            self._history.append(record)
        else:
            self._history[self._history_index] = record
        self._history_index = (self._history_index + 1) % self.HISTORY_SIZE

    @property
    def camera_ready(self) -> bool:
        return self._camera_ready

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def to_dict(self) -> dict:
        return {
            "enabled": self._enabled,
            "cameraReady": self._camera_ready,
            "modelLoaded": self._model_loaded,
            "lastDiseaseId": self._last_disease_id,
            "lastDiseaseName": DISEASE_LABELS[self._last_disease_id],
            "lastDiseaseNameCn": DISEASE_LABELS_CN[self._last_disease_id],
            "lastConfidence": round(self._last_confidence, 4),
            "lastDetectionTime": time.strftime("%H:%M:%S", time.localtime(self._last_detection_time))
                                  if self._last_detection_time > 0 else "",
            "treatment": TREATMENTS[self._last_disease_id],
            "totalDetections": self._total_detections,
            "diseaseDetections": self._disease_detections,
            "autoDetect": self._auto_detect,
            "detectInterval": self._detect_interval,
            "confidenceThreshold": self._confidence_threshold,
            "buzzerEnabled": self._buzzer_enabled,
            "engineType": self._engine_type,
        }

    def close(self):
        if self._capture and self._capture.isOpened():
            self._capture.release()
