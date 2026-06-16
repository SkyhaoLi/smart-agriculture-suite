"""
智润智慧农业套件 - Atlas 200I DK A2 版
植物病害检测模块 - 使用NPU(Ascend 310B)进行AI推理

对应原ESP32项目的 PlantDoctorModule.h/PlantDoctorModule.cpp
ESP32使用TFLite Micro + DVP摄像头, Atlas 200I DK A2使用:
- MIPI-CSI摄像头 (原生支持, 非DVP)
- ACL Python SDK 或 ONNX Runtime 加载病害检测模型
- NPU 8 TOPS INT8 推理, 性能远超ESP32

5类草莓病害 + MobileNetV2特征匹配泛化识别
"""

import cv2
import time
import json
import logging
import threading
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

from config.app_types import DiseaseClass, DiseaseResult, DISEASE_NAMES
from config.hardware_config import MODEL_DIR

logger = logging.getLogger(__name__)

DISEASE_LABELS = ["Healthy", "Strawberry_Anthracnose", "Strawberry_Gray_Mold",
                  "Strawberry_Leaf_Scorch", "Strawberry_Powdery_Mildew"]
DISEASE_LABELS_CN = ["健康", "其他病害", "未训练类别", "叶部病斑类", "未训练类别"]
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
    """植物病害检测 - Atlas 200I DK A2 NPU推理版 + 特征匹配泛化"""

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
        self._camera_type = "none"  # "local", "rtsp", "none"
        self._capture = None
        self._last_uploaded_frame = None

        # ACL/ONNX推理引擎 (根据实际环境选择)
        self._inference_engine = None
        self._engine_type = "none"  # "acl", "onnx", "opencv_dnn"

        self._last_disease_id = 0
        self._last_confidence = 0.0
        self._last_all_probs = None
        self._last_detection_time = 0.0
        self._total_detections = 0
        self._disease_detections = 0
        self._last_detect_time = 0.0

        self._history: List[DetectionRecord] = []
        self._history_index = 0

        # 特征匹配 (泛化识别)
        self._feature_extractor = None
        self._feature_library_loaded = False
        self._last_feature_matches = []

    def begin(self, camera_id=0, enabled: bool = True) -> bool:
        """初始化摄像头和推理引擎

        camera_id: int (本地摄像头ID) 或 str (RTSP URL)
        """
        self._enabled = enabled

        if not self._enabled:
            return True

        # 如果camera_id是字符串且以rtsp://开头, 使用RTSP
        if isinstance(camera_id, str) and (camera_id.startswith("rtsp://") or camera_id.startswith("http://")):
            self._camera_id = camera_id
        else:
            self._camera_id = camera_id

        if not self._enabled:
            return True

        success = True
        if not self._setup_camera():
            success = False
        if not self._setup_inference():
            logger.warning("推理引擎未加载, 病害检测不可用")
            success = False

        # 初始化特征匹配器
        self._setup_feature_matcher()

        return success

    def _setup_feature_matcher(self):
        """初始化 MobileNetV2 特征匹配器"""
        try:
            from src.ai.disease_feature_extractor import DiseaseFeatureExtractor
            model_path = f"{self._model_dir}/mobilenetv2_features.onnx"
            library_path = f"{self._model_dir}/disease_features.json"
            self._feature_extractor = DiseaseFeatureExtractor(model_path, library_path)
            if self._feature_extractor.begin():
                self._feature_library_loaded = True
                stats = self._feature_extractor.get_library_stats()
                logger.info(f"特征匹配器就绪: {stats['total_diseases']} 种病害, "
                            f"{stats['total_features']} 条特征")
            else:
                logger.warning("特征匹配器初始化失败")
        except Exception as e:
            logger.warning(f"特征匹配器不可用: {e}")

    def match_disease(self, frame: np.ndarray) -> List[dict]:
        """使用特征匹配识别病害 (泛化到多作物)

        Args:
            frame: BGR 格式原始图像

        Returns:
            [{disease, similarity, source, treatment}, ...]
        """
        if not self._feature_library_loaded or self._feature_extractor is None:
            logger.warning(f"Feature matching not ready: loaded={self._feature_library_loaded}, extractor={self._feature_extractor is not None}")
            return []
        logger.info(f"Running feature matching on frame shape: {frame.shape}")
        result = self._feature_extractor.match(frame, top_k=3)
        logger.info(f"Feature matching returned {len(result)} results")
        return result

    def detect_from_image(self, image_bytes: bytes) -> Optional[DiseaseResult]:
        """从上传的图片字节进行病害检测 (不依赖摄像头)"""
        if not self._enabled or not self._model_loaded:
            return None

        start = time.time()

        # 解码图片
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("图片解码失败")
            return None

        # 保存原图
        self._last_uploaded_frame = frame

        # 1. 原有模型推理 (草莓5类)
        image = self._preprocess_frame(frame)
        if image is None:
            return None
        confidences = self._run_inference(image)
        if confidences is None:
            return None
        disease_id = int(np.argmax(confidences))
        confidence = float(confidences[disease_id])

        # 2. 特征匹配 (泛化识别)
        feature_matches = self.match_disease(frame)
        self._last_feature_matches = feature_matches

        # 3. 综合判断: 如果原模型置信度低且有特征匹配结果, 使用特征匹配
        final_disease_id = disease_id
        final_confidence = confidence
        if confidence < 0.5 and feature_matches:
            best_match = feature_matches[0]
            match_name = best_match.get("disease", "")
            for i, label in enumerate(DISEASE_LABELS_CN):
                if label in match_name or match_name in label:
                    final_disease_id = i
                    final_confidence = best_match["similarity"]
                    break
            else:
                if "健康" not in match_name and best_match["similarity"] > 0.5:
                    final_disease_id = 1
                    final_confidence = best_match["similarity"]
            logger.info(f"原模型置信度低({confidence:.2f}), "
                        f"特征匹配: {match_name} ({final_confidence:.2f})")

        self._last_disease_id = final_disease_id
        self._last_confidence = final_confidence
        self._last_all_probs = confidences.tolist()
        self._last_detection_time = time.time()
        self._total_detections += 1

        # 记录历史
        self._add_to_history(final_disease_id, final_confidence)

        # 病害告警
        if final_disease_id > 0 and final_confidence >= self._confidence_threshold:
            self._disease_detections += 1
            self._trigger_alarm()

        elapsed = (time.time() - start) * 1000
        logger.info(f"图片检测完成 {elapsed:.0f}ms -> "
                     f"{DISEASE_LABELS_CN[final_disease_id]} ({final_confidence * 100:.1f}%)")

        return DiseaseResult(
            disease_class=DiseaseClass(final_disease_id),
            confidence=final_confidence,
            all_probs=confidences.tolist(),
            timestamp=time.time(),
        )

    def get_last_uploaded_jpeg(self) -> Optional[bytes]:
        """获取最后上传图片的JPEG编码 (供Web返回)"""
        frame = getattr(self, '_last_uploaded_frame', None)
        if frame is None:
            return None
        _, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """将BGR帧预处理为模型输入格式 (1,3,96,96 float32 NCHW)"""
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.INPUT_SIZE, self.INPUT_SIZE))
            frame = frame.astype(np.float32) / 127.5 - 1.0
            frame = np.transpose(frame, (2, 0, 1))
            frame = np.expand_dims(frame, axis=0)
            return frame
        except Exception as e:
            logger.error(f"图片预处理失败: {e}")
            return None

    def _setup_camera(self) -> bool:
        """初始化摄像头 - 支持MIPI-CSI(本地)和大华RTSP(网络)"""
        camera_source = str(self._camera_id)

        # 判断是否为RTSP网络摄像头 (大华等)
        if camera_source.startswith("rtsp://") or camera_source.startswith("http://"):
            return self._setup_rtsp_camera(camera_source)

        # 本地MIPI-CSI摄像头
        return self._setup_local_camera(int(self._camera_id))

    def _setup_local_camera(self, camera_id: int) -> bool:
        """初始化本地MIPI-CSI摄像头"""
        try:
            self._capture = cv2.VideoCapture(camera_id)
            if self._capture.isOpened():
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._camera_ready = True
                self._camera_type = "local"
                logger.info(f"本地摄像头{camera_id}初始化成功")
                return True
            else:
                logger.warning(f"本地摄像头{camera_id}打开失败")
                self._camera_ready = False
                return False
        except Exception as e:
            logger.warning(f"本地摄像头初始化异常: {e}")
            self._camera_ready = False
            return False

    def _setup_rtsp_camera(self, rtsp_url: str) -> bool:
        """初始化大华RTSP网络摄像头"""
        try:
            # 大华摄像头RTSP格式: rtsp://user:password@ip:port/cam/realmonitor?channel=1&subtype=0
            self._capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if self._capture.isOpened():
                self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟
                self._start_frame_reader()
                self._camera_ready = True
                self._camera_type = "rtsp"
                logger.info(f"RTSP摄像头初始化成功: {rtsp_url[:50]}...")
                return True
            else:
                logger.warning(f"RTSP摄像头连接失败: {rtsp_url[:50]}")
                self._camera_ready = False
                return False
        except Exception as e:
            logger.warning(f"RTSP摄像头初始化异常: {e}")
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

        # 获取原始帧 (用于特征匹配)
        frame = self.capture_image()

        # 2. 推理 (原有草莓模型)
        confidences = self._run_inference(image)
        if confidences is None:
            return None

        # 3. 取最大置信度类别
        disease_id = int(np.argmax(confidences))
        confidence = float(confidences[disease_id])

        # 4. 特征匹配 (泛化识别)
        feature_matches = []
        if frame is not None:
            feature_matches = self.match_disease(frame)
            self._last_feature_matches = feature_matches
            if feature_matches:
                logger.info(f"特征匹配结果: {feature_matches[0]['disease']} ({feature_matches[0]['similarity']:.2f})")
        else:
            logger.warning("frame is None, skipping feature matching")

        # 5. 综合判断: 如果原模型置信度低且有特征匹配结果, 使用特征匹配
        final_disease_id = disease_id
        final_confidence = confidence
        if confidence < 0.5 and feature_matches:
            best_match = feature_matches[0]
            # 映射特征匹配结果到disease_id
            match_name = best_match.get("disease", "")
            for i, label in enumerate(DISEASE_LABELS_CN):
                if label in match_name or match_name in label:
                    final_disease_id = i
                    final_confidence = best_match["similarity"]
                    break
            else:
                # 特征匹配发现病害但不在5类中, 标记为其他病害(id=1)
                if "健康" not in match_name and best_match["similarity"] > 0.5:
                    final_disease_id = 1
                    final_confidence = best_match["similarity"]
            logger.info(f"置信度低({confidence:.2f}), 使用特征匹配: "
                        f"{match_name} ({final_confidence:.2f})")

        self._last_disease_id = final_disease_id
        self._last_confidence = final_confidence
        self._last_all_probs = confidences.tolist()
        self._last_detection_time = time.time()
        self._total_detections += 1

        # 6. 记录历史
        self._add_to_history(final_disease_id, final_confidence)

        # 7. 病害告警
        if final_disease_id > 0 and final_confidence >= self._confidence_threshold:
            self._disease_detections += 1
            self._trigger_alarm()

        elapsed = (time.time() - start) * 1000
        match_info = ""
        if feature_matches:
            match_info = f" [特征匹配: {feature_matches[0]['disease']} {feature_matches[0]['similarity']:.2f}]"
        logger.info(f"病害检测完成 {elapsed:.0f}ms -> "
                     f"{DISEASE_LABELS_CN[final_disease_id]} ({final_confidence * 100:.1f}%){match_info}")

        return DiseaseResult(
            disease_class=DiseaseClass(final_disease_id),
            confidence=final_confidence,
            all_probs=confidences.tolist(),
            timestamp=time.time(),
        )

    def _start_frame_reader(self):
        if getattr(self, '_frame_reader_started', False):
            return
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._frame_reader_started = True

        def reader():
            while self._frame_reader_started and self._capture and self._capture.isOpened():
                ret, frame = self._capture.read()
                if ret and frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame

        threading.Thread(target=reader, daemon=True).start()

    def capture_image(self) -> Optional[np.ndarray]:
        """捕获一帧原始图像 (供Web API返回JPEG)"""
        frame = getattr(self, '_latest_frame', None)
        lock = getattr(self, '_frame_lock', None)
        if frame is not None and lock is not None:
            with lock:
                return self._latest_frame.copy()
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

        frame = self.capture_image()
        if frame is None:
            logger.debug("摄像头拍照失败")
            return None

        return self._preprocess_frame(frame)

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
            # 仅当输出不是概率时才做softmax (检查sum是否接近1.0)
            if abs(float(probs.sum()) - 1.0) > 0.05:
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
            if abs(float(probs.sum()) - 1.0) > 0.05:
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

            # 仅当输出不是概率时才做softmax
            if abs(float(probs.sum()) - 1.0) > 0.05:
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
        # 综合判断: 如果原模型置信度低且有特征匹配结果，使用特征匹配
        final_disease = DISEASE_LABELS_CN[self._last_disease_id]
        final_confidence = self._last_confidence
        final_treatment = TREATMENTS[self._last_disease_id]
        final_crop = "草莓"  # 默认作物

        if self._last_confidence < 0.5 and self._last_feature_matches:
            best_match = self._last_feature_matches[0]
            final_disease = best_match["disease"]
            final_confidence = best_match["similarity"]
            final_treatment = best_match.get("treatment", "")
            # 从病害名称提取作物
            if "___" in best_match.get("source", ""):
                final_crop = best_match["source"].split("___")[0]
            else:
                # 从病害中文名推断作物
                for crop_name in ["番茄", "马铃薯", "玉米", "葡萄", "苹果", "桃", "辣椒", "草莓"]:
                    if crop_name in final_disease:
                        final_crop = crop_name
                        break

        result = {
            "enabled": self._enabled,
            "cameraReady": self._camera_ready,
            "modelLoaded": self._model_loaded,
            "lastDiseaseId": self._last_disease_id,
            "lastDiseaseName": DISEASE_LABELS[self._last_disease_id],
            "lastDiseaseNameCn": DISEASE_LABELS_CN[self._last_disease_id],
            "lastConfidence": round(self._last_confidence, 4),
            "lastDetectionTime": time.strftime("%H:%M:%S", time.localtime(self._last_detection_time))
                                  if self._last_detection_time > 0 else "",
            "allProbs": self._last_all_probs,
            "treatment": TREATMENTS[self._last_disease_id],
            "totalDetections": self._total_detections,
            "diseaseDetections": self._disease_detections,
            "autoDetect": self._auto_detect,
            "detectInterval": self._detect_interval,
            "confidenceThreshold": self._confidence_threshold,
            "buzzerEnabled": self._buzzer_enabled,
            "engineType": self._engine_type,
            # 综合识别结果
            "finalCrop": final_crop,
            "finalDisease": final_disease,
            "finalConfidence": round(final_confidence, 4),
            "finalTreatment": final_treatment,
            # 特征匹配 (泛化识别)
            "featureMatchEnabled": self._feature_library_loaded,
            "featureMatches": [
                {"disease": m["disease"], "similarity": m["similarity"],
                 "treatment": m.get("treatment", "")}
                for m in self._last_feature_matches
            ],
            # 检测历史
            "detectionHistory": [
                {"diseaseId": r.disease_id, "confidence": r.confidence, "timestamp": r.timestamp}
                for r in reversed(self._history)
            ],
        }
        # 添加特征库统计
        if self._feature_extractor:
            result["featureLibrary"] = self._feature_extractor.get_library_stats()
        return result

    def close(self):
        if self._capture and self._capture.isOpened():
            self._capture.release()
