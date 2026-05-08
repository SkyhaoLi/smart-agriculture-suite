#include "PlantDoctorModule.h"

#include <math.h>

// ESP32-S3 使用DVP接口摄像头 (硬件限制, 不支持USB摄像头)
// Atlas 200I DK A2 版使用USB摄像头 (通过OpenCV/VideoCapture读取)
#include "esp_camera.h"
#include "esp_heap_caps.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "plant_disease_model.h"

namespace agri {

namespace {
constexpr int kTensorArenaSize = 150 * 1024;
constexpr int kHistorySize = 10;
constexpr int kNumClasses = PLANT_MODEL_NUM_CLASSES;
constexpr int kInputWidth = PLANT_MODEL_INPUT_WIDTH;
constexpr int kInputHeight = PLANT_MODEL_INPUT_HEIGHT;
constexpr int kInputChannels = PLANT_MODEL_INPUT_CHANNELS;
}  // namespace

const char* PlantDoctorModule::diseaseLabel(int diseaseId) {
    static const char* kLabels[] = {
        "Healthy",
        "Strawberry_Anthracnose",
        "Strawberry_Gray_Mold",
        "Strawberry_Leaf_Scorch",
        "Strawberry_Powdery_Mildew"
    };
    const int count = sizeof(kLabels) / sizeof(kLabels[0]);
    return kLabels[constrain(diseaseId, 0, count - 1)];
}

const char* PlantDoctorModule::diseaseLabelCn(int diseaseId) {
    static const char* kLabels[] = {
        "健康",
        "草莓炭疽病",
        "草莓灰霉病",
        "草莓叶焦病",
        "草莓白粉病"
    };
    const int count = sizeof(kLabels) / sizeof(kLabels[0]);
    return kLabels[constrain(diseaseId, 0, count - 1)];
}

const char* PlantDoctorModule::treatment(int diseaseId) {
    static const char* kSuggestions[] = {
        "叶片健康。保持通风和定期检查。",
        "清除病株残体，避免伤口感染。施用咪鲜胺等杀菌剂。",
        "降低湿度，增加通风。及时摘除病果，施用嘧霉胺等杀菌剂。",
        "移除感染叶片，降低叶面湿度。必要时施用杀菌剂。",
        "移除病叶，改善通风。喷施硫制剂或三唑类杀菌剂。"
    };
    const int count = sizeof(kSuggestions) / sizeof(kSuggestions[0]);
    return kSuggestions[constrain(diseaseId, 0, count - 1)];
}

void PlantDoctorModule::begin(const PinConfig& pins, const PlantDoctorConfig& config) {
    pins_ = pins;
    config_ = config;

    if (pins_.buzzerPin >= 0) {
        pinMode(pins_.buzzerPin, OUTPUT);
        digitalWrite(pins_.buzzerPin, LOW);
    }

    if (config_.enabled) {
        setupCamera();
        setupTflite();
    }
}

void PlantDoctorModule::update(unsigned long nowMs, float lightValue) {
    lightValue_ = lightValue;
    if (!config_.enabled || !config_.autoDetect || !cameraReady_ || !modelLoaded_) {
        return;
    }
    if (nowMs - lastDetectMs_ < static_cast<unsigned long>(config_.detectIntervalSec) * 1000UL) {
        return;
    }
    performDetection();
    lastDetectMs_ = nowMs;
}

void PlantDoctorModule::setConfig(const PlantDoctorConfig& config) {
    const bool enablingNow = !config_.enabled && config.enabled;
    const bool disablingNow = config_.enabled && !config.enabled;
    config_ = config;

    if (enablingNow) {
        setupCamera();
        setupTflite();
    } else if (disablingNow) {
        deinit();
    }
}

void PlantDoctorModule::deinit() {
    if (cameraReady_) {
        esp_camera_deinit();
        cameraReady_ = false;
        Serial.println("[PlantDoctor] camera deinitialized");
    }
    modelLoaded_ = false;
    input_ = nullptr;
    output_ = nullptr;
    interpreter_ = nullptr;
    // Note: tensor arena memory is kept allocated to avoid PSRAM fragmentation
    Serial.println("[PlantDoctor] module deinitialized");
}

bool PlantDoctorModule::performDetection() {
    if (!config_.enabled || !cameraReady_ || !modelLoaded_ || input_ == nullptr) {
        return false;
    }

    const unsigned long start = millis();
    if (!captureAndPreprocess(input_->data.int8)) {
        return false;
    }

    float confidences[kNumClasses] = {};
    const int diseaseId = runInference(confidences);
    const float confidence = confidences[diseaseId];

    lastDiseaseId_ = diseaseId;
    lastConfidence_ = confidence;
    lastDetectionTime_ = timestampFromMillis(millis());
    ++totalDetections_;
    addToHistory(diseaseId, confidence);

    if (diseaseId > 0 && confidence >= config_.confidenceThreshold) {
        ++diseaseDetections_;
        triggerAlarm(diseaseId);
    }

    Serial.printf("[PlantDoctor] detection finished in %lu ms -> %s (%.1f%%)\n",
                  millis() - start, diseaseLabel(diseaseId), confidence * 100.0f);
    return true;
}

void PlantDoctorModule::handleCapture(WebServer& server) {
    if (!cameraReady_) {
        server.send(503, "application/json", "{\"error\":\"Camera not ready\"}");
        return;
    }

    camera_fb_t* frameBuffer = esp_camera_fb_get();
    if (frameBuffer == nullptr) {
        server.send(500, "application/json", "{\"error\":\"Capture failed\"}");
        return;
    }

    server.sendHeader("Content-Disposition", "inline; filename=capture.jpg");
    server.setContentLength(frameBuffer->len);
    server.send(200, "image/jpeg", "");
    server.client().write(frameBuffer->buf, frameBuffer->len);

    esp_camera_fb_return(frameBuffer);
}

void PlantDoctorModule::writeStatus(JsonDocument& doc) const {
    doc["enabled"] = config_.enabled;
    doc["cameraReady"] = cameraReady_;
    doc["modelLoaded"] = modelLoaded_;
    doc["lightValue"] = lightValue_;
    doc["lastDiseaseId"] = lastDiseaseId_;
    doc["lastDiseaseName"] = diseaseLabel(lastDiseaseId_);
    doc["lastDiseaseNameCn"] = diseaseLabelCn(lastDiseaseId_);
    doc["lastConfidence"] = lastConfidence_;
    doc["lastDetectionTime"] = lastDetectionTime_;
    doc["treatment"] = treatment(lastDiseaseId_);
    doc["totalDetections"] = totalDetections_;
    doc["diseaseDetections"] = diseaseDetections_;
    doc["autoDetect"] = config_.autoDetect;
    doc["detectInterval"] = config_.detectIntervalSec;
    doc["confidenceThreshold"] = config_.confidenceThreshold;
    doc["buzzerEnabled"] = config_.buzzerEnabled;
}

void PlantDoctorModule::writeHistory(JsonDocument& doc) const {
    JsonArray history = doc.to<JsonArray>();
    for (int i = 0; i < kHistorySize; ++i) {
        const int idx = (historyIndex_ - 1 - i + kHistorySize) % kHistorySize;
        if (history_[idx].timestamp.isEmpty()) {
            continue;
        }
        JsonObject item = history.add<JsonObject>();
        item["diseaseId"] = history_[idx].diseaseId;
        item["diseaseName"] = diseaseLabel(history_[idx].diseaseId);
        item["diseaseNameCn"] = diseaseLabelCn(history_[idx].diseaseId);
        item["confidence"] = history_[idx].confidence;
        item["timestamp"] = history_[idx].timestamp;
    }
}

void PlantDoctorModule::setupCamera() {
    if (cameraReady_ || !pins_.camera.enabled) {
        return;
    }

    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = pins_.camera.d0;
    config.pin_d1 = pins_.camera.d1;
    config.pin_d2 = pins_.camera.d2;
    config.pin_d3 = pins_.camera.d3;
    config.pin_d4 = pins_.camera.d4;
    config.pin_d5 = pins_.camera.d5;
    config.pin_d6 = pins_.camera.d6;
    config.pin_d7 = pins_.camera.d7;
    config.pin_xclk = pins_.camera.xclk;
    config.pin_pclk = pins_.camera.pclk;
    config.pin_vsync = pins_.camera.vsync;
    config.pin_href = pins_.camera.href;
    config.pin_sccb_sda = pins_.camera.sccbSda;
    config.pin_sccb_scl = pins_.camera.sccbScl;
    config.pin_pwdn = pins_.camera.pwdn;
    config.pin_reset = pins_.camera.reset;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_96X96;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    const esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[PlantDoctor] camera init failed: 0x%x\n", err);
        cameraReady_ = false;
        return;
    }

    sensor_t* sensor = esp_camera_sensor_get();
    if (sensor != nullptr) {
        sensor->set_brightness(sensor, 0);
        sensor->set_contrast(sensor, 0);
        sensor->set_saturation(sensor, 0);
        sensor->set_whitebal(sensor, 1);
        sensor->set_awb_gain(sensor, 1);
        sensor->set_wb_mode(sensor, 0);
    }

    cameraReady_ = true;
}

void PlantDoctorModule::setupTflite() {
    if (modelLoaded_) {
        return;
    }

    static tflite::MicroErrorReporter microErrorReporter;
    static tflite::MicroMutableOpResolver<15> resolver;
    static bool resolverInitialized = false;

    if (!resolverInitialized) {
        resolver.AddAdd();
        resolver.AddConv2D();
        resolver.AddDepthwiseConv2D();
        resolver.AddAveragePool2D();
        resolver.AddFullyConnected();
        resolver.AddMaxPool2D();
        resolver.AddMul();
        resolver.AddPack();
        resolver.AddPad();
        resolver.AddRelu();
        resolver.AddRelu6();
        resolver.AddReshape();
        resolver.AddShape();
        resolver.AddSoftmax();
        resolver.AddStridedSlice();
        resolverInitialized = true;
    }

    errorReporter_ = &microErrorReporter;

    if (tensorArena_ == nullptr) {
        tensorArena_ = static_cast<uint8_t*>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM));
        if (tensorArena_ == nullptr) {
            tensorArena_ = static_cast<uint8_t*>(malloc(kTensorArenaSize));
        }
    }
    if (tensorArena_ == nullptr) {
        Serial.println("[PlantDoctor] tensor arena allocation failed");
        return;
    }

    model_ = tflite::GetModel(plant_disease_model_tflite);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[PlantDoctor] model schema mismatch: %d vs %d\n", model_->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroInterpreter staticInterpreter(model_, resolver, tensorArena_, kTensorArenaSize, errorReporter_);
    interpreter_ = &staticInterpreter;

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        Serial.println("[PlantDoctor] AllocateTensors failed");
        return;
    }

    input_ = interpreter_->input(0);
    output_ = interpreter_->output(0);
    modelLoaded_ = input_ != nullptr && output_ != nullptr;

    if (modelLoaded_) {
        Serial.printf("[PlantDoctor] model ready, arena used %d bytes\n", interpreter_->arena_used_bytes());
    }
}

bool PlantDoctorModule::captureAndPreprocess(int8_t* inputBuffer) {
    camera_fb_t* frameBuffer = esp_camera_fb_get();
    if (frameBuffer == nullptr) {
        Serial.println("[PlantDoctor] camera capture failed");
        return false;
    }

    uint16_t* pixels = reinterpret_cast<uint16_t*>(frameBuffer->buf);
    int index = 0;
    const float inputScale = input_->params.scale;
    const int32_t inputZeroPoint = input_->params.zero_point;

    auto quantize = [inputScale, inputZeroPoint](uint8_t channel) -> int8_t {
        const float normalized = (static_cast<float>(channel) / 127.5f) - 1.0f;
        int32_t quantized = static_cast<int32_t>(roundf(normalized / inputScale)) + inputZeroPoint;
        quantized = constrain(quantized, -128, 127);
        return static_cast<int8_t>(quantized);
    };

    for (int y = 0; y < kInputHeight; ++y) {
        for (int x = 0; x < kInputWidth; ++x) {
            const uint16_t pixel = pixels[y * kInputWidth + x];
            const uint8_t r = static_cast<uint8_t>(((pixel >> 11) & 0x1F) << 3);
            const uint8_t g = static_cast<uint8_t>(((pixel >> 5) & 0x3F) << 2);
            const uint8_t b = static_cast<uint8_t>((pixel & 0x1F) << 3);

            if (kInputChannels >= 1) {
                inputBuffer[index++] = quantize(r);
            }
            if (kInputChannels >= 2) {
                inputBuffer[index++] = quantize(g);
            }
            if (kInputChannels >= 3) {
                inputBuffer[index++] = quantize(b);
            }
        }
    }

    esp_camera_fb_return(frameBuffer);
    return true;
}

int PlantDoctorModule::runInference(float confidences[]) {
    if (interpreter_->Invoke() != kTfLiteOk) {
        Serial.println("[PlantDoctor] inference failed");
        return 0;
    }

    int8_t* outputData = output_->data.int8;
    const float scale = output_->params.scale;
    const int32_t zeroPoint = output_->params.zero_point;

    int maxIndex = 0;
    float maxValue = -1000.0f;
    for (int i = 0; i < kNumClasses; ++i) {
        confidences[i] = (outputData[i] - zeroPoint) * scale;
        if (confidences[i] > maxValue) {
            maxValue = confidences[i];
            maxIndex = i;
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < kNumClasses; ++i) {
        confidences[i] = expf(confidences[i] - maxValue);
        sum += confidences[i];
    }
    for (int i = 0; i < kNumClasses; ++i) {
        confidences[i] /= sum;
    }

    return maxIndex;
}

void PlantDoctorModule::triggerAlarm(int diseaseId) const {
    if (!config_.buzzerEnabled || pins_.buzzerPin < 0 || diseaseId <= 0) {
        return;
    }

    for (int i = 0; i < 3; ++i) {
        digitalWrite(pins_.buzzerPin, HIGH);
        delay(100);
        digitalWrite(pins_.buzzerPin, LOW);
        delay(100);
    }
}

void PlantDoctorModule::addToHistory(int diseaseId, float confidence) {
    history_[historyIndex_].diseaseId = diseaseId;
    history_[historyIndex_].confidence = confidence;
    history_[historyIndex_].timestamp = timestampFromMillis(millis());
    historyIndex_ = (historyIndex_ + 1) % kHistorySize;
}

String PlantDoctorModule::timestampFromMillis(unsigned long ms) {
    const unsigned long seconds = ms / 1000UL;
    const unsigned long minutes = seconds / 60UL;
    const unsigned long hours = minutes / 60UL;

    char buffer[20];
    snprintf(buffer, sizeof(buffer), "%02lu:%02lu:%02lu", hours % 24UL, minutes % 60UL, seconds % 60UL);
    return String(buffer);
}

}  // namespace agri
