#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>
#include <WebServer.h>

#include "AppConfig.h"
#include "AppTypes.h"

struct TfLiteTensor;

namespace tflite {
class ErrorReporter;
class MicroInterpreter;
class Model;
}  // namespace tflite

namespace agri {

class PlantDoctorModule {
public:
    struct DetectionRecord {
        int diseaseId = 0;
        float confidence = 0.0f;
        String timestamp;
    };

    void begin(const PinConfig& pins, const PlantDoctorConfig& config);
    void update(unsigned long nowMs, float lightValue);
    void setConfig(const PlantDoctorConfig& config);
    const PlantDoctorConfig& config() const { return config_; }
    bool enabled() const { return config_.enabled; }
    bool cameraReady() const { return cameraReady_; }
    bool modelLoaded() const { return modelLoaded_; }
    bool performDetection();
    void handleCapture(WebServer& server);
    void writeStatus(JsonDocument& doc) const;
    void writeHistory(JsonDocument& doc) const;

    static const char* diseaseLabel(int diseaseId);
    static const char* diseaseLabelCn(int diseaseId);
    static const char* treatment(int diseaseId);

private:
    void setupCamera();
    void setupTflite();
    void deinit();
    bool captureAndPreprocess(int8_t* inputBuffer);
    int runInference(float confidences[]);
    void triggerAlarm(int diseaseId) const;
    void addToHistory(int diseaseId, float confidence);
    static String timestampFromMillis(unsigned long ms);

    PinConfig pins_{};
    PlantDoctorConfig config_{};
    bool cameraReady_ = false;
    bool modelLoaded_ = false;
    float lightValue_ = 0.0f;
    int lastDiseaseId_ = 0;
    float lastConfidence_ = 0.0f;
    String lastDetectionTime_;
    int totalDetections_ = 0;
    int diseaseDetections_ = 0;
    unsigned long lastDetectMs_ = 0;
    DetectionRecord history_[10];
    int historyIndex_ = 0;
    uint8_t* tensorArena_ = nullptr;
    tflite::ErrorReporter* errorReporter_ = nullptr;
    const tflite::Model* model_ = nullptr;
    tflite::MicroInterpreter* interpreter_ = nullptr;
    TfLiteTensor* input_ = nullptr;
    TfLiteTensor* output_ = nullptr;
};

}  // namespace agri
