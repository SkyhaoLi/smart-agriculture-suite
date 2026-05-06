#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <Preferences.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "ActuatorController.h"
#include "AnomalyModule.h"
#include "AppConfig.h"
#include "ConfigManager.h"
#include "FusionModule.h"
#include "GrowthModule.h"
#include "IrrigationModule.h"
#include "LearningModule.h"
#include "OtaManager.h"
#include "PlantDoctorModule.h"
#include "Sensors.h"
#include "WebFrontend.h"

using namespace agri;

namespace {

constexpr int kScreenWidth = 128;
constexpr int kScreenHeight = 64;
constexpr int kScreenAddress = 0x3C;

PinConfig gPins = defaultPins();
SystemConfig gSystemConfig = defaultSystemConfig();

WebServer gServer(80);
Adafruit_SSD1306 gDisplay(kScreenWidth, kScreenHeight, &Wire, -1);
ConfigManager gConfigManager;
OtaManager gOta;

SensorHub gSensorHub;
ActuatorController gActuator;
IrrigationModule gIrrigation;
AnomalyModule gAnomaly;
GrowthModule gGrowth;
LearningModule gLearning;
FusionModule gFusion;
PlantDoctorModule gPlantDoctor;

bool gDisplayReady = false;
bool gWifiConnected = false;
String gIpAddress = "offline";
unsigned long gLastDisplayRefreshMs = 0;
unsigned long gLastDisplayPageMs = 0;
uint8_t gDisplayPage = 0;

void sendJson(int code, JsonDocument& doc) {
    String body;
    serializeJson(doc, body);
    gServer.send(code, "application/json", body);
}

void sendError(int code, const char* message) {
    JsonDocument doc;
    doc["error"] = message;
    sendJson(code, doc);
}

bool parseJsonBody(JsonDocument& doc) {
    if (!gServer.hasArg("plain")) {
        sendError(400, "No body");
        return false;
    }

    const DeserializationError error = deserializeJson(doc, gServer.arg("plain"));
    if (error) {
        sendError(400, "Invalid JSON");
        return false;
    }
    return true;
}

void connectWiFi() {
    // Prefer NVS-stored credentials over hardcoded defaults
    String ssid = gConfigManager.loadWifiSsid();
    String password = gConfigManager.loadWifiPassword();
    if (ssid.isEmpty()) {
        ssid = kWifiSsid;
        password = kWifiPassword;
    }

    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid.c_str(), password.c_str());

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        ++attempts;
    }

    gWifiConnected = WiFi.status() == WL_CONNECTED;
    gIpAddress = gWifiConnected ? WiFi.localIP().toString() : String("offline");
}

void setupDisplay() {
    if (!gDisplay.begin(SSD1306_SWITCHCAPVCC, kScreenAddress)) {
        gDisplayReady = false;
        return;
    }

    gDisplayReady = true;
    gDisplay.clearDisplay();
    gDisplay.setTextSize(1);
    gDisplay.setTextColor(SSD1306_WHITE);
    gDisplay.setCursor(0, 0);
    gDisplay.println("Smart Agriculture");
    gDisplay.println("Unified Suite");
    gDisplay.display();
}

void buildSensorJson(JsonObject sensors) {
    const SensorSnapshot& snapshot = gSensorHub.snapshot();
    sensors["airTemp"] = snapshot.airTemp;
    sensors["airHumi"] = snapshot.airHumi;
    sensors["soilHumi"] = snapshot.soilHumi;
    sensors["liquidLevel"] = snapshot.liquidLevel;
    sensors["lightValue"] = snapshot.lightValue;
    sensors["updatedAtMs"] = snapshot.updatedAtMs;
}

void buildActuatorJson(JsonObject actuator) {
    const ActuatorStatus& status = gActuator.status();
    actuator["valveOn"] = status.valveOn;
    actuator["pumpOn"] = status.pumpOn;
    actuator["autoMode"] = status.autoMode;
    actuator["manualValve"] = status.manualValve;
    actuator["manualPump"] = status.manualPump;
    actuator["lowLiquidLock"] = status.lowLiquidLock;
    actuator["timedRunActive"] = status.timedRunActive;
    actuator["source"] = controlSourceName(status.source);
    actuator["secondsRemaining"] = secondsRemaining(millis(), status.activeUntilMs);
}

void buildOverallStatus(JsonDocument& doc) {
    doc["project"] = "smart-agriculture-suite";
    doc["hardwareProfile"] = hardwareProfileName(gPins.profile);
    doc["wifiConnected"] = gWifiConnected;
    doc["ipAddress"] = gIpAddress;
    doc["lightSensorReady"] = gSensorHub.lightReady();
    doc["lastAirFrame"] = gSensorHub.lastAirFrame();

    buildSensorJson(doc["sensors"].to<JsonObject>());
    buildActuatorJson(doc["actuator"].to<JsonObject>());
    JsonDocument irrigationDoc;
    JsonDocument anomalyDoc;
    JsonDocument growthDoc;
    JsonDocument learningDoc;
    JsonDocument fusionDoc;
    JsonDocument plantDoc;
    gIrrigation.writeStatus(irrigationDoc);
    gAnomaly.writeStatus(anomalyDoc);
    gGrowth.writeStatus(growthDoc);
    gLearning.writeStatus(learningDoc);
    gFusion.writeStatus(fusionDoc);
    gPlantDoctor.writeStatus(plantDoc);
    doc["modules"]["irrigation"].set(irrigationDoc.as<JsonVariantConst>());
    doc["modules"]["anomaly"].set(anomalyDoc.as<JsonVariantConst>());
    doc["modules"]["growth"].set(growthDoc.as<JsonVariantConst>());
    doc["modules"]["learning"].set(learningDoc.as<JsonVariantConst>());
    doc["modules"]["fusion"].set(fusionDoc.as<JsonVariantConst>());
    doc["modules"]["plantDoctor"].set(plantDoc.as<JsonVariantConst>());
}

void drawHeader(const __FlashStringHelper* title) {
    gDisplay.clearDisplay();
    gDisplay.setTextSize(1);
    gDisplay.setCursor(0, 0);
    gDisplay.println(title);
    gDisplay.drawLine(0, 9, 127, 9, SSD1306_WHITE);
}

void refreshDisplay(unsigned long nowMs) {
    if (!gDisplayReady) {
        return;
    }
    if (nowMs - gLastDisplayPageMs >= kDisplayPageIntervalMs) {
        gDisplayPage = (gDisplayPage + 1) % 6;
        gLastDisplayPageMs = nowMs;
    }

    switch (gDisplayPage) {
        case 0: {
            drawHeader(F("Dashboard"));
            const SensorSnapshot& s = gSensorHub.snapshot();
            const ActuatorStatus& a = gActuator.status();
            gDisplay.setCursor(0, 14);
            gDisplay.printf("T %.1fC H %.0f%%", s.airTemp, s.airHumi);
            gDisplay.setCursor(0, 24);
            gDisplay.printf("Soil %.0f Lvl %.0f", s.soilHumi, s.liquidLevel);
            gDisplay.setCursor(0, 34);
            gDisplay.printf("Light %.0f %s", s.lightValue, gIrrigation.isDay() ? "Day" : "Night");
            gDisplay.setCursor(0, 44);
            gDisplay.printf("Pump %s", a.pumpOn ? "ON" : "OFF");
            gDisplay.setCursor(0, 54);
            gDisplay.printf("%s %s", gWifiConnected ? "WiFi" : "NoWiFi", controlSourceName(a.source));
            break;
        }
        case 1: {
            drawHeader(F("Irrigation"));
            const ActuatorStatus& a = gActuator.status();
            gDisplay.setCursor(0, 14);
            gDisplay.printf("Mode %s", a.autoMode ? "AUTO" : "MANUAL");
            gDisplay.setCursor(0, 24);
            gDisplay.printf("Rule %s", gIrrigation.enabled() ? "ENABLED" : "OFF");
            gDisplay.setCursor(0, 34);
            gDisplay.printf("Need %s", gIrrigation.shouldWater() ? "YES" : "NO");
            gDisplay.setCursor(0, 44);
            gDisplay.printf("Warn %s", gIrrigation.liquidWarn() ? "LOW LIQ" : "OK");
            gDisplay.setCursor(0, 54);
            gDisplay.print(gIrrigation.reason());
            break;
        }
        case 2: {
            drawHeader(F("Anomaly"));
            gDisplay.setCursor(0, 14);
            gDisplay.printf("Level %s", AnomalyModule::alertLevelName(gAnomaly.level()));
            gDisplay.setCursor(0, 24);
            gDisplay.printf("Score %.2f", gAnomaly.iforestScore());
            gDisplay.setCursor(0, 34);
            gDisplay.printf("Trained %s", gAnomaly.iforestTrained() ? "YES" : "NO");
            gDisplay.setCursor(0, 44);
            gDisplay.printf("Samples %d", gAnomaly.totalSamples());
            gDisplay.setCursor(0, 54);
            gDisplay.printf("Alerts %d", gAnomaly.totalAnomalies());
            break;
        }
        case 3: {
            drawHeader(F("Growth"));
            JsonDocument doc;
            gGrowth.writeStatus(doc);
            gDisplay.setCursor(0, 14);
            gDisplay.print(doc["crop"].as<const char*>());
            gDisplay.setCursor(0, 24);
            gDisplay.print(doc["stageName"].as<const char*>());
            gDisplay.setCursor(0, 34);
            gDisplay.printf("Day %d", doc["dayOfGrowth"].as<int>());
            gDisplay.setCursor(0, 44);
            gDisplay.printf("GDD %.0f", doc["cumulativeGdd"].as<float>());
            gDisplay.setCursor(0, 54);
            gDisplay.printf("Yield %.0f", doc["yieldScore"].as<float>());
            break;
        }
        case 4: {
            drawHeader(F("Learn/Fusion"));
            JsonDocument learningDoc;
            JsonDocument fusionDoc;
            gLearning.writeStatus(learningDoc);
            gFusion.writeStatus(fusionDoc);
            gDisplay.setCursor(0, 14);
            gDisplay.printf("Learn %s", learningDoc["autoControlEnabled"].as<bool>() ? "AUTO" : "ADVISE");
            gDisplay.setCursor(0, 24);
            gDisplay.printf("Ep %d", learningDoc["totalEpisodes"].as<int>());
            gDisplay.setCursor(0, 34);
            gDisplay.printf("Act %s", learningDoc["lastActionName"].as<const char*>());
            gDisplay.setCursor(0, 44);
            gDisplay.printf("Fusion %s", fusionDoc["decisionName"].as<const char*>());
            gDisplay.setCursor(0, 54);
            gDisplay.printf("Conf %.0f%%", fusionDoc["confidence"].as<float>() * 100.0f);
            break;
        }
        case 5:
        default: {
            drawHeader(F("Plant Doctor"));
            JsonDocument doc;
            gPlantDoctor.writeStatus(doc);
            gDisplay.setCursor(0, 14);
            gDisplay.printf("Enable %s", doc["enabled"].as<bool>() ? "YES" : "NO");
            gDisplay.setCursor(0, 24);
            gDisplay.printf("Camera %s", doc["cameraReady"].as<bool>() ? "READY" : "NA");
            gDisplay.setCursor(0, 34);
            gDisplay.printf("Model %s", doc["modelLoaded"].as<bool>() ? "READY" : "NA");
            gDisplay.setCursor(0, 44);
            gDisplay.print(doc["lastDiseaseName"].as<const char*>());
            gDisplay.setCursor(0, 54);
            gDisplay.printf("Conf %.0f%%", doc["lastConfidence"].as<float>() * 100.0f);
            break;
        }
    }

    gDisplay.display();
}

void registerRoutes() {
    gServer.enableCORS(true);

    gServer.on("/", HTTP_GET, []() {
        gServer.send_P(200, "text/html", kWebHtml);
    });

    gServer.on("/api/status", HTTP_GET, []() {
        JsonDocument doc;
        buildOverallStatus(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/system/modules", HTTP_GET, []() {
        JsonDocument doc;
        doc["ruleEngineEnabled"] = gIrrigation.enabled();
        doc["learningAutoEnabled"] = gLearning.config().autoControlEnabled;
        doc["fusionAutoEnabled"] = gFusion.autoControlEnabled();
        doc["plantDoctorEnabled"] = gPlantDoctor.config().enabled;
        sendJson(200, doc);
    });

    gServer.on("/api/system/modules", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }

        if (body["ruleEngineEnabled"].is<bool>()) {
            gIrrigation.setEnabled(body["ruleEngineEnabled"].as<bool>());
        }
        if (body["learningAutoEnabled"].is<bool>()) {
            LearningConfig config = gLearning.config();
            config.autoControlEnabled = body["learningAutoEnabled"].as<bool>();
            gLearning.setConfig(config);
            gConfigManager.saveLearning(config);
        }
        if (body["fusionAutoEnabled"].is<bool>()) {
            gFusion.setAutoControlEnabled(body["fusionAutoEnabled"].as<bool>());
        }
        if (body["plantDoctorEnabled"].is<bool>()) {
            PlantDoctorConfig config = gPlantDoctor.config();
            config.enabled = body["plantDoctorEnabled"].as<bool>();
            gPlantDoctor.setConfig(config);
            gConfigManager.savePlantDoctor(config);
        }

        gConfigManager.saveSystemFlags(gIrrigation.enabled(), gFusion.autoControlEnabled());

        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    gServer.on("/api/irrigation/status", HTTP_GET, []() {
        JsonDocument doc;
        buildSensorJson(doc["sensors"].to<JsonObject>());
        buildActuatorJson(doc["actuator"].to<JsonObject>());
        JsonDocument moduleDoc;
        gIrrigation.writeStatus(moduleDoc);
        doc["module"].set(moduleDoc.as<JsonVariantConst>());
        sendJson(200, doc);
    });

    gServer.on("/api/irrigation/mode", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }
        if (body["auto"].is<bool>()) {
            gActuator.setAutoMode(body["auto"].as<bool>());
        }
        JsonDocument response;
        response["success"] = true;
        response["autoMode"] = gActuator.status().autoMode;
        sendJson(200, response);
    });

    gServer.on("/api/irrigation/pump", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }
        if (body["state"].is<bool>()) {
            gActuator.setManualCombined(body["state"].as<bool>());
        }
        JsonDocument response;
        response["success"] = true;
        response["manualValve"] = gActuator.status().manualValve;
        response["manualPump"] = gActuator.status().manualPump;
        sendJson(200, response);
    });

    gServer.on("/api/irrigation/valve", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }
        if (body["state"].is<bool>()) {
            gActuator.setManualValve(body["state"].as<bool>());
        }
        JsonDocument response;
        response["success"] = true;
        response["manualValve"] = gActuator.status().manualValve;
        sendJson(200, response);
    });

    gServer.on("/api/irrigation/pump_only", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }
        if (body["state"].is<bool>()) {
            gActuator.setManualPump(body["state"].as<bool>());
        }
        JsonDocument response;
        response["success"] = true;
        response["manualPump"] = gActuator.status().manualPump;
        sendJson(200, response);
    });

    gServer.on("/api/irrigation/config", HTTP_GET, []() {
        JsonDocument doc;
        gIrrigation.writeStatus(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/irrigation/config", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }
        if (!gIrrigation.updateConfigFromJson(body.as<JsonVariantConst>())) {
            sendError(400, "No valid parameters");
            return;
        }
        gConfigManager.saveIrrigation(gIrrigation.thresholdConfig());
        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    gServer.on("/api/anomaly/status", HTTP_GET, []() {
        JsonDocument doc;
        gAnomaly.writeStatus(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/anomaly/alerts", HTTP_GET, []() {
        JsonDocument doc;
        gAnomaly.writeAlerts(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/anomaly/sensor", HTTP_GET, []() {
        const String sensorName = gServer.arg("name");
        JsonDocument doc;
        if (!gAnomaly.writeSensorDetail(sensorName, doc)) {
            sendError(404, "Sensor not found");
            return;
        }
        sendJson(200, doc);
    });

    gServer.on("/api/anomaly/clear", HTTP_POST, []() {
        gAnomaly.clear();
        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    gServer.on("/api/growth/status", HTTP_GET, []() {
        JsonDocument doc;
        gGrowth.writeStatus(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/growth/history", HTTP_GET, []() {
        JsonDocument doc;
        gGrowth.writeHistory(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/growth/prediction", HTTP_GET, []() {
        JsonDocument doc;
        gGrowth.writePrediction(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/growth/crop", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }
        if (!body["cropId"].is<int>()) {
            sendError(400, "cropId required");
            return;
        }
        gGrowth.setCrop(body["cropId"].as<int>());
        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    gServer.on("/api/growth/reset", HTTP_POST, []() {
        gGrowth.reset();
        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    gServer.on("/api/learning/status", HTTP_GET, []() {
        JsonDocument doc;
        gLearning.writeStatus(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/learning/qtable", HTTP_GET, []() {
        JsonDocument doc;
        gLearning.writeQTableSummary(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/learning/params", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }

        LearningConfig config = gLearning.config();
        if (body["alpha"].is<float>() || body["alpha"].is<double>() || body["alpha"].is<int>()) {
            config.alpha = body["alpha"].as<float>();
        }
        if (body["gamma"].is<float>() || body["gamma"].is<double>() || body["gamma"].is<int>()) {
            config.gamma = body["gamma"].as<float>();
        }
        if (body["epsilon"].is<float>() || body["epsilon"].is<double>() || body["epsilon"].is<int>()) {
            config.epsilon = body["epsilon"].as<float>();
        }
        if (body["targetSoil"].is<float>() || body["targetSoil"].is<double>() || body["targetSoil"].is<int>()) {
            config.targetSoil = body["targetSoil"].as<float>();
        }
        if (body["soilTolerance"].is<float>() || body["soilTolerance"].is<double>() || body["soilTolerance"].is<int>()) {
            config.soilTolerance = body["soilTolerance"].as<float>();
        }
        if (body["decisionInterval"].is<unsigned long>() || body["decisionInterval"].is<int>()) {
            config.decisionIntervalMs = body["decisionInterval"].as<unsigned long>() * 1000UL;
        }
        if (body["autoControlEnabled"].is<bool>()) {
            config.autoControlEnabled = body["autoControlEnabled"].as<bool>();
        }

        gLearning.setConfig(config);
        gConfigManager.saveLearning(config);

        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    gServer.on("/api/learning/feedback", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }
        const bool positive = body["positive"].is<bool>() ? body["positive"].as<bool>() : true;
        gLearning.recordUserFeedback(positive);
        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    gServer.on("/api/learning/reset", HTTP_POST, []() {
        gLearning.reset();
        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    gServer.on("/api/fusion/status", HTTP_GET, []() {
        JsonDocument doc;
        gFusion.writeStatus(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/fusion/sensors", HTTP_GET, []() {
        JsonDocument doc;
        gFusion.writeSensors(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/fusion/config", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }
        if (body["autoControlEnabled"].is<bool>()) {
            gFusion.setAutoControlEnabled(body["autoControlEnabled"].as<bool>());
        }
        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    gServer.on("/api/plant/status", HTTP_GET, []() {
        JsonDocument doc;
        gPlantDoctor.writeStatus(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/plant/history", HTTP_GET, []() {
        JsonDocument doc;
        gPlantDoctor.writeHistory(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/plant/detect", HTTP_GET, []() {
        if (!gPlantDoctor.performDetection()) {
            sendError(503, "Plant doctor unavailable");
            return;
        }
        JsonDocument doc;
        gPlantDoctor.writeStatus(doc);
        sendJson(200, doc);
    });

    gServer.on("/api/plant/capture", HTTP_GET, []() {
        gPlantDoctor.handleCapture(gServer);
    });

    gServer.on("/api/plant/config", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }

        PlantDoctorConfig config = gPlantDoctor.config();
        if (body["enabled"].is<bool>()) {
            config.enabled = body["enabled"].as<bool>();
        }
        if (body["autoDetect"].is<bool>()) {
            config.autoDetect = body["autoDetect"].as<bool>();
        }
        if (body["detectInterval"].is<int>()) {
            config.detectIntervalSec = body["detectInterval"].as<int>();
        }
        if (body["confidenceThreshold"].is<float>() || body["confidenceThreshold"].is<double>() || body["confidenceThreshold"].is<int>()) {
            config.confidenceThreshold = body["confidenceThreshold"].as<float>();
        }
        if (body["buzzerEnabled"].is<bool>()) {
            config.buzzerEnabled = body["buzzerEnabled"].as<bool>();
        }

        gPlantDoctor.setConfig(config);
        gConfigManager.savePlantDoctor(config);
        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    // ── WiFi configuration ─────────────────────────────────────────────
    gServer.on("/api/system/wifi", HTTP_GET, []() {
        JsonDocument doc;
        String ssid = gConfigManager.loadWifiSsid();
        doc["ssid"] = ssid.isEmpty() ? kWifiSsid : ssid;
        doc["hasPassword"] = !gConfigManager.loadWifiPassword().isEmpty();
        sendJson(200, doc);
    });

    gServer.on("/api/system/wifi", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }
        if (!body["ssid"].is<const char*>() || !body["password"].is<const char*>()) {
            sendError(400, "ssid and password required");
            return;
        }
        const char* ssid = body["ssid"].as<const char*>();
        const char* pass = body["password"].as<const char*>();
        gConfigManager.saveWiFi(ssid, pass);

        JsonDocument response;
        response["success"] = true;
        response["message"] = "WiFi credentials saved. Reboot to apply.";
        sendJson(200, response);
    });

    // ── OTA firmware update ────────────────────────────────────────────
    gServer.on("/api/ota/update", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }
        if (!body["url"].is<const char*>()) {
            sendError(400, "url required");
            return;
        }
        const char* url = body["url"].as<const char*>();
        if (!gOta.updateFromUrl(url)) {
            JsonDocument err;
            err["error"] = gOta.lastError();
            sendJson(500, err);
            return;
        }
        // On success, device reboots — this line is never reached
        JsonDocument response;
        response["success"] = true;
        sendJson(200, response);
    });

    gServer.on("/api/ota/status", HTTP_GET, []() {
        JsonDocument doc;
        gOta.writeStatus(doc);
        sendJson(200, doc);
    });

    // ── Factory reset ──────────────────────────────────────────────────
    gServer.on("/api/system/factory-reset", HTTP_POST, []() {
        gConfigManager.factoryReset();
        JsonDocument response;
        response["success"] = true;
        response["message"] = "Factory reset complete. Rebooting in 3s...";
        sendJson(200, response);
        delay(3000);
        ESP.restart();
    });

    // ── Fusion NN weight update ────────────────────────────────────────
    gServer.on("/api/fusion/weights", HTTP_POST, []() {
        JsonDocument body;
        if (!parseJsonBody(body)) {
            return;
        }

        // Save the full weight set to NVS so they persist across reboots
        Preferences prefs;
        if (prefs.begin("fusion_nn", false)) {
            if (body["weightsIH"].is<JsonArrayConst>()) {
                JsonArrayConst wih = body["weightsIH"].as<JsonArrayConst>();
                int i = 0;
                for (JsonArrayConst row : wih) {
                    int h = 0;
                    for (float val : row) {
                        char key[8];
                        snprintf(key, sizeof(key), "w%d%d", i, h);
                        prefs.putFloat(key, val);
                        ++h;
                    }
                    ++i;
                }
            }
            if (body["biasH"].is<JsonArrayConst>()) {
                int h = 0;
                for (float val : body["biasH"].as<JsonArrayConst>()) {
                    char key[8];
                    snprintf(key, sizeof(key), "bh%d", h);
                    prefs.putFloat(key, val);
                    ++h;
                }
            }
            if (body["weightsHO"].is<JsonArrayConst>()) {
                int h = 0;
                for (JsonArrayConst row : body["weightsHO"].as<JsonArrayConst>()) {
                    int o = 0;
                    for (float val : row) {
                        char key[8];
                        snprintf(key, sizeof(key), "v%d%d", h, o);
                        prefs.putFloat(key, val);
                        ++o;
                    }
                    ++h;
                }
            }
            if (body["biasO"].is<JsonArrayConst>()) {
                int o = 0;
                for (float val : body["biasO"].as<JsonArrayConst>()) {
                    char key[8];
                    snprintf(key, sizeof(key), "bo%d", o);
                    prefs.putFloat(key, val);
                    ++o;
                }
            }
            prefs.putUInt("ver", 1);
            prefs.end();
        }

        // Reload weights into the running FusionModule
        gFusion.loadNetworkFromPrefs();

        JsonDocument response;
        response["success"] = true;
        response["message"] = "Fusion NN weights updated and persisted";
        sendJson(200, response);
    });

    gServer.onNotFound([]() {
        sendError(404, "Route not found");
    });

    gServer.begin();
}

}  // namespace

void setup() {
    Serial.begin(kSerialBaud);
    delay(300);
    randomSeed(static_cast<unsigned long>(micros()));

    Serial.println();
    Serial.println("========================================");
    Serial.println(" Smart Agriculture Suite");
    Serial.println(" Unified PlatformIO Project");
    Serial.println("========================================");
    Serial.printf("Hardware profile: %s\n", hardwareProfileName(gPins.profile));

    Wire.begin(gPins.i2cSda, gPins.i2cScl);
    setupDisplay();

    // Load persisted configuration from NVS
    gConfigManager.begin();
    gSystemConfig.irrigation = gConfigManager.loadIrrigation();
    gSystemConfig.learning = gConfigManager.loadLearning();
    gSystemConfig.plantDoctor = gConfigManager.loadPlantDoctor();
    gSystemConfig.ruleEngineEnabled = gConfigManager.loadRuleEngineEnabled();
    gSystemConfig.fusionAutoEnabled = gConfigManager.loadFusionAutoEnabled();

    gIrrigation.setConfig(gSystemConfig.irrigation);
    gIrrigation.setEnabled(gSystemConfig.ruleEngineEnabled);
    gSensorHub.begin(gPins);
    gActuator.begin(gPins);
    gAnomaly.begin(gPins.buzzerPin);
    gGrowth.begin();
    gLearning.begin(gSystemConfig.learning);
    gFusion.begin(gSystemConfig.fusionAutoEnabled);
    gPlantDoctor.begin(gPins, gSystemConfig.plantDoctor);
    gOta.begin();

    connectWiFi();
    registerRoutes();

    Serial.printf("WiFi: %s\n", gWifiConnected ? gIpAddress.c_str() : "offline");
}

void loop() {
    gServer.handleClient();

    const unsigned long now = millis();
    const bool sampleUpdated = gSensorHub.update(now);
    if (sampleUpdated) {
        gIrrigation.update(gSensorHub.snapshot());
    }

    gActuator.update(gIrrigation.liquidWarn(), gIrrigation.enabled() && gIrrigation.shouldWater(), now);

    gAnomaly.update(gSensorHub.snapshot(), sampleUpdated, now);
    gGrowth.update(gSensorHub.snapshot(), sampleUpdated, now);
    gLearning.update(gSensorHub.snapshot(), sampleUpdated, now, gActuator);
    gFusion.update(gSensorHub.snapshot(), sampleUpdated, now, gActuator);
    gPlantDoctor.update(now, gSensorHub.snapshot().lightValue);

    if (now - gLastDisplayRefreshMs >= kDisplayRefreshIntervalMs) {
        refreshDisplay(now);
        gLastDisplayRefreshMs = now;
    }

    delay(10);
}
