#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>

#include "AppTypes.h"
#include "AppConfig.h"
#include "Sensors.h"
#include "ActuatorController.h"
#include "GrowthModule.h"
#include "WorldModelClient.h"
#include "WebFrontend.h"

using namespace agri;

// ============================================================================
// 全局对象
// ============================================================================
static Adafruit_SSD1306 gDisplay(128, 64, &Wire, -1);
static WebServer gServer(80);
static SensorHub gSensors;
static ActuatorController gActuator;
static GrowthModule gGrowth;
static WorldModelClient gWorldModel;
static Preferences gPrefs;

static PinConfig gPins;
static SystemConfig gConfig;
static WorldModelResponse gLastResponse;

static bool gDisplayReady = false;
static int gDisplayPage = 0;
static unsigned long gLastDisplayMs = 0;
static unsigned long gLastDisplayPageMs = 0;
static unsigned long gLastWorldModelMs = 0;
static unsigned long gLastIrrigationMs = 0;
static bool gFallbackShouldWater = false;
static String gFallbackReason;

// ============================================================================
// WiFi连接
// ============================================================================
void connectWifi() {
    gPrefs.begin("wifi_cfg", true);
    String ssid = gPrefs.getString("ssid", "");
    String pass = gPrefs.getString("pass", "");
    gPrefs.end();

    if (ssid.length() == 0) {
        Serial.println("[WiFi] 未配置WiFi, 跳过连接");
        return;
    }

    Serial.printf("[WiFi] 连接到 %s ...\n", ssid.c_str());
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid.c_str(), pass.c_str());

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] 已连接, IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WiFi] 连接失败");
    }
}

// ============================================================================
// 规则引擎回退 (当Atlas不可用时)
// ============================================================================
void updateFallbackIrrigation(const SensorSnapshot& snap) {
    const auto& cfg = gConfig.irrigation;

    if (snap.fault.airFault() && snap.fault.soilFault()) {
        gFallbackShouldWater = false;
        gFallbackReason = "传感器故障,安全停止";
        return;
    }

    bool isDay = snap.isDay;
    float tempThreshold = isDay ? cfg.dayAirTempThreshold : cfg.nightAirTempThreshold;
    float humiThreshold = isDay ? cfg.dayAirHumiThreshold : cfg.nightAirHumiThreshold;
    float soilThreshold = isDay ? cfg.daySoilHumiThreshold : cfg.nightSoilHumiThreshold;

    bool tempPass = snap.fault.airFault() || snap.airTemp > tempThreshold;
    bool humiPass = snap.fault.airFault() || snap.airHumi < humiThreshold;
    bool soilPass = snap.fault.soilFault() || snap.soilHumi < soilThreshold;

    gFallbackShouldWater = tempPass && humiPass && soilPass;

    if (gFallbackShouldWater) gFallbackReason = "规则引擎:需要灌溉";
    else gFallbackReason = "规则引擎:无需灌溉";
}

// ============================================================================
// OLED显示
// ============================================================================
void updateDisplay(unsigned long nowMs) {
    if (!gDisplayReady) return;
    if (nowMs - gLastDisplayMs < kDisplayRefreshIntervalMs) return;
    gLastDisplayMs = nowMs;

    if (nowMs - gLastDisplayPageMs > kDisplayPageIntervalMs) {
        gDisplayPage = (gDisplayPage + 1) % 4;
        gLastDisplayPageMs = nowMs;
    }

    const auto& snap = gSensors.snapshot();
    const auto& act = gActuator.status();

    gDisplay.clearDisplay();
    gDisplay.setTextSize(1);
    gDisplay.setTextColor(SSD1306_WHITE);
    gDisplay.setCursor(0, 0);

    switch (gDisplayPage) {
        case 0: {
            // 传感器概览
            gDisplay.println("=== Sensors ===");
            if (snap.fault.airFault())
                gDisplay.println("Air:  FAULT");
            else
                gDisplay.printf("Air:  %.1fC %.0f%%\n", snap.airTemp, snap.airHumi);
            if (snap.fault.soilFault())
                gDisplay.println("Soil: FAULT");
            else
                gDisplay.printf("Soil: %.0f%%\n", snap.soilHumi);
            if (snap.fault.lightFault())
                gDisplay.println("Light:FAULT");
            else
                gDisplay.printf("Light:%.0f lux\n", snap.lightValue);
            break;
        }
        case 1: {
            // 执行器状态
            gDisplay.println("=== Actuator ===");
            gDisplay.printf("Valve:%s Pump:%s\n", act.valveOn ? "ON" : "OFF", act.pumpOn ? "ON" : "OFF");
            gDisplay.printf("Mode: %s\n", act.autoMode ? "AUTO" : "MANUAL");
            gDisplay.printf("Src:  %s\n", controlSourceName(act.source));
            if (act.timedRunActive) {
                unsigned long rem = secondsRemaining(nowMs, act.activeUntilMs);
                gDisplay.printf("Timer:%lus\n", rem);
            }
            break;
        }
        case 2: {
            // 世界模型
            gDisplay.println("=== World Model ===");
            gDisplay.printf("Atlas: %s\n", gWorldModel.isConnected() ? "OK" : "OFF");
            if (gLastResponse.valid) {
                gDisplay.printf("Disease: %s\n", gLastResponse.diseaseName);
                gDisplay.printf("Conf: %.0f%%\n", gLastResponse.diseaseConfidence * 100);
                gDisplay.printf("Action: %d (%ds)\n", (int)gLastResponse.action, gLastResponse.actionDurationSec);
                gDisplay.printf("Risk: %.0f%%\n", gLastResponse.environmentRisk * 100);
            } else {
                gDisplay.println("No data yet");
            }
            break;
        }
        case 3: {
            // 生长状态
            gDisplay.println("=== Growth ===");
            gDisplay.printf("Crop: %s\n", cropNameCn((CropType)gGrowth.currentCropIndex()));
            gDisplay.printf("Stage: %s\n", stageNameCn(gGrowth.currentStage()));
            gDisplay.printf("Day: %d\n", gGrowth.currentDayOfGrowth());
            gDisplay.printf("GDD: %.1f\n", gGrowth.cumulativeGdd());
            gDisplay.printf("Yield: %.0f\n", gGrowth.yieldScore());
            gDisplay.printf("WiFi: %s\n", WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString().c_str() : "OFF");
            break;
        }
    }

    gDisplay.display();
}

// ============================================================================
// HTTP API路由
// ============================================================================
void handleStatus() {
    StaticJsonDocument<1024> doc;
    const auto& snap = gSensors.snapshot();
    const auto& act = gActuator.status();

    doc["air_temp"] = snap.airTemp;
    doc["air_humi"] = snap.airHumi;
    doc["soil_humi"] = snap.soilHumi;
    doc["light"] = snap.lightValue;
    doc["is_day"] = snap.isDay;
    doc["uptime_ms"] = millis();
    doc["ip"] = WiFi.localIP().toString();
    doc["rssi"] = WiFi.RSSI();
    doc["crop_id"] = gGrowth.currentCropIndex();

    // 故障状态
    JsonObject faults = doc.createNestedObject("faults");
    faults["air"] = snap.fault.airFault();
    faults["soil"] = snap.fault.soilFault();
    faults["light"] = snap.fault.lightFault();

    // 执行器
    JsonObject actObj = doc.createNestedObject("actuator");
    actObj["valve_on"] = act.valveOn;
    actObj["pump_on"] = act.pumpOn;
    actObj["auto_mode"] = act.autoMode;
    actObj["source"] = controlSourceName(act.source);
    actObj["timed_active"] = act.timedRunActive;

    // 世界模型
    if (gLastResponse.valid) {
        JsonObject wm = doc.createNestedObject("world_model");
        wm["disease_id"] = gLastResponse.diseaseId;
        wm["disease_name"] = gLastResponse.diseaseName;
        wm["disease_confidence"] = gLastResponse.diseaseConfidence;
        wm["treatment"] = gLastResponse.treatment;
        wm["action"] = (int)gLastResponse.action;
        wm["duration"] = gLastResponse.actionDurationSec;
        wm["action_confidence"] = gLastResponse.actionConfidence;
        wm["reason"] = gLastResponse.actionReason;
        wm["pred_soil"] = gLastResponse.predictedSoilHumi;
        wm["pred_temp"] = gLastResponse.predictedAirTemp;
        wm["pred_humi"] = gLastResponse.predictedAirHumi;
        wm["risk"] = gLastResponse.environmentRisk;
    }

    // 生长
    JsonObject grow = doc.createNestedObject("growth");
    gGrowth.writeStatus(grow);

    String out;
    serializeJson(doc, out);
    gServer.send(200, "application/json", out);
}

void handleIrrigationMode() {
    if (!gServer.hasArg("plain")) { gServer.send(400, "text/plain", "missing body"); return; }
    StaticJsonDocument<128> doc;
    deserializeJson(doc, gServer.arg("plain"));
    bool autoMode = doc["auto"] | true;
    gActuator.setAutoMode(autoMode);
    gServer.send(200, "application/json", "{\"ok\":true}");
}

void handleManualPump() {
    if (!gServer.hasArg("plain")) { gServer.send(400, "text/plain", "missing body"); return; }
    StaticJsonDocument<64> doc;
    deserializeJson(doc, gServer.arg("plain"));
    bool on = doc["on"] | false;
    gActuator.setManualPump(on);
    gServer.send(200, "application/json", "{\"ok\":true}");
}

void handleIrrigationConfig() {
    if (gServer.method() == HTTP_GET) {
        StaticJsonDocument<256> doc;
        const auto& c = gConfig.irrigation;
        doc["day_temp"] = c.dayAirTempThreshold;
        doc["day_humi"] = c.dayAirHumiThreshold;
        doc["day_soil"] = c.daySoilHumiThreshold;
        doc["night_temp"] = c.nightAirTempThreshold;
        doc["night_humi"] = c.nightAirHumiThreshold;
        doc["night_soil"] = c.nightSoilHumiThreshold;
        String out;
        serializeJson(doc, out);
        gServer.send(200, "application/json", out);
    } else {
        if (!gServer.hasArg("plain")) { gServer.send(400, "text/plain", "missing body"); return; }
        StaticJsonDocument<256> doc;
        deserializeJson(doc, gServer.arg("plain"));
        auto& c = gConfig.irrigation;
        c.dayAirTempThreshold = doc["day_temp"] | c.dayAirTempThreshold;
        c.dayAirHumiThreshold = doc["day_humi"] | c.dayAirHumiThreshold;
        c.daySoilHumiThreshold = doc["day_soil"] | c.daySoilHumiThreshold;
        c.nightAirTempThreshold = doc["night_temp"] | c.nightAirTempThreshold;
        c.nightAirHumiThreshold = doc["night_humi"] | c.nightAirHumiThreshold;
        c.nightSoilHumiThreshold = doc["night_soil"] | c.nightSoilHumiThreshold;
        gServer.send(200, "application/json", "{\"ok\":true}");
    }
}

void handleAtlasConfig() {
    if (!gServer.hasArg("plain")) { gServer.send(400, "text/plain", "missing body"); return; }
    StaticJsonDocument<128> doc;
    deserializeJson(doc, gServer.arg("plain"));
    const char* host = doc["host"] | "";
    uint16_t port = doc["port"] | 8080;
    gWorldModel.setServer(host, port);

    gPrefs.begin("atlas_cfg", false);
    gPrefs.putString("host", host);
    gPrefs.putUShort("port", port);
    gPrefs.end();

    gServer.send(200, "application/json", "{\"ok\":true}");
}

void handleGrowthCrop() {
    if (!gServer.hasArg("plain")) { gServer.send(400, "text/plain", "missing body"); return; }
    StaticJsonDocument<64> doc;
    deserializeJson(doc, gServer.arg("plain"));
    uint8_t crop = doc["crop"] | 0;
    gGrowth.setCrop(crop);
    gServer.send(200, "application/json", "{\"ok\":true}");
}

void handleGrowthReset() {
    gGrowth.reset();
    gServer.send(200, "application/json", "{\"ok\":true}");
}

void handleWifi() {
    if (!gServer.hasArg("plain")) { gServer.send(400, "text/plain", "missing body"); return; }
    StaticJsonDocument<128> doc;
    deserializeJson(doc, gServer.arg("plain"));
    const char* ssid = doc["ssid"] | "";
    const char* pass = doc["pass"] | "";

    gPrefs.begin("wifi_cfg", false);
    gPrefs.putString("ssid", ssid);
    gPrefs.putString("pass", pass);
    gPrefs.end();

    gServer.send(200, "application/json", "{\"ok\":true}");
}

void registerRoutes() {
    gServer.on("/", HTTP_GET, []() {
        gServer.send_P(200, "text/html", kWebHtml);
    });
    gServer.on("/api/status", HTTP_GET, handleStatus);
    gServer.on("/api/irrigation/mode", HTTP_POST, handleIrrigationMode);
    gServer.on("/api/irrigation/pump", HTTP_POST, handleManualPump);
    gServer.on("/api/irrigation/config", HTTP_ANY, handleIrrigationConfig);
    gServer.on("/api/atlas/config", HTTP_POST, handleAtlasConfig);
    gServer.on("/api/growth/crop", HTTP_POST, handleGrowthCrop);
    gServer.on("/api/growth/reset", HTTP_POST, handleGrowthReset);
    gServer.on("/api/wifi", HTTP_POST, handleWifi);
    gServer.begin();
}

// ============================================================================
// setup / loop
// ============================================================================
void setup() {
    Serial.begin(kSerialBaud);
    delay(500);

    Serial.println();
    Serial.println("========================================");
    Serial.println("  Smart Agriculture Suite");
    Serial.println("  World Model Edition");
    Serial.println("========================================");

    // I2C + Display
    gPins = defaultPins();
    Wire.begin(gPins.i2cSda, gPins.i2cScl);

    gDisplayReady = gDisplay.begin(SSD1306_SWITCHCAPVCC, 0x3C);
    if (gDisplayReady) {
        gDisplay.clearDisplay();
        gDisplay.setTextSize(1);
        gDisplay.setTextColor(SSD1306_WHITE);
        gDisplay.setCursor(0, 0);
        gDisplay.println("Smart Agriculture");
        gDisplay.println("Starting...");
        gDisplay.display();
    }

    // 配置
    gConfig = defaultSystemConfig();
    gPrefs.begin("atlas_cfg", true);
    String atlasHost = gPrefs.getString("host", kDefaultAtlasHost);
    uint16_t atlasPort = gPrefs.getUShort("port", kDefaultAtlasPort);
    gPrefs.end();

    // 初始化模块
    gSensors.begin(gPins);
    gActuator.begin(gPins);
    gGrowth.begin();
    gWorldModel.begin(atlasHost.c_str(), atlasPort);

    // WiFi
    connectWifi();

    // HTTP路由
    registerRoutes();

    Serial.println("[OK] 所有模块初始化完成");
    Serial.printf("[INFO] Atlas服务器: %s:%u\n", atlasHost.c_str(), atlasPort);
}

void loop() {
    unsigned long now = millis();

    // 1. HTTP
    gServer.handleClient();

    // 2. 传感器采样
    bool sampleUpdated = gSensors.update(now);

    // 3. 规则引擎回退
    if (sampleUpdated) {
        updateFallbackIrrigation(gSensors.snapshot());
    }

    // 4. 世界模型查询
    if (sampleUpdated && gConfig.worldModelEnabled &&
        now - gLastWorldModelMs >= gConfig.worldModelIntervalMs) {
        gLastWorldModelMs = now;

        const auto& snap = gSensors.snapshot();
        if (gWorldModel.sendSensorData(snap, gGrowth.currentStage(),
                                        gGrowth.currentCropIndex(), gLastResponse)) {
            // 世界模型返回有效响应
            if (gLastResponse.action != IrrigationAction::Off &&
                gLastResponse.actionDurationSec > 0) {
                ControlSource src = ControlSource::WorldModel;
                gActuator.startTimedRun(src, gLastResponse.actionDurationSec, now);
            }
        }
    }

    // 5. 执行器更新
    gActuator.update(gFallbackShouldWater, now);

    // 6. 显示刷新
    updateDisplay(now);
}
