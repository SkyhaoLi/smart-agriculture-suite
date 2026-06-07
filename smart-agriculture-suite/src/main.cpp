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
#include "AtlasCDC.h"

using namespace agri;

// USB CDC 开启后 Serial 走 USB (给Atlas)，调试日志改用 Serial0 (UART0 → CH343 → 电脑)
// 注意: Serial1 已被空气传感器占用 (GPIO 17/18)，不可复用
#define DebugSerial Serial0

// ============================================================================
// 全局对象
// ============================================================================
static Adafruit_SSD1306 gDisplay(128, 64, &Wire, -1);
static WebServer gServer(80);
static SensorHub gSensors;
static ActuatorController gActuator;
static GrowthModule gGrowth;
static WorldModelClient gWorldModel;
static AtlasCDC gAtlasCDC;
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
    // 清除旧的 WiFi 配置（首次运行）
    gPrefs.begin("wifi_cfg", false);
    gPrefs.clear();
    gPrefs.end();

    gPrefs.begin("wifi_cfg", true);
    String ssid = gPrefs.getString("ssid", "");
    String pass = gPrefs.getString("pass", "");
    gPrefs.end();

    // 默认 WiFi 配置
    if (ssid.length() == 0) {
        ssid = "helloo";
        pass = "88888888";
        DebugSerial.println("[WiFi] 使用默认WiFi配置");
    }

    DebugSerial.printf("[WiFi] 连接到 %s ...\n", ssid.c_str());
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid.c_str(), pass.c_str());

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        DebugSerial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        DebugSerial.printf("\n[WiFi] 已连接, IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        DebugSerial.println("\n[WiFi] 连接失败");
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
    bool liquidOk = snap.fault.liquidFault() || snap.liquidLevel >= cfg.liquidLevelThreshold;

    gFallbackShouldWater = liquidOk && tempPass && humiPass && soilPass;

    if (!liquidOk) gFallbackReason = "液位不足";
    else if (gFallbackShouldWater) gFallbackReason = "规则引擎:需要灌溉";
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
            if (snap.fault.liquidFault())
                gDisplay.println("Liq:  FAULT");
            else
                gDisplay.printf("Liq:  %.0f%%\n", snap.liquidLevel);
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
                gDisplay.printf("Timer:lus\n", rem);
            }
            if (act.lowLiquidLock) gDisplay.println("** SAFETY LOCK **");
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
    JsonDocument doc;
    const auto& snap = gSensors.snapshot();
    const auto& act = gActuator.status();

    doc["air_temp"] = snap.airTemp;
    doc["air_humi"] = snap.airHumi;
    doc["soil_humi"] = snap.soilHumi;
    doc["liquid_level"] = snap.liquidLevel;
    doc["light"] = snap.lightValue;
    doc["is_day"] = snap.isDay;
    doc["uptime_ms"] = millis();
    doc["ip"] = WiFi.localIP().toString();
    doc["rssi"] = WiFi.RSSI();
    doc["crop_id"] = gGrowth.currentCropIndex();

    // 故障状态
    JsonObject faults = doc["faults"].to<JsonObject>();
    faults["air"] = snap.fault.airFault();
    faults["soil"] = snap.fault.soilFault();
    faults["liquid"] = snap.fault.liquidFault();
    faults["light"] = snap.fault.lightFault();

    // 执行器
    JsonObject actObj = doc["actuator"].to<JsonObject>();
    actObj["valve_on"] = act.valveOn;
    actObj["pump_on"] = act.pumpOn;
    actObj["auto_mode"] = act.autoMode;
    actObj["source"] = controlSourceName(act.source);
    actObj["timed_active"] = act.timedRunActive;
    actObj["safety_lock"] = act.lowLiquidLock;

    // 世界模型
    if (gLastResponse.valid) {
        JsonObject wm = doc["world_model"].to<JsonObject>();
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
    JsonObject grow = doc["growth"].to<JsonObject>();
    gGrowth.writeStatus(grow);

    String out;
    serializeJson(doc, out);
    gServer.send(200, "application/json", out);
}

void handleIrrigationMode() {
    if (!gServer.hasArg("plain")) { gServer.send(400, "text/plain", "missing body"); return; }
    JsonDocument doc;
    deserializeJson(doc, gServer.arg("plain"));
    bool autoMode = doc["auto"] | true;
    gActuator.setAutoMode(autoMode);
    gServer.send(200, "application/json", "{\"ok\":true}");
}

void handleManualPump() {
    if (!gServer.hasArg("plain")) { gServer.send(400, "text/plain", "missing body"); return; }
    JsonDocument doc;
    deserializeJson(doc, gServer.arg("plain"));
    bool on = doc["on"] | false;
    gActuator.setManualPump(on);
    gServer.send(200, "application/json", "{\"ok\":true}");
}

void handleIrrigationConfig() {
    if (gServer.method() == HTTP_GET) {
        JsonDocument doc;
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
        JsonDocument doc;
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
    JsonDocument doc;
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
    JsonDocument doc;
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
    JsonDocument doc;
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
    // 调试串口: Serial0 (UART0 → CH343 → 电脑 COM 口, GPIO 43/44)
    DebugSerial.begin(kSerialBaud, SERIAL_8N1, 44, 43);
    delay(500);

    DebugSerial.println();
    DebugSerial.println("========================================");
    DebugSerial.println("  Smart Agriculture Suite");
    DebugSerial.println("  World Model Edition");
    DebugSerial.println("========================================");

    // USB CDC: Serial → Atlas (/dev/ttyACM0)
    gAtlasCDC.begin();
    DebugSerial.println("[CDC] Atlas USB CDC 已初始化");

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

    DebugSerial.println("[OK] 所有模块初始化完成");
    DebugSerial.printf("[INFO] Atlas服务器: %s:%u\n", atlasHost.c_str(), atlasPort);
}

void loop() {
    unsigned long now = millis();

    // 1. HTTP
    gServer.handleClient();

    // 2. Atlas USB CDC 指令处理
    gAtlasCDC.update(gSensors.snapshot());

    // 3. 传感器采样
    bool sampleUpdated = gSensors.update(now);

    // 3. 规则引擎回退 + Atlas 实时推送
    if (sampleUpdated) {
        updateFallbackIrrigation(gSensors.snapshot());
        gAtlasCDC.pushData(gSensors.snapshot(), gActuator.status());
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
    bool lowLiquid = !gSensors.snapshot().fault.liquidFault() &&
                     gSensors.snapshot().liquidLevel < gConfig.liquidSafetyThreshold;
    gActuator.update(lowLiquid, gFallbackShouldWater, now);

    // 6. 显示刷新
    updateDisplay(now);

    // 7. 串口打印传感器详细调试信息 (每5秒)
    static unsigned long lastPrintMs = 0;
    if (now - lastPrintMs >= 5000) {
        lastPrintMs = now;

        // 传感器详细调试
        gSensors.printDebugInfo(now);

        const auto& act = gActuator.status();

        // 执行器状态
        DebugSerial.println("\n【执行器状态】");
        DebugSerial.printf("  电磁阀 (GPIO%d):   %s\n", gPins.valvePin, act.valveOn ? "✅ 开启" : "⬜ 关闭");
        DebugSerial.printf("  水泵   (GPIO%d):   %s\n", gPins.pumpPin, act.pumpOn ? "✅ 开启" : "⬜ 关闭");
        DebugSerial.printf("  蜂鸣器 (GPIO%d):   %s\n", gPins.buzzerPin, digitalRead(gPins.buzzerPin) ? "响" : "静");
        DebugSerial.printf("  自动模式:    %s\n", act.autoMode ? "是" : "否");
        DebugSerial.printf("  控制源:      %s\n", controlSourceName(act.source));
        DebugSerial.printf("  定时运行:    %s\n", act.timedRunActive ? "是" : "否");
        DebugSerial.printf("  液位安全锁:  %s\n", act.lowLiquidLock ? "⚠️ 激活" : "未激活");

        // 生长状态
        DebugSerial.println("\n【生长状态】");
        DebugSerial.printf("  当前作物:    %s (%s)\n",
                      cropName(static_cast<CropType>(gGrowth.currentCropIndex())),
                      cropNameCn(static_cast<CropType>(gGrowth.currentCropIndex())));
        DebugSerial.printf("  生长阶段:    %s (%s)\n", stageName(gGrowth.currentStage()), stageNameCn(gGrowth.currentStage()));
        DebugSerial.printf("  生长天数:    %d 天\n", gGrowth.currentDayOfGrowth());
        DebugSerial.printf("  累计GDD:     %.1f\n", gGrowth.cumulativeGdd());
        DebugSerial.printf("  产量评分:    %.1f\n", gGrowth.yieldScore());

        // 网络状态
        DebugSerial.println("\n【网络状态】");
        DebugSerial.printf("  WiFi 状态:   %s\n", WiFi.status() == WL_CONNECTED ? "✅ 已连接" : "❌ 未连接");
        if (WiFi.status() == WL_CONNECTED) {
            DebugSerial.printf("  IP 地址:     %s\n", WiFi.localIP().toString().c_str());
            DebugSerial.printf("  信号强度:    %d dBm\n", WiFi.RSSI());
            DebugSerial.printf("  Web 访问:    http://%s\n", WiFi.localIP().toString().c_str());
        }
        DebugSerial.println("========================================\n");
    }
}
