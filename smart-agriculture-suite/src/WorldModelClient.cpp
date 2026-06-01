#include "WorldModelClient.h"

namespace agri {

void WorldModelClient::begin(const char* host, uint16_t port) {
    strlcpy(host_, host, sizeof(host_));
    port_ = port;
}

void WorldModelClient::setServer(const char* host, uint16_t port) {
    strlcpy(host_, host, sizeof(host_));
    port_ = port;
}

bool WorldModelClient::sendSensorData(const SensorSnapshot& snapshot, GrowthStage stage,
                                       uint8_t cropId, WorldModelResponse& response) {
    if (strlen(host_) == 0) {
        strlcpy(lastError_, "no server configured", sizeof(lastError_));
        return false;
    }

    char url[128];
    snprintf(url, sizeof(url), "http://%s:%u/api/predict", host_, port_);

    // 构建请求JSON
    StaticJsonDocument<512> req;
    req["air_temp"] = snapshot.airTemp;
    req["air_humi"] = snapshot.airHumi;
    req["soil_humi"] = snapshot.soilHumi;
    req["liquid_level"] = snapshot.liquidLevel;
    req["light"] = snapshot.lightValue;
    req["is_day"] = snapshot.isDay;
    req["growth_stage"] = static_cast<int>(stage);
    req["crop_id"] = static_cast<int>(cropId);

    // 故障标记
    JsonObject faults = req.createNestedObject("faults");
    faults["air"] = snapshot.fault.airFault();
    faults["soil"] = snapshot.fault.soilFault();
    faults["liquid"] = snapshot.fault.liquidFault();
    faults["light"] = snapshot.fault.lightFault();

    String body;
    serializeJson(req, body);

    HTTPClient http;
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(5000);

    int code = http.POST(body);
    bool ok = false;

    if (code == 200) {
        String respBody = http.getString();
        ok = parseResponse(respBody, response);
        if (ok) {
            connected_ = true;
            lastSuccessMs_ = millis();
        }
    } else {
        snprintf(lastError_, sizeof(lastError_), "HTTP %d", code);
    }

    http.end();
    return ok;
}

bool WorldModelClient::parseResponse(const String& json, WorldModelResponse& response) {
    StaticJsonDocument<1024> doc;
    if (deserializeJson(doc, json) != DeserializationError::Ok) {
        strlcpy(lastError_, "json parse error", sizeof(lastError_));
        return false;
    }

    // 病害诊断
    response.diseaseId = doc["disease"]["id"] | 0;
    response.diseaseConfidence = doc["disease"]["confidence"] | 0.0f;
    strlcpy(response.diseaseName, doc["disease"]["name"] | "", sizeof(response.diseaseName));
    strlcpy(response.treatment, doc["disease"]["treatment"] | "", sizeof(response.treatment));

    // 灌溉决策
    response.action = static_cast<IrrigationAction>(doc["irrigation"]["action"] | 0);
    response.actionDurationSec = doc["irrigation"]["duration_sec"] | 0;
    response.actionConfidence = doc["irrigation"]["confidence"] | 0.0f;
    strlcpy(response.actionReason, doc["irrigation"]["reason"] | "", sizeof(response.actionReason));

    // 世界模型预测
    response.predictedSoilHumi = doc["prediction"]["soil_humi"] | 0.0f;
    response.predictedAirTemp = doc["prediction"]["air_temp"] | 0.0f;
    response.predictedAirHumi = doc["prediction"]["air_humi"] | 0.0f;
    response.environmentRisk = doc["prediction"]["risk"] | 0.0f;

    response.valid = true;
    response.timestampMs = millis();
    return true;
}

}  // namespace agri
