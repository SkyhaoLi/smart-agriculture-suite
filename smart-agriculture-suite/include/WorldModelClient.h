#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>

#include "AppConfig.h"
#include "AppTypes.h"

namespace agri {

class WorldModelClient {
public:
    void begin(const char* host, uint16_t port);
    void setServer(const char* host, uint16_t port);
    bool sendSensorData(const SensorSnapshot& snapshot, GrowthStage stage,
                        uint8_t cropId, WorldModelResponse& response);
    bool isConnected() const { return connected_; }
    unsigned long lastSuccessMs() const { return lastSuccessMs_; }
    const char* lastError() const { return lastError_; }

private:
    bool parseResponse(const String& json, WorldModelResponse& response);

    char host_[64] = "";
    uint16_t port_ = 8080;
    bool connected_ = false;
    unsigned long lastSuccessMs_ = 0;
    char lastError_[64] = "";
};

}  // namespace agri
