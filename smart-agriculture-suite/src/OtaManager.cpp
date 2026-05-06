#include "OtaManager.h"

#include <HTTPClient.h>
#include <HTTPUpdate.h>

namespace agri {

void OtaManager::begin() {
    initialized_ = true;
    Serial.println("[OTA] manager initialized");
}

bool OtaManager::updateFromUrl(const char* url) {
    if (!initialized_ || url == nullptr || strlen(url) == 0) {
        snprintf(lastError_, sizeof(lastError_), "Invalid parameters");
        return false;
    }

    Serial.printf("[OTA] starting update from: %s\n", url);

    WiFiClient client;
    t_httpUpdate_return result = httpUpdate.update(client, url);

    switch (result) {
        case HTTP_UPDATE_OK:
            // Device will reboot — we never reach here
            Serial.println("[OTA] update OK, rebooting...");
            return true;

        case HTTP_UPDATE_NO_UPDATES:
            snprintf(lastError_, sizeof(lastError_), "No update available");
            Serial.println("[OTA] no update available");
            return false;

        case HTTP_UPDATE_FAILED:
            snprintf(lastError_, sizeof(lastError_), "Update failed: %s",
                     httpUpdate.getLastErrorString().c_str());
            Serial.printf("[OTA] failed: %s\n", httpUpdate.getLastErrorString().c_str());
            return false;

        default:
            snprintf(lastError_, sizeof(lastError_), "Unknown result: %d", result);
            return false;
    }
}

void OtaManager::writeStatus(JsonDocument& doc) const {
    doc["initialized"] = initialized_;
    if (lastError_[0] != '\0') {
        doc["lastError"] = lastError_;
    }
}

}  // namespace agri
