#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>

namespace agri {

class OtaManager {
public:
    void begin();

    /// Trigger OTA update from the given HTTP URL.
    /// Returns true if the update was attempted (does NOT return on success — device reboots).
    /// On failure, returns false and sets lastError().
    bool updateFromUrl(const char* url);

    const char* lastError() const { return lastError_; }

    void writeStatus(JsonDocument& doc) const;

private:
    bool initialized_ = false;
    char lastError_[128] = {};
};

}  // namespace agri
