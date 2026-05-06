#pragma once

#include <Arduino.h>
#include <Preferences.h>
#include "AppTypes.h"

namespace agri {

class ConfigManager {
public:
    void begin() {
        Preferences prefs;
        prefs.begin(kAgriNs, true);
        prefs.end();
    }

    // ── Irrigation ──────────────────────────────────────────────────
    void saveIrrigation(const IrrigationThresholdConfig& config) {
        Preferences prefs;
        prefs.begin(kAgriNs, false);
        prefs.putFloat("irr_lqt",  config.liquidLevelThreshold);
        prefs.putFloat("irr_ldt",  config.lightDayThreshold);
        prefs.putFloat("irr_datt", config.dayAirTempThreshold);
        prefs.putFloat("irr_daht", config.dayAirHumiThreshold);
        prefs.putFloat("irr_dsht", config.daySoilHumiThreshold);
        prefs.putFloat("irr_natt", config.nightAirTempThreshold);
        prefs.putFloat("irr_naht", config.nightAirHumiThreshold);
        prefs.putFloat("irr_nsht", config.nightSoilHumiThreshold);
        prefs.end();
    }

    IrrigationThresholdConfig loadIrrigation() {
        IrrigationThresholdConfig config;
        Preferences prefs;
        prefs.begin(kAgriNs, true);
        config.liquidLevelThreshold  = prefs.getFloat("irr_lqt",  config.liquidLevelThreshold);
        config.lightDayThreshold     = prefs.getFloat("irr_ldt",  config.lightDayThreshold);
        config.dayAirTempThreshold   = prefs.getFloat("irr_datt", config.dayAirTempThreshold);
        config.dayAirHumiThreshold   = prefs.getFloat("irr_daht", config.dayAirHumiThreshold);
        config.daySoilHumiThreshold  = prefs.getFloat("irr_dsht", config.daySoilHumiThreshold);
        config.nightAirTempThreshold = prefs.getFloat("irr_natt", config.nightAirTempThreshold);
        config.nightAirHumiThreshold = prefs.getFloat("irr_naht", config.nightAirHumiThreshold);
        config.nightSoilHumiThreshold = prefs.getFloat("irr_nsht", config.nightSoilHumiThreshold);
        prefs.end();
        return config;
    }

    // ── Learning ────────────────────────────────────────────────────
    void saveLearning(const LearningConfig& config) {
        Preferences prefs;
        prefs.begin(kAgriNs, false);
        prefs.putFloat("lrn_alpha",  config.alpha);
        prefs.putFloat("lrn_gamma",  config.gamma);
        prefs.putFloat("lrn_eps",    config.epsilon);
        prefs.putFloat("lrn_epsd",   config.epsilonDecay);
        prefs.putFloat("lrn_epsm",   config.epsilonMin);
        prefs.putFloat("lrn_tgtso",  config.targetSoil);
        prefs.putFloat("lrn_soiltol", config.soilTolerance);
        prefs.putULong("lrn_decms",  config.decisionIntervalMs);
        prefs.putBool("lrn_auto",    config.autoControlEnabled);
        prefs.end();
    }

    LearningConfig loadLearning() {
        LearningConfig config;
        Preferences prefs;
        prefs.begin(kAgriNs, true);
        config.alpha               = prefs.getFloat("lrn_alpha",  config.alpha);
        config.gamma               = prefs.getFloat("lrn_gamma",  config.gamma);
        config.epsilon             = prefs.getFloat("lrn_eps",    config.epsilon);
        config.epsilonDecay        = prefs.getFloat("lrn_epsd",   config.epsilonDecay);
        config.epsilonMin          = prefs.getFloat("lrn_epsm",   config.epsilonMin);
        config.targetSoil          = prefs.getFloat("lrn_tgtso",  config.targetSoil);
        config.soilTolerance       = prefs.getFloat("lrn_soiltol", config.soilTolerance);
        config.decisionIntervalMs  = prefs.getULong("lrn_decms",  config.decisionIntervalMs);
        config.autoControlEnabled  = prefs.getBool("lrn_auto",    config.autoControlEnabled);
        prefs.end();
        return config;
    }

    // ── Plant Doctor ────────────────────────────────────────────────
    void savePlantDoctor(const PlantDoctorConfig& config) {
        Preferences prefs;
        prefs.begin(kAgriNs, false);
        prefs.putBool("pd_enabled", config.enabled);
        prefs.putBool("pd_autodet", config.autoDetect);
        prefs.putInt("pd_detsec",   config.detectIntervalSec);
        prefs.putFloat("pd_confth", config.confidenceThreshold);
        prefs.putBool("pd_buzzer",  config.buzzerEnabled);
        prefs.end();
    }

    PlantDoctorConfig loadPlantDoctor() {
        PlantDoctorConfig config;
        Preferences prefs;
        prefs.begin(kAgriNs, true);
        config.enabled             = prefs.getBool("pd_enabled", config.enabled);
        config.autoDetect          = prefs.getBool("pd_autodet", config.autoDetect);
        config.detectIntervalSec   = prefs.getInt("pd_detsec",   config.detectIntervalSec);
        config.confidenceThreshold = prefs.getFloat("pd_confth", config.confidenceThreshold);
        config.buzzerEnabled       = prefs.getBool("pd_buzzer",  config.buzzerEnabled);
        prefs.end();
        return config;
    }

    // ── System flags ────────────────────────────────────────────────
    void saveSystemFlags(bool ruleEngineEnabled, bool fusionAutoEnabled) {
        Preferences prefs;
        prefs.begin(kAgriNs, false);
        prefs.putBool("sys_rule",  ruleEngineEnabled);
        prefs.putBool("sys_fusion", fusionAutoEnabled);
        prefs.end();
    }

    bool loadRuleEngineEnabled() {
        Preferences prefs;
        prefs.begin(kAgriNs, true);
        bool val = prefs.getBool("sys_rule", true);
        prefs.end();
        return val;
    }

    bool loadFusionAutoEnabled() {
        Preferences prefs;
        prefs.begin(kAgriNs, true);
        bool val = prefs.getBool("sys_fusion", false);
        prefs.end();
        return val;
    }

    // ── WiFi credentials (separate namespace) ───────────────────────
    void saveWiFi(const char* ssid, const char* password) {
        Preferences prefs;
        prefs.begin(kWifiNs, false);
        prefs.putString("ssid", ssid);
        prefs.putString("pass", password);
        prefs.end();
    }

    String loadWifiSsid() {
        Preferences prefs;
        prefs.begin(kWifiNs, true);
        String ssid = prefs.getString("ssid", "");
        prefs.end();
        return ssid;
    }

    String loadWifiPassword() {
        Preferences prefs;
        prefs.begin(kWifiNs, true);
        String password = prefs.getString("pass", "");
        prefs.end();
        return password;
    }

    // ── Factory reset ───────────────────────────────────────────────
    void factoryReset() {
        Preferences prefs;
        prefs.begin(kAgriNs, false);
        prefs.clear();
        prefs.end();

        prefs.begin(kWifiNs, false);
        prefs.clear();
        prefs.end();
    }

private:
    static constexpr const char* kAgriNs = "agri_cfg";
    static constexpr const char* kWifiNs = "wifi_cfg";
};

}  // namespace agri
