#include "Sensors.h"

// USB CDC 开启后 Serial 走 USB (给Atlas)，调试输出用 Serial0 (UART0 → CH343 → 电脑)
#define DebugSerial Serial0

namespace agri {

void SensorHub::begin(const PinConfig& pins) {
    pins_ = pins;

    Serial1.begin(kAirSensorBaud, SERIAL_8N1, pins_.airRx, pins_.airTx);

    if (pins_.soilPin >= 0) pinMode(pins_.soilPin, INPUT);
    if (pins_.liquidPin >= 0) pinMode(pins_.liquidPin, INPUT);

    lightReady_ = lightMeter_.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, 0x23, &Wire);
}

bool SensorHub::update(unsigned long nowMs) {
    if (nowMs - lastSampleMs_ < kSensorSampleIntervalMs) return false;
    lastSampleMs_ = nowMs;

    // 空气传感器
    if (Serial1.available()) {
        lastAirFrame_ = Serial1.readStringUntil('\n');
        lastAirFrame_.trim();
        if (lastAirFrame_.length() > 0) {
            parseAirFrame(lastAirFrame_, nowMs);
        }
    }

    // 土壤湿度
    if (pins_.soilPin >= 0) {
        int raw = readAnalogAverage(pins_.soilPin);
        float val = map(raw, 3500, 1500, 0, 100);
        val = constrain(val, 0.0f, 100.0f);
        if (val > 0.5f) {  // 基本有效性检查
            snapshot_.soilHumi = val;
            lastSoilOkMs_ = nowMs;
        }
    }

    // 液位
    if (pins_.liquidPin >= 0) {
        int raw = readAnalogAverage(pins_.liquidPin);
        float val = map(raw, 500, 3500, 0, 100);
        val = constrain(val, 0.0f, 100.0f);
        snapshot_.liquidLevel = val;
        lastLiquidOkMs_ = nowMs;
    }

    // 光照
    if (lightReady_) {
        float lux = lightMeter_.readLightLevel();
        if (lux >= 0) {
            snapshot_.lightValue = lux;
            snapshot_.isDay = lux >= 200.0f;
            lastLightOkMs_ = nowMs;
        }
    }

    snapshot_.updatedAtMs = nowMs;
    checkFaults(nowMs);
    return true;
}

void SensorHub::parseAirFrame(const String& frame, unsigned long nowMs) {
    // 跳过前导非ASCII字节
    String cleanFrame = "";
    for (unsigned int i = 0; i < frame.length(); i++) {
        char c = frame.charAt(i);
        if ((c >= 0x20 && c <= 0x7E) || c == 0x0D || c == 0x0A) {
            cleanFrame += c;
        }
    }

    float temp = 0, humi = 0;
    bool parsed = false;

    // 格式: "R:029.6RH 026.2C"
    if (cleanFrame.indexOf("R:") >= 0 && cleanFrame.indexOf("RH") >= 0) {
        int rIdx = cleanFrame.indexOf("R:");
        int rhIdx = cleanFrame.indexOf("RH");
        int cIdx = cleanFrame.indexOf("C", rhIdx);

        if (rIdx >= 0 && rhIdx > rIdx && cIdx > rhIdx) {
            humi = cleanFrame.substring(rIdx + 2, rhIdx).toFloat();
            temp = cleanFrame.substring(rhIdx + 2, cIdx).toFloat();
            parsed = true;
        }
    }

    // 保存有效数据
    if (parsed && temp > -40.0f && temp < 80.0f && humi >= 0.0f && humi <= 100.0f) {
        snapshot_.airTemp = temp;
        snapshot_.airHumi = humi;
        lastAirOkMs_ = nowMs;
    }
}

void SensorHub::checkFaults(unsigned long nowMs) {
    snapshot_.fault.airTimeout = (nowMs - lastAirOkMs_ > kSensorFaultTimeoutMs);
    snapshot_.fault.soilTimeout = (pins_.soilPin < 0) || (nowMs - lastSoilOkMs_ > kSensorFaultTimeoutMs);
    snapshot_.fault.liquidTimeout = (pins_.liquidPin < 0) || (nowMs - lastLiquidOkMs_ > kSensorFaultTimeoutMs);
    snapshot_.fault.lightTimeout = !lightReady_ || (nowMs - lastLightOkMs_ > kSensorFaultTimeoutMs);

    // 范围检查 (仅在传感器有数据时检查)
    if (!snapshot_.fault.airTimeout) {
        snapshot_.fault.airRange = (snapshot_.airTemp < -20.0f || snapshot_.airTemp > 60.0f);
    }
    if (!snapshot_.fault.soilTimeout) {
        snapshot_.fault.soilRange = (snapshot_.soilHumi < 0.0f || snapshot_.soilHumi > 100.0f);
    }
    if (!snapshot_.fault.liquidTimeout) {
        snapshot_.fault.liquidRange = (snapshot_.liquidLevel < 0.0f || snapshot_.liquidLevel > 100.0f);
    }
    if (!snapshot_.fault.lightTimeout) {
        snapshot_.fault.lightRange = (snapshot_.lightValue < 0.0f || snapshot_.lightValue > 200000.0f);
    }
}

void SensorHub::printDebugInfo(unsigned long nowMs) const {
    DebugSerial.println("\n┌─────────────────────────────────────────┐");
    DebugSerial.println("│         传感器详细调试信息              │");
    DebugSerial.println("└─────────────────────────────────────────┘");

    // 空气传感器
    DebugSerial.println("\n【空气传感器 - UART】");
    DebugSerial.printf("  引脚:        RX=GPIO%d, TX=GPIO%d\n", pins_.airRx, pins_.airTx);
    DebugSerial.printf("  波特率:      %d bps\n", kAirSensorBaud);
    DebugSerial.printf("  RX 引脚电平: %d\n", digitalRead(pins_.airRx));
    DebugSerial.printf("  TX 引脚电平: %d\n", digitalRead(pins_.airTx));
    DebugSerial.printf("  缓冲区数据:  %d 字节\n", Serial1.available());
    DebugSerial.printf("  最后收到:    \"%s\"\n", lastAirFrame_.c_str());
    DebugSerial.printf("  最后成功:    %lu ms前\n", nowMs - lastAirOkMs_);
    DebugSerial.printf("  超时阈值:    %lu ms\n", kSensorFaultTimeoutMs);
    DebugSerial.printf("  状态:        %s\n", snapshot_.fault.airTimeout ? "❌ 超时故障" : "✅ 正常");
    if (!snapshot_.fault.airTimeout) {
        DebugSerial.printf("  温度:        %.1f °C %s\n", snapshot_.airTemp, snapshot_.fault.airRange ? "(范围异常!)" : "");
        DebugSerial.printf("  湿度:        %.1f %% %s\n", snapshot_.airHumi, snapshot_.fault.airRange ? "(范围异常!)" : "");
    }

    // 土壤传感器
    DebugSerial.println("\n【土壤湿度传感器 - ADC】");
    DebugSerial.printf("  引脚:        GPIO%d\n", pins_.soilPin);
    DebugSerial.printf("  引脚电平:    %d\n", digitalRead(pins_.soilPin));
    int soilRaw = (pins_.soilPin >= 0) ? analogRead(pins_.soilPin) : -1;
    DebugSerial.printf("  ADC 原始值:  %d (0-4095)\n", soilRaw);
    DebugSerial.printf("  ADC 电压:    %.2f V\n", soilRaw * 3.3f / 4095.0f);
    DebugSerial.printf("  最后成功:    %lu ms前\n", nowMs - lastSoilOkMs_);
    DebugSerial.printf("  映射范围:    3500(干) → 1500(湿)\n");
    DebugSerial.printf("  状态:        %s\n", snapshot_.fault.soilTimeout ? "❌ 超时故障" : "✅ 正常");
    if (!snapshot_.fault.soilTimeout) {
        DebugSerial.printf("  湿度值:      %.1f %%\n", snapshot_.soilHumi);
    }

    // 液位传感器
    DebugSerial.println("\n【液位传感器 - ADC】");
    DebugSerial.printf("  引脚:        GPIO%d\n", pins_.liquidPin);
    DebugSerial.printf("  引脚电平:    %d\n", digitalRead(pins_.liquidPin));
    int liquidRaw = (pins_.liquidPin >= 0) ? analogRead(pins_.liquidPin) : -1;
    DebugSerial.printf("  ADC 原始值:  %d (0-4095)\n", liquidRaw);
    DebugSerial.printf("  ADC 电压:    %.2f V\n", liquidRaw * 3.3f / 4095.0f);
    DebugSerial.printf("  最后成功:    %lu ms前\n", nowMs - lastLiquidOkMs_);
    DebugSerial.printf("  映射范围:    500(空) → 3500(满)\n");
    DebugSerial.printf("  状态:        %s\n", snapshot_.fault.liquidTimeout ? "❌ 超时故障" : "✅ 正常");
    if (!snapshot_.fault.liquidTimeout) {
        DebugSerial.printf("  液位值:      %.1f %%\n", snapshot_.liquidLevel);
    }

    // 光照传感器
    DebugSerial.println("\n【光照传感器 - I2C BH1750】");
    DebugSerial.printf("  引脚:        SDA=GPIO%d, SCL=GPIO%d\n", pins_.i2cSda, pins_.i2cScl);
    DebugSerial.printf("  初始化:      %s\n", lightReady_ ? "成功" : "失败");
    DebugSerial.printf("  SDA 电平:    %d\n", digitalRead(pins_.i2cSda));
    DebugSerial.printf("  SCL 电平:    %d\n", digitalRead(pins_.i2cScl));
    DebugSerial.printf("  最后成功:    %lu ms前\n", nowMs - lastLightOkMs_);
    DebugSerial.printf("  状态:        %s\n", snapshot_.fault.lightTimeout ? "❌ 超时故障" : "✅ 正常");
    if (!snapshot_.fault.lightTimeout) {
        DebugSerial.printf("  光照值:      %.1f lux\n", snapshot_.lightValue);
        DebugSerial.printf("  是否白天:    %s (阈值 200 lux)\n", snapshot_.isDay ? "是" : "否");
    }

    // 故障汇总
    DebugSerial.println("\n【故障汇总】");
    DebugSerial.printf("  空气:  超时=%d  范围=%d\n", snapshot_.fault.airTimeout, snapshot_.fault.airRange);
    DebugSerial.printf("  土壤:  超时=%d  范围=%d\n", snapshot_.fault.soilTimeout, snapshot_.fault.soilRange);
    DebugSerial.printf("  液位:  超时=%d  范围=%d\n", snapshot_.fault.liquidTimeout, snapshot_.fault.liquidRange);
    DebugSerial.printf("  光照:  超时=%d  范围=%d\n", snapshot_.fault.lightTimeout, snapshot_.fault.lightRange);
    DebugSerial.printf("  综合:  %s\n", snapshot_.fault.anyFault() ? "❌ 有故障" : "✅ 全部正常");
}

int SensorHub::readAnalogAverage(int pin, int samples) {
    long sum = 0;
    for (int i = 0; i < samples; i++) {
        sum += analogRead(pin);
        delayMicroseconds(500);
    }
    return sum / samples;
}

}  // namespace agri
