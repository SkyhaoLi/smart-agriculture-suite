#include "AtlasCDC.h"

// Serial (USB CDC) 映射说明:
//   platformio.ini 中 ARDUINO_USB_CDC_ON_BOOT=1 使默认 Serial 走 USB CDC
//   Atlas 端通过 /dev/ttyACM0 读写
//   调试日志改走 Serial0 (UART0 → CH343 → 电脑 COM 口)

namespace agri {

void AtlasCDC::begin(unsigned long baud) {
    // USB CDC 的 baud 参数实际无效，USB 全速通信由硬件决定
    // 但调用 begin() 是必要的以初始化 CDC 接口
    Serial.begin(baud);

    // 不要写 while(!Serial); — USB CDC 断开后会卡死
    delay(2000);  // 等待 USB 节点在 Linux 端注册

    Serial.println("ESP32-S3 USB CDC Ready");
    Serial.flush();
    ready_ = true;
}

void AtlasCDC::update(const SensorSnapshot& snap) {
    if (Serial.available() <= 0) return;

    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) return;

    // 清空残留数据
    while (Serial.available()) Serial.read();

    handleCommand(cmd, snap);
}

void AtlasCDC::handleCommand(const String& cmd, const SensorSnapshot& snap) {
    lastCmdMs_ = millis();
    cmdCount_++;

    if (cmd == "PING") {
        Serial.println("PONG");
        Serial.flush();
    }
    else if (cmd == "START_TASK") {
        delay(500);
        Serial.println("TASK_COMPLETED_SUCCESSFULLY");
        Serial.flush();
    }
    else if (cmd == "GET_DATA") {
        char buf[96];
        if (snap.fault.airFault()) {
            snprintf(buf, sizeof(buf), "DATA:no_sensor_data");
        } else {
            snprintf(buf, sizeof(buf), "DATA:temp=%.1f,humi=%.1f,soil=%.1f,light=%.0f",
                     snap.airTemp, snap.airHumi, snap.soilHumi, snap.lightValue);
        }
        Serial.println(buf);
        Serial.flush();
    }
    else {
        Serial.println("ERROR:unknown_command");
        Serial.flush();
    }
}

void AtlasCDC::pushData(const SensorSnapshot& snap, const ActuatorStatus& act) {
    if (!ready_) return;

    char buf[128];
    snprintf(buf, sizeof(buf),
             "PUSH:{\"temp\":%.1f,\"humi\":%.1f,\"soil\":%.1f,\"light\":%.0f,"
             "\"liquid\":%.0f,\"valve\":%d,\"pump\":%d}",
             snap.airTemp, snap.airHumi, snap.soilHumi, snap.lightValue,
             snap.liquidLevel, act.valveOn ? 1 : 0, act.pumpOn ? 1 : 0);
    Serial.println(buf);
    Serial.flush();
}

}  // namespace agri
