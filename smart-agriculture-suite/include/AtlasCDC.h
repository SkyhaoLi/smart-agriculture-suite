#pragma once

#include <Arduino.h>
#include "AppTypes.h"

namespace agri {

// ESP32-S3 USB CDC 与 Atlas 200 DK 通信模块
// Atlas 端识别为 /dev/ttyACM0，使用 Python os.read() + select() 通信
//
// 指令协议 (以 \n 结尾的文本行):
//   PING        → PONG
//   START_TASK  → TASK_COMPLETED_SUCCESSFULLY
//   GET_DATA    → DATA:temp=XX.X,humi=XX.X,soil=XX.X,light=XX.X
//   任意未知     → ERROR:unknown_command

class AtlasCDC {
public:
    void begin(unsigned long baud = 115200);
    void update(const SensorSnapshot& snap);

    // 实时推送传感器数据 (传感器采样更新时调用)
    void pushData(const SensorSnapshot& snap, const ActuatorStatus& act);

    bool isReady() const { return ready_; }
    unsigned long lastCommandMs() const { return lastCmdMs_; }
    unsigned long commandCount() const { return cmdCount_; }

private:
    void handleCommand(const String& cmd, const SensorSnapshot& snap);

    bool ready_ = false;
    unsigned long lastCmdMs_ = 0;
    unsigned long cmdCount_ = 0;
};

}  // namespace agri
