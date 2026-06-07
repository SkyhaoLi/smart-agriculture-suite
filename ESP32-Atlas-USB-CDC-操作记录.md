# ESP32-S3 ↔ Atlas 200 DK USB CDC 通信 — 操作记录

**日期**: 2026-06-07  
**目标**: 实现 ESP32-S3 通过 USB CDC 与 Atlas 200 DK 双向通信，并搭建实时数据仪表盘

---

## 一、硬件连接

```
┌────────────┐  网线   ┌─────────────┐  USB 线   ┌──────────────────────────┐
│  电脑       │ ──────→ │ Atlas 200 DK│ ────────→ │ ESP32-S3  USB 口 (CDC)   │
│ (Windows)  │         │ /dev/ttyACM0│          │       Serial             │
└─────┬──────┘         └─────────────┘          │                          │
      │                                         │  COM 口 (CH343 UART)     │
      │         USB 线 (烧录 + 调试)            │       Serial0            │
      └────────────────────────────────────────→│                          │
                                                └──────────────────────────┘
```

| 通道 | ESP32 对象 | 用途 | 引脚 |
|------|-----------|------|------|
| USB CDC | `Serial` | Atlas 通信 | GPIO 19/20 (原生 USB) |
| UART0 | `Serial0` | 调试日志 → 电脑 | GPIO 43/44 (CH343) |
| UART1 | `Serial1` | 空气传感器 | GPIO 17/18 |

---

## 二、修改的文件

### 2.1 platformio.ini — 启用 USB CDC

新增两个编译宏：

```ini
build_flags =
    ...
    -DARDUINO_USB_MODE=1          # 启用硬件 USB
    -DARDUINO_USB_CDC_ON_BOOT=1   # Serial 映射到 USB CDC
```

开启后 `Serial.print()` 通过 USB 数据线发送，不再走传统 TX/RX 针脚。

### 2.2 新增 include/AtlasCDC.h + src/AtlasCDC.cpp

USB CDC 通信模块，支持：

- **指令响应协议** (Atlas 发指令 → ESP32 回复)：

| 指令 | 回复 | 说明 |
|------|------|------|
| `PING` | `PONG` | 心跳检测 |
| `START_TASK` | `TASK_COMPLETED_SUCCESSFULLY` | 触发任务 |
| `GET_DATA` | `DATA:temp=XX.X,humi=XX.X,soil=XX.X,light=XX.X` | 获取传感器数据 |
| 未知指令 | `ERROR:unknown_command` | 未识别 |

- **实时数据推送** (ESP32 主动 → Atlas)：

每 2 秒传感器采样更新时自动发送：
```
PUSH:{"temp":25.9,"humi":34.3,"soil":100.0,"light":96,"liquid":80,"valve":0,"pump":0}
```

### 2.3 src/main.cpp — 集成 Atlas CDC

- 新增 `#include "AtlasCDC.h"` 和 `static AtlasCDC gAtlasCDC`
- 调试输出改用 `DebugSerial` (= `Serial0`，UART0 → CH343 → 电脑)
- `setup()` 中调用 `gAtlasCDC.begin()`
- `loop()` 中调用 `gAtlasCDC.update()` 处理指令
- 传感器采样更新时调用 `gAtlasCDC.pushData()` 推送数据

### 2.4 src/Sensors.cpp — 调试输出重定向

`printDebugInfo()` 中所有 `Serial.` 改为 `DebugSerial.` (= `Serial0`)，避免调试信息发给 Atlas。

### 2.5 Atlas 端 Python 文件

| 文件 | 用途 |
|------|------|
| `esp32_link.py` | `ESP32Link` 通信类，封装 send/get_sensor_data/ping |
| `test_esp32.py` | 指令测试脚本 (4 项测试) |
| `quick_test.py` | 快速测试脚本 |
| `debug_cdc.py` | 原始字节调试脚本 |
| `dashboard_server.py` | **实时数据仪表盘服务器** |

---

## 三、遇到的问题与解决

### 3.1 PlatformIO 项目路径错误

**现象**: `NotPlatformIOProjectError: platformio.ini file has not been found`  
**原因**: 在根目录执行 pio 命令，但 `platformio.ini` 在 `smart-agriculture-suite` 子目录  
**解决**: 使用 `--project-dir` 参数或 `cd` 到子目录

### 3.2 Serial1 引脚冲突

**现象**: `Serial1` 已被空气传感器占用 (GPIO 17/18, 9600 baud)  
**解决**: 调试输出改用 `Serial0` (UART0 → GPIO 43/44)，避免冲突

### 3.3 USB CDC 终端回环 (关键问题)

**现象**: ESP32 发送的数据被 Atlas Linux 内核 echo 回来，导致指令被覆盖  
**表现**: 
- PING 正常但 GET_DATA 返回 unknown_command
- 调试发现 ESP32 收到了自己发出去的 ECHO 数据

**根因**: Linux `stty` 默认开启 `echo` 和 `onlcr`，串口数据被内核回环  
**解决**: 在 Python 打开串口时设置：
```python
os.system('stty -F /dev/ttyACM0 115200 cs8 -cstopb -parenb -icanon min 0 time 0 -echo -echoe -echok -echoctl -echoke -onlcr')
```

### 3.4 固件未更新 (clean build)

**现象**: 修改代码后烧录，ESP32 仍运行旧代码  
**原因**: PlatformIO 增量编译未检测到变化  
**解决**: 先 clean 再编译烧录
```powershell
pio run -e esp32-s3-controller -t clean
pio run -e esp32-s3-controller -t upload
```

---

## 四、测试结果

Atlas 上执行 `python3 /opt/Zhirun/test_esp32.py`：

```
=== ESP32 USB CDC 测试 ===
设备: /dev/ttyACM0

  ✅ PING: [PONG]
  ✅ START_TASK: [TASK_COMPLETED_SUCCESSFULLY]
  ✅ GET_DATA: [DATA:temp=25.9,humi=34.3,soil=100.0,light=94]
  ✅ UNKNOWN: [ERROR:unknown_command]

结果: 4/4 通过
```

---

## 五、实时仪表盘

### 启动

```bash
# Atlas 上执行
python3 /opt/Zhirun/dashboard_server.py
```

### 访问

浏览器打开 `http://192.168.137.100:8088`

### 功能

- ESP32 每 2 秒推送传感器数据到 Atlas
- 网页每 2 秒轮询刷新
- 显示: 温度、空气湿度、土壤湿度、光照、液位、电磁阀状态、水泵状态
- 连接状态指示 (绿灯/红灯)
- 数据阈值变色 (正常/警告/危险)

### 数据流

```
ESP32 传感器 (2s)
    → AtlasCDC::pushData() → Serial.println("PUSH:{...}")
    → USB CDC /dev/ttyACM0
    → Atlas 后台线程读取解析
    → HTTP /api/data 返回 JSON
    → 浏览器 fetch 轮询更新仪表盘
```

---

## 六、Atlas SSH 连接信息

| 项目 | 值 |
|------|-----|
| IP | 192.168.137.100 |
| 用户 | root |
| 密码 | Mind@123 |
| 项目路径 | /opt/Zhirun |

---

## 七、烧录命令速查

```powershell
# 编译 + 烧录 (在 smart-agriculture-suite 目录下)
pio run -e esp32-s3-controller -t upload

# clean 后重新编译烧录
pio run -e esp32-s3-controller -t clean; pio run -e esp32-s3-controller -t upload

# 仅编译不烧录
pio run -e esp32-s3-controller
```
