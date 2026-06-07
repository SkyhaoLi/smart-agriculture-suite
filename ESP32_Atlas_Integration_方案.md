# ESP32-S3 + Atlas 200I DK A2 整合方案

> **架构**: ESP32采集传感器 → WiFi HTTP → Atlas世界模型推理 → Web仪表盘显示
> 
> **目标**: ESP32作为传感器网关采集数据，Atlas负责AI推理，Web页面展示全部结果
> 
> **安全**: Atlas仅接网线和USB摄像头，不接任何GPIO/传感器/执行器，零风险

---

## 📋 目录

- [系统架构](#系统架构)
- [ESP32端说明](#esp32端说明)
- [Atlas端说明](#atlas端说明)
- [通信协议](#通信协议)
- [实施步骤](#实施步骤)
- [验证测试](#验证测试)

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    ESP32-S3 (传感器网关)                         │   │
│   │  ┌──────────────────────────────────────────────────────────┐  │   │
│   │  │ GPIO1: 土壤湿度 ADC                                        │  │   │
│   │  │ GPIO2: 液位 ADC                                           │  │   │
│   │  │ GPIO8/9: I2C (BH1750 + SSD1306 OLED)                      │  │   │
│   │  │ GPIO17/18: UART (空气温湿度传感器)                          │  │   │
│   │  └──────────────────────────────────────────────────────────┘  │   │
│   │                              │                                  │   │
│   │                              │ 传感器数据                      │   │
│   │                              ▼                                  │   │
│   │  ┌──────────────────────────────────────────────────────────┐  │   │
│   │  │ 1. 读取传感器                                             │  │   │
│   │  │ 2. 本地OLED显示 (可选)                                     │  │   │
│   │  │ 3. 本地Web面板 (可选)                                      │  │   │
│   │  │ 4. HTTP POST到Atlas ──────────────────────────────►       │  │   │
│   │  └──────────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    │ WiFi HTTP POST                    │
│                                    │ 每5秒发送一次                      │
│                                    ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Atlas 200I DK A2 (推理服务器)                 │   │
│   │                                                                  │   │
│   │   接口使用情况:                                                  │   │
│   │   ├─ 网口: ✅ 千兆以太网 (接路由器)                              │   │
│   │   ├─ GPIO: ❌ 全部空闲 (不接线)                                  │   │
│   │   ├─ I2C:  ❌ 全部空闲 (不接线)                                  │   │
│   │   ├─ UART: ❌ 全部空闲 (不接线)                                  │   │
│   │   ├─ USB:  ✅ 可选 (接USB摄像头做病害检测)                       │   │
│   │   └─ ADC:  ❌ 不使用 (无需外接ADS1115)                           │   │
│   │                                                                  │   │
│   │   运行服务:                                                      │   │
│   │   ├─ server.py: HTTP服务器 (接收ESP32数据)                      │   │
│   │   ├─ world_model.py: 世界模型推理                               │   │
│   │   └─ dashboard.py: Web仪表盘 (展示+控制)                        │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│                              Web浏览器                                   │
│                              (仪表盘显示)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔌 ESP32端说明

### 接线状态

**✅ 不需要任何改动** - ESP32端接线已经完成且正常工作

| 模块 | ESP32 GPIO | 状态 | 说明 |
|------|-----------|------|------|
| OLED | GPIO8(SDA)/GPIO9(SCL) | ✅ 已完成 | I2C 0x3C |
| BH1750 | GPIO8(SDA)/GPIO9(SCL) | ✅ 已完成 | I2C 0x23 |
| 空气温湿度 | GPIO17(RX)/GPIO18(TX) | ✅ 已完成 | UART 9600 |
| 土壤湿度 | GPIO1(ADC) | ✅ 已完成 | 5次采样均值 |
| 液位 | GPIO2(ADC) | ✅ 已完成 | 5次采样均值 |

### ESP32需要做什么

1. **修改 `include/AppConfig.h`** - 添加Atlas服务器IP配置
2. **修改 `src/main.cpp`** - 添加HTTP POST发送到Atlas的功能
3. **保持现有传感器读取逻辑不变**
4. **保持现有本地OLED/Web功能不变（可选）**

### ESP32固件修改说明

```cpp
// 新增配置项 (include/AppConfig.h)
const char* kAtlasServerHost = "192.168.1.100";  // Atlas IP地址
const int   kAtlasServerPort = 8080;              // Atlas 端口
const unsigned long kAtlasSendIntervalMs = 5000;  // 发送到Atlas间隔
```

```cpp
// 新增函数 (src/main.cpp)
// 定时将传感器数据发送到Atlas服务器
void sendSensorDataToAtlas() {
    // 从SensorHub获取最新数据
    // 构建JSON
    // HTTP POST到 http://<AtlasIP>:8080/api/esp32/data
    // 处理响应（可选：显示推理结果）
}
```

---

## 💻 Atlas端说明

### 接口使用状态

**✅ 安全** - Atlas只接网线和USB摄像头，不接触任何GPIO/传感器

| 接口 | 状态 | 说明 |
|------|------|------|
| 千兆网口 | ✅ 使用 | 通过路由器连接ESP32 |
| GPIO全部 | ❌ 空闲 | 不接线 |
| I2C全部 | ❌ 空闲 | 不接线 |
| UART全部 | ❌ 空闲 | 不接线 |
| USB摄像头 | ✅ 可选 | 用于病害检测 |
| ADC | ❌ 不使用 | 无需外接模块 |

### Atlas需要做什么

1. **修改 `server.py`** - 添加 `/api/esp32/data` 端点接收ESP32数据
2. **整合 `world_model.py`** - 收到数据后自动触发推理
3. **修改 `dashboard.py`** - 添加ESP32数据接收和显示功能
4. **保持现有Web服务不变**

### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `server.py` | 添加 `/api/esp32/data` POST端点 |
| `world_model.py` | 已有，验证兼容性 |
| `dashboard.py` | 添加ESP32数据源支持 |

---

## 📡 通信协议

### ESP32 → Atlas 数据格式

```json
POST /api/esp32/data
Content-Type: application/json

{
    "air_temp": 25.3,
    "air_humi": 60.2,
    "soil_humi": 47.0,
    "liquid_level": 82.0,
    "light": 543.0,
    "is_day": true,
    "uptime_ms": 12345678
}
```

### Atlas → ESP32 响应格式（可选）

```json
{
    "success": true,
    "prediction": {
        "soil_humi_predicted": 45.2,
        "risk_level": "low",
        "irrigation_recommendation": "no_action"
    }
}
```

### Atlas Web API 完整端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/esp32/data` | POST | 接收ESP32传感器数据 |
| `/api/predict` | POST | 世界模型推理（可选） |
| `/api/status` | GET | 获取全部状态 |
| `/api/sensors` | GET | 获取传感器数据 |
| `/dashboard` | GET | Web仪表盘页面 |
| `/api/irrigation/status` | GET | 灌溉状态 |
| `/api/plant/status` | GET | 病害检测状态 |

---

## 📝 实施步骤

### 阶段一：Atlas端准备

#### 1.1 备份现有代码
```bash
cd D:/Projects/new.06.06.03/smart-agriculture-suite-main
cp -r smart-agriculture-atlas200dk smart-agriculture-atlas200dk.bak
```

#### 1.2 修改 server.py（添加ESP32数据接收端点）

```python
# 在 server.py 中添加

# 全局变量 - 存储ESP32最新数据
esp32_latest_data = {
    "air_temp": None,
    "air_humi": None,
    "soil_humi": None,
    "liquid_level": None,
    "light": None,
    "is_day": True,
    "uptime_ms": 0,
    "last_update": None
}

# 在 do_POST 中添加
def do_POST(self):
    if self.path == '/api/esp32/data':
        self._handle_esp32_sensor_data()
    elif self.path == '/api/predict':
        self._handle_predict()
    # ... 其他端点

def _handle_esp32_sensor_data(self):
    """接收ESP32传感器数据"""
    try:
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len)
        data = json.loads(body)
        
        # 更新全局数据
        esp32_latest_data.update({
            "air_temp": data.get("air_temp"),
            "air_humi": data.get("air_humi"),
            "soil_humi": data.get("soil_humi"),
            "liquid_level": data.get("liquid_level"),
            "light": data.get("light"),
            "is_day": data.get("is_day", True),
            "uptime_ms": data.get("uptime_ms", 0),
            "last_update": time.time()
        })
        
        # 自动触发世界模型推理
        if world_model and esp32_latest_data["air_temp"] is not None:
            result = world_model.predict(esp32_latest_data)
            # 可选：存储推理结果
        
        self._send_json(200, {
            "success": True,
            "received_at": time.time()
        })
    except Exception as e:
        self._send_json(400, {"error": str(e)})
```

#### 1.3 验证Atlas服务

```bash
cd D:/Projects/new.06.06.03/smart-agriculture-suite-main/smart-agriculture-atlas200dk
python3 server.py --port 8080
```

测试端点：
```bash
curl http://localhost:8080/api/health
```

---

### 阶段二：ESP32端准备

#### 2.1 修改 include/AppConfig.h

```cpp
// 添加Atlas服务器配置
namespace agri {

// ... 现有代码 ...

// ---------------- Atlas 服务器配置 ----------------
constexpr const char* kAtlasServerHost = "192.168.1.100";  // TODO: 修改为实际Atlas IP
constexpr int   kAtlasServerPort = 8080;
constexpr unsigned long kAtlasSendIntervalMs = 5000;  // 5秒发送一次

}  // namespace agri
```

#### 2.2 修改 src/main.cpp

在 `setup()` 函数中添加Atlas发送初始化，在 `loop()` 中添加定时发送逻辑：

```cpp
// 在文件顶部添加
#include <HTTPClient.h>

namespace {
    // ... 现有变量 ...
    unsigned long gLastAtlasSendMs = 0;
    bool gAtlasConnected = false;
}

// 在 setup() 中添加
void setup() {
    // ... 现有代码 ...
    Serial.println("[Atlas] Connection configured");
}

// 在 loop() 中添加
void loop() {
    // ... 现有代码 ...
    
    // 发送到Atlas
    const unsigned long now = millis();
    if (now - gLastAtlasSendMs >= kAtlasSendIntervalMs) {
        sendToAtlas();
        gLastAtlasSendMs = now;
    }
}

// 添加新函数
void sendToAtlas() {
    const SensorSnapshot& s = gSensorHub.snapshot();
    
    StaticJsonDocument<256> doc;
    doc["air_temp"] = s.airTemp;
    doc["air_humi"] = s.airHumi;
    doc["soil_humi"] = s.soilHumi;
    doc["liquid_level"] = s.liquidLevel;
    doc["light"] = s.lightValue;
    doc["is_day"] = s.isDay;
    doc["uptime_ms"] = millis();
    
    HTTPClient http;
    String url = String("http://") + kAtlasServerHost + ":" + kAtlasServerPort + "/api/esp32/data";
    
    if (http.begin(url)) {
        http.addHeader("Content-Type", "application/json");
        int code = http.POST(doc.as<String>());
        
        if (code == 200) {
            gAtlasConnected = true;
            Serial.printf("[Atlas] Data sent successfully (HTTP %d)\n", code);
        } else {
            Serial.printf("[Atlas] Send failed (HTTP %d)\n", code);
        }
        http.end();
    } else {
        Serial.println("[Atlas] Connection failed");
    }
}
```

#### 2.3 编译测试

```bash
cd D:/Projects/new.06.5.22（改）
pio run  # 编译
pio run -t upload  # 烧录
pio device monitor  # 查看日志
```

---

### 阶段三：联合调试

#### 3.1 启动顺序

1. 先启动Atlas服务器：
```bash
cd D:/Projects/new.06.06.03/smart-agriculture-suite-main/smart-agriculture-atlas200dk
python3 server.py --port 8080
```

2. 确认Atlas IP（在Atlas上运行）：
```bash
hostname -I
```

3. 修改ESP32的 `AppConfig.h` 中的 `kAtlasServerHost` 为实际IP

4. 烧录ESP32固件

5. 观察串口日志：
```
[Atlas] Data sent successfully (HTTP 200)
```

6. 打开Atlas的Web仪表盘：
```
http://<Atlas_IP>:8080/dashboard
```

---

## ✅ 验证测试

### 测试清单

| # | 测试项 | 验证方法 | 预期结果 |
|---|--------|----------|----------|
| 1 | ESP32传感器读取 | 查看OLED显示 | 6个传感器数据正常显示 |
| 2 | ESP32本地Web | 浏览器访问ESP32 IP | 传感器数据卡片正常显示 |
| 3 | Atlas服务器启动 | 串口/终端运行server.py | "推理服务器启动"提示 |
| 4 | Atlas Web访问 | 浏览器访问Atlas IP:8080 | 仪表盘页面正常加载 |
| 5 | ESP32→Atlas数据 | 查看ESP32串口日志 | "[Atlas] Data sent successfully" |
| 6 | Atlas接收数据 | Atlas日志显示收到请求 | 传感器数据正确解析 |
| 7 | 世界模型推理 | 查看Atlas响应 | 推理结果返回 |
| 8 | 仪表盘显示 | Atlas Web页面刷新 | ESP32数据显示 |

### 完整测试流程

```bash
# 1. 启动Atlas
ssh atlas-user@<atlas-ip>
cd ~/smart-agriculture-atlas200dk
python3 server.py --port 8080 &

# 2. 确认Atlas IP
hostname -I
# 例如: 192.168.1.100

# 3. 修改ESP32 AppConfig.h
# 将 kAtlasServerHost 改为 "192.168.1.100"

# 4. 烧录ESP32
cd D:/Projects/new.06.5.22（改）
pio run -t upload

# 5. 观察ESP32串口
pio device monitor -b 115200

# 6. 预期日志输出
[Atlas] Connection configured
[WiFi] connected (IP显示)
[Atlas] Data sent successfully (HTTP 200)
[Atlas] Data sent successfully (HTTP 200)
...

# 7. 访问Atlas Web仪表盘
# 浏览器输入: http://192.168.1.100:8080/dashboard
```

---

## ⚠️ 注意事项

### 1. 网络要求
- ESP32和Atlas必须在**同一局域网**内
- 路由器需开启DHCP（通常是默认开启）
- ESP32只支持**2.4GHz WiFi**，不支持5GHz

### 2. IP配置
- Atlas IP如果是动态分配，建议在路由器上绑定静态IP
- 或在Atlas上设置静态IP

### 3. 发送频率
- 默认5秒发送一次
- 可在ESP32的 `kAtlasSendIntervalMs` 调整
- 建议不要低于3秒，避免数据量过大

### 4. 错误处理
- ESP32发送失败时会打印错误日志，不影响本地运行
- Atlas服务异常时ESP32会继续尝试重连

---

## 📁 修改文件清单

```
D:/Projects/new.06.06.03/smart-agriculture-suite-main/
│
├── smart-agriculture-atlas200dk/
│   ├── server.py          [修改] 添加 /api/esp32/data 端点
│   ├── world_model.py     [保持] 世界模型推理
│   └── 备份/
│       └── server.py.bak  [自动] 备份原文件
│
D:/Projects/new.06.5.22（改）/
│
├── include/
│   └── AppConfig.h        [修改] 添加Atlas服务器配置
│
└── src/
    └── main.cpp           [修改] 添加HTTP POST到Atlas功能
```

---

## 📞 技术支持

如遇问题，请检查：

1. 网络连通性：`ping <atlas-ip>`
2. Atlas服务状态：`curl http://localhost:8080/api/health`
3. ESP32日志：查看串口输出的 `[Atlas]` 相关日志
4. 防火墙设置：确认Atlas的8080端口未被防火墙阻止

---

*文档版本: 1.0*
*创建日期: 2026/06/03*
*适用版本: ESP32-S3传感器项目 + Atlas世界模型项目*