[使用说明.md](https://github.com/user-attachments/files/27590030/default.md)
# 智润 — 智慧农业边缘AI灌溉控制系统 使用说明

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [PC模拟器使用](#pc模拟器使用)
  - [安装依赖](#安装依赖)
  - [启动命令](#启动命令)
  - [Web控制台功能](#web控制台功能)
  - [传感器注入测试](#传感器注入测试)
  - [时间加速](#时间加速)
- [Atlas 200I DK A2 使用](#atlas-200i-dk-a2-使用)
  - [部署到开发板](#部署到开发板)
  - [启动运行](#启动运行)
- [AI模型训练工具](#ai模型训练工具)
  - [植物病害模型训练](#植物病害模型训练)
  - [融合神经网络训练](#融合神经网络训练)
- [对比实验工具](#对比实验工具)
  - [灌溉策略基准测试](#灌溉策略基准测试)
  - [联邦学习模拟](#联邦学习模拟)
- [大屏实时展示](#大屏实时展示)
- [室外部署传感器选型](#室外部署传感器选型)
- [REST API 参考](#rest-api-参考)
- [常见问题](#常见问题)

---

## 项目简介

智润是一套完整的边缘AI智慧农业灌溉控制系统，支持两个硬件平台：

| 平台 | 语言 | 用途 |
|------|------|------|
| **Atlas 200I DK A2** | Python | 华为昇腾开发板部署，NPU加速推理 |
| **PC模拟器** | Python / Flask | 无硬件开发调试，功能演示 |

系统集成8大AI模块：
1. **规则灌溉引擎** — 日夜分时阈值判断
2. **Q-Learning 强化学习** — 900状态×4动作，在线学习最优灌溉策略
3. **卡尔曼滤波 + 神经网络融合** — 5通道传感器数据融合
4. **3层异常检测** — 均值/Z-Score/Isolation Forest
5. **GDD作物生长预测** — 积温模型追踪生长阶段
6. **ONNX 植物病害识别** — CNN模型，5类草莓病害
7. **MobileNetV2 特征匹配** — 1280维特征向量，余弦相似度匹配，跨作物23类病害泛化识别
8. **GRU 环境状态预测** — 纯NumPy实现，预测未来15/30/60分钟的气温、气湿、土湿、光照

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                            传感器层 (Field Edge)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ 空气温湿度│ │ 土壤湿度  │ │ 液位     │ │ 光照BH1750│ │ USB摄像头│  │
│  │ UART     │ │ ADC/ADS  │ │ ADC/ADS  │ │ I2C      │ │ 相机     │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       └────────────┴────────────┴────────────┴────────────┘         │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ 原始数据
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         边缘推理层 (Edge AI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ Kalman 滤波  │  │ Isolation   │  │ Q-Learning  │                  │
│  │ 5通道融合    │  │ Forest 异常 │  │ 在线学习    │                  │
│  │ + NN 决策   │  │ 检测(10树)  │  │ 900×4 状态  │                  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │
│         └────────────────┼────────────────┘                         │
│                          ▼                                           │
│              ┌───────────────────────┐   ┌─────────────┐            │
│              │   执行器仲裁调度器     │   │ ONNX / NPU  │            │
│              │ 安全锁>手动>AI>规则   │   │ 病害识别    │            │
│              └───────────┬───────────┘   │ 5类草莓病害 │            │
│                          │               └─────────────┘            │
│  ┌───────────┐  ┌───────┴───────┐  ┌─────────────┐                 │
│  │ GDD 生长  │  │ 电磁阀 + 水泵 │  │ JSON 文件   │                 │
│  │ 预测      │  │ GPIO 驱动     │  │ 持久化      │                 │
│  └───────────┘  └───────────────┘  └─────────────┘                 │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ HTTP
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         可视化层 (Visualization)                     │
│  ┌──────────────────────┐    ┌──────────────────────┐               │
│  │  Atlas Web 仪表板    │    │  PC 模拟器           │               │
│  │  Flask + 单页应用    │    │  Flask + Chart.js    │               │
│  │  暗色主题 9 页面     │    │  暗色主题 9 页面     │               │
│  └──────────────────────┘    └──────────────────────┘               │
│                   REST API (30+ 端点，两平台一致)                     │
└─────────────────────────────────────────────────────────────────────┘
```

**执行器优先级**：安全锁定 > 手动 > AI学习/融合 > 规则引擎

**联邦学习架构**（多设备协作）：
```
田块A(Atlas#1)  田块B(Atlas#2)  田块C(Atlas#3)
    │               │               │
    └─── 定期上传 Q-Table (仅参数) ──┘
                    ▼
            聚合服务器 (FedAvg)
            Q_avg = mean(Q_i)
                    │
    ┌─── 下发聚合后的 Q-Table ───┐
    ▼               ▼               ▼
继续本地学习    继续本地学习    继续本地学习
```
核心价值：多块田独立学习，共享经验，不传原始数据

**离线运行保障**：所有AI功能完全离线运行，WiFi仅用于远程查看。

| 功能 | 离线可用 | 推理位置 |
|------|---------|---------|
| 规则灌溉 | ✅ | 本地 |
| Q-Learning | ✅ | 本地查表 |
| 传感器融合 | ✅ | Kalman + NN本地前向 |
| 病害识别 | ✅ | ONNX / NPU 本地推理 |
| 异常检测 | ✅ | Isolation Forest本地建树 |
| 生长预测 | ✅ | GDD + 线性回归本地 |

**两平台技术栈对比**：

| 层级 | Atlas版 | 模拟器 |
|------|---------|--------|
| 语言 | Python 3 | Python 3 |
| AI推理 | ACL NPU / ONNX Runtime | TFLite / NumPy |
| Web | Flask | Flask |
| 持久化 | JSON文件 | JSON文件 |
| 显示 | OLED / HDMI | 浏览器 |

---

## 环境要求

### PC模拟器
- Python 3.8+
- 依赖包：flask, flask-cors, Pillow, numpy

### Atlas 200I DK A2
- Python 3.10+
- 依赖包：pyserial, smbus2, gpiod, numpy, opencv-python, flask

---

## PC模拟器使用

### 安装依赖

```bash
cd simulator
pip install -r requirements.txt
```

或使用项目虚拟环境（已预装所有依赖）：

```bash
source ../venv/bin/activate
```

### 启动命令

```bash
# 默认启动（实时速度，自动打开浏览器）
python run.py

# 加速60倍（1秒=1分钟），指定端口
python run.py --time-scale 60 --port 8080

# 加速3600倍（1秒=1小时），不自动打开浏览器
python run.py --time-scale 3600 --no-browser
```

**启动参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--time-scale N` | 1.0 | 时间加速倍率。1=实时，60=1分钟/秒，3600=1小时/秒 |
| `--port PORT` | 自动(5000-5010) | HTTP服务端口 |
| `--no-browser` | false | 不自动打开浏览器 |

启动后终端会显示：
```
[System] all modules initialized
[System] simulation loop started
[Main] opening browser: http://localhost:5000/
[Main] Smart Agriculture Simulator running (time_scale=1.0x)
[Main] Press Ctrl+C to stop
```

按 `Ctrl+C` 停止模拟器，当前配置会自动保存到 `simulator/data/config.json`。

### Web控制台功能

浏览器打开 `http://localhost:5000/` 后，顶部有9个功能标签页：

#### 1. 仪表盘 (Dashboard)
- 5个传感器实时数值（空气温度、空气湿度、土壤湿度、液位、光照）
- 执行器状态（阀门、水泵开关状态及控制来源）
- 各模块运行状态概览

#### 2. 数据大屏 (Charts)
- **土壤湿度趋势图** — 实时曲线，100个数据点滚动显示
- **空气温湿度图** — 双Y轴，温度(°C)和湿度(%)同图展示
- **学习奖励曲线** — Q-Learning累积奖励变化
- **融合对比图** — 卡尔曼滤波前后的传感器值对比

图表每3秒自动刷新，可直观观察系统运行趋势。

#### 3. 灌溉控制 (Irrigation)
- **模式切换** — 自动/手动模式切换按钮
- **手动控制** — 水泵开/关、阀门开/关按钮（带状态高亮）
- **灌溉阈值配置** — 日间/夜间模式的温度、湿度、土壤湿度阈值
- **规则引擎开关** — 启用/禁用规则灌溉

> **注意**：手动操作水泵或阀门时，系统会自动切换到手动模式。

#### 4. 异常检测 (Anomaly)
- 3层检测状态（均值偏离、Z-Score、Isolation Forest）
- 当前告警列表
- 各传感器异常详情（点击传感器名称查看）
- 清除告警按钮

#### 5. 生长预测 (Growth)
- 当前作物选择（番茄、生菜、辣椒、黄瓜、草莓）
- GDD（积温）进度条和生长阶段
- 历史生长数据
- 生长预测曲线

#### 6. Q-Learning (Learning)
- 当前策略参数（α学习率、γ折扣因子、ε探索率）
- Q-Table热力图 — 900个状态×4个动作的价值矩阵可视化
- 最近决策解释 — 显示当前状态、选择的动作及原因
- 自动控制开关
- 参数调节滑块
- 用户反馈按钮（正面/负面奖励）
- 重置学习按钮

#### 7. 融合模块 (Fusion)
- 卡尔曼滤波后的传感器估计值
- 神经网络置信度（3个最优传感器的权重）
- 自动控制开关
- 5通道滤波值实时图表
- 手动调整NN权重

#### 8. 植物医生 (Plant Doctor)
- 病害检测开关和自动检测开关
- **上传图片检测** — 选择叶片图片进行病害识别
- **Grad-CAM热力图** — 显示模型关注区域（遮挡敏感度方法）
- **综合识别** — 原始CNN模型 + MobileNetV2特征匹配，自动选择最佳结果
- 检测结果（作物名称、病害名称、置信度、治疗建议）
- 检测历史记录
- 支持8种作物23类病害：番茄(10类)、马铃薯(3类)、玉米(4类)、葡萄(4类)、苹果(4类)、桃(2类)、辣椒(2类)、草莓(2类)

#### 9. 系统配置 (Config)
- 模块开关（规则引擎、Q-Learning自动、融合自动、植物医生）
- WiFi配置（模拟器存根）
- 恢复出厂设置
- OTA更新（模拟器不支持）

### 传感器注入测试

可通过API强制设定传感器值，用于测试特定场景：

```bash
# 注入传感器值（模拟高温低湿场景）
curl -X POST http://localhost:5000/api/sensor/inject \
  -H "Content-Type: application/json" \
  -d '{"airTemp": 35.0, "airHumi": 30.0, "soilHumi": 20.0}'

# 恢复自动模拟
curl -X DELETE http://localhost:5000/api/sensor/inject
```

支持的传感器字段：`airTemp`、`airHumi`、`soilHumi`、`liquidLevel`、`lightValue`

### 时间加速

通过API动态调整时间倍率：

```bash
# 设置为100倍速
curl -X POST http://localhost:5000/api/simulator/time-scale \
  -H "Content-Type: application/json" \
  -d '{"scale": 100}'

# 查看模拟器状态
curl http://localhost:5000/api/simulator/status
```

返回值说明：
- `timeScale` — 当前加速倍率
- `uptime` — 实际运行时间（秒）
- `simHours` — 模拟时间（小时）

---

## Atlas 200I DK A2 使用

### 部署到开发板

**方式一：SSH远程部署**

```bash
cd smart-agriculture-atlas200dk

# 编辑 tools/deploy.py 中的板卡IP和密码
# 然后执行一键部署
python tools/deploy.py
```

**方式二：手动部署**

```bash
# 在Atlas板上安装依赖
scp -r ./* user@atlas-ip:~/smart-agriculture/
ssh user@atlas-ip
cd ~/smart-agriculture
pip3 install -r requirements.txt
```

**一键安装依赖**（在Atlas板上执行）：

```bash
bash tools/install.sh
```

### 启动运行

```bash
# 默认启动（Profile 1: ControllerKit）
python3 main.py

# 指定硬件配置和端口
python3 main.py --profile 2 --port 8080

# Profile 说明
# 1 = ControllerKit (灌溉控制)
# 2 = HybridDevKit (灌溉+摄像头)
# 3 = CameraEyeStandalone (纯摄像头)
```

**设置开机自启**：

```bash
sudo cp tools/agri-atlas.service /etc/systemd/system/
sudo systemctl enable agri-atlas
sudo systemctl start agri-atlas
```

---

## AI模型训练工具

### 植物病害模型训练

训练植物叶片病害分类模型。

**原始CNN模型**（5类草莓病害）：
- 模型架构：Conv2D(32) → Conv2D(64) → Conv2D(64) → Dense(64) → Dense(5)，输入96×96×3 RGB
- 训练完成后将ONNX模型放置到 `smart-agriculture-atlas200dk/models/` 目录即可使用

**MobileNetV2特征匹配**（23类跨作物病害）：
- 使用预训练MobileNetV2提取1280维特征向量
- 通过余弦相似度匹配特征库中的已知病害
- 特征库支持在线注册新病害（小样本学习）
- 构建特征库：`python disease_feature_extractor.py <model_path> <dataset_dir> [output_path]`

**综合识别逻辑**：当原始CNN模型置信度 < 50% 且特征匹配有结果时，使用特征匹配结果作为最终诊断。

Atlas版支持三种推理引擎（按优先级自动选择）：
1. **ACL NPU** — 华为昇腾NPU加速推理（推荐）
2. **ONNX Runtime** — 通用推理后端
3. **OpenCV DNN** — 备用方案

### 融合神经网络训练

传感器融合神经网络（5输入→8隐藏→3输出），训练后权重自动保存为JSON格式。

---

## 对比实验工具

### 灌溉策略基准测试

比较3种灌溉策略的效果：纯规则、规则+Q-Learning、规则+融合。

```bash
cd simulator

# 运行基准测试（默认模拟72小时）
python benchmark.py

# 指定模拟时长和加速倍率
python benchmark.py --hours 168 --time-scale 3600
```

测试完成后会生成HTML报告，包含：
- 各策略的土壤湿度稳定性对比
- 灌溉次数和用水量统计
- 累积奖励曲线
- 实时数据图表

### 联邦学习模拟

模拟多设备联邦学习场景，验证FedAvg算法对Q-Table的聚合效果。

```bash
cd simulator

# 运行联邦学习模拟（默认5台设备，10轮通信）
python federated_learning.py

# 自定义参数
python federated_learning.py --devices 10 --rounds 20 --hours 48
```

模拟内容：
- N台虚拟边缘设备各自独立学习
- 每轮通信后通过FedAvg聚合Q-Table
- 对比聚合前后的策略表现
- 生成HTML对比报告

---

## REST API 参考

两个平台提供完全一致的REST API（30+端点）。以下以模拟器 `http://localhost:5000` 为例。

### 系统状态

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/status` | 获取全部状态（传感器+执行器+所有模块） |
| GET | `/api/system/modules` | 获取模块开关状态 |
| POST | `/api/system/modules` | 设置模块开关 |
| GET | `/api/system/wifi` | 获取WiFi配置 |
| POST | `/api/system/wifi` | 设置WiFi配置 |
| POST | `/api/system/factory-reset` | 恢复出厂设置 |

### 灌溉控制

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/irrigation/status` | 灌溉状态 |
| POST | `/api/irrigation/mode` | 切换自动/手动 `{"auto": true/false}` |
| POST | `/api/irrigation/pump` | 开关水泵 `{"state": true/false}` |
| POST | `/api/irrigation/valve` | 开关阀门 `{"state": true/false}` |
| POST | `/api/irrigation/pump_only` | 仅开水泵 `{"state": true/false}` |
| GET | `/api/irrigation/config` | 获取灌溉阈值配置 |
| POST | `/api/irrigation/config` | 更新灌溉阈值配置 |

### 异常检测

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/anomaly/status` | 异常检测状态 |
| GET | `/api/anomaly/alerts` | 当前告警列表 |
| GET | `/api/anomaly/sensor?name=airTemp` | 传感器异常详情 |
| POST | `/api/anomaly/clear` | 清除所有告警 |

### 生长预测

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/growth/status` | 生长状态 |
| GET | `/api/growth/history` | 历史数据 |
| GET | `/api/growth/prediction` | 生长预测 |
| POST | `/api/growth/crop` | 切换作物 `{"cropId": 0-4}` |
| POST | `/api/growth/reset` | 重置生长数据 |

### Q-Learning

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/learning/status` | 学习状态 |
| GET | `/api/learning/qtable` | Q-Table数据 |
| GET | `/api/learning/explain` | 最近决策解释 |
| POST | `/api/learning/params` | 更新学习参数 |
| POST | `/api/learning/feedback` | 用户反馈 `{"positive": true/false}` |
| POST | `/api/learning/reset` | 重置Q-Table |

### 融合模块

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/fusion/status` | 融合状态 |
| GET | `/api/fusion/sensors` | 各通道滤波值 |
| POST | `/api/fusion/config` | 设置自动控制 |
| POST | `/api/fusion/weights` | 更新NN权重 |

### 植物医生

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/plant/status` | 植物医生状态 |
| GET | `/api/plant/history` | 检测历史 |
| POST | `/api/plant/detect` | 上传图片检测（支持multipart文件、raw bytes、base64 JSON） |
| POST | `/api/plant/detect_gradcam` | 上传图片检测+Grad-CAM热力图（同上三种格式） |
| GET | `/api/plant/capture` | 拍照检测（Atlas用摄像头，模拟器返回503） |
| POST | `/api/plant/config` | 配置植物医生参数 |

### 环境状态预测（世界模型）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/worldmodel/status` | 世界模型状态（预测值、置信度、训练信息） |
| GET | `/api/worldmodel/history` | 历史传感器数据 |
| POST | `/api/worldmodel/config` | 配置世界模型参数 |

### 模拟器专属

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/sensor/inject` | 注入传感器值 |
| DELETE | `/api/sensor/inject` | 恢复自动模拟 |
| GET | `/api/simulator/status` | 模拟器状态（时间倍率、运行时长） |
| POST | `/api/simulator/time-scale` | 设置时间加速 `{"scale": 100}` |

### API调用示例

```bash
# 获取全部状态
curl http://localhost:5000/api/status

# 切换到手动模式
curl -X POST http://localhost:5000/api/irrigation/mode \
  -H "Content-Type: application/json" -d '{"auto": false}'

# 开启水泵
curl -X POST http://localhost:5000/api/irrigation/pump \
  -H "Content-Type: application/json" -d '{"state": true}'

# 上传图片进行病害检测（multipart文件方式）
curl -X POST http://localhost:5000/api/plant/detect \
  -F "file=@leaf_image.jpg"

# 上传图片进行病害检测（base64方式）
curl -X POST http://localhost:5000/api/plant/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -w0 leaf_image.jpg)'"}'

# 注入传感器值（模拟干旱场景）
curl -X POST http://localhost:5000/api/sensor/inject \
  -H "Content-Type: application/json" \
  -d '{"soilHumi": 15.0, "airTemp": 38.0}'

# 恢复出厂设置
curl -X POST http://localhost:5000/api/system/factory-reset
```

---

## 性能指标

### Atlas 200I DK A2 推理性能

| 指标 | 值 | 备注 |
|------|-----|------|
| ONNX 病害推理总耗时 | ~50–150 ms | 含相机采集+预处理+推理 |
| ACL NPU 推理 | ~10–30 ms | INT8量化模型，NPU加速 |
| MobileNetV2 特征提取 | ~100–200 ms | 1280维特征向量+余弦相似度匹配 |
| 融合NN单次前向 | <1 ms | 5→8→3全连接，75个float参数 |
| Q-Learning一步更新 | <0.1 ms | 900×4 Q-Table，单次查表+更新 |
| Isolation Forest单次评分 | ~1–3 ms | 10棵树，深度8 |
| GRU 推理 60步 | ~0.2 ms | 纯NumPy实现，1220个参数 |
| GRU 训练一步 | ~5 ms | batch=8, seq_len=15 |

### 内存占用

| 指标 | 值 |
|------|-----|
| Q-Table内存 | 14.4 KB (900×4×4 bytes) |
| Isolation Forest内存 | ~25 KB (10棵树) |
| 融合网络权重 | ~0.3 KB (75个float参数) |
| 特征库内存 | ~900 KB (23类×184条×1280维) |
| GRU 参数 | ~5 KB (1220个float32) |
| 传感器缓冲区 | ~29 KB (1440分钟×5维) |

### 时序参数

| 事件 | 间隔 | 说明 |
|------|------|------|
| 传感器采样 | 2 s | 每2秒采样一次 |
| 融合决策 | 10 s | Kalman+NN融合周期 |
| Isolation Forest运行 | 60 s | 异常检测周期 |
| 病害自动检测 | 可配置 | 默认60秒 |
| Q-Table保存 | 每50 episode | JSON文件持久化 |
| Web仪表板刷新 | 3 s | 前端自动刷新 |

### Q-Learning参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 状态空间 | 5×4×5×3×3 = 900 | 温度×湿度×土壤×光照×时段 |
| 动作空间 | 4 | Off(0s) / Low(30s) / Medium(60s) / High(120s) |
| 学习率 α | 0.1 | 可通过API修改 |
| 折扣因子 γ | 0.9 | 可通过API修改 |
| 初始探索率 ε | 0.3 | 自适应衰减（平均奖励>2.0时加速） |

### Isolation Forest参数

| 参数 | 值 |
|------|-----|
| 树数量 | 10 |
| 最大深度 | 8 |
| 训练缓冲区 | 200样本 |
| 触发训练阈值 | ≥50样本 |
| 异常分数阈值 | 0.65 |

---

## 演示流程（3分钟）

### 准备工作

1. **启动模拟器**：`cd simulator && python run.py`（功能最全）
2. **或启动Atlas版**：`cd smart-agriculture-atlas200dk && python3 main.py`
3. **浏览器访问**：打开 `http://localhost:5000/` 或 `http://<Atlas-IP>:8080/`
4. **准备样本**：真实草莓叶片或高清打印图片（用于病害检测）

### 演示脚本

| 时间 | 环节 | 内容 |
|------|------|------|
| 0:00–0:30 | 系统启动 | 启动模拟器/Atlas版，浏览器打开Web仪表板 |
| 0:30–1:00 | Web仪表板 | 浏览器展示9个标签页，3秒自动刷新，暗色主题 |
| 1:00–1:40 | 病害检测 | 上传叶片图片→ONNX推理→病害识别+置信度+治疗建议 |
| 1:40–2:20 | Q-Learning | 展示Q-Table热力图、探索率衰减、用户反馈干预学习 |
| 2:20–3:00 | 异常注入 | 注入极端值→三层检测告警→安全锁定 |

### 话术要点

- **病害识别**："ONNX/NPU模型在边缘设备上推理，5类草莓病害，全程无需云端"
- **Q-Learning**："900状态4动作，一步更新<0.1ms，Q-Table持久化到JSON文件，断电不丢失"
- **异常检测**："边缘设备自研Isolation Forest，10棵树深度8，从实时数据自动建树"
- **离线能力**："所有AI功能完全离线运行，灌溉、学习、检测、告警一切照常"

---

## 大屏实时展示

系统提供独立的大屏展示页面，专为比赛/演示场景设计，适合在大屏幕或投影仪上实时展示。

### 访问方式

```
http://localhost:5000/dashboard          # PC模拟器
http://<Atlas-IP>:8080/dashboard         # Atlas开发板
```

### 大屏功能

| 区域 | 内容 |
|------|------|
| 顶部标题栏 | 系统名称、硬件平台、运行时长、模拟时间、时间倍率、全屏按钮 |
| 传感器仪表盘 | 5个大数字显示（温度、湿度、土壤、液位、光照），带SVG环形进度条 |
| 实时趋势图 | 土壤湿度折线图，含目标值参考线，60个数据点滚动显示 |
| 执行器状态 | 水泵/阀门开关、自动/手动模式、控制来源、剩余时间 |
| AI模块状态 | 6个模块的开关状态和关键指标（ε探索率、IF分数、生长阶段等） |
| 告警信息栏 | 实时告警滚动显示，无告警时显示"系统运行正常" |

页面每3秒自动刷新，支持F11全屏显示。访问后点击右上角"全屏"按钮即可进入大屏模式。

### 大屏显示硬件推荐

#### 方案一：电视/显示器 + 迷你主机（推荐）

| 组件 | 推荐 | 价格 |
|------|------|------|
| 显示屏 | 55寸4K智能电视（小米/海信/创维） | 1500-3000元 |
| 迷你主机 | Intel N100迷你主机（4GB+128GB） | 500-800元 |
| 无线键鼠 | 用于操作和全屏切换 | 50-100元 |

优点：画质好、可视角度大、可同时显示其他内容。适合固定展厅。

#### 方案二：投影仪方案

| 组件 | 推荐 | 价格 |
|------|------|------|
| 投影仪 | 极米/当贝/坚果 1080p智能投影 | 2000-4000元 |
| 投影幕布 | 100寸电动幕布 | 200-500元 |
| 迷你主机 | 同上 | 500-800元 |

优点：画面大、视觉冲击强。适合答辩/演讲场景。

#### 方案三：树莓派直连（最省钱）

| 组件 | 推荐 | 价格 |
|------|------|------|
| 开发板 | 树莓派4B/5（2GB+即可） | 300-600元 |
| 显示屏 | 10寸HDMI显示屏 | 200-400元 |
| SD卡 | 32GB Class10 | 30元 |

树莓派运行Chromium全屏打开大屏页面，功耗低（5V/3A），可24小时运行。

#### 方案四：平板电脑（最便携）

直接用iPad或安卓平板的浏览器打开大屏页面，全屏显示即可。适合外出演示。

> **网络要求**：大屏设备需与运行模拟器/Atlas的设备在同一局域网。

---

## 室外部署传感器选型

实际室外部署需要能承受刮风下雨的工业级传感器。以下按项目5个传感器接口分别推荐。

### 传感器总览

| 传感器 | 推荐型号 | 接口兼容 | 防护等级 | 价格(元) |
|--------|---------|---------|---------|---------|
| 空气温湿度 | 建大仁科 RS-WS-N01 (TTL款) | UART直连 | IP65 | 80-250 |
| 土壤湿度 | 精讯畅通 JXBS-3001-TR (0-3.3V款) | ADC直连 | IP68(埋土) | 80-200 |
| 液位 | 投入式液位变送器 0-3V (1米量程) | ADC直连 | IP68 | 50-200 |
| 光照 | TSL2591模块(装在防水盒玻璃窗后) | I2C直连 | 需加防护 | 15-35 |
| 摄像头 | USB摄像头 (海康/大恒/罗技) | USB直连 | IP65+ | 300-2000 |

### 空气温湿度传感器

**首选：建大仁科 RS-WS-N01 TTL款**

- 温度范围 -40~+80°C，精度 ±0.5°C；湿度范围 0-100%RH，精度 ±3%
- IP65防水探头，可直接暴露在室外
- TTL款直接输出 `"Temp:XX.X, Humi:XX.X"`，与固件 `parseAirFrame()` 完全兼容
- 淘宝搜索：`"建大仁科 温湿度传感器 TTL"`

如购买RS485款（更常见），需加 MAX485模块（3-5元）将RS485转为TTL接GPIO17/18。

### 土壤湿度传感器

**首选：精讯畅通 JXBS-3001-TR FDR电容式**

- IP68等级，可长期埋在土里不腐蚀，环氧树脂密封探头
- 选购 **0-3.3V输出款**，直接接GPIO1，无需分压
- 淘宝搜索：`"精讯畅通 土壤水分传感器 0-3.3V"`
- 装上后需重新校准 `sensor_hub.py` 中的ADC映射值

### 液位传感器

**首选：投入式液位变送器**

- 扩散硅/陶瓷压力式，IP68可长期浸泡
- 选购 **0-3V输出款**，接GPIO2
- 根据水箱高度选量程（0-1m / 0-2m / 0-5m）
- 通常需12V供电，用MT3608升压模块从电池升压
- 淘宝搜索：`"投入式液位传感器 0-3V IP68 1米"`

### 光照传感器

**首选：TSL2591模块**

- 量程0-188,000 lux（远超BH1750的65k），室外阳光直射不会溢出
- I2C地址0x29，与BH1750(0x23)不冲突
- 室内防护：装在防水盒内，通过玻璃/亚克力窗口透光
- 如需专业级防护，用建大仁科 RS-GZ-N01 光照变送器（IP65，~100-250元）

### 摄像头

**推荐：USB工业摄像头**

- 统一使用USB摄像头，Atlas 200I DK A2 通过OpenCV `cv2.VideoCapture()` 读取
- 经济款：罗技 C920/C930e USB摄像头 (~300-600元)
- 工业款：海康威视 USB工业相机 (~400-1000元)
- 专业款：大恒图像 工业USB相机 (全局快门, ~800-2000元)
- 如需户外安装，加装防水外壳 (IP65+)
- 2.8mm镜头适合近距离拍摄叶片

### 防水外壳与供电

| 组件 | 推荐 | 价格 |
|------|------|------|
| 主控防水盒 | IP65 ABS盒 150×110×70mm | 20-40元 |
| 太阳能板 | 6V/3W单晶硅 | 15-35元 |
| 电池 | 18650 3400mAh | 10-25元 |
| 充电模块 | CN3065太阳能充电 | 3-8元 |
| 稳压 | HT7333(3.3V) + MT3608(升压12V) | 6-15元 |
| 防水接头 | GX12-4航空插头 ×5 | 25-50元 |
| 户外线缆 | 硅胶4芯线 ~10米 | 10-30元 |

**总计约 350-1000元/套**（批量采购可更低）。

### 代码适配提醒

接上真实传感器后需修改代码：

1. **ADC映射校准** — `sensor_hub.py` 中土壤湿度和液位的映射参数需根据实际输出调整
2. **I2C驱动** — 如用TSL2591替换BH1750，需修改I2C地址和读取逻辑
3. **RS485通信** — 如温湿度传感器是RS485款，需加MAX485模块并添加Modbus解析
4. **采样间隔** — 室外部署建议将采样间隔从2秒改为60-300秒（1-5分钟）以省电

---

## 常见问题

### Q: 模拟器启动报错 `ModuleNotFoundError: No module named 'flask'`
A: 安装依赖：`pip install -r simulator/requirements.txt`，或使用项目虚拟环境：`source venv/bin/activate`

### Q: 浏览器打开后页面空白
A: 检查终端是否有错误输出。确认模型文件存在。

### Q: 灌溉控制按钮点击没反应
A: 检查浏览器控制台(F12)是否有网络错误。确认模拟器正在运行。手动操作水泵/阀门会自动切换到手动模式。

### Q: Q-Learning 训练很慢
A: 使用时间加速：`python run.py --time-scale 3600`，让模拟以1小时/秒的速度运行。在Web控制台的"数据大屏"标签页可以观察学习奖励曲线的变化。

### Q: 植物医生检测返回 503 错误
A: 确认植物医生模块已启用（系统配置标签页）。确认模型文件存在且完整。

### Q: 如何重置所有配置？
A: Web控制台"系统配置"标签页点击"恢复出厂设置"，或删除 `simulator/data/config.json` 文件后重启。

### Q: 两个平台的API完全兼容吗？
A: 是的。Atlas 200I DK A2 和 PC模拟器提供相同的30+ REST API端点，Web前端代码也完全一致。模拟器额外增加了传感器注入和时间加速端点。

### Q: 如何查看Q-Learning的学习效果？
A: 在"Q-Learning"标签页查看Q-Table热力图，颜色越深表示该状态-动作对的价值越高。同时可以观察"数据大屏"中的学习奖励曲线，奖励上升说明策略在改善。

### Q: 联邦学习模拟的作用是什么？
A: 模拟多台边缘设备各自学习灌溉策略，然后通过FedAvg算法聚合Q-Table。这展示了在不共享原始数据的情况下，多设备协作提升策略的可行性。运行 `python federated_learning.py` 生成对比报告。
