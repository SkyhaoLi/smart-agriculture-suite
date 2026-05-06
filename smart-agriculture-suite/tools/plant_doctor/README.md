# Plant Doctor Tools

离线工具集，用于植物病害检测模型的训练、量化和固件集成。

## 工具清单

| 文件 | 用途 |
|------|------|
| `train_model.py` | 训练草莓病害分类 CNN 并导出 INT8 TFLite 模型 |
| `tflite_to_header.py` | 将 `.tflite` 模型转为固件可直接包含的 `.h` 文件 |

---

## 完整训练工作流

### 1. 环境准备

```bash
# 建议使用 Python 3.9+，安装 TensorFlow 2.x
pip install tensorflow
# 如需 GPU 加速（NVIDIA）：
pip install tensorflow[and-cuda]
```

### 2. 准备数据集

按以下目录结构组织图片，每个子目录对应一种病害类别：

```
data_dir/
  Healthy/                    # 健康叶片
  Strawberry_Anthracnose/     # 草莓炭疽病
  Strawberry_Gray_Mold/       # 草莓灰霉病
  Strawberry_Leaf_Scorch/     # 草莓叶焦病
  Strawberry_Powdery_Mildew/  # 草莓白粉病
```

**数据采集建议：**

- 每类至少 **200 张** 图片，推荐 500+ 张以获得较好泛化能力
- 拍摄条件应尽量多样：不同光照、角度、生长阶段
- 图片分辨率不需要统一，脚本会自动缩放到 96x96
- 建议格式：JPG / PNG
- 病害图片应覆盖早期、中期和晚期症状
- 健康叶片应包含不同品种和叶龄

### 3. 训练模型

```bash
python tools/plant_doctor/train_model.py \
  --data_dir ./dataset \
  --epochs 30 \
  --batch_size 32 \
  --output model.tflite \
  --num_classes 5
```

**命令行参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | （必填） | 数据集根目录 |
| `--epochs` | 30 | 最大训练轮数（含早停） |
| `--batch_size` | 32 | 批大小 |
| `--output` | model.tflite | 输出 TFLite 模型路径 |
| `--num_classes` | 5 | 预期类别数（仅校验用，实际由目录决定） |
| `--learning_rate` | 0.001 | 初始学习率 |
| `--report` | training_report.txt | 训练报告输出路径 |

训练完成后会生成：

- `model.tflite` — INT8 量化模型，可直接部署到 ESP32-S3
- `training_report.txt` — 包含准确率、损失曲线、混淆矩阵
- `best_model.keras` — 训练过程中验证精度最高的 Keras 模型

### 4. 转换为 C 头文件

```powershell
# Windows PowerShell
python .\tools\plant_doctor\tflite_to_header.py `
  --input model.tflite `
  --output ..\..\include\plant_disease_model.h `
  --symbol plant_disease_model_tflite
```

```bash
# Linux / macOS
python tools/plant_doctor/tflite_to_header.py \
  --input model.tflite \
  --output include/plant_disease_model.h \
  --symbol plant_disease_model_tflite
```

### 5. 更新固件

将新模型集成到固件需要以下步骤：

#### 5a. 替换模型头文件

`tflite_to_header.py` 会自动生成包含模型常量数组的 `.h` 文件。确保：

- 符号名为 `plant_disease_model_tflite`（与固件代码一致）
- `PLANT_MODEL_NUM_CLASSES` 宏与实际类别数匹配
- `PLANT_MODEL_INPUT_WIDTH` / `PLANT_MODEL_INPUT_HEIGHT` / `PLANT_MODEL_INPUT_CHANNELS` 保持为 96 / 96 / 3

#### 5b. 更新类别标签

编辑 `src/PlantDoctorModule.cpp`，更新以下静态数组以匹配新模型的类别顺序：

```cpp
// diseaseLabel() 中的英文标签
static const char* kLabels[] = {
    "Healthy",
    "Strawberry_Anthracnose",
    "Strawberry_Gray_Mold",
    "Strawberry_Leaf_Scorch",
    "Strawberry_Powdery_Mildew"
};

// diseaseLabelCn() 中的中文标签
static const char* kLabels[] = {
    "健康",
    "草莓炭疽病",
    "草莓灰霉病",
    "草莓叶焦病",
    "草莓白粉病"
};

// treatment() 中的防治建议
static const char* kSuggestions[] = {
    "叶片健康。保持通风和定期检查。",
    "清除病株残体，避免伤口感染。施用咪鲜胺等杀菌剂。",
    "降低湿度，增加通风。及时摘除病果，施用嘧霉胺等杀菌剂。",
    "移除感染叶片，降低叶面湿度。必要时施用杀菌剂。",
    "移除病叶，改善通风。喷施硫制剂或三唑类杀菌剂。"
};
```

#### 5c. 更新 TFLite Op Resolver

如果新模型引入了额外的算子，需在 `setupTflite()` 的 `MicroMutableOpResolver` 中注册。
当前架构（Conv2D + MaxPool2D + Dense + Softmax + Reshape）已包含在现有的 15 个算子中。

#### 5d. 编译并烧录

```bash
# 使用 PlatformIO
pio run -t upload
```

---

## 模型规格

| 项目 | 值 |
|------|-----|
| 输入尺寸 | 96 x 96 x 3 (RGB) |
| 输入数据类型 | int8，范围 [-1, 1] 对应量化值 |
| 输出类别数 | 5（默认），可自定义 |
| 量化方式 | INT8 全整数量化（输入输出均为 int8） |
| 推理内存 | 约 150 KB tensor arena |

### 模型架构

```
Input(96, 96, 3)
  -> Rescaling(1/127.5, -1)     # 归一化到 [-1, 1]
  -> Conv2D(16, 3x3, relu)      # 48x48x16
  -> MaxPool2D(2x2)             # 24x24x16
  -> Conv2D(32, 3x3, relu)      # 24x24x32
  -> MaxPool2D(2x2)             # 12x12x32
  -> Conv2D(64, 3x3, relu)      # 12x12x64
  -> MaxPool2D(2x2)             # 6x6x64
  -> Flatten                     # 2304
  -> Dense(64, relu)            # 64
  -> Dropout(0.3)
  -> Dense(num_classes, softmax) # num_classes
```

数据增强（仅训练阶段）：
- 随机水平 + 垂直翻转
- 随机旋转 ±15 度
- 随机亮度 ±20%
- 随机缩放 ±10%

### 类别标签映射

| 索引 | 英文名称 | 中文名称 | 防治要点 |
|------|----------|----------|----------|
| 0 | Healthy | 健康 | 保持通风和定期检查 |
| 1 | Strawberry_Anthracnose | 炭疽病 | 清除病残体，施用铜制剂 |
| 2 | Strawberry_Gray_Mold | 灰霉病 | 降低湿度，摘除病果，施用灰霉病药剂 |
| 3 | Strawberry_Leaf_Scorch | 叶焦病 | 清除病叶，减少叶面湿度，必要时施用杀菌剂 |
| 4 | Strawberry_Powdery_Mildew | 白粉病 | 改善通风，预防性施用硫制剂 |

---

## 常见问题

**Q: 训练精度不高怎么办？**

- 增加每类样本数量（建议 500+）
- 增加 `--epochs`（如 50 或 100）
- 尝试更小的学习率 `--learning_rate 0.0005`
- 检查数据质量，确保标注正确

**Q: 模型太大，ESP32-S3 放不下？**

- 当前模型量化后约 70-80 KB，远小于 150 KB tensor arena 限制
- 如果增加类别或通道后模型变大，可减少卷积滤波器数量

**Q: 如何添加新的病害类别？**

1. 在数据目录中新建子目录，放入对应图片
2. 重新运行训练脚本
3. 更新固件中的标签和建议数组
4. 更新 `PLANT_MODEL_NUM_CLASSES` 宏
