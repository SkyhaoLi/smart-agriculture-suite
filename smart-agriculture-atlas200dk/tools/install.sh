#!/bin/bash
# 智润智慧农业套件 - Atlas 200I DK A2 一键安装脚本
# 在Atlas 200I DK A2开发板上运行 (Ubuntu 22.04)

set -e

echo "========================================="
echo " 智润智慧农业套件 - Atlas 200I DK A2"
echo " 依赖安装脚本"
echo "========================================="

# 系统依赖
echo "[1/4] 安装系统依赖..."
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-venv \
    i2c-tools \
    gpiod \
    libgpiod2 \
    python3-libgpiod \
    python3-smbus2 \
    python3-serial \
    python3-numpy \
    python3-opencv \
    python3-flask

# Atlas 200I DK A2 ACL SDK (如果已安装CANN)
if [ -d /usr/local/Ascend ]; then
    echo "[2/4] 检测到Ascend CANN, 配置ACL环境..."
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    echo "ACL环境已加载"
else
    echo "[2/4] 未检测到Ascend CANN, 病害检测将使用CPU推理"
fi

# ONNX Runtime (可选, 用于CPU推理后备)
echo "[3/4] 安装Python依赖..."
pip3 install --user -r requirements.txt

# 创建数据目录
echo "[4/4] 创建数据目录..."
sudo mkdir -p /etc/agri-atlas
sudo mkdir -p /var/lib/agri-atlas
sudo mkdir -p /var/log/agri-atlas
sudo mkdir -p /opt/agri-atlas/models
sudo chown -R $USER:$USER /var/lib/agri-atlas /var/log/agri-atlas /opt/agri-atlas

# 模型文件 (如需NPU推理, 需要将ONNX模型转为OM格式)
if [ -d /usr/local/Ascend ]; then
    echo ""
    echo "提示: 要使用NPU推理, 需将ONNX模型转换为OM格式:"
    echo "  atc --model=plant_disease_model.onnx --output=plant_disease_model.om --framework=5 --soc=Ascend310B1"
    echo "  将OM文件放入 /opt/agri-atlas/models/"
fi

echo ""
echo "安装完成!"
echo "启动: python3 main.py"
echo "Web仪表盘: http://<Atlas IP>:8080"
