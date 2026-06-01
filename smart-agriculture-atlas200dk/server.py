#!/usr/bin/env python3
"""
智润智慧农业 - 世界模型推理服务器
Atlas 200I DK A2 版

HTTP服务器: 接收ESP32传感器数据, 运行世界模型推理, 返回决策
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_model import WorldModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("agri_server")

# 全局世界模型实例
world_model: WorldModel = None


class AgriHandler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""

    def log_message(self, format, *args):
        logger.debug(f"{self.address_string()} {format % args}")

    def _send_json(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/health':
            self._send_json(200, {
                "status": "ok",
                "model_loaded": world_model.is_loaded if world_model else False,
                "uptime": time.time() - start_time,
            })
        elif self.path == '/api/model/info':
            info = world_model.get_info() if world_model else {}
            self._send_json(200, info)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == '/api/predict':
            self._handle_predict()
        elif self.path == '/api/train/step':
            self._handle_train_step()
        elif self.path == '/api/model/save':
            self._handle_model_save()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_predict(self):
        """处理预测请求"""
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)
        except Exception as e:
            self._send_json(400, {"error": f"invalid json: {e}"})
            return

        try:
            result = world_model.predict(data)
            self._send_json(200, result)
        except Exception as e:
            logger.error(f"预测错误: {e}", exc_info=True)
            self._send_json(500, {"error": str(e)})

    def _handle_train_step(self):
        """处理训练步请求 (批量数据)"""
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)
        except Exception as e:
            self._send_json(400, {"error": f"invalid json: {e}"})
            return

        try:
            result = world_model.train_step(data)
            self._send_json(200, result)
        except Exception as e:
            logger.error(f"训练错误: {e}", exc_info=True)
            self._send_json(500, {"error": str(e)})

    def _handle_model_save(self):
        """保存模型"""
        try:
            world_model.save()
            self._send_json(200, {"ok": True})
        except Exception as e:
            self._send_json(500, {"error": str(e)})


start_time = time.time()


def main():
    global world_model

    parser = argparse.ArgumentParser(description="智润世界模型推理服务器")
    parser.add_argument('--host', default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=8080, help='监听端口')
    parser.add_argument('--model-dir', default='./models', help='模型目录')
    parser.add_argument('--device', default='cpu', help='推理设备 (cpu/npu)')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 初始化世界模型
    logger.info("正在加载世界模型...")
    world_model = WorldModel(
        model_dir=args.model_dir,
        device=args.device,
    )
    world_model.load()
    logger.info(f"世界模型加载完成: {world_model.get_info()}")

    # 启动HTTP服务器
    server = HTTPServer((args.host, args.port), AgriHandler)
    logger.info(f"推理服务器启动: http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("服务器停止")
        world_model.save()
        server.server_close()


if __name__ == '__main__':
    main()
