#!/usr/bin/env python3
"""
ESP32 实时数据仪表盘服务器

ESP32 通过 USB CDC 每 2 秒推送一次传感器数据 (PUSH:{json})
本脚本后台读取数据，HTTP 服务提供实时仪表盘页面。

用法:
    python3 dashboard_server.py

浏览器访问:
    http://<atlas_ip>:8088
"""

import json
import os
import sys
import threading
import time
import select
from http.server import HTTPServer, BaseHTTPRequestHandler

# ============================================================================
# 配置
# ============================================================================
CDC_PORT = '/dev/ttyACM0'
HTTP_PORT = 8088

# ============================================================================
# 全局数据 (线程安全由 GIL 保证)
# ============================================================================
latest_data = {
    'temp': 0.0, 'humi': 0.0, 'soil': 0.0, 'light': 0.0,
    'liquid': 0.0, 'valve': 0, 'pump': 0,
    'connected': False, 'updated_at': 0
}
cdc_lock = threading.Lock()

# ============================================================================
# USB CDC 读取线程
# ============================================================================
def cdc_reader():
    """后台线程: 持续读取 ESP32 USB CDC 数据"""
    global latest_data

    while True:
        try:
            # 配置串口
            os.system(f'stty -F {CDC_PORT} 115200 cs8 -cstopb -parenb -icanon min 0 time 0 -echo -echoe -echok -echoctl -echoke -onlcr')
            os.system(f'chmod 666 {CDC_PORT}')
            fd = os.open(CDC_PORT, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)

            # 清空缓冲区
            time.sleep(1)
            try:
                while os.read(fd, 4096):
                    pass
            except BlockingIOError:
                pass

            print(f'[CDC] 已连接 {CDC_PORT}')
            with cdc_lock:
                latest_data['connected'] = True

            buf = b''
            while True:
                r, _, _ = select.select([fd], [], [], 0.1)
                if r:
                    try:
                        chunk = os.read(fd, 4096)
                        if chunk:
                            buf += chunk
                            # 逐行处理
                            while b'\n' in buf:
                                line, buf = buf.split(b'\n', 1)
                                line = line.decode('utf-8', errors='replace').strip()
                                if line.startswith('PUSH:'):
                                    _parse_push(line[5:])
                        else:
                            # 设备断开
                            break
                    except BlockingIOError:
                        pass

        except Exception as e:
            print(f'[CDC] 连接失败: {e}')
        finally:
            with cdc_lock:
                latest_data['connected'] = False
            try:
                os.close(fd)
            except:
                pass

        print('[CDC] 5 秒后重连...')
        time.sleep(5)


def _parse_push(json_str):
    """解析 PUSH: 后面的 JSON 数据"""
    global latest_data
    try:
        data = json.loads(json_str)
        with cdc_lock:
            latest_data.update(data)
            latest_data['updated_at'] = time.time()
    except json.JSONDecodeError:
        pass

# ============================================================================
# HTML 仪表盘
# ============================================================================
DASHBOARD_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>智慧农业实时监控</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
.header { background: linear-gradient(135deg, #1e3a5f, #0f172a); padding: 20px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; }
.header h1 { font-size: 20px; font-weight: 600; }
.status { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.dot { width: 10px; height: 10px; border-radius: 50%; }
.dot.on { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
.dot.off { background: #ef4444; box-shadow: 0 0 8px #ef4444; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; padding: 24px; max-width: 900px; margin: 0 auto; }
.card { background: #1e293b; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #334155; transition: border-color 0.3s; }
.card:hover { border-color: #3b82f6; }
.card .icon { font-size: 28px; margin-bottom: 8px; }
.card .label { font-size: 12px; color: #94a3b8; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }
.card .value { font-size: 32px; font-weight: 700; color: #f1f5f9; }
.card .unit { font-size: 14px; color: #64748b; margin-left: 4px; }
.card.warn .value { color: #f59e0b; }
.card.danger .value { color: #ef4444; }
.card.good .value { color: #22c55e; }
.actuator-row { display: flex; gap: 16px; padding: 0 24px 24px; max-width: 900px; margin: 0 auto; }
.actuator-card { flex: 1; background: #1e293b; border-radius: 12px; padding: 16px 20px; display: flex; align-items: center; gap: 16px; border: 1px solid #334155; }
.actuator-card .icon { font-size: 32px; }
.actuator-card .info { flex: 1; }
.actuator-card .info .name { font-size: 13px; color: #94a3b8; }
.actuator-card .info .state { font-size: 20px; font-weight: 700; }
.actuator-card .state.on { color: #22c55e; }
.actuator-card .state.off { color: #64748b; }
.footer { text-align: center; padding: 16px; color: #475569; font-size: 12px; }
</style>
</head>
<body>
<div class="header">
    <h1>&#127793; 智慧农业实时监控</h1>
    <div class="status">
        <div class="dot" id="connDot"></div>
        <span id="connText">连接中...</span>
    </div>
</div>
<div class="grid">
    <div class="card" id="cardTemp">
        <div class="icon">&#127777;&#65039;</div>
        <div class="label">温度</div>
        <div><span class="value" id="vTemp">--</span><span class="unit">&#176;C</span></div>
    </div>
    <div class="card" id="cardHumi">
        <div class="icon">&#128167;</div>
        <div class="label">空气湿度</div>
        <div><span class="value" id="vHumi">--</span><span class="unit">%</span></div>
    </div>
    <div class="card" id="cardSoil">
        <div class="icon">&#127793;</div>
        <div class="label">土壤湿度</div>
        <div><span class="value" id="vSoil">--</span><span class="unit">%</span></div>
    </div>
    <div class="card" id="cardLight">
        <div class="icon">&#9728;&#65039;</div>
        <div class="label">光照</div>
        <div><span class="value" id="vLight">--</span><span class="unit">lux</span></div>
    </div>
    <div class="card" id="cardLiquid">
        <div class="icon">&#128164;</div>
        <div class="label">液位</div>
        <div><span class="value" id="vLiquid">--</span><span class="unit">%</span></div>
    </div>
</div>
<div class="actuator-row">
    <div class="actuator-card">
        <div class="icon">&#128167;</div>
        <div class="info">
            <div class="name">电磁阀</div>
            <div class="state" id="vValve">--</div>
        </div>
    </div>
    <div class="actuator-card">
        <div class="icon">&#9881;&#65039;</div>
        <div class="info">
            <div class="name">水泵</div>
            <div class="state" id="vPump">--</div>
        </div>
    </div>
</div>
<div class="footer" id="footer">等待数据...</div>
<script>
function refresh(){
    fetch('/api/data').then(r=>r.json()).then(d=>{
        document.getElementById('connDot').className='dot '+(d.connected?'on':'off');
        document.getElementById('connText').textContent=d.connected?'ESP32 已连接':'ESP32 离线';
        setVal('vTemp',d.temp,'cardTemp',d.temp>35?'danger':d.temp<10?'warn':'');
        setVal('vHumi',d.humi,'cardHumi',d.humi>80?'warn':'');
        setVal('vSoil',d.soil,'cardSoil',d.soil<30?'danger':d.soil<50?'warn':'good');
        setVal('vLight',Math.round(d.light),'cardLight','');
        setVal('vLiquid',d.liquid,'cardLiquid',d.liquid<20?'danger':d.liquid<40?'warn':'good');
        setAct('vValve',d.valve);
        setAct('vPump',d.pump);
        if(d.updated_at>0){
            let ago=Math.round(Date.now()/1000-d.updated_at);
            document.getElementById('footer').textContent='最后更新: '+ago+' 秒前';
        }
    }).catch(()=>{
        document.getElementById('connDot').className='dot off';
        document.getElementById('connText').textContent='请求失败';
    });
}
function setVal(id,val,cardId,cls){
    document.getElementById(id).textContent=typeof val==='number'?val.toFixed(1):val;
    let c=document.getElementById(cardId);
    c.className='card '+(cls||'');
}
function setAct(id,on){
    let el=document.getElementById(id);
    el.textContent=on?'开启':'关闭';
    el.className='state '+(on?'on':'off');
}
refresh();
setInterval(refresh,2000);
</script>
</body>
</html>'''

# ============================================================================
# HTTP 服务器
# ============================================================================
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/data':
            with cdc_lock:
                data = dict(latest_data)
            self._send_json(200, data)
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode('utf-8'))
        else:
            self._send_json(404, {'error': 'not found'})

    def _send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # 静默常规日志, 只打印错误
        if args and '200' not in str(args[0]):
            super().log_message(format, *args)

# ============================================================================
# 启动
# ============================================================================
def main():
    print(f'=== ESP32 实时数据仪表盘 ===')
    print(f'USB CDC: {CDC_PORT}')
    print(f'HTTP 端口: {HTTP_PORT}')
    print()

    # 启动 CDC 读取线程
    t = threading.Thread(target=cdc_reader, daemon=True)
    t.start()

    # 启动 HTTP 服务
    server = HTTPServer(('0.0.0.0', HTTP_PORT), DashboardHandler)
    print(f'[HTTP] 仪表盘地址: http://0.0.0.0:{HTTP_PORT}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n已停止')
        server.server_close()


if __name__ == '__main__':
    main()
