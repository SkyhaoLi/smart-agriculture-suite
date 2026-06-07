#!/usr/bin/env python3
"""快速 USB CDC 测试 - 简化版"""
import os, time, select

PORT = '/dev/ttyACM0'
os.system(f'stty -F {PORT} 115200 cs8 -cstopb -parenb -icanon min 0 time 0 -echo -echoe -echok -echoctl -echoke -onlcr')
os.system(f'chmod 666 {PORT}')
fd = os.open(PORT, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)

# 彻底清空缓冲区 (ESP32 刚重启会发启动消息)
for _ in range(5):
    time.sleep(0.5)
    try:
        while os.read(fd, 4096):
            pass
    except BlockingIOError:
        pass

def send_cmd(cmd, timeout=3.0):
    data = (cmd + '\n').encode('utf-8')
    try:
        os.write(fd, data)
    except BlockingIOError:
        select.select([], [fd], [], 1.0)
        os.write(fd, data)
    deadline = time.time() + timeout
    buf = b''
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], min(deadline - time.time(), 0.05))
        if r:
            try:
                c = os.read(fd, 4096)
                if c:
                    buf += c
                    if b'\n' in buf:
                        break
            except BlockingIOError:
                pass
    return buf.decode('utf-8', errors='replace').strip()

print("=== ESP32 USB CDC Quick Test ===\n")
for cmd in ["PING", "START_TASK", "GET_DATA", "UNKNOWN"]:
    resp = send_cmd(cmd)
    print(f"  {cmd:15s} -> [{resp}]")
    time.sleep(0.3)

os.close(fd)
print("\nDone.")
