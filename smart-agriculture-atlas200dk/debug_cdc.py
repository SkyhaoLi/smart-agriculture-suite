#!/usr/bin/env python3
"""调试 USB CDC - 查看原始响应"""
import os, time, select

PORT = '/dev/ttyACM0'
os.system(f'stty -F {PORT} 115200 cs8 -cstopb -parenb -icanon min 0 time 0 -echo -echoe -echok -echoctl -echoke -onlcr')
os.system(f'chmod 666 {PORT}')
fd = os.open(PORT, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)

# 清空
time.sleep(2)
try:
    while os.read(fd, 4096):
        pass
except BlockingIOError:
    pass

def raw_send(cmd, timeout=3.0):
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
    return buf

print("=== USB CDC Raw Debug ===\n")

# 1. 发 PING 看原始字节
resp = raw_send('PING')
print(f"PING raw bytes: {resp!r}")
print(f"PING hex:       {resp.hex()}")
print(f"PING decoded:   {resp.decode('utf-8', errors='replace').strip()}")
time.sleep(0.5)

# 2. 发 HELLO 看响应
resp = raw_send('HELLO')
print(f"HELLO raw bytes: {resp!r}")
print(f"HELLO decoded:   {resp.decode('utf-8', errors='replace').strip()}")
time.sleep(0.5)

# 3. 发 GET_DATA
resp = raw_send('GET_DATA')
print(f"GET_DATA raw bytes: {resp!r}")
print(f"GET_DATA decoded:   {resp.decode('utf-8', errors='replace').strip()}")

os.close(fd)
