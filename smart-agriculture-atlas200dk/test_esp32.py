#!/usr/bin/env python3
"""
ESP32-S3 USB CDC 通信测试脚本

在 Atlas 200 DK 上运行:
    python3 test_esp32.py

在电脑上通过 SSH 运行:
    python3 test_esp32.py /dev/ttyACM0
"""

import sys
import os

# 直接使用 os 模块, 不依赖 pyserial
def test_esp32(port='/dev/ttyACM0'):
    """测试 ESP32 USB CDC 通信"""

    print(f"=== ESP32 USB CDC 测试 ===")
    print(f"设备: {port}")
    print()

    # 检查设备是否存在
    if not os.path.exists(port):
        print(f"❌ 设备 {port} 不存在!")
        print("   可能原因:")
        print("   1. USB 线是充电线不是数据线")
        print("   2. ESP32 固件未重新烧录")
        print("   3. ESP32 没有重启")
        print("   4. 插错口了 (应插 ESP32 的 USB 口)")
        return False

    # 设置串口参数
    os.system(f'stty -F {port} 115200 cs8 -cstopb -parenb -icanon min 0 time 0 -echo -echoe -echok -echoctl -echoke -onlcr')
    os.system(f'chmod 666 {port}')

    import time
    import select

    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)

    # 等待启动消息
    print("等待 ESP32 启动...")
    time.sleep(1)

    # 清空缓冲区
    try:
        while os.read(fd, 4096):
            pass
    except BlockingIOError:
        pass

    def send_cmd(cmd, timeout=3.0):
        """发送指令并读取回复"""
        # 清空残留
        try:
            while os.read(fd, 4096):
                pass
        except BlockingIOError:
            pass

        # 发送 (非阻塞写可能抛 BlockingIOError)
        data = (cmd + '\n').encode('utf-8')
        try:
            os.write(fd, data)
        except BlockingIOError:
            select.select([], [fd], [], 1.0)
            os.write(fd, data)

        # 读取回复
        deadline = time.time() + timeout
        buf = b''
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [],
                                    min(deadline - time.time(), 0.05))
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

    # 测试指令
    tests = [
        ("PING",       "PONG"),
        ("START_TASK", "TASK_COMPLETED_SUCCESSFULLY"),
        ("GET_DATA",   None),  # 动态数据, 不做精确匹配
        ("UNKNOWN",    "ERROR:unknown_command"),
    ]

    results = []
    for cmd, expected in tests:
        resp = send_cmd(cmd)
        if expected is None:
            ok = resp.startswith("DATA:")
            status = "✅" if ok else "❌"
            print(f"  {status} {cmd}: [{resp}]")
        else:
            ok = (resp == expected)
            status = "✅" if ok else "❌"
            print(f"  {status} {cmd}: [{resp}]" +
                  ("" if ok else f" (期望: {expected})"))
        results.append(ok)

    os.close(fd)

    print()
    passed = sum(results)
    total = len(results)
    print(f"结果: {passed}/{total} 通过")
    return all(results)


if __name__ == '__main__':
    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
    ok = test_esp32(port)
    sys.exit(0 if ok else 1)
