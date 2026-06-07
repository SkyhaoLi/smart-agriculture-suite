"""
ESP32-S3 USB CDC 通信模块 (Atlas 端)

Atlas 200 DK 通过 /dev/ttyACM0 与 ESP32-S3 双向通信。
使用 Python 内置 os/select 模块，无需安装 pyserial。

用法:
    from esp32_link import ESP32Link

    esp = ESP32Link('/dev/ttyACM0')
    print(esp.send('PING'))        # -> PONG
    print(esp.send('GET_DATA'))    # -> DATA:temp=26.1,humi=34.9,soil=45.2,light=320
    esp.close()
"""

import os
import time
import select


class ESP32Link:
    """Atlas <-> ESP32-S3 USB CDC 通信"""

    def __init__(self, port='/dev/ttyACM0', baud=115200):
        """打开 USB CDC 设备文件

        Args:
            port: 设备路径, 默认 /dev/ttyACM0
            baud: 波特率 (USB CDC 实际由硬件决定, 此参数仅为 stty 兼容)
        """
        os.system(f'stty -F {port} {baud} cs8 -cstopb -parenb -icanon min 0 time 0 -echo -echoe -echok -echoctl -echoke -onlcr')
        os.system(f'chmod 666 {port}')
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        self.port = port

    def drain(self):
        """清空接收缓冲区中的残留数据"""
        try:
            while True:
                d = os.read(self.fd, 4096)
                if not d:
                    break
        except BlockingIOError:
            pass

    def read_line(self, timeout=3.0):
        """读取一行回复 (直到 \\n 或超时)

        Args:
            timeout: 超时秒数, 默认 3.0

        Returns:
            解码后的字符串 (已 strip)
        """
        deadline = time.time() + timeout
        buf = b''
        while time.time() < deadline:
            r, _, _ = select.select([self.fd], [], [],
                                    min(deadline - time.time(), 0.05))
            if r:
                try:
                    c = os.read(self.fd, 4096)
                    if c:
                        buf += c
                        if b'\n' in buf:
                            break
                except BlockingIOError:
                    pass
        return buf.decode('utf-8', errors='replace').strip()

    def send(self, cmd, timeout=3.0):
        """发送指令并读取回复

        Args:
            cmd: 指令字符串 (不需要手动加 \\n)
            timeout: 回复超时秒数

        Returns:
            ESP32 的回复字符串
        """
        self.drain()
        os.write(self.fd, (cmd + '\n').encode('utf-8'))
        return self.read_line(timeout=timeout)

    def get_sensor_data(self):
        """获取传感器数据, 返回解析后的字典

        Returns:
            dict: {'temp': float, 'humi': float, 'soil': float, 'light': float}
                  如果传感器故障返回 None
        """
        resp = self.send('GET_DATA')
        if not resp.startswith('DATA:'):
            return None

        data = {}
        for pair in resp[5:].split(','):
            if '=' in pair:
                k, v = pair.split('=', 1)
                try:
                    data[k] = float(v)
                except ValueError:
                    data[k] = v
        return data

    def ping(self):
        """心跳检测, 返回 True 表示 ESP32 在线"""
        return self.send('PING') == 'PONG'

    def close(self):
        """关闭设备文件"""
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f'ESP32Link({self.port!r})'
