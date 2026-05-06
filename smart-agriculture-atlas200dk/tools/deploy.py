#!/usr/bin/env python3
"""
部署脚本 - 将项目上传到Atlas 200I DK A2开发板
使用paramiko SSH连接

用法:
  python3 deploy.py                          # 默认 192.168.137.100
  python3 deploy.py --ip 192.168.1.10        # 自定义IP
  python3 deploy.py --ip 192.168.1.10 --run  # 部署后自动启动
"""

import os
import sys
import argparse
import tarfile
import io
import time

try:
    import paramiko
except ImportError:
    print("请先安装paramiko: pip install paramiko")
    sys.exit(1)

REMOTE_DIR = "/home/HwHiAiUser/smart-agriculture-atlas200dk"
TAR_NAME = "smart-agriculture-atlas200dk.tar.gz"


def create_tarball(project_dir):
    """在内存中创建tar.gz"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for root, dirs, files in os.walk(project_dir):
            # 跳过__pycache__
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for f in files:
                if f.endswith('.pyc'):
                    continue
                filepath = os.path.join(root, f)
                arcname = os.path.relpath(filepath, os.path.dirname(project_dir))
                tar.add(filepath, arcname=arcname)
    buf.seek(0)
    return buf


def deploy(ip, user, password, project_dir, run_after=False):
    print(f"正在连接 Atlas 200I DK A2 ({ip})...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(ip, username=user, password=password, timeout=10,
                    allow_agent=False, look_for_keys=False)
    except Exception as e:
        print(f"连接失败: {e}")
        print(f"请检查:")
        print(f"  1. Atlas板已开机并通过USB/网线连接")
        print(f"  2. 默认IP 192.168.137.100 可达 (ping测试)")
        print(f"  3. 网络接口在同一网段 (sudo ip addr add 192.168.137.x/24 dev <interface>)")
        print(f"\n手动部署方法:")
        print(f"  scp -r {project_dir} {user}@{ip}:{REMOTE_DIR}")
        return False

    print("连接成功!")

    # 1. 创建远程目录
    print(f"创建远程目录 {REMOTE_DIR}...")
    ssh.exec_command(f"mkdir -p {REMOTE_DIR}")
    time.sleep(1)

    # 2. 上传项目
    print("上传项目文件...")
    sftp = ssh.open_sftp()

    def upload_dir(local_dir, remote_dir):
        for item in os.listdir(local_dir):
            local_path = os.path.join(local_dir, item)
            if item == '__pycache__' or item.endswith('.pyc'):
                continue
            remote_path = f"{remote_dir}/{item}"
            if os.path.isdir(local_path):
                try:
                    sftp.stat(remote_path)
                except FileNotFoundError:
                    sftp.mkdir(remote_path)
                upload_dir(local_path, remote_path)
            else:
                print(f"  {remote_path}")
                sftp.put(local_path, remote_path)

    upload_dir(project_dir, REMOTE_DIR)
    sftp.close()
    print("上传完成!")

    # 3. 安装依赖
    print("安装系统依赖...")
    cmds = [
        f"sudo apt update && sudo apt install -y python3-pip i2c-tools gpiod libgpiod2 python3-smbus2 python3-serial python3-numpy python3-opencv python3-flask",
        f"pip3 install --user pyserial smbus2 gpiod numpy opencv-python flask",
    ]
    for cmd in cmds:
        print(f"  执行: {cmd[:60]}...")
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            err = stderr.read().decode()[-200:]
            print(f"  警告 (exit={exit_code}): {err}")

    # 4. 创建数据目录
    print("创建数据目录...")
    ssh.exec_command(f"sudo mkdir -p /var/lib/agri-atlas /var/log/agri-atlas /opt/agri-atlas/models")
    ssh.exec_command(f"sudo chown -R {user}:{user} /var/lib/agri-atlas /var/log/agri-atlas /opt/agri-atlas")

    # 5. 启动 (可选)
    if run_after:
        print("启动智慧农业套件...")
        ssh.exec_command(f"cd {REMOTE_DIR} && nohup python3 main.py --profile 2 > /var/log/agri-atlas/main.log 2>&1 &")
        time.sleep(2)
        stdin, stdout, stderr = ssh.exec_command("ps aux | grep 'main.py' | grep -v grep")
        ps_output = stdout.read().decode()
        if 'main.py' in ps_output:
            print("启动成功!")
        else:
            print("启动可能失败, 请检查日志: /var/log/agri-atlas/main.log")

    print(f"\n部署完成!")
    print(f"  项目路径: {REMOTE_DIR}")
    print(f"  启动命令: cd {REMOTE_DIR} && python3 main.py --profile 2")
    print(f"  Web仪表盘: http://{ip}:8080")

    ssh.close()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='部署智慧农业套件到Atlas 200I DK A2')
    parser.add_argument('--ip', default='192.168.137.100', help='Atlas板IP地址')
    parser.add_argument('--user', default='HwHiAiUser', help='SSH用户名')
    parser.add_argument('--password', default='Mind@123', help='SSH密码')
    parser.add_argument('--run', action='store_true', help='部署后自动启动')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deploy(args.ip, args.user, args.password, project_dir, run_after=args.run)
