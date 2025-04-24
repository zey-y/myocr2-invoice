#!/bin/bash

# 定义标志文件路径
FLAG_FILE="/usr/src/app/.installed"
# 如果标志文件不存在，则安装依赖
if [ ! -f "$FLAG_FILE" ]; then
  echo "Installing Python dependencies..."
#  pip install --upgrade pip
  pip install -r requirements.txt -i $1 --no-cache-dir --progress-bar off
  touch "$FLAG_FILE"  # 创建标志文件
else
  echo "Dependencies already installed. Skipping pip install."
fi
# 启动应用
exec gunicorn -w 4 -b 0.0.0.0:5000 main:app
