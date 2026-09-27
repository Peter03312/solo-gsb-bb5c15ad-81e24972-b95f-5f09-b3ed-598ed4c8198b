#!/bin/sh
# 容器启动时安装依赖（满足“容器启动时安装依赖”的要求），随后启动服务。
set -e

echo "[entrypoint] 正在安装依赖（requirements.txt）..."
python -m pip install -r /app/requirements.txt

echo "[entrypoint] 启动 API 服务..."
exec "$@"
