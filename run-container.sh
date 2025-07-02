#!/bin/bash

# GPU Docker 开发环境启动脚本
# 用法: ./run-container.sh [用户名] [密码] [工作目录] [端口前缀] [pip镜像源URL] [pip可信主机] [apt镜像源URL] [npm注册表URL] [npm可信主机]

set -e

# 默认参数
DEFAULT_USER="devuser"
DEFAULT_PASSWORD="changeme"
DEFAULT_WORKSPACE_DIR="./workspace"
DEFAULT_PORT_PREFIX="80"

# 解析参数
DEV_USER=${1:-$DEFAULT_USER}
DEV_PASSWORD=${2:-$DEFAULT_PASSWORD}
WORKSPACE_DIR=${3:-$DEFAULT_WORKSPACE_DIR}
PORT_PREFIX=${4:-$DEFAULT_PORT_PREFIX}
PIP_INDEX_URL=${5:-""}
PIP_TRUSTED_HOST=${6:-""}
APT_MIRROR_URL=${7:-""}
NPM_REGISTRY_URL=${8:-""}
NPM_TRUSTED_HOST=${9:-""}

# 构造端口号
CODE_SERVER_PORT="${PORT_PREFIX}80"
JUPYTER_PORT="${PORT_PREFIX}88"
TENSORBOARD_PORT="${PORT_PREFIX}06"
SSH_PORT="${PORT_PREFIX}22"

# 获取当前用户的UID和GID
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

# 如果是root用户，使用安全的默认值避免冲突
if [ "$CURRENT_UID" -eq 0 ]; then
    echo "检测到root用户，使用安全的默认UID/GID..."
    # 查找可用的UID，从1000开始（避免系统保留ID）
    SAFE_UID=1000
    while id -u $SAFE_UID &>/dev/null; do
        SAFE_UID=$((SAFE_UID+1))
    done
    
    # 查找可用的GID，从1000开始
    SAFE_GID=1000
    while getent group $SAFE_GID &>/dev/null; do
        SAFE_GID=$((SAFE_GID+1))
    done
    
    CURRENT_UID=$SAFE_UID
    CURRENT_GID=$SAFE_GID
    echo "使用安全的UID: $CURRENT_UID, GID: $CURRENT_GID"
fi

# 获取宿主机IP地址
# 优先使用hostname -I获取第一个非回环IP地址
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}' 2>/dev/null)

# 如果hostname -I失败，尝试从网络接口获取IP
if [ -z "$HOST_IP" ]; then
    # 获取默认路由的网络接口，然后获取该接口的IP
    DEFAULT_INTERFACE=$(ip route | grep '^default' | awk '{print $5}' | head -n1 2>/dev/null)
    if [ -n "$DEFAULT_INTERFACE" ]; then
        HOST_IP=$(ip addr show "$DEFAULT_INTERFACE" 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1 | head -n1)
    fi
fi

# 如果还是获取不到，尝试获取第一个非回环的网络接口IP
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(ip addr show 2>/dev/null | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | cut -d'/' -f1 | head -n1)
fi

# 最后的备选方案使用localhost
if [ -z "$HOST_IP" ]; then
    HOST_IP="localhost"
    echo "警告: 无法获取宿主机IP地址，使用localhost"
fi

# 创建工作目录
mkdir -p "$WORKSPACE_DIR"/{projects,data,models,notebooks,tensorboard_logs}
mkdir -p ./shared
mkdir -p ./tmp

# 确保共享目录权限正确（避免容器内权限冲突）
if [ -d "./shared" ]; then
    echo "设置共享目录权限..."
    chmod 777 ./shared 2>/dev/null || echo "警告: 无法设置共享目录权限，继续执行..."
    # 如果不是root用户，尝试设置所有权
    if [ "$CURRENT_UID" -ne 0 ]; then
        chown "$CURRENT_UID:$CURRENT_GID" ./shared 2>/dev/null || echo "注意: 无法设置共享目录所有权，继续执行..."
    fi
fi

echo "==============================================="
echo "启动 GPU Docker 开发环境"
echo "==============================================="
echo "用户名: $DEV_USER"
echo "密码: $DEV_PASSWORD"
echo "工作目录: $WORKSPACE_DIR"
echo "宿主机IP: $HOST_IP"
echo ""
echo "共享目录映射:"
echo "  $WORKSPACE_DIR -> /home/$DEV_USER/workspace (读写)"
echo "  ./shared -> /home/$DEV_USER/shared-ro (只读)"
echo "  ./tmp -> /home/$DEV_USER/shared-rw (读写)"
echo "端口配置:"
echo "  VSCode:      http://$HOST_IP:$CODE_SERVER_PORT"
echo "  Jupyter:     http://$HOST_IP:$JUPYTER_PORT"
echo "  TensorBoard: http://$HOST_IP:$TENSORBOARD_PORT"
echo "  SSH:         ssh -p $SSH_PORT $DEV_USER@$HOST_IP"
if [ -n "$APT_MIRROR_URL" ] || [ -n "$NPM_REGISTRY_URL" ] || [ -n "$PIP_INDEX_URL" ]; then
echo "内网镜像源配置:"
[ -n "$APT_MIRROR_URL" ] && echo "  APT Mirror:   $APT_MIRROR_URL"
[ -n "$NPM_REGISTRY_URL" ] && echo "  NPM Registry: $NPM_REGISTRY_URL" && [ -n "$NPM_TRUSTED_HOST" ] && echo "  NPM Host:     $NPM_TRUSTED_HOST"
[ -n "$PIP_INDEX_URL" ] && echo "  PIP Index:    $PIP_INDEX_URL" && [ -n "$PIP_TRUSTED_HOST" ] && echo "  PIP Host:     $PIP_TRUSTED_HOST"
fi
echo "==============================================="

# 检查Docker和nvidia-docker
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装或未启动"
    exit 1
fi

# 构建镜像
# echo "构建 Docker 镜像..."
# docker build -t gpu-dev-env:latest .

# 运行容器
echo "启动容器..."
docker run -d \
    --name "gpu-dev-$DEV_USER" \
    --hostname "gpu-dev-$DEV_USER" \
    --gpus all \
    -p "$CODE_SERVER_PORT:8080" \
    -p "$JUPYTER_PORT:8888" \
    -p "$TENSORBOARD_PORT:6006" \
    -p "$SSH_PORT:22" \
    -e "DEV_USER=$DEV_USER" \
    -e "DEV_PASSWORD=$DEV_PASSWORD" \
    -e "PASSWORD=$DEV_PASSWORD" \
    -e "DEV_UID=$CURRENT_UID" \
    -e "DEV_GID=$CURRENT_GID" \
    -e "ENABLE_JUPYTER=true" \
    -e "ENABLE_TENSORBOARD=true" \
    -e "WORKSPACE_DIR=/home/$DEV_USER/workspace" \
    $([ -n "$APT_MIRROR_URL" ] && echo "-e APT_MIRROR_URL=$APT_MIRROR_URL") \
    $([ -n "$NPM_REGISTRY_URL" ] && echo "-e NPM_REGISTRY_URL=$NPM_REGISTRY_URL") \
    $([ -n "$NPM_TRUSTED_HOST" ] && echo "-e NPM_TRUSTED_HOST=$NPM_TRUSTED_HOST") \
    $([ -n "$PIP_INDEX_URL" ] && echo "-e PIP_INDEX_URL=$PIP_INDEX_URL") \
    $([ -n "$PIP_TRUSTED_HOST" ] && echo "-e PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST") \
    -v "$(realpath $WORKSPACE_DIR):/home/$DEV_USER/workspace:rw" \
    -v "$(realpath ./shared):/home/$DEV_USER/shared-ro:rw" \
    -v "$(realpath ./tmp):/home/$DEV_USER/shared-rw:rw" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --shm-size=32g \
    --restart unless-stopped \
    connermo/ai4s-gpu-dev:latest

echo ""
echo "容器启动完成！"
echo "请等待几秒钟让服务完全启动..."
sleep 5

echo ""
echo "检查容器状态:"
docker ps | grep "gpu-dev-$DEV_USER" || echo "容器可能未正常启动，请检查日志: docker logs gpu-dev-$DEV_USER"

echo ""
echo "查看日志:"
echo "  docker logs -f gpu-dev-$DEV_USER"
echo ""
echo "停止容器:"
echo "  docker stop gpu-dev-$DEV_USER"
echo ""
echo "删除容器:"
echo "  docker rm gpu-dev-$DEV_USER" 