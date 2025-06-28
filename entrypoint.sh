#!/bin/bash
set -e

# 设置默认环境变量
export DEV_USER=${DEV_USER:-devuser}
export DEV_PASSWORD=${DEV_PASSWORD:-changeme}
export DEV_UID=${DEV_UID:-1000}
export DEV_GID=${DEV_GID:-1000}
export ENABLE_JUPYTER=${ENABLE_JUPYTER:-true}
export ENABLE_TENSORBOARD=${ENABLE_TENSORBOARD:-true}
export WORKSPACE_DIR=${WORKSPACE_DIR:-/home/$DEV_USER/workspace}

echo "Starting GPU Docker Development Environment..."
echo "User: $DEV_USER (UID: $DEV_UID, GID: $DEV_GID)"
echo "Workspace: $WORKSPACE_DIR"
echo "Services: code-server, ssh$([ "$ENABLE_JUPYTER" = "true" ] && echo ", jupyter")$([ "$ENABLE_TENSORBOARD" = "true" ] && echo ", tensorboard")"

# 创建或修改用户
if id "$DEV_USER" &>/dev/null; then
    echo "User $DEV_USER already exists, updating UID/GID..."
    usermod -u $DEV_UID $DEV_USER
    groupmod -g $DEV_GID $DEV_USER
else
    echo "Creating user $DEV_USER..."
    getent group $DEV_USER >/dev/null || groupadd -g $DEV_GID $DEV_USER
    useradd -m -s /bin/bash -u $DEV_UID -g $DEV_GID $DEV_USER
    echo "$DEV_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

# 设置用户密码
echo "$DEV_USER:$DEV_PASSWORD" | chpasswd

# 确保工作目录存在并设置权限
mkdir -p $WORKSPACE_DIR/{projects,data,models,notebooks,tensorboard_logs}
chown -R $DEV_UID:$DEV_GID $WORKSPACE_DIR

# 确保用户家目录权限正确
chown -R $DEV_UID:$DEV_GID /home/$DEV_USER

# 创建日志目录
mkdir -p /var/log/supervisor
chmod 755 /var/log/supervisor

# 检查GPU是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "Warning: nvidia-smi not found. GPU might not be available."
fi

# 显示访问信息
echo ""
echo "==============================================="
echo "GPU Docker Development Environment Started!"
echo "==============================================="
echo "Access URLs:"
echo "  VSCode:      http://localhost:8080"
if [ "$ENABLE_JUPYTER" = "true" ]; then
echo "  Jupyter Lab: http://localhost:8888"
fi
if [ "$ENABLE_TENSORBOARD" = "true" ]; then
echo "  TensorBoard: http://localhost:6006"
fi
echo "  SSH:         ssh -p 22 $DEV_USER@localhost"
echo ""
echo "Login credentials:"
echo "  Username: $DEV_USER"
echo "  Password: $DEV_PASSWORD"
echo "==============================================="

# 执行传入的命令
exec "$@" 