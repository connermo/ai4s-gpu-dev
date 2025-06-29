#!/bin/bash
set -e

# 设置默认环境变量
export DEV_USER=${DEV_USER:-devuser}
export DEV_PASSWORD=${DEV_PASSWORD:-changeme}
export PASSWORD=${PASSWORD:-$DEV_PASSWORD}
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
    
    # 检查并分配可用的 UID
    if id -u $DEV_UID &>/dev/null; then
        echo "UID $DEV_UID is already in use, finding next available UID..."
        ORIGINAL_UID=$DEV_UID
        while id -u $DEV_UID &>/dev/null; do
            DEV_UID=$((DEV_UID+1))
        done
        echo "Using UID $DEV_UID instead of $ORIGINAL_UID"
    fi
    
    # 检查并分配可用的 GID
    if getent group $DEV_GID &>/dev/null; then
        echo "GID $DEV_GID is already in use, finding next available GID..."
        ORIGINAL_GID=$DEV_GID
        while getent group $DEV_GID &>/dev/null; do
            DEV_GID=$((DEV_GID+1))
        done
        echo "Using GID $DEV_GID instead of $ORIGINAL_GID"
    fi
    
    getent group $DEV_USER >/dev/null || groupadd -g $DEV_GID $DEV_USER
    useradd -m -s /bin/bash -u $DEV_UID -g $DEV_GID $DEV_USER
    echo "$DEV_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

# 设置用户密码
echo "$DEV_USER:$DEV_PASSWORD" | chpasswd

# 如果不是devuser，复制配置文件
if [ "$DEV_USER" != "devuser" ]; then
    echo "Copying configuration files from devuser to $DEV_USER..."
    # 复制配置文件和目录
    cp -r /home/devuser/.bashrc /home/$DEV_USER/
    cp -r /home/devuser/.profile /home/$DEV_USER/
    cp -r /home/devuser/.bash_logout /home/$DEV_USER/ 2>/dev/null || true
    cp -r /home/devuser/.config /home/$DEV_USER/
    cp -r /home/devuser/.jupyter /home/$DEV_USER/
    cp -r /home/devuser/.ssh /home/$DEV_USER/
    cp -r /home/devuser/scripts /home/$DEV_USER/
    
    # 确保权限正确
    chown -R $DEV_UID:$DEV_GID /home/$DEV_USER
fi

# 确保用户设置目录存在并优化VSCode配置
mkdir -p /home/$DEV_USER/.local/share/code-server/User
echo '{"markdown.preview.openMarkdownLinks": "inPreview","markdown.preview.scrollPreviewWithEditor": true,"markdown.preview.markEditorSelection": true,"workbench.editorAssociations": {"*.md": "default"},"security.workspace.trust.enabled": false}' > /home/$DEV_USER/.local/share/code-server/User/settings.json
chown -R $DEV_UID:$DEV_GID /home/$DEV_USER/.local

# 确保工作目录存在并设置权限
mkdir -p $WORKSPACE_DIR/{projects,data,models,notebooks,tensorboard_logs}
chown -R $DEV_UID:$DEV_GID $WORKSPACE_DIR

# 确保共享目录存在并设置权限
mkdir -p /home/$DEV_USER/shared-ro
mkdir -p /home/$DEV_USER/shared-rw
chown -R $DEV_UID:$DEV_GID /home/$DEV_USER/shared-ro
chown -R $DEV_UID:$DEV_GID /home/$DEV_USER/shared-rw
chmod 755 /home/$DEV_USER/shared-ro
chmod 755 /home/$DEV_USER/shared-rw

# 配置apt内网源
if [ -n "$APT_MIRROR_URL" ]; then
    echo "Configuring apt with internal mirror: $APT_MIRROR_URL"
    
    # 备份原始sources.list
    cp /etc/apt/sources.list /etc/apt/sources.list.backup
    
    # 创建新的sources.list
    cat > /etc/apt/sources.list << EOF
# Internal APT Mirror
deb $APT_MIRROR_URL jammy main restricted universe multiverse
deb $APT_MIRROR_URL jammy-updates main restricted universe multiverse
deb $APT_MIRROR_URL jammy-backports main restricted universe multiverse
deb $APT_MIRROR_URL jammy-security main restricted universe multiverse
EOF
    
    echo "APT sources.list updated:"
    cat /etc/apt/sources.list
    
    # 更新apt缓存
    apt-get update || echo "Warning: apt update failed, continuing..."
else
    echo "No internal apt mirror configured. Using default Ubuntu repositories."
fi

# 配置npm内网源
if [ -n "$NPM_REGISTRY_URL" ]; then
    echo "Configuring npm with internal registry: $NPM_REGISTRY_URL"
    
    # 为用户配置npm
    su - $DEV_USER -c "npm config set registry $NPM_REGISTRY_URL"
    
    # 如果指定了可信主机，配置证书验证
    if [ -n "$NPM_TRUSTED_HOST" ]; then
        su - $DEV_USER -c "npm config set strict-ssl false"
        echo "NPM SSL verification disabled for internal registry"
    fi
    
    # 显示npm配置
    echo "NPM configuration:"
    su - $DEV_USER -c "npm config list | grep registry"
else
    echo "No internal npm registry configured. Using default npm registry."
fi

# 配置pip内网镜像源
if [ -n "$PIP_INDEX_URL" ]; then
    echo "Configuring pip with internal mirror: $PIP_INDEX_URL"
    
    # 创建pip配置目录
    mkdir -p /home/$DEV_USER/.pip
    
    # 创建pip.conf配置文件
    cat > /home/$DEV_USER/.pip/pip.conf << EOF
[global]
index-url = $PIP_INDEX_URL
EOF
    
    # 如果指定了可信主机，添加到配置
    if [ -n "$PIP_TRUSTED_HOST" ]; then
        echo "trusted-host = $PIP_TRUSTED_HOST" >> /home/$DEV_USER/.pip/pip.conf
    fi
    
    # 设置权限
    chown -R $DEV_UID:$DEV_GID /home/$DEV_USER/.pip
    chmod 755 /home/$DEV_USER/.pip
    chmod 644 /home/$DEV_USER/.pip/pip.conf
    
    echo "Pip configuration created:"
    cat /home/$DEV_USER/.pip/pip.conf
else
    echo "No internal pip mirror configured. Using default PyPI."
fi

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

# 获取宿主机IP地址
# 在容器内，尝试获取默认网关IP（通常是宿主机IP）
HOST_IP=$(ip route | grep '^default' | awk '{print $3}' 2>/dev/null | head -n1)

# 如果获取不到网关IP，尝试从/etc/hosts获取
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(getent hosts host.docker.internal 2>/dev/null | awk '{print $1}')
fi

# 如果还是获取不到，使用占位符
if [ -z "$HOST_IP" ]; then
    HOST_IP="<宿主机IP>"
    echo "提示: 请将下方访问地址中的'<宿主机IP>'替换为实际的宿主机IP地址"
fi

# 显示访问信息
echo ""
echo "==============================================="
echo "GPU Docker Development Environment Started!"
echo "==============================================="
echo "Shared Directories:"
echo "  Workspace:   $WORKSPACE_DIR"
echo "  Shared (RO): /home/$DEV_USER/shared-ro"
echo "  Shared (RW): /home/$DEV_USER/shared-rw"
echo ""
echo "Access URLs:"
echo "  VSCode:      http://$HOST_IP:8080"
if [ "$ENABLE_JUPYTER" = "true" ]; then
echo "  Jupyter Lab: http://$HOST_IP:8888"
fi
if [ "$ENABLE_TENSORBOARD" = "true" ]; then
echo "  TensorBoard: http://$HOST_IP:6006"
fi
echo "  SSH:         ssh -p 22 $DEV_USER@$HOST_IP"
echo ""
echo "Login credentials:"
echo "  Username: $DEV_USER"
echo "  Password: $DEV_PASSWORD"
echo "  Final UID: $DEV_UID"
echo "  Final GID: $DEV_GID"
echo "==============================================="

# 执行传入的命令
exec "$@" 