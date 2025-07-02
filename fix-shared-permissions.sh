#!/bin/bash

# 修复共享目录权限的脚本
# 在启动容器前运行此脚本

set -e

echo "修复共享目录权限..."

# 创建必要的目录
mkdir -p ./shared
mkdir -p ./tmp
mkdir -p ./workspace

# 获取当前用户UID/GID或使用默认值
TARGET_UID=${1:-1000}
TARGET_GID=${2:-1000}

echo "目标UID: $TARGET_UID"
echo "目标GID: $TARGET_GID"

# 设置目录权限
echo "设置 ./shared 权限..."
chmod 777 ./shared
sudo chown $TARGET_UID:$TARGET_GID ./shared 2>/dev/null || \
    chown $TARGET_UID:$TARGET_GID ./shared 2>/dev/null || \
    echo "注意: 无法设置./shared所有权，可能需要sudo权限"

echo "设置 ./tmp 权限..."
chmod 777 ./tmp
sudo chown $TARGET_UID:$TARGET_GID ./tmp 2>/dev/null || \
    chown $TARGET_UID:$TARGET_GID ./tmp 2>/dev/null || \
    echo "注意: 无法设置./tmp所有权，可能需要sudo权限"

echo "设置 ./workspace 权限..."
chmod 755 ./workspace
sudo chown -R $TARGET_UID:$TARGET_GID ./workspace 2>/dev/null || \
    chown -R $TARGET_UID:$TARGET_GID ./workspace 2>/dev/null || \
    echo "注意: 无法设置./workspace所有权，可能需要sudo权限"

echo ""
echo "权限设置完成！现在可以安全启动容器了"
echo "使用方法:"
echo "  ./fix-shared-permissions.sh [UID] [GID]"
echo "  ./run-container.sh"
echo ""
echo "当前目录权限:"
ls -la . | grep -E "(shared|tmp|workspace)" 