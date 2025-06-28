# 🐳 Docker镜像构建指南

## 构建策略

### 选项1：GitHub Actions自动构建（推荐）

1. **设置仓库secrets**（如果推送到Docker Hub）：
   ```bash
   DOCKER_USERNAME: 你的Docker Hub用户名
   DOCKER_PASSWORD: 你的Docker Hub密码
   ```

2. **推送代码触发构建**：
   ```bash
   git push origin main
   # 或发布版本tag
   git tag v1.0.0
   git push origin v1.0.0
   ```

3. **镜像自动推送到**：
   - GitHub Container Registry: `ghcr.io/connermo/ai4s-gpu-dev:latest`
   - 或Docker Hub: `connermo/ai4s-gpu-dev:latest`

### 选项2：本地构建测试

```bash
# 使用优化的Dockerfile
docker build -f Dockerfile.optimized -t ai4s-gpu-dev:test .

# 或使用原始Dockerfile
docker build -t ai4s-gpu-dev:original .
```

## 构建时间预估

| 构建方式 | 预估时间 | 成功率 |
|---------|---------|--------|
| GitHub Actions (首次) | 3-4小时 | 85% |
| GitHub Actions (缓存) | 30-60分钟 | 95% |
| 本地构建 | 2-3小时 | 90% |

## 优化建议

### 减少构建时间
1. **使用多阶段构建**：`Dockerfile.optimized`
2. **启用BuildKit缓存**：工作流中已配置
3. **减少不必要的依赖**：按需安装

### 提高成功率
1. **增加超时时间**：设置为360分钟（6小时）
2. **错误重试**：网络问题自动重试
3. **分阶段验证**：每个阶段单独测试

## 监控构建状态

- 访问Actions页面：[构建状态](https://github.com/connermo/ai4s-gpu-dev/actions)
- 构建徽章：[![Build Status](https://github.com/connermo/ai4s-gpu-dev/workflows/Build%20and%20Push%20Docker%20Image/badge.svg)](https://github.com/connermo/ai4s-gpu-dev/actions)

## 镜像使用

```bash
# 从GitHub Container Registry拉取
docker pull ghcr.io/connermo/ai4s-gpu-dev:latest

# 运行镜像
./run-container.sh devuser your_password ./workspace 80
``` 