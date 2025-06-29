# GPU Docker 开发环境

一个支持GPU的独立Docker开发环境，集成了VSCode Server、Jupyter Lab、TensorBoard和SSH服务，支持多用户和环境变量配置。

## 🚀 特性

- **GPU支持**: 基于NVIDIA CUDA 11.8镜像，支持GPU加速
- **多服务集成**: VSCode Server + Jupyter Lab + TensorBoard + SSH
- **多用户支持**: 通过环境变量配置不同用户
- **灵活配置**: 支持自定义用户ID、密码、工作目录等
- **持久化存储**: 用户数据持久化保存
- **开箱即用**: 预装深度学习常用库

## 📦 预装软件

- **开发环境**: VSCode Server, SSH, Git, Vim
- **Python生态**: Python 3.12, Miniconda, Jupyter Lab
- **深度学习**: PyTorch (CUDA 12.4), TensorBoard, Transformers, DGL, PyTorch Geometric
- **科学计算**: NumPy, SciPy, SymPy, JAX, Numba, Dask
- **数据科学**: Pandas, Polars, Scikit-learn, XGBoost, LightGBM, Statsmodels
- **可视化**: Matplotlib, Seaborn, Plotly, Bokeh, Mayavi, VTK, py3Dmol
- **化学信息学**: RDKit, DeepChem, ASE, Pymatgen, OpenBabel
- **生物信息学**: BioPython, Scanpy, AnnData, MDTraj
- **优化工具**: CVXPY, Ray
- **专业工具**: Mendeleev, PeriodTable, Pint (单位转换)
- **其他**: OpenCV, W&B, Accelerate, Diffusers

## 🛠 快速开始

### 方法1: 使用便捷脚本（推荐）

```bash
# 给脚本执行权限
chmod +x run-container.sh

# 启动默认用户环境
./run-container.sh

# 启动自定义用户环境
./run-container.sh myuser mypassword ./my-workspace 90

# 启动带内网镜像源的环境
./run-container.sh myuser mypass ./workspace 80 \
    "http://internal-pypi.com/simple/" \
    "internal-pypi.com" \
    "http://internal-apt.com/ubuntu" \
    "http://internal-npm.com/" \
    "internal-npm.com"
```

**脚本参数说明：**
- 参数1：用户名（默认：devuser）
- 参数2：密码（默认：changeme）
- 参数3：工作目录（默认：./workspace）
- 参数4：端口前缀（默认：80，对应8080、8888、8006、8022端口）
- 参数5：pip内网镜像源URL（可选）
- 参数6：pip可信主机（可选）
- 参数7：apt内网镜像源URL（可选）
- 参数8：npm内网注册表URL（可选）
- 参数9：npm可信主机（可选）

### 方法2: 使用Docker命令

```bash
# 构建镜像
docker build -t gpu-dev-env:latest .

# 运行容器
docker run -d \
    --name gpu-dev-myuser \
    --gpus all \
    -p 8080:8080 \
    -p 8888:8888 \
    -p 6006:6006 \
    -p 2222:22 \
    -e DEV_USER=myuser \
    -e DEV_PASSWORD=mypassword \
    -e DEV_UID=$(id -u) \
    -e DEV_GID=$(id -g) \
    -v ./workspace:/home/myuser/workspace:rw \
    -v ./shared:/shared:ro \
    --shm-size=32g \
    gpu-dev-env:latest
```

### 方法3: 使用Docker Compose（多用户）

```bash
# 启动多用户环境
docker-compose up -d

# 停止环境
docker-compose down
```

## 🌍 访问服务

启动后可以通过以下方式访问各项服务：

> **获取宿主机IP地址**：
> - Linux/macOS: `hostname -I | awk '{print $1}'` 或 `ip addr show | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | cut -d'/' -f1 | head -n1`
> - Windows: `ipconfig` 查看IPv4地址
> - 本地开发可以使用 `localhost` 或 `127.0.0.1`

| 服务 | 默认端口 | 访问地址 |
|------|---------|----------|
| VSCode Server | 8080 | http://`<宿主机IP>`:8080 |
| Jupyter Lab | 8888 | http://`<宿主机IP>`:8888 |
| TensorBoard | 6006 | http://`<宿主机IP>`:6006 |
| SSH | 22 | `ssh -p 22 用户名@<宿主机IP>` |

## ⚙️ 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEV_USER` | devuser | 容器内用户名 |
| `DEV_PASSWORD` | changeme | 用户密码（用于VSCode、Jupyter、SSH） |
| `DEV_UID` | 1000 | 用户UID（建议设置为宿主机用户UID） |
| `DEV_GID` | 1000 | 用户GID（建议设置为宿主机用户GID） |
| `ENABLE_JUPYTER` | true | 是否启用Jupyter Lab |
| `ENABLE_TENSORBOARD` | true | 是否启用TensorBoard |
| `WORKSPACE_DIR` | /home/用户名/workspace | 工作目录路径 |
| `APT_MIRROR_URL` | - | apt内网镜像源地址（可选） |
| `NPM_REGISTRY_URL` | - | npm内网注册表地址（可选） |
| `NPM_TRUSTED_HOST` | - | npm可信主机（可选） |
| `PIP_INDEX_URL` | - | pip内网镜像源地址（可选） |
| `PIP_TRUSTED_HOST` | - | pip可信主机（可选） |

## 📁 目录结构

```
gpu-docker/
├── Dockerfile              # 主镜像定义
├── docker-compose.yml      # 多用户编排文件
├── run-container.sh        # 快速启动脚本
├── entrypoint.sh           # 容器入口脚本
├── supervisord.conf        # 服务管理配置
├── config/                 # 配置文件目录
│   └── code-server/
│       └── config.yaml     # VSCode Server配置
├── scripts/                # 辅助脚本
│   └── start-services.sh   # 服务启动脚本
├── data/                   # 用户数据目录（自动创建）
│   ├── user1/             # 用户1工作空间
│   └── user2/             # 用户2工作空间
└── shared/                # 共享数据目录
```

## 🔧 使用示例

### 单用户快速启动

```bash
# 启动用户alice的开发环境，端口前缀90
./run-container.sh alice password123 ./alice-workspace 90

# 访问地址：
# VSCode: http://<宿主机IP>:9080
# Jupyter: http://<宿主机IP>:9088
# TensorBoard: http://<宿主机IP>:9006
# SSH: ssh -p 9022 alice@<宿主机IP>
```

### 多用户并行使用

```bash
# 启动用户1
./run-container.sh user1 pass1 ./data/user1 80

# 启动用户2  
./run-container.sh user2 pass2 ./data/user2 81

# 用户1访问：http://<宿主机IP>:8080
# 用户2访问：http://<宿主机IP>:8180
```

### 配置内网镜像源

```bash
# 使用便捷脚本配置所有内网镜像源
./run-container.sh myuser mypass ./workspace 80 \
    "http://your-internal-pypi.com/simple/" \
    "your-internal-pypi.com" \
    "http://your-internal-apt.com/ubuntu" \
    "http://your-internal-npm.com/" \
    "your-internal-npm.com"

# 只配置部分镜像源（传空字符串跳过不需要的参数）
./run-container.sh myuser mypass ./workspace 80 \
    "http://your-internal-pypi.com/simple/" \
    "your-internal-pypi.com" \
    "" \
    "http://your-internal-npm.com/" \
    "your-internal-npm.com"

# 使用Docker命令配置所有内网镜像源
docker run -d \
    --name gpu-dev-myuser \
    --gpus all \
    -p 8080:8080 \
    -p 8888:8888 \
    -e DEV_USER=myuser \
    -e DEV_PASSWORD=mypass \
    -e APT_MIRROR_URL=http://your-internal-apt.com/ubuntu \
    -e NPM_REGISTRY_URL=http://your-internal-npm.com/ \
    -e NPM_TRUSTED_HOST=your-internal-npm.com \
    -e PIP_INDEX_URL=http://your-internal-pypi.com/simple/ \
    -e PIP_TRUSTED_HOST=your-internal-pypi.com \
    -v ./workspace:/home/myuser/workspace:rw \
    --shm-size=32g \
    connermo/ai4s-gpu-dev:latest

# 使用Docker Compose配置
# 在docker-compose.yml中取消注释并修改以下行：
# - APT_MIRROR_URL=http://your-internal-apt.com/ubuntu
# - NPM_REGISTRY_URL=http://your-internal-npm.com/
# - NPM_TRUSTED_HOST=your-internal-npm.com
# - PIP_INDEX_URL=http://your-internal-pypi.com/simple/
# - PIP_TRUSTED_HOST=your-internal-pypi.com
```

### 验证内网镜像源配置

```bash
# 进入容器验证配置
docker exec -it gpu-dev-myuser bash

# 验证APT配置
cat /etc/apt/sources.list
apt update  # 测试apt源可用性

# 验证NPM配置
npm config list | grep registry
npm config get registry
npm info express  # 测试npm源可用性

# 验证PIP配置
pip config list
cat ~/.pip/pip.conf
pip install requests  # 测试pip源可用性
```

### 内网镜像源地址格式说明

**APT镜像源：**
- 格式：`http://your-mirror-host/ubuntu`
- 常见示例：
  - 阿里云：`http://mirrors.aliyun.com/ubuntu`
  - 清华大学：`http://mirrors.tuna.tsinghua.edu.cn/ubuntu`
  - 网易：`http://mirrors.163.com/ubuntu`

**NPM注册表：**
- 格式：`http://your-npm-registry/`
- 常见示例：
  - 淘宝：`https://registry.npmmirror.com/`
  - 腾讯：`http://mirrors.cloud.tencent.com/npm/`
  - 华为：`https://mirrors.huaweicloud.com/repository/npm/`

**PIP镜像源：**
- 格式：`http://your-pypi-mirror/simple/`
- 常见示例：
  - 阿里云：`http://mirrors.aliyun.com/pypi/simple/`
  - 清华大学：`https://pypi.tuna.tsinghua.edu.cn/simple/`
  - 豆瓣：`http://pypi.douban.com/simple/`

**注意事项：**
1. 如果使用HTTPS镜像源但证书有问题，可以通过trusted-host参数跳过证书验证
2. APT镜像源配置会在容器启动时自动更新包列表
3. NPM和PIP配置是用户级别的，不会影响系统级安装
4. 内网环境建议同时配置所有三种镜像源以获得最佳体验

### GPU测试

在Jupyter或VSCode中运行以下代码测试GPU：

```python
import torch
import numpy as np
import sys

# 检查Python和CUDA版本
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")

# 创建GPU张量进行测试
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print(f"GPU calculation result shape: {z.shape}")
    print("GPU test completed successfully!")
```

### AI for Science工具测试

```python
# 测试科学计算工具
import numpy as np
import scipy
import jax.numpy as jnp
from jax import grad, jit
import sympy as sp

print("=== 基础科学计算 ===")
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")

# 测试JAX
@jit
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000.0)
result = selu(x)
print(f"JAX computation completed: {result.shape}")

# 测试化学信息学
print("\n=== 化学信息学 ===")
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mol = Chem.MolFromSmiles('CCO')  # 乙醇
    print(f"分子量: {Descriptors.MolWt(mol):.2f}")
    print("RDKit工作正常")
except ImportError:
    print("RDKit未正确安装")

# 测试材料科学
print("\n=== 材料科学 ===")
try:
    from pymatgen.core import Structure, Lattice
    from ase import Atoms
    print("ASE和Pymatgen工作正常")
except ImportError:
    print("材料科学工具未正确安装")

# 测试图神经网络
print("\n=== 图神经网络 ===")
try:
    import torch_geometric
    import dgl
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    print(f"DGL version: {dgl.__version__}")
except ImportError:
    print("图神经网络库未正确安装")

# 测试生物信息学
print("\n=== 生物信息学 ===")
try:
    from Bio.Seq import Seq
    dna = Seq("AGTACACTGGT")
    print(f"DNA序列: {dna}")
    print(f"转录RNA: {dna.transcribe()}")
    print("BioPython工作正常")
except ImportError:
    print("BioPython未正确安装")

print("\n=== 测试完成 ===")
```

## 🛡️ 安全注意事项

1. **密码安全**: 请修改默认密码，使用强密码
2. **端口安全**: 在生产环境中，建议使用防火墙限制端口访问
3. **数据权限**: 确保挂载目录的权限设置正确
4. **网络安全**: 在公网环境中使用时，建议配置SSL证书

## 🔍 故障排除

### 容器无法启动

```bash
# 查看容器日志（替换为实际用户名，如：gpu-dev-devuser）
docker logs gpu-dev-<用户名>

# 检查GPU驱动
nvidia-smi

# 检查Docker GPU支持（使用当前CUDA版本）
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi

# 检查Docker是否正常运行
docker --version
docker ps
```

### 服务无法访问

```bash
# 检查端口占用（Linux/macOS）
netstat -tlnp | grep :8080
# 或者使用ss命令
ss -tlnp | grep :8080

# Windows检查端口占用
netstat -ano | findstr :8080

# 检查容器是否运行
docker ps | grep gpu-dev

# 进入容器调试（替换为实际容器名）
docker exec -it gpu-dev-<用户名> bash

# 在容器内检查服务状态
docker exec gpu-dev-<用户名> supervisorctl status

# 重启特定服务
docker exec gpu-dev-<用户名> supervisorctl restart code-server
docker exec gpu-dev-<用户名> supervisorctl restart jupyter
docker exec gpu-dev-<用户名> supervisorctl restart tensorboard

# 重启所有服务
docker exec gpu-dev-<用户名> supervisorctl restart all
```

### 权限问题

```bash
# 修复工作目录权限（替换为实际工作目录路径）
sudo chown -R $(id -u):$(id -g) <工作目录路径>

# 示例：
sudo chown -R $(id -u):$(id -g) ./workspace
sudo chown -R $(id -u):$(id -g) ./data/user1
sudo chown -R $(id -u):$(id -g) ./shared
sudo chown -R $(id -u):$(id -g) ./tmp

# 检查目录权限
ls -la ./workspace
ls -la ./shared
ls -la ./tmp
```

### 共享内存不足

```bash
# 检查容器内共享内存使用情况
docker exec gpu-dev-<用户名> df -h /dev/shm

# 如果出现 "No space left on device" 错误
# 可以尝试增加共享内存大小（在启动时添加）
--shm-size=64g  # 根据需要调整大小
```

### 常见错误及解决方案

**错误：`docker: Error response from daemon: could not select device driver "nvidia"`**
```bash
# 解决方案：安装或重新安装NVIDIA Container Toolkit
# Ubuntu/Debian:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**错误：端口已被占用**
```bash
# 查找占用端口的进程
sudo lsof -i :8080
# 终止占用端口的进程
sudo kill -9 <PID>
# 或使用不同的端口前缀启动
./run-container.sh myuser password ./workspace 81  # 使用81xx端口
```

## 📝 开发建议

1. **工作目录**: 将代码放在挂载的工作目录中，确保数据持久化
2. **GPU内存**: 使用TensorBoard监控GPU使用情况
3. **Jupyter插件**: 可以安装额外的Jupyter扩展
4. **VSCode插件**: 在VSCode中安装Python、Docker等扩展

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## �� 许可证

MIT License 