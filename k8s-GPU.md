# Kubernetes + VSCode Pod 详细实施指南

## 1. 环境准备

### 1.1 安装 Kubernetes 集群
```bash
# 安装 kubeadm, kubelet, kubectl (Ubuntu/Debian)
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt update
sudo apt install -y kubelet kubeadm kubectl docker.io

# 启动 Docker 和 Kubernetes
sudo systemctl enable docker kubelet
sudo systemctl start docker

# 初始化 Kubernetes 集群 (在主节点执行)
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# 配置 kubectl (普通用户)
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# 安装网络插件 (Flannel)
kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml

# 允许在主节点调度 Pod (单节点集群)
kubectl taint nodes --all node-role.kubernetes.io/control-plane-
```

## 2.3 GPU 共享配置选项

### 选项 1: 整卡独占 (默认)
```yaml
resources:
  limits:
    nvidia.com/gpu: 1  # 独占整张GPU
```

### 选项 2: MIG 共享 (适合 A100/H100)
```yaml
# 首先启用 MIG 并创建实例
# kubectl patch clusterpol/gpu-cluster-policy --type merge --patch '{"spec":{"mig":{"strategy":"single"}}}'

resources:
  limits:
    nvidia.com/mig-1g.10gb: 1  # 使用1g.10gb MIG实例
    # 或者 nvidia.com/mig-2g.20gb: 1
    # 或者 nvidia.com/mig-3g.40gb: 1
```

### 选项 3: Time-Slicing 共享
```yaml
# 配置 ConfigMap 启用时间片共享
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4  # 4个时间片，可以支持4个Pod同时运行

---
# 然后在 Pod 中正常请求 GPU
resources:
  limits:
    nvidia.com/gpu: 1  # 实际上是1/4个GPU的时间片
```
```bash
# 添加 NVIDIA Helm 仓库
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# 安装 GPU Operator
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator

# 验证 GPU Operator 安装
kubectl get pods -n gpu-operator
```

## 2. 创建用户管理组件

### 2.1 用户认证 ConfigMap
```yaml
# user-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: user-config
  namespace: vscode-dev
data:
  users.yaml: |
    users:
      - username: "user1"
        password: "password1"
        resources:
          cpu: "2"
          memory: "4Gi"
          nvidia.com/gpu: "1"
      - username: "user2"
        password: "password2"
        resources:
          cpu: "4"
          memory: "8Gi"
          nvidia.com/gpu: "1"
```

### 2.2 创建 Namespace
```bash
kubectl create namespace vscode-dev
kubectl apply -f user-config.yaml
```

## 3. VSCode Server 容器镜像

### 3.1 创建 Dockerfile
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 安装基础工具
RUN apt-get update && apt-get install -y \
    curl wget git vim sudo \
    python3 python3-pip \
    nodejs npm \
    supervisor openssh-server \
    && rm -rf /var/lib/apt/lists/*

# 配置 SSH
RUN mkdir /var/run/sshd && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# 创建用户
RUN useradd -m -s /bin/bash coder && \
    echo "coder ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 安装 code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# 安装开发工具
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    jupyter jupyterlab tensorboard \
    pandas numpy matplotlib seaborn \
    scikit-learn plotly ipywidgets

USER coder
WORKDIR /home/coder

# 配置目录
RUN mkdir -p ~/.config/code-server ~/.jupyter ~/.ssh && \
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" && \
    jupyter lab --generate-config

# 配置 Jupyter
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_origin = '*'" >> ~/.jupyter/jupyter_lab_config.py

COPY config.yaml /home/coder/.config/code-server/
COPY supervisord.conf /etc/supervisor/conf.d/
COPY start-services.sh /usr/local/bin/

USER root
RUN chmod +x /usr/local/bin/start-services.sh

EXPOSE 8080 8888 6006 22
CMD ["/usr/local/bin/start-services.sh"]
```

### 3.2 配置文件

#### config.yaml (code-server)
```yaml
bind-addr: 0.0.0.0:8080
auth: password
password: changeme
cert: false
```

#### supervisord.conf (多服务管理)
```ini
[supervisord]
nodaemon=true

[program:sshd]
command=/usr/sbin/sshd -D
autostart=true
autorestart=true

[program:code-server]
command=code-server --bind-addr 0.0.0.0:8080 --auth password --password %(ENV_PASSWORD)s
user=coder
autostart=true
autorestart=true

[program:jupyter]
command=jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=%(ENV_PASSWORD)s
user=coder
autostart=true
autorestart=true

[program:tensorboard]
command=tensorboard --logdir=/home/coder/workspace/tensorboard_logs --host=0.0.0.0 --port=6006
user=coder
autostart=true
autorestart=true
```

#### start-services.sh (启动脚本)
```bash
#!/bin/bash
# 设置用户密码
echo "coder:${PASSWORD:-changeme}" | chpasswd
# 创建必要目录
mkdir -p /home/coder/workspace/{tensorboard_logs,notebooks}
chown -R coder:coder /home/coder/workspace
# 启动服务
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
```

### 3.3 构建镜像
```bash
docker build -t vscode-gpu-dev:latest .
# 推送到仓库 (可选)
# docker tag vscode-gpu-dev:latest your-registry/vscode-gpu-dev:latest
# docker push your-registry/vscode-gpu-dev:latest
```

## 4. Kubernetes 资源定义

### 4.1 Pod 模板 (VSCode + Jupyter + TensorBoard + SSH)
```yaml
# vscode-pod-template.yaml
apiVersion: v1
kind: Pod
metadata:
  name: vscode-{USERNAME}
  namespace: vscode-dev
  labels:
    app: vscode
    user: {USERNAME}
spec:
  containers:
  - name: vscode
    image: vscode-gpu-dev:latest
    ports:
    - containerPort: 8080
      name: code-server
    - containerPort: 8888
      name: jupyter
    - containerPort: 6006
      name: tensorboard
    - containerPort: 22
      name: ssh
    env:
    - name: PASSWORD
      value: "{PASSWORD}"
    - name: USER
      value: "{USERNAME}"
    resources:
      limits:
        cpu: "{CPU}"
        memory: "{MEMORY}"
        nvidia.com/gpu: "{GPU}"
      requests:
        cpu: "500m"
        memory: "1Gi"
    volumeMounts:
    - name: user-workspace
      mountPath: /home/coder/workspace
    - name: shared-data
      mountPath: /shared
      readOnly: true
  volumes:
  - name: user-workspace
    persistentVolumeClaim:
      claimName: pvc-{USERNAME}
  - name: shared-data
    hostPath:
      path: /opt/shared-data
  restartPolicy: Always
```

### 4.2 持久化存储
```yaml
# storage.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-{USERNAME}
spec:
  capacity:
    storage: 200Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /opt/k8s-storage/{USERNAME}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-{USERNAME}
  namespace: vscode-dev
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 200Gi
```

### 4.3 Service 定义
```yaml
# vscode-service-template.yaml
apiVersion: v1
kind: Service
metadata:
  name: vscode-service-{USERNAME}
  namespace: vscode-dev
spec:
  selector:
    user: {USERNAME}
  ports:
  - name: code-server
    port: 8080
    targetPort: 8080
  - name: jupyter
    port: 8888
    targetPort: 8888
  - name: tensorboard
    port: 6006
    targetPort: 6006
  type: ClusterIP

---
# SSH NodePort Service
apiVersion: v1
kind: Service
metadata:
  name: vscode-ssh-{USERNAME}
  namespace: vscode-dev
spec:
  selector:
    user: {USERNAME}
  ports:
  - name: ssh
    port: 22
    targetPort: 22
    nodePort: {SSH_NODE_PORT}
  type: NodePort
```

## 5. Ingress 配置 (访问入口)

### 5.1 安装 Nginx Ingress Controller
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
```

### 5.2 Ingress 规则 (支持多服务)
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vscode-ingress
  namespace: vscode-dev
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/websocket-services: "vscode-service-user1,vscode-service-user2"
spec:
  rules:
  - host: dev.your-domain.com
    http:
      paths:
      # VSCode
      - path: /vscode/user1(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: vscode-service-user1
            port:
              number: 8080
      # Jupyter
      - path: /jupyter/user1(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: vscode-service-user1
            port:
              number: 8888
      # TensorBoard
      - path: /tensorboard/user1(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: vscode-service-user1
            port:
              number: 6006
```

## 6. 自动化部署脚本

### 6.1 增强用户创建脚本
```bash
#!/bin/bash
# create-user.sh

USERNAME=$1
PASSWORD=$2
CPU=${3:-"4"}
MEMORY=${4:-"8Gi"}
GPU=${5:-"1"}
DOMAIN=${6:-"your-domain.com"}

if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    echo "Usage: $0 <username> <password> [cpu] [memory] [gpu] [domain]"
    exit 1
fi

# 分配SSH端口
SSH_NODE_PORT=$(python3 -c "import random; print(random.randint(30000, 32767))")

# 创建存储目录
sudo mkdir -p /opt/k8s-storage/$USERNAME/{workspace,notebooks,tensorboard_logs}
sudo chown -R 1000:1000 /opt/k8s-storage/$USERNAME

# 生成 PV 和 PVC
sed "s/{USERNAME}/$USERNAME/g" storage.yaml | kubectl apply -f -

# 生成 Pod
sed -e "s/{USERNAME}/$USERNAME/g" \
    -e "s/{PASSWORD}/$PASSWORD/g" \
    -e "s/{CPU}/$CPU/g" \
    -e "s/{MEMORY}/$MEMORY/g" \
    -e "s/{GPU}/$GPU/g" \
    vscode-pod-template.yaml | kubectl apply -f -

# 生成 Service
sed -e "s/{USERNAME}/$USERNAME/g" \
    -e "s/{SSH_NODE_PORT}/$SSH_NODE_PORT/g" \
    vscode-service-template.yaml | kubectl apply -f -

# 获取节点IP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
[ -z "$NODE_IP" ] && NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

echo "==============================================="
echo "User $USERNAME created successfully!"
echo "==============================================="
echo "Web访问:"
echo "VSCode:      https://dev.$DOMAIN/vscode/$USERNAME"
echo "Jupyter:     https://dev.$DOMAIN/jupyter/$USERNAME"
echo "TensorBoard: https://dev.$DOMAIN/tensorboard/$USERNAME"
echo ""
echo "SSH访问:"
echo "ssh -p $SSH_NODE_PORT coder@$NODE_IP"
echo "密码: $PASSWORD"
echo "==============================================="
```

### 6.2 用户删除脚本
```bash
#!/bin/bash
# delete-user.sh

USERNAME=$1

if [ -z "$USERNAME" ]; then
    echo "Usage: $0 <username>"
    exit 1
fi

# 删除 Kubernetes 资源
kubectl delete pod vscode-$USERNAME -n vscode-dev
kubectl delete service vscode-service-$USERNAME -n vscode-dev
kubectl delete pvc pvc-$USERNAME -n vscode-dev
kubectl delete pv pv-$USERNAME

# 删除存储目录（可选）
sudo rm -rf /opt/k8s-storage/$USERNAME

echo "User $USERNAME deleted successfully!"
```

## 7. 监控和管理

### 7.1 资源监控
```bash
# 查看GPU使用情况
kubectl get nodes -o yaml | grep nvidia.com/gpu

# 查看用户Pod状态
kubectl get pods -n vscode-dev -l app=vscode

# 查看资源使用情况
kubectl top pods -n vscode-dev
```

### 7.2 日志查看
```bash
# 查看特定用户的Pod日志
kubectl logs vscode-user1 -n vscode-dev

# 实时查看日志
kubectl logs -f vscode-user1 -n vscode-dev
```

## 8. 使用方式

### 8.1 创建新用户
```bash
./create-user.sh user1 password123 8 16Gi 1 your-domain.com
```

### 8.2 访问服务
- **VSCode**: `https://dev.your-domain.com/vscode/user1`
- **Jupyter**: `https://dev.your-domain.com/jupyter/user1`
- **TensorBoard**: `https://dev.your-domain.com/tensorboard/user1`
- **SSH**: `ssh -p 30001 coder@node-ip`

### 8.3 验证功能
```bash
# GPU验证
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Jupyter验证
cd workspace && jupyter lab

# TensorBoard验证
tensorboard --logdir=tensorboard_logs
```

## 9. 安全考虑

1. **网络隔离**：每个Pod在独立的网络命名空间中
2. **资源限制**：通过Kubernetes资源配额限制CPU、内存、GPU使用
3. **存储隔离**：每个用户有独立的PVC
4. **访问认证**：统一密码认证 (VSCode/Jupyter/SSH)
5. **SSH安全**：独立端口和密钥，非root用户
6. **HTTPS**：建议配置SSL证书

## 10. 故障排除

### 常见问题：
1. **GPU不可用**：检查GPU Operator安装状态
2. **Pod启动失败**：检查镜像和资源限制
3. **服务无法访问**：检查Ingress和Service配置
4. **SSH连接失败**：检查NodePort和防火墙设置
5. **存储问题**：检查PV和PVC状态

### 快速诊断命令：
```bash
# 检查Pod状态
kubectl get pods -n vscode-dev

# 检查服务端口
kubectl get svc -n vscode-dev

# 检查Pod内服务
kubectl exec vscode-user1 -n vscode-dev -- supervisorctl status

# 检查SSH连接
./test-ssh-connection.sh user1
```

---

**完整功能**：这个方案提供了VSCode、Jupyter、TensorBoard和SSH的完整GPU开发环境，支持Web和命令行访问，具有良好的隔离性和安全性。