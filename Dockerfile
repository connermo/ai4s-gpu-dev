FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装基础工具和依赖
RUN apt-get update && apt-get install -y \
    curl wget git vim sudo \
    nodejs npm \
    supervisor openssh-server \
    tzdata locales \
    build-essential cmake \
    bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda 并配置mamba加速依赖解析
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -tip && \
    conda config --set always_yes yes --set changeps1 no && \
    conda config --set solver libmamba && \
    conda install -c conda-forge mamba python=3.12 && \
    conda clean -afy

# 配置时区和语言环境
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# 配置 SSH
RUN mkdir /var/run/sshd && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 创建默认用户
RUN useradd -m -s /bin/bash -u 1000 devuser && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 安装 code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# 分批使用mamba安装科学计算库（减少单次依赖解析复杂度）
# 核心数值计算库
RUN mamba install -c conda-forge \
    numpy scipy pandas sympy \
    && conda clean -afy

# Jupyter和可视化工具
RUN mamba install -c conda-forge \
    jupyter jupyterlab \
    matplotlib seaborn plotly bokeh \
    ipywidgets ipykernel \
    && conda clean -afy

# 机器学习和图像处理
RUN mamba install -c conda-forge \
    scikit-learn scikit-image \
    opencv numba \
    && conda clean -afy

# 数据处理和网络分析
RUN mamba install -c conda-forge \
    h5py netcdf4 zarr dask \
    networkx openbabel \
    && conda clean -afy

# 化学信息学工具
RUN mamba install -c conda-forge rdkit \
    && conda clean -afy

# 使用mamba安装深度学习框架
RUN mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia && \
    conda clean -afy

# 安装AI for Science专用工具
RUN pip install --no-cache-dir \
    # 基础AI工具
    tensorboard \
    transformers datasets \
    accelerate \
    diffusers \
    wandb \
    # 图神经网络
    torch-geometric \
    dgl \
    # 化学信息学
    deepchem \
    # 物理/材料科学
    ase \
    pymatgen \
    # 生物信息学 (已移除以减少安装时间)
    # biopython \
    # scanpy \
    # anndata \
    # 分子动力学和模拟
    mdtraj \
    # 3D可视化
    py3Dmol \
    # 优化工具
    cvxpy \
    # 机器学习扩展
    xgboost \
    lightgbm \
    # 数据处理
    polars \
    # 科学可视化
    mayavi \
    vtk \
    # 统计分析
    statsmodels \
    # 并行计算
    ray[default] \
    # 专业科学工具
    mendeleev \
    periodictable \
    pint

# 切换到用户环境
USER devuser
WORKDIR /home/devuser

# 初始化conda环境
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc

# 创建必要的目录
RUN mkdir -p ~/.config/code-server ~/.jupyter ~/.ssh \
    workspace/projects workspace/data workspace/models \
    workspace/notebooks workspace/tensorboard_logs

# 生成SSH密钥
RUN ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# 创建优化的VSCode用户设置
RUN mkdir -p ~/.local/share/code-server/User
RUN echo '{"markdown.preview.openMarkdownLinks": "inPreview","markdown.preview.scrollPreviewWithEditor": true,"markdown.preview.markEditorSelection": true,"workbench.editorAssociations": {"*.md": "default"},"security.workspace.trust.enabled": false}' > ~/.local/share/code-server/User/settings.json

# 配置 Jupyter
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_origin = '*'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py

# 复制配置文件
COPY --chown=devuser:devuser config/ /home/devuser/.config/
COPY --chown=devuser:devuser scripts/ /home/devuser/scripts/

# 切换回root用户进行最终配置
USER root

# 复制supervisor配置
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 创建VSDA占位符文件以防止404错误
RUN mkdir -p /usr/lib/code-server/lib/vscode/out/vs/workbench/services/extensions/worker
RUN echo "// VSDA placeholder file" > /usr/lib/code-server/lib/vscode/out/vs/workbench/services/extensions/worker/vsda.js
RUN echo -e '\x00asm\x01\x00\x00\x00' > /usr/lib/code-server/lib/vscode/out/vs/workbench/services/extensions/worker/vsda_bg.wasm

# 复制entrypoint脚本
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh /home/devuser/scripts/*.sh

# 暴露端口
# 8080: code-server
# 8888: jupyter lab
# 6006: tensorboard
# 22: ssh
EXPOSE 8080 8888 6006 22

# 设置入口点
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["supervisord"] 