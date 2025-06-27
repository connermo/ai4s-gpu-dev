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

# 安装 Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -tip && \
    conda config --set always_yes yes --set changeps1 no && \
    conda update -q conda && \
    conda install python=3.12 && \
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

# 安装科学计算基础库
RUN conda install -c conda-forge \
    jupyter jupyterlab \
    pandas numpy scipy sympy \
    matplotlib seaborn plotly bokeh \
    scikit-learn scikit-image \
    ipywidgets ipykernel \
    opencv \
    jax jaxlib \
    numba \
    h5py netcdf4 zarr dask \
    networkx \
    openbabel \
    && conda clean -afy

# 安装深度学习框架
RUN conda install -c pytorch pytorch torchvision torchaudio pytorch-cuda=12.4 && \
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
    rdkit-pypi \
    deepchem \
    # 物理/材料科学
    ase \
    pymatgen \
    # 生物信息学
    biopython \
    scanpy \
    anndata \
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
COPY supervisord.conf /etc/supervisor/conf.d/
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