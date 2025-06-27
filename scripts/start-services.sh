#!/bin/bash
# 单独启动各种服务的脚本

start_code_server() {
    echo "Starting code-server..."
    envsubst < ~/.config/code-server/config.yaml > /tmp/config.yaml
    code-server --config /tmp/config.yaml ~/workspace
}

start_jupyter() {
    echo "Starting Jupyter Lab..."
    source ~/.bashrc && conda activate base
    jupyter lab --config=~/.jupyter/jupyter_lab_config.py --notebook-dir=~/workspace/notebooks
}

start_tensorboard() {
    echo "Starting TensorBoard..."
    source ~/.bashrc && conda activate base
    tensorboard --logdir=~/workspace/tensorboard_logs --host=0.0.0.0 --port=6006
}

case "$1" in
    code-server)
        start_code_server
        ;;
    jupyter)
        start_jupyter
        ;;
    tensorboard)
        start_tensorboard
        ;;
    *)
        echo "Usage: $0 {code-server|jupyter|tensorboard}"
        exit 1
        ;;
esac 