#!/bin/bash
# 启动 Streamlit 应用，监听 0.0.0.0 以便在 sealos devbox 等环境中访问

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 使用 python3 -m，避免 streamlit 可执行文件未加入 PATH（如 pip --user 安装）
python3 -m streamlit run src/zhisaotong_agent/app.py --server.address 0.0.0.0 --server.port 8501
