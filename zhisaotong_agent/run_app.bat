@echo off
REM Start Streamlit app on 0.0.0.0 (e.g. devbox / LAN). Same behavior as run_app.sh

cd /d "%~dp0"
python -m streamlit run src\zhisaotong_agent\app.py --server.address 0.0.0.0 --server.port 8501
