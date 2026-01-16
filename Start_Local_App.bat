@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run superlig_local_app.py
