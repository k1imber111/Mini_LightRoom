@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Первый запуск: создаю окружение и ставлю библиотеки, это пара минут...
    py -3.12 -m venv .venv || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
)
start "" ".venv\Scripts\pythonw.exe" main.py %*
