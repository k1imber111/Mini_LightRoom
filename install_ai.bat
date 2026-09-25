@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Сначала запустите start.bat: он создаст окружение программы.
    pause
    exit /b 1
)
echo Ставлю библиотеки ИИ для распознавания сцен: PyTorch с CUDA, около 3 ГБ, один раз...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-ai.txt
echo.
echo Готово. Перезапустите Mini LightRoom и нажмите кнопку «Сцены».
pause
