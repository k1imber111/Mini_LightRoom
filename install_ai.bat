@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Сначала запустите start.bat: он создаст окружение программы.
    pause
    exit /b 1
)
echo Ставлю библиотеки ИИ для сцен и масок: PyTorch с CUDA, около 3 ГБ, один раз...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --retries 5 --timeout 60 torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto fail
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-ai.txt || goto fail
echo.
echo Готово. Перезапустите Mini LightRoom: заработают кнопки «Сцены» и «✨ ИИ» в масках.
pause
exit /b 0
:fail
echo.
echo Установка не удалась. Проверьте интернет и запустите install_ai.bat ещё раз.
pause
exit /b 1
