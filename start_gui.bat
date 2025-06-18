@echo off
echo ============================================================
echo    WiFi Indoor Localization System
echo ============================================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting the GUI system...
echo The web interface will open automatically in your browser
echo.
echo Press Ctrl+C to stop the server
echo.

python localization_api.py

pause 