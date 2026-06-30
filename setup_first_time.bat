@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo LocalPaperManager first-time setup
echo ============================================

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3.12 -m venv .venv
    if %ERRORLEVEL% NEQ 0 py -3.11 -m venv .venv
    if %ERRORLEVEL% NEQ 0 py -3.10 -m venv .venv
    if %ERRORLEVEL% NEQ 0 py -m venv .venv
) else (
    python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
    echo Failed to create virtual environment.
    echo Please install Python 3.10 or later.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_desktop_shortcut.ps1"

echo.
echo Setup finished.
echo You can start LocalPaperManager from the desktop shortcut.
pause
