@echo off
echo ======================================
echo Crypto Trading Bot - Continuous Mode
echo ======================================
echo.
echo Running continuous trading cycles...
echo Press Ctrl+C to stop
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run continuous mode with 60-minute interval
python main.py --mode continuous --interval 60

pause
