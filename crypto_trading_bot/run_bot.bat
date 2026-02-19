@echo off
echo ============================================
echo Crypto Trading Bot - Setup & Run Script
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Check for venv
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo Choose an option:
echo 1. Run GUI Dashboard (Streamlit)
echo 2. Run single trading cycle
echo 3. Start scheduled bot (daily runs)
echo 4. Run backtest
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Starting GUI Dashboard...
    streamlit run gui\app.py
) else if "%choice%"=="2" (
    echo Running single trading cycle...
    python main.py --mode run
) else if "%choice%"=="3" (
    echo Starting scheduled bot...
    echo Commands to stop: Press Ctrl+C
    python main.py --mode schedule
) else if "%choice%"=="4" (
    set /p symbol="Enter symbol (default BTC-USD): "
    set /p days="Enter days for backtest (default 90): "
    if "%symbol%"=="" set symbol=BTC-USD
    if "%days%"=="" set days=90
    echo Running backtest for %symbol% for %days% days...
    python main.py --mode backtest --symbol %symbol% --days %days%
) else (
    echo Exiting...
)

pause
