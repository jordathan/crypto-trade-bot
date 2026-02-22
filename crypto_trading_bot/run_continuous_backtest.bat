@echo off
echo ================================================
echo Crypto Trading Bot - Continuous Backtesting
echo ================================================
echo.
echo Running continuous backtests every 5 minutes...
echo Testing: BTC, ETH, BNB, SOL, ADA, XRP, and more
echo Press Ctrl+C to stop
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run continuous backtest mode with 5-minute interval, 90-day lookback
python main.py --mode continuous-backtest --interval 5 --days 90

pause
