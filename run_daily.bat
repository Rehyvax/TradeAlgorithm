@echo off
cd /d "%~dp0"

if not exist "data\live" mkdir "data\live"

echo ===== RUN START %date% %time% =====>> "data\live\cron.log"

REM Print which python is being used
"%~dp0.venv\Scripts\python.exe" --version >> "data\live\cron.log" 2>&1
"%~dp0.venv\Scripts\python.exe" -c "import sys; print(sys.executable)" >> "data\live\cron.log" 2>&1

REM Run the bot with the venv python
"%~dp0.venv\Scripts\python.exe" daily_execution.py >> "data\live\cron.log" 2>&1

echo ===== RUN END %date% %time% =====>> "data\live\cron.log"
pause