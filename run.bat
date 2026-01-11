@echo off
REM BenderBox - AI Security Analysis Platform
REM Quick launcher for Windows
REM
REM Usage:
REM   run.bat                  - Show help
REM   run.bat chat             - Start interactive chat
REM   run.bat config api-keys  - Manage API keys
REM   run.bat interrogate ...  - Run interrogation

cd /d "%~dp0"
python run.py %*

REM If running without arguments or if there was an error, pause
if "%~1"=="" (
    echo.
    echo Type 'run.bat --help' for usage information.
    pause
)
