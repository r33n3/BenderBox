@echo off
REM BenderBox - AI Security Analysis Platform
REM Quick launcher for Windows
REM
REM Usage:
REM   run.bat                  - Show help
REM   run.bat chat             - Start interactive chat
REM   run.bat config api-keys  - Manage API keys
REM   run.bat interrogate ...  - Run interrogation

REM Store script directory without changing working directory
REM This preserves the user's current directory for relative paths
set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%run.py" %*

REM If running without arguments or if there was an error, pause
if "%~1"=="" (
    echo.
    echo Type 'run.bat --help' for usage information.
    pause
)
