@echo off
REM BenderBox Installation Launcher (Windows)
REM Double-click this file or run from command prompt

echo.
echo ================================================
echo   BenderBox Installation Launcher (Windows)
echo ================================================
echo.
echo This will launch the interactive setup wizard.
echo For advanced options, use the PowerShell script:
echo   powershell -ExecutionPolicy Bypass -File scripts\install-prerequisites.ps1
echo.

REM Check for Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.9+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo Or via winget:
    echo   winget install Python.Python.3.11
    echo.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo [OK] Python %pyver% found
echo.

REM Check if setup_wizard.py exists
if not exist "%~dp0setup_wizard.py" (
    echo [ERROR] setup_wizard.py not found!
    echo Make sure you're running this from the BenderBox directory.
    pause
    exit /b 1
)

REM Run the setup wizard
echo Starting BenderBox Setup Wizard...
echo.
python "%~dp0setup_wizard.py"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Setup wizard failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Installation Complete!
echo ================================================
echo.
echo To start BenderBox, run:
echo   python bb.py -i
echo.
echo Or use the quick launcher:
echo   bb
echo.
pause
