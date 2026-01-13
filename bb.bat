@echo off
REM BenderBox - Quick launcher with interactive mode
REM
REM Usage:
REM   bb              - Start interactive chat (NLP mode)
REM   bb -i           - Same as above
REM   bb analyze ...  - Run specific command
REM   bb --help       - Show all commands

REM Store script directory without changing working directory
REM This preserves the user's current directory for relative paths
set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%bb.py" %*
