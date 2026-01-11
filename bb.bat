@echo off
REM BenderBox - Quick launcher with interactive mode
REM
REM Usage:
REM   bb              - Start interactive chat (NLP mode)
REM   bb -i           - Same as above
REM   bb analyze ...  - Run specific command
REM   bb --help       - Show all commands

cd /d "%~dp0"
python bb.py %*
