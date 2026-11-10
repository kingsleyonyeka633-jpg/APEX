@echo off
title Apex paper trading bot
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
  echo Python is not installed. Get it from python.org/downloads and tick "Add python.exe to PATH".
  pause
  exit /b
)
if not exist .installed (
  echo First time only: installing the tools the bot needs...
  pip install -r requirements.txt
  if errorlevel 1 ( echo Install failed. Send me the message above. & pause & exit /b )
  echo done > .installed
)
echo Starting. Your browser will open the dashboard. Keep this window open.
python -m apex auto
pause
