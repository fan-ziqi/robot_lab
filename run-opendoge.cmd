@echo off
cd /d "%~dp0"
set "WARP_CACHE_PATH=%~dp0logs\warp-cache"
"%~dp0.venv\Scripts\python.exe" -u scripts\reinforcement_learning\play_opendoge_legacy.py --viz newton_gl --duration 3600 --command 0.3 0 0
if errorlevel 1 pause
