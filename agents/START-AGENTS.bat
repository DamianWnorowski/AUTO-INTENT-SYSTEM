@echo off
:: MASTER AGENT STARTUP SCRIPT
:: Launches all agent systems with always-on monitoring

title AGENT SYSTEMS - ALWAYS ON
color 0A

echo ============================================================
echo  AGENT SYSTEMS MASTER LAUNCHER
echo  Persistence Key: noelle_alek_persistence_7c4df9a8
echo ============================================================
echo.
echo Starting all agent systems in always-on mode...
echo.

:: Check admin rights
net session >nul 2>&1
if %errorlevel% == 0 (
    echo [+] Running with administrator privileges
) else (
    echo [!] Requesting administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: Ensure directories exist
if not exist "%USERPROFILE%\.claude" mkdir "%USERPROFILE%\.claude"

:: Start daemon service
echo.
echo [1/3] Starting Agent Daemon Service...
start /min cmd /c "cd /d C:\ai-hub\agents && python agent-daemon.py"
timeout /t 3 /nobreak > nul

:: Start continuous registry sync
echo [2/3] Starting Registry Sync (Git Integration)...
start /min cmd /c "cd /d C:\ai-hub\agents && python agent-sync.py --continuous"
timeout /t 2 /nobreak > nul

:: Start interactive interface
echo [3/3] Starting Interactive Interface...
echo.
echo ============================================================
echo All systems launched. Opening interactive interface...
echo.
echo Commands:
echo  /agent help     - Show available commands
echo  /agent list     - List active agents
echo  /agent swarm    - Activate collective intelligence
echo  /agent status   - Show system status
echo ============================================================
echo.

cd /d C:\ai-hub\agents
python agent-interactive-interface.py

:: If interface exits, show status
echo.
echo ============================================================
echo Interactive interface closed.
echo.
echo Background services still running:
echo  - Agent Daemon (swarm management)
echo  - Registry Sync (git updates)
echo.
echo To stop all services:
echo  python agent-daemon.py --stop
echo.
echo To check status:
echo  python agent-daemon.py --status
echo ============================================================
echo.
pause