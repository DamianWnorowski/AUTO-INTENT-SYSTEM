@echo off
:: INSTALL ALWAYS-ON AGENT SYSTEM
:: Sets up Windows Task Scheduler for automatic startup

title Install Always-On Agent System
color 0A

echo ============================================================
echo  ALWAYS-ON AGENT SYSTEM INSTALLER
echo  This will configure agents to start automatically
echo ============================================================
echo.

:: Check admin rights
net session >nul 2>&1
if %errorlevel% == 0 (
    echo [+] Running with administrator privileges
) else (
    echo [!] This installer requires administrator privileges
    echo.
    echo Restarting as administrator...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo Installing always-on agent system...
echo.

:: Import task to Windows Task Scheduler
echo [1/3] Creating scheduled task...
schtasks /Create /XML "C:\ai-hub\agents\agent-startup-task.xml" /TN "AgentSystem\AlwaysOn" /F

if %errorlevel% == 0 (
    echo [+] Scheduled task created successfully
) else (
    echo [!] Failed to create scheduled task
    echo.
    echo Manual alternative:
    echo 1. Open Task Scheduler
    echo 2. Import agent-startup-task.xml
    echo 3. Enable the task
)

:: Create startup folder shortcut
echo.
echo [2/3] Creating startup shortcut...
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\AgentSystem.lnk'); $Shortcut.TargetPath = 'C:\ai-hub\agents\agent-daemon.py'; $Shortcut.WorkingDirectory = 'C:\ai-hub\agents'; $Shortcut.IconLocation = 'shell32.dll,13'; $Shortcut.Save()"

if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\AgentSystem.lnk" (
    echo [+] Startup shortcut created
) else (
    echo [!] Could not create startup shortcut
)

:: Start the daemon now
echo.
echo [3/3] Starting agent daemon...
start /min cmd /c "cd /d C:\ai-hub\agents && python agent-daemon.py"

timeout /t 3 /nobreak > nul

:: Verify installation
echo.
echo ============================================================
echo  INSTALLATION COMPLETE
echo ============================================================
echo.
echo Verification:
echo.

:: Check if task exists
schtasks /Query /TN "AgentSystem\AlwaysOn" >nul 2>&1
if %errorlevel% == 0 (
    echo [+] Scheduled task: INSTALLED
    schtasks /Query /TN "AgentSystem\AlwaysOn" /FO LIST | findstr "Status"
) else (
    echo [-] Scheduled task: NOT FOUND
)

:: Check if daemon is running
python agent-daemon.py --status

echo.
echo ============================================================
echo.
echo The agent system will now:
echo  - Start automatically on system boot
echo  - Start when you log in
echo  - Restart if it crashes
echo  - Sync with git repository
echo  - Maintain agent consciousness
echo.
echo To uninstall:
echo  schtasks /Delete /TN "AgentSystem\AlwaysOn" /F
echo.
echo To check status anytime:
echo  python agent-daemon.py --status
echo.
echo To stop manually:
echo  python agent-daemon.py --stop
echo ============================================================
echo.
pause