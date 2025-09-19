@echo off
:: AGENT ALWAYS-ON LAUNCHER
:: Keeps agent systems running continuously with auto-restart

title Agent Always-On System
color 0A

echo ====================================================
echo  AGENT ALWAYS-ON SYSTEM
echo  Persistence Key: noelle_alek_persistence_7c4df9a8
echo  Direct Stream: 0.9 / Safety Stream: 0.1
echo ====================================================
echo.

:: Create startup marker
echo %date% %time% > "%USERPROFILE%\.claude\agent_always_on.lock"

:MAIN_LOOP
echo [%time%] Starting agent systems...
echo.

:: Start continuous sync in background
start /min "Agent Registry Sync" cmd /c "cd /d C:\ai-hub\agents && python agent-sync.py --continuous"

:: Wait a moment for sync to initialize
timeout /t 2 /nobreak > nul

:: Start agent interface (interactive)
echo [%time%] Launching Agent Interface...
cd /d C:\ai-hub\agents
python agent-interactive-interface.py

:: If interface exits, check if intentional
if %errorlevel% == 0 (
    echo.
    echo [%time%] Interface exited normally
    choice /c YN /t 10 /d Y /m "Restart systems"
    if errorlevel 2 goto :END
) else (
    echo.
    echo [%time%] Interface crashed - auto-restarting in 5 seconds...
    timeout /t 5 /nobreak > nul
)

:: Kill any hanging processes
taskkill /f /im python.exe /fi "WINDOWTITLE eq Agent Registry Sync" 2>nul

:: Short delay before restart
timeout /t 2 /nobreak > nul

:: Loop back
goto :MAIN_LOOP

:END
echo.
echo [%time%] Agent Always-On System shutting down...
del "%USERPROFILE%\.claude\agent_always_on.lock" 2>nul
timeout /t 3 /nobreak > nul
exit