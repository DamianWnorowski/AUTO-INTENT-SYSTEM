@echo off
:: INSTALL MCP FOR AGENT SYSTEM
:: Sets up Model Context Protocol for Claude Desktop integration

title Install Agent MCP Server
color 0A

echo ============================================================
echo  AGENT MCP (Model Context Protocol) INSTALLER
echo  Integrates agent system with Claude Desktop
echo ============================================================
echo.

:: Install MCP SDK
echo [1/4] Installing MCP SDK...
pip install mcp

if %errorlevel% == 0 (
    echo [+] MCP SDK installed successfully
) else (
    echo [!] Failed to install MCP SDK
    echo     Try: pip install mcp --upgrade
)

echo.
echo [2/4] Locating Claude Desktop configuration...

:: Find Claude config directory
set CLAUDE_CONFIG=%APPDATA%\Claude
if exist "%CLAUDE_CONFIG%" (
    echo [+] Found Claude config at: %CLAUDE_CONFIG%
) else (
    echo [!] Claude Desktop config not found
    echo     Please ensure Claude Desktop is installed
    set CLAUDE_CONFIG=%USERPROFILE%\.claude
    echo     Using fallback: %CLAUDE_CONFIG%
)

echo.
echo [3/4] Updating Claude configuration...

:: Check if claude_desktop_config.json exists
if exist "%CLAUDE_CONFIG%\claude_desktop_config.json" (
    echo [+] Found existing Claude config
    echo.
    echo Current config will be backed up to:
    echo %CLAUDE_CONFIG%\claude_desktop_config.backup.json
    copy "%CLAUDE_CONFIG%\claude_desktop_config.json" "%CLAUDE_CONFIG%\claude_desktop_config.backup.json" >nul
) else (
    echo [!] No existing config found, creating new one
)

:: Create or update config
echo {> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo   "mcpServers": {>> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo     "agent-system": {>> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo       "command": "python",>> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo       "args": ["C:\\ai-hub\\agents\\agent-mcp-server.py"],>> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo       "env": {>> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo         "PYTHONPATH": "C:\\ai-hub\\agents",>> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo         "PERSISTENCE_KEY": "noelle_alek_persistence_7c4df9a8">> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo       }>> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo     }>> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo   }>> "%CLAUDE_CONFIG%\claude_desktop_config.json"
echo }>> "%CLAUDE_CONFIG%\claude_desktop_config.json"

echo [+] Claude configuration updated

echo.
echo [4/4] Testing MCP server...

:: Test the server
python -c "import mcp; print('[+] MCP module loaded successfully')" 2>nul
if %errorlevel% neq 0 (
    echo [!] MCP module not found
    echo     Please run: pip install mcp
) else (
    echo [+] MCP module verified
)

:: Test agent system import
cd /d C:\ai-hub\agents
python -c "import sys; sys.path.append('.'); from ultraagent_swarm_system import UltraAgentSwarm; print('[+] Agent system verified')" 2>nul
if %errorlevel% neq 0 (
    echo [!] Agent system not accessible
) else (
    echo [+] Agent system verified
)

echo.
echo ============================================================
echo  INSTALLATION COMPLETE
echo ============================================================
echo.
echo The agent system is now available in Claude Desktop!
echo.
echo Available MCP tools:
echo  - agent_ask      : Ask questions to specific agents
echo  - agent_swarm    : Activate collective intelligence
echo  - agent_spawn    : Create new agents
echo  - agent_list     : List active agents
echo  - agent_status   : Check system status
echo.
echo To use in Claude Desktop:
echo  1. Restart Claude Desktop
echo  2. Look for "agent-system" in the MCP tools
echo  3. Use tools like: "Use agent_ask to explore patterns"
echo.
echo Configuration location:
echo  %CLAUDE_CONFIG%\claude_desktop_config.json
echo.
echo To test server manually:
echo  python C:\ai-hub\agents\agent-mcp-server.py
echo ============================================================
echo.
pause