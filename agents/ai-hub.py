#!/usr/bin/env python3
"""
AI-HUB MASTER LAUNCHER
======================
Central launcher for all AI agents
"""

import os
import sys
import json
import subprocess

# Add AI-HUB to path
sys.path.insert(0, "C:/AI-HUB/agents/core")
sys.path.insert(0, "C:/AI-HUB/agents/commands")

def load_registry():
    """Load command registry from AI-HUB"""
    registry_path = "C:/AI-HUB/agents/commands/command_registry.json"
    with open(registry_path, 'r') as f:
        return json.load(f)

def list_commands():
    """List all available AI-HUB commands"""
    registry = load_registry()
    
    print("\n" + "="*60)
    print("AI-HUB COMMAND CENTER")
    print("="*60)
    print("\nAVAILABLE AGENTS:")
    
    for cmd, info in registry["commands"].items():
        print(f"  {cmd:15} - {info['description']}")
    
    print("\nUsage: python ai-hub.py <command> [args]")
    print("       python ai-hub.py list")

def run_command(command, args):
    """Run specified command from AI-HUB"""
    registry = load_registry()
    
    if command not in registry["commands"]:
        print(f"Unknown command: {command}")
        list_commands()
        return
    
    cmd_info = registry["commands"][command]
    script_path = cmd_info["path"]
    
    print(f"\nAI-HUB: Running {command}")
    print(f"Description: {cmd_info['description']}")
    print("-"*60)
    
    subprocess.run([sys.executable, script_path] + args)

def main():
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        list_commands()
    else:
        command = sys.argv[1]
        args = sys.argv[2:]
        run_command(command, args)

if __name__ == "__main__":
    main()
