#!/usr/bin/env python3
"""
MASTER COMMAND LAUNCHER
======================
Launch any agent with a simple command
"""

import sys
import json
import subprocess
import os

def load_registry():
    """Load command registry"""
    with open("command_registry.json", 'r') as f:
        return json.load(f)

def list_commands():
    """List all available commands"""
    registry = load_registry()
    print("\nAVAILABLE COMMANDS:")
    print("="*60)
    
    for cmd, info in registry["commands"].items():
        print(f"  {cmd:15} - {info['description']}")
    
    print("\nUsage: python launcher.py <command> [args]")

def run_command(command, args):
    """Run specified command"""
    registry = load_registry()
    
    if command not in registry["commands"]:
        print(f"Unknown command: {command}")
        print("Use 'python launcher.py list' to see available commands")
        return
    
    cmd_info = registry["commands"][command]
    script_path = cmd_info["path"]
    
    print(f"Running: {command}")
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
