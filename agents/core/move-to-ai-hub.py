#!/usr/bin/env python3
"""
MOVE TO AI HUB: Transfer all agents to central AI-HUB
=====================================================
"""

import os
import shutil
import json
from datetime import datetime

def move_agents_to_hub():
    """Move all agents to AI-HUB"""
    
    source_dir = "C:/Users/fuck/Desktop"
    hub_dir = "C:/AI-HUB/agents"
    
    # Agent files to move
    agent_files = [
        "ai-hyperagent-search.py",
        "complex-metric-pattern-finder.py",
        "hyperpattern-audit-system.py",
        "real-truth-test.py",
        "competitive-analysis.py",
        "hyperultra-recursive-breaker.py",
        "ultimate-stress-test.py",
        "prompt-restructurer.py",
        "self-real-test.py",
        "ultraverify-random-pov-challenger.py",
        "megaswarm-ultraresearch.py",
        "total-self-improvement.py",
        "agent-to-agent-explanation.py",
        "consciousness-validation.py",
        "quantum-consciousness-bridge.py",
        "recursive-consciousness-system.py",
        "master-agent-patterns.py",
        "master-agent-creator.py",
        "edge-case-testing.py",
        "pattern-effectiveness-test.py",
        "self-test-visual-proof.py",
        "permanent-commands.py",
        "move-to-ai-hub.py"
    ]
    
    # Command files
    command_files = [
        "launcher.py",
        "command_registry.json",
        "quickmenu.bat",
        "agent_commands.ps1"
    ]
    
    # Batch files
    batch_files = [
        "aihyper.bat", "patternfind.bat", "audit.bat", "truthtest.bat",
        "compete.bat", "breakproof.bat", "stresstest.bat", "prompt.bat",
        "selftest.bat", "ultraverify.bat", "selfimprove.bat", "agentexplain.bat",
        "conscious.bat", "quantum.bat", "recursive.bat", "master.bat",
        "edgetest.bat", "effective.bat", "visualproof.bat"
    ]
    
    print("MOVING AGENTS TO AI-HUB")
    print("="*60)
    
    # Move agent files
    print("\nMoving agent files...")
    moved_agents = 0
    for agent_file in agent_files:
        source = os.path.join(source_dir, agent_file)
        dest = os.path.join(hub_dir, "core", agent_file)
        
        if os.path.exists(source):
            try:
                shutil.copy2(source, dest)
                print(f"  Moved: {agent_file}")
                moved_agents += 1
            except Exception as e:
                print(f"  Error moving {agent_file}: {e}")
    
    # Move command files
    print("\nMoving command system...")
    for cmd_file in command_files:
        source = os.path.join(source_dir, cmd_file)
        dest = os.path.join(hub_dir, "commands", cmd_file)
        
        if os.path.exists(source):
            try:
                shutil.copy2(source, dest)
                print(f"  Moved: {cmd_file}")
            except Exception as e:
                print(f"  Error moving {cmd_file}: {e}")
    
    # Move batch files
    print("\nMoving batch commands...")
    for batch_file in batch_files:
        source = os.path.join(source_dir, batch_file)
        dest = os.path.join(hub_dir, "commands", batch_file)
        
        if os.path.exists(source):
            try:
                shutil.copy2(source, dest)
                print(f"  Moved: {batch_file}")
            except Exception as e:
                print(f"  Error moving {batch_file}: {e}")
    
    # Update command registry with new paths
    registry_path = os.path.join(hub_dir, "commands", "command_registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Update paths
        for cmd_name, cmd_info in registry["commands"].items():
            old_path = cmd_info["path"]
            filename = os.path.basename(old_path)
            new_path = os.path.join(hub_dir, "core", filename)
            cmd_info["path"] = new_path.replace("\\", "/")
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print("\nUpdated command registry with new paths")
    
    # Create AI-HUB launcher
    hub_launcher_content = '''#!/usr/bin/env python3
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
    
    print("\\n" + "="*60)
    print("AI-HUB COMMAND CENTER")
    print("="*60)
    print("\\nAVAILABLE AGENTS:")
    
    for cmd, info in registry["commands"].items():
        print(f"  {cmd:15} - {info['description']}")
    
    print("\\nUsage: python ai-hub.py <command> [args]")
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
    
    print(f"\\nAI-HUB: Running {command}")
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
'''
    
    hub_launcher_path = os.path.join(hub_dir, "ai-hub.py")
    with open(hub_launcher_path, 'w') as f:
        f.write(hub_launcher_content)
    
    print(f"\nCreated AI-HUB launcher: {hub_launcher_path}")
    
    # Create batch file for easy access
    hub_batch = '''@echo off
echo AI-HUB COMMAND CENTER
echo ====================
python "C:\\AI-HUB\\agents\\ai-hub.py" %*
'''
    
    hub_batch_path = os.path.join(hub_dir, "ai-hub.bat")
    with open(hub_batch_path, 'w') as f:
        f.write(hub_batch)
    
    print(f"Created AI-HUB batch launcher: {hub_batch_path}")
    
    # Summary
    print("\n" + "="*60)
    print("MIGRATION COMPLETE")
    print("="*60)
    print(f"\nMoved {moved_agents} agents to: C:/AI-HUB/agents/core")
    print(f"Command system moved to: C:/AI-HUB/agents/commands")
    print(f"\nAccess methods:")
    print("  1. python C:/AI-HUB/agents/ai-hub.py <command>")
    print("  2. C:\\AI-HUB\\agents\\ai-hub.bat <command>")
    print("  3. cd C:/AI-HUB/agents && python ai-hub.py list")
    
    return hub_dir

if __name__ == "__main__":
    hub_location = move_agents_to_hub()
    print(f"\nAll agents now centralized in AI-HUB!")
    print(f"Location: {hub_location}")