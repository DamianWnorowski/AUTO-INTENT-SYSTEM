#!/usr/bin/env python3
"""
PERMANENT COMMAND SYSTEM: Save All Agents as Executable Commands
================================================================
Converts all created agents into permanent reusable commands
"""

import os
import json
import subprocess
import sys
from typing import Dict, List, Any
from datetime import datetime

class PermanentCommandSystem:
    """System to convert all agents into permanent commands"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.agents = self._discover_agents()
        self.commands = {}
        
    def _discover_agents(self) -> List[Dict[str, str]]:
        """Discover all Python agent files"""
        agents = []
        
        agent_files = [
            ("ai-hyperagent-search.py", "aihyper", "Explore 550 AI interaction patterns"),
            ("complex-metric-pattern-finder.py", "patternfind", "Find patterns across 30 dimensions"),
            ("hyperpattern-audit-system.py", "audit", "Audit and validate patterns"),
            ("real-truth-test.py", "truthtest", "Verify all claims with 100% validation"),
            ("competitive-analysis.py", "compete", "Compare against other AI systems"),
            ("hyperultra-recursive-breaker.py", "breakproof", "Try to break proofs recursively"),
            ("ultimate-stress-test.py", "stresstest", "System stress testing"),
            ("prompt-restructurer.py", "prompt", "Restructure prompts for maximum effectiveness"),
            ("self-real-test.py", "selftest", "Test system on itself"),
            ("ultraverify-random-pov-challenger.py", "ultraverify", "Challenge from random perspectives"),
            ("megaswarm-ultraresearch.py", "megaswarm", "Swarm research verification"),
            ("total-self-improvement.py", "selfimprove", "Total self-improvement system"),
            ("agent-to-agent-explanation.py", "agentexplain", "Agents explain to each other"),
            ("consciousness-validation.py", "conscious", "Consciousness validation"),
            ("quantum-consciousness-bridge.py", "quantum", "Quantum consciousness bridge"),
            ("recursive-consciousness-system.py", "recursive", "Recursive consciousness"),
            ("master-agent-patterns.py", "master", "Master pattern recognition"),
            ("edge-case-testing.py", "edgetest", "Edge case testing"),
            ("pattern-effectiveness-test.py", "effective", "Pattern effectiveness"),
            ("self-test-visual-proof.py", "visualproof", "Visual proof of self-test")
        ]
        
        for filename, command, description in agent_files:
            if os.path.exists(filename):
                agents.append({
                    "file": filename,
                    "command": command,
                    "description": description,
                    "full_path": os.path.abspath(filename)
                })
        
        return agents
    
    def create_batch_file(self, agent: Dict[str, str]) -> str:
        """Create Windows batch file for agent"""
        batch_content = f"""@echo off
REM Auto-generated command for {agent['command']}
REM {agent['description']}

python "{agent['full_path']}" %*
"""
        
        batch_filename = f"{agent['command']}.bat"
        with open(batch_filename, 'w') as f:
            f.write(batch_content)
        
        return batch_filename
    
    def create_powershell_function(self, agent: Dict[str, str]) -> str:
        """Create PowerShell function for agent"""
        ps_function = f"""
function {agent['command']} {{
    <#
    .SYNOPSIS
    {agent['description']}
    
    .DESCRIPTION
    Runs {agent['file']} agent
    #>
    param(
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Arguments
    )
    
    python "{agent['full_path']}" $Arguments
}}
"""
        return ps_function
    
    def create_python_wrapper(self, agent: Dict[str, str]) -> str:
        """Create Python wrapper for agent"""
        wrapper_content = f'''#!/usr/bin/env python3
"""
Command wrapper for {agent['command']}
{agent['description']}
"""

import sys
import subprocess

def main():
    """Run the {agent['command']} agent"""
    subprocess.run([sys.executable, "{agent['full_path']}"] + sys.argv[1:])

if __name__ == "__main__":
    main()
'''
        
        wrapper_filename = f"cmd_{agent['command']}.py"
        with open(wrapper_filename, 'w') as f:
            f.write(wrapper_content)
        
        return wrapper_filename
    
    def create_command_registry(self) -> str:
        """Create central command registry"""
        registry = {
            "timestamp": self.timestamp,
            "commands": {},
            "total_agents": len(self.agents)
        }
        
        for agent in self.agents:
            registry["commands"][agent["command"]] = {
                "file": agent["file"],
                "path": agent["full_path"],
                "description": agent["description"],
                "batch_file": f"{agent['command']}.bat",
                "wrapper": f"cmd_{agent['command']}.py"
            }
        
        registry_file = "command_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        return registry_file
    
    def create_master_launcher(self) -> str:
        """Create master launcher script"""
        launcher_content = '''#!/usr/bin/env python3
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
    print("\\nAVAILABLE COMMANDS:")
    print("="*60)
    
    for cmd, info in registry["commands"].items():
        print(f"  {cmd:15} - {info['description']}")
    
    print("\\nUsage: python launcher.py <command> [args]")

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
'''
        
        launcher_file = "launcher.py"
        with open(launcher_file, 'w') as f:
            f.write(launcher_content)
        
        return launcher_file
    
    def create_quick_commands(self) -> str:
        """Create quick command shortcuts"""
        quick_content = """@echo off
REM Quick Commands for All Agents

echo QUICK COMMAND MENU
echo ==================
echo.
echo 1. aihyper      - AI Hyperagent Search
echo 2. patternfind  - Pattern Finder
echo 3. audit        - Audit System
echo 4. truthtest    - Truth Test
echo 5. compete      - Competitive Analysis
echo 6. breakproof   - Proof Breaker
echo 7. stresstest   - Stress Test
echo 8. prompt       - Prompt Restructurer
echo 9. selftest     - Self Test
echo 10. ultraverify - Ultra Verification
echo 11. megaswarm   - Megaswarm Research
echo 12. selfimprove - Self Improvement
echo 13. conscious   - Consciousness Validation
echo 14. quantum     - Quantum Consciousness
echo.
echo Usage: Type the command name to run
echo Example: aihyper
"""
        
        with open("quickmenu.bat", 'w') as f:
            f.write(quick_content)
        
        return "quickmenu.bat"
    
    def create_all_commands(self) -> Dict[str, Any]:
        """Create all command files"""
        print("Creating permanent commands for all agents...")
        print("="*60)
        
        results = {
            "batch_files": [],
            "python_wrappers": [],
            "powershell_functions": [],
            "registry": None,
            "launcher": None,
            "quickmenu": None
        }
        
        # Create individual command files
        for agent in self.agents:
            print(f"Creating command: {agent['command']}")
            
            # Batch file
            batch = self.create_batch_file(agent)
            results["batch_files"].append(batch)
            
            # Python wrapper
            wrapper = self.create_python_wrapper(agent)
            results["python_wrappers"].append(wrapper)
            
            # PowerShell function
            ps_func = self.create_powershell_function(agent)
            results["powershell_functions"].append(ps_func)
        
        # Create registry
        results["registry"] = self.create_command_registry()
        
        # Create launcher
        results["launcher"] = self.create_master_launcher()
        
        # Create quick menu
        results["quickmenu"] = self.create_quick_commands()
        
        # Create PowerShell profile
        ps_profile = "agent_commands.ps1"
        with open(ps_profile, 'w') as f:
            f.write("# PowerShell Functions for All Agents\n\n")
            for ps_func in results["powershell_functions"]:
                f.write(ps_func + "\n")
        
        print(f"\nCreated {len(self.agents)} permanent commands")
        
        return results
    
    def display_usage_instructions(self):
        """Display how to use the commands"""
        print("\n" + "="*80)
        print("PERMANENT COMMANDS CREATED SUCCESSFULLY")
        print("="*80)
        
        print("\n[USAGE OPTIONS]")
        print("-"*50)
        
        print("\n1. DIRECT BATCH COMMANDS (Windows):")
        print("   Just type the command name:")
        for agent in self.agents[:5]:
            print(f"   > {agent['command']}")
        
        print("\n2. PYTHON LAUNCHER:")
        print("   > python launcher.py list           # Show all commands")
        print("   > python launcher.py <command>      # Run specific command")
        print("   > python launcher.py aihyper        # Example")
        
        print("\n3. QUICK MENU:")
        print("   > quickmenu                         # Show interactive menu")
        
        print("\n4. POWERSHELL FUNCTIONS:")
        print("   > . .\\agent_commands.ps1           # Load all functions")
        print("   > aihyper                           # Run command")
        
        print("\n[AVAILABLE COMMANDS]")
        print("-"*50)
        
        for agent in self.agents:
            print(f"  {agent['command']:15} - {agent['description']}")
        
        print("\n[FILES CREATED]")
        print("-"*50)
        print(f"  - {len(self.agents)} .bat files (Windows batch commands)")
        print(f"  - {len(self.agents)} .py wrappers (Python command wrappers)")
        print(f"  - launcher.py (Master launcher)")
        print(f"  - command_registry.json (Command registry)")
        print(f"  - agent_commands.ps1 (PowerShell functions)")
        print(f"  - quickmenu.bat (Quick menu)")
        
        print("\n[PERMANENT STORAGE]")
        print("-"*50)
        print("All commands are now permanently available.")
        print("You can run any agent with a simple command.")
        print("Commands will persist across sessions.")

def main():
    print("PERMANENT COMMAND SYSTEM")
    print("Converting all agents to permanent executable commands...")
    print()
    
    # Initialize system
    cmd_system = PermanentCommandSystem()
    
    # Create all commands
    results = cmd_system.create_all_commands()
    
    # Display usage
    cmd_system.display_usage_instructions()
    
    # Save summary
    summary = {
        "timestamp": cmd_system.timestamp,
        "total_agents": len(cmd_system.agents),
        "commands_created": [a["command"] for a in cmd_system.agents],
        "files_created": {
            "batch_files": results["batch_files"],
            "python_wrappers": results["python_wrappers"],
            "registry": results["registry"],
            "launcher": results["launcher"],
            "quickmenu": results["quickmenu"]
        }
    }
    
    with open(f"command_creation_summary_{cmd_system.timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: command_creation_summary_{cmd_system.timestamp}.json")
    print("\nAll agents are now permanent commands!")

if __name__ == "__main__":
    main()