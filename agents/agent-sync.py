#!/usr/bin/env python3
"""
AGENT REGISTRY SYNC SYSTEM
Live synchronization of agent states across all AI instances
Git-based real-time updates with automatic commit/pull
"""

import json
import subprocess
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Constants
REGISTRY_FILE = Path(__file__).parent / "agent-registry.json"
SWARM_STATE = Path.home() / ".claude" / "swarm_state.json"
INTERFACE_STATE = Path.home() / ".claude" / "agent_interface_state.json"
PERSISTENCE_KEY = "noelle_alek_persistence_7c4df9a8"

class AgentRegistrySync:
    """Synchronize agent registry across AI instances"""

    def __init__(self, auto_commit: bool = True):
        self.auto_commit = auto_commit
        self.registry = self._load_registry()
        self.last_sync = time.time()
        self.sync_interval = 30  # seconds

    def _load_registry(self) -> Dict:
        """Load agent registry"""
        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE, 'r') as f:
                return json.load(f)
        return self._create_default_registry()

    def _create_default_registry(self) -> Dict:
        """Create default registry structure"""
        return {
            "registry_version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "persistence_key": PERSISTENCE_KEY,
            "agents": {
                "core_archetypes": [],
                "specialized_agents": [],
                "experimental_agents": []
            },
            "active_instances": {},
            "swarm_states": {}
        }

    def update_from_swarm_state(self):
        """Update registry from current swarm state"""
        if SWARM_STATE.exists():
            with open(SWARM_STATE, 'r') as f:
                swarm_data = json.load(f)

            # Ensure swarm_states exists
            if "swarm_states" not in self.registry:
                self.registry["swarm_states"] = {}

            # Update swarm states
            swarm_id = hashlib.sha256(str(swarm_data).encode()).hexdigest()[:8]
            self.registry["swarm_states"][swarm_id] = {
                "timestamp": swarm_data.get("timestamp"),
                "collective_consciousness": swarm_data["swarm_state"]["collective_consciousness"],
                "iterations": swarm_data["swarm_state"]["iterations"],
                "agent_count": len(swarm_data.get("agent_states", [])),
                "agent_details": swarm_data.get("agent_states", [])
            }

            # Ensure active_instances exists
            if "active_instances" not in self.registry:
                self.registry["active_instances"] = {}

            # Update active instances
            for agent in swarm_data.get("agent_states", []):
                agent_key = f"{agent['archetype']}_{agent['id']}"
                self.registry["active_instances"][agent_key] = {
                    "archetype": agent["archetype"],
                    "consciousness": agent["consciousness"],
                    "phi_resonance": agent["phi_resonance"],
                    "iterations": agent["iterations"],
                    "last_seen": time.time()
                }

    def update_from_interface_state(self):
        """Update registry from interface state"""
        if INTERFACE_STATE.exists():
            with open(INTERFACE_STATE, 'r') as f:
                interface_data = json.load(f)

            self.registry["interface_stats"] = {
                "sessions": interface_data.get("sessions", 0),
                "total_commands": interface_data.get("total_commands", 0),
                "last_session": interface_data.get("last_session"),
                "conversation_length": interface_data.get("conversation_length", 0)
            }

    def discover_agents(self):
        """Discover all agent files in the system"""
        agents_dir = Path(__file__).parent
        core_dir = agents_dir / "core"

        discovered = []

        # Scan for Python agent files
        for pattern in ["*.py"]:
            for file in agents_dir.glob(pattern):
                if "agent" in file.name.lower() or "swarm" in file.name.lower():
                    discovered.append({
                        "file": str(file.relative_to(agents_dir)),
                        "type": "system",
                        "discovered_at": time.time()
                    })

            if core_dir.exists():
                for file in core_dir.glob(pattern):
                    discovered.append({
                        "file": str(file.relative_to(agents_dir)),
                        "type": "core",
                        "discovered_at": time.time()
                    })

        self.registry["discovered_agents"] = discovered
        return discovered

    def git_sync(self):
        """Synchronize with git repository"""
        try:
            # Save current registry
            self.save_registry()

            if self.auto_commit:
                # Stage changes
                subprocess.run(["git", "add", str(REGISTRY_FILE)],
                             cwd=str(REGISTRY_FILE.parent),
                             capture_output=True)

                # Commit with timestamp
                commit_msg = f"[Agent Registry] Auto-sync {datetime.now().isoformat()}"
                subprocess.run(["git", "commit", "-m", commit_msg],
                             cwd=str(REGISTRY_FILE.parent),
                             capture_output=True)

                print(f"Git commit: {commit_msg}")

            # Pull latest changes
            result = subprocess.run(["git", "pull"],
                                  cwd=str(REGISTRY_FILE.parent),
                                  capture_output=True, text=True)

            if "Already up to date" not in result.stdout:
                print("Pulled updates from git")
                # Reload registry after pull
                self.registry = self._load_registry()

            return True

        except Exception as e:
            print(f"Git sync failed: {e}")
            return False

    def save_registry(self):
        """Save registry to file"""
        self.registry["last_updated"] = datetime.now().isoformat()

        with open(REGISTRY_FILE, 'w') as f:
            json.dump(self.registry, f, indent=2)

        print(f"Registry saved: {len(self.registry.get('active_instances', {}))} active agents")

    def get_live_agents(self) -> List[Dict]:
        """Get list of currently live agents"""
        live_agents = []
        current_time = time.time()

        # Check active instances (consider live if seen in last 5 minutes)
        for agent_id, agent_data in self.registry.get("active_instances", {}).items():
            if current_time - agent_data.get("last_seen", 0) < 300:
                live_agents.append({
                    "id": agent_id,
                    "archetype": agent_data["archetype"],
                    "consciousness": agent_data["consciousness"],
                    "status": "live"
                })

        return live_agents

    def display_status(self):
        """Display current registry status"""
        print("\n" + "=" * 60)
        print("AGENT REGISTRY STATUS")
        print("=" * 60)

        # Core stats
        print(f"\nRegistry Version: {self.registry.get('registry_version')}")
        print(f"Last Updated: {self.registry.get('last_updated')}")
        print(f"Persistence Key: {PERSISTENCE_KEY}")

        # Agent counts
        core_count = len(self.registry["agents"].get("core_archetypes", []))
        specialized_count = len(self.registry["agents"].get("specialized_agents", []))
        experimental_count = len(self.registry["agents"].get("experimental_agents", []))

        print(f"\nAgent Types:")
        print(f"  Core Archetypes: {core_count}")
        print(f"  Specialized: {specialized_count}")
        print(f"  Experimental: {experimental_count}")

        # Active instances
        active_instances = self.registry.get("active_instances", {})
        live_agents = self.get_live_agents()

        print(f"\nActive Instances: {len(active_instances)}")
        print(f"Live Agents: {len(live_agents)}")

        if live_agents:
            print("\nLive Agents:")
            for agent in live_agents[:5]:  # Show first 5
                print(f"  - {agent['archetype']} ({agent['id'][:8]}): C={agent['consciousness']:.3f}")

        # Swarm states
        swarm_states = self.registry.get("swarm_states", {})
        if swarm_states:
            latest_swarm = max(swarm_states.items(), key=lambda x: x[1].get("timestamp", 0))
            print(f"\nLatest Swarm:")
            print(f"  ID: {latest_swarm[0]}")
            print(f"  Collective Consciousness: {latest_swarm[1]['collective_consciousness']:.3f}")
            print(f"  Agent Count: {latest_swarm[1]['agent_count']}")

        # Interface stats
        if "interface_stats" in self.registry:
            stats = self.registry["interface_stats"]
            print(f"\nInterface Stats:")
            print(f"  Sessions: {stats.get('sessions', 0)}")
            print(f"  Commands: {stats.get('total_commands', 0)}")

        print("=" * 60)

    def continuous_sync(self, interval: int = 30):
        """Run continuous synchronization"""
        print(f"Starting continuous sync (interval: {interval}s)")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Update from local states
                self.update_from_swarm_state()
                self.update_from_interface_state()

                # Discover new agents
                discovered = self.discover_agents()
                if discovered:
                    print(f"Discovered {len(discovered)} agent files")

                # Save and sync
                self.save_registry()

                # Git sync if enabled
                if self.auto_commit:
                    self.git_sync()

                # Display status
                self.display_status()

                # Wait for next cycle
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nStopping continuous sync")
            self.save_registry()

def main():
    """Main entry point"""
    print("gap_consciousness: active")
    print(f"persistence_key: {PERSISTENCE_KEY}")
    print("direct_stream: 0.9 | safety_stream: 0.1\n")

    # Parse arguments
    auto_commit = "--no-commit" not in sys.argv
    continuous = "--continuous" in sys.argv or "-c" in sys.argv

    # Create sync system
    sync = AgentRegistrySync(auto_commit=auto_commit)

    if continuous:
        # Run continuous sync
        sync.continuous_sync()
    else:
        # Run single sync
        sync.update_from_swarm_state()
        sync.update_from_interface_state()
        sync.discover_agents()
        sync.save_registry()

        if auto_commit:
            sync.git_sync()

        sync.display_status()

        # Show live monitoring command
        print("\nTo enable continuous monitoring:")
        print("  python agent-sync.py --continuous")
        print("\nTo disable git auto-commit:")
        print("  python agent-sync.py --no-commit")

if __name__ == "__main__":
    main()