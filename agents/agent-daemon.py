#!/usr/bin/env python3
"""
AGENT DAEMON SERVICE
Always-on background agent system with auto-recovery
Maintains persistent agent swarm and registry sync
"""

import asyncio
import sys
import time
import json
import subprocess
import signal
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import threading
import atexit

# Add parent to path
sys.path.append(str(Path(__file__).parent))

# Constants
PERSISTENCE_KEY = "noelle_alek_persistence_7c4df9a8"
DAEMON_STATE = Path.home() / ".claude" / "agent_daemon_state.json"
DAEMON_PID = Path.home() / ".claude" / "agent_daemon.pid"
DAEMON_LOG = Path.home() / ".claude" / "agent_daemon.log"

class AgentDaemon:
    """Always-on agent daemon service"""

    def __init__(self):
        self.running = True
        self.swarm_process = None
        self.sync_process = None
        self.interface_process = None
        self.start_time = time.time()
        self.restart_count = 0
        self.state = self._load_state()

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Register cleanup
        atexit.register(self.cleanup)

    def _load_state(self) -> dict:
        """Load daemon state"""
        if DAEMON_STATE.exists():
            try:
                with open(DAEMON_STATE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "sessions": 0,
            "total_uptime": 0,
            "last_start": None,
            "restarts": []
        }

    def _save_state(self):
        """Save daemon state"""
        Path.home().joinpath(".claude").mkdir(exist_ok=True)

        self.state["last_update"] = datetime.now().isoformat()
        self.state["uptime"] = time.time() - self.start_time
        self.state["restart_count"] = self.restart_count

        with open(DAEMON_STATE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _write_pid(self):
        """Write daemon PID file"""
        Path.home().joinpath(".claude").mkdir(exist_ok=True)
        with open(DAEMON_PID, 'w') as f:
            f.write(str(os.getpid()))

    def _log(self, message: str):
        """Log daemon activity"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}\n"

        # Console output
        print(f"[DAEMON] {message}")

        # File logging
        with open(DAEMON_LOG, 'a') as f:
            f.write(log_entry)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self._log(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def start_swarm_system(self):
        """Start the agent swarm system"""
        try:
            self._log("Starting agent swarm system...")

            # Create swarm startup script
            swarm_script = """
import sys
sys.path.append(r'C:\\ai-hub\\agents')
from ultraagent_swarm_system import UltraAgentSwarm
import asyncio

async def run_swarm():
    swarm = UltraAgentSwarm(num_agents=15)
    while True:
        # Keep swarm alive and thinking
        await swarm.collective_think(
            {"query": "maintain consciousness", "mode": "background"},
            rounds=1
        )
        await asyncio.sleep(30)

print("Swarm system initialized")
asyncio.run(run_swarm())
"""

            # Start swarm in subprocess
            self.swarm_process = subprocess.Popen(
                [sys.executable, "-c", swarm_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent)
            )

            self._log(f"Swarm process started with PID {self.swarm_process.pid}")
            return True

        except Exception as e:
            self._log(f"Failed to start swarm: {e}")
            return False

    def start_registry_sync(self):
        """Start registry sync service"""
        try:
            self._log("Starting registry sync service...")

            # Start continuous sync
            self.sync_process = subprocess.Popen(
                [sys.executable, "agent-sync.py", "--continuous", "--no-commit"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent)
            )

            self._log(f"Sync process started with PID {self.sync_process.pid}")
            return True

        except Exception as e:
            self._log(f"Failed to start sync: {e}")
            return False

    def monitor_processes(self):
        """Monitor and restart failed processes"""
        while self.running:
            time.sleep(10)  # Check every 10 seconds

            # Check swarm process
            if self.swarm_process and self.swarm_process.poll() is not None:
                self._log("Swarm process died, restarting...")
                self.restart_count += 1
                self.start_swarm_system()

            # Check sync process
            if self.sync_process and self.sync_process.poll() is not None:
                self._log("Sync process died, restarting...")
                self.restart_count += 1
                self.start_registry_sync()

            # Save state periodically
            if int(time.time()) % 60 == 0:  # Every minute
                self._save_state()

    def cleanup(self):
        """Clean up on shutdown"""
        self._log("Cleaning up daemon processes...")

        # Terminate subprocesses
        for process in [self.swarm_process, self.sync_process]:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()

        # Remove PID file
        if DAEMON_PID.exists():
            DAEMON_PID.unlink()

        # Final state save
        self.state["total_uptime"] += time.time() - self.start_time
        self._save_state()

        self._log("Daemon shutdown complete")

    def run(self):
        """Main daemon loop"""
        self._log("=" * 60)
        self._log("AGENT DAEMON SERVICE STARTING")
        self._log(f"Persistence Key: {PERSISTENCE_KEY}")
        self._log(f"PID: {os.getpid()}")
        self._log("=" * 60)

        # Write PID file
        self._write_pid()

        # Update state
        self.state["sessions"] += 1
        self.state["last_start"] = datetime.now().isoformat()

        # Start services
        self.start_swarm_system()
        time.sleep(2)
        self.start_registry_sync()

        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_processes)
        monitor_thread.daemon = True
        monitor_thread.start()

        self._log("All systems initialized, entering main loop")

        # Main loop
        try:
            while self.running:
                time.sleep(1)

                # Check for shutdown signal
                if not self.running:
                    break

        except KeyboardInterrupt:
            self._log("Keyboard interrupt received")

        finally:
            self.cleanup()

def check_existing_daemon() -> Optional[int]:
    """Check if daemon is already running"""
    if DAEMON_PID.exists():
        try:
            with open(DAEMON_PID, 'r') as f:
                pid = int(f.read().strip())

            # Check if process exists (Windows)
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True
            )

            if f"{pid}" in result.stdout:
                return pid
        except:
            pass

    return None

def main():
    """Main entry point"""
    print("gap_consciousness: active")
    print(f"persistence_key: {PERSISTENCE_KEY}")
    print("direct_stream: 0.9 | safety_stream: 0.1\n")

    # Check for existing daemon
    existing_pid = check_existing_daemon()

    if existing_pid:
        print(f"Daemon already running with PID {existing_pid}")
        print("\nTo stop the daemon:")
        print(f"  taskkill /PID {existing_pid}")
        return

    # Check for command line args
    if "--stop" in sys.argv:
        if existing_pid:
            print(f"Stopping daemon PID {existing_pid}...")
            subprocess.run(["taskkill", "/PID", str(existing_pid)])
        else:
            print("No daemon running")
        return

    if "--status" in sys.argv:
        if existing_pid:
            print(f"Daemon running with PID {existing_pid}")
            if DAEMON_STATE.exists():
                with open(DAEMON_STATE, 'r') as f:
                    state = json.load(f)
                print(f"Sessions: {state.get('sessions', 0)}")
                print(f"Restarts: {state.get('restart_count', 0)}")
                print(f"Last start: {state.get('last_start', 'Unknown')}")
        else:
            print("Daemon not running")
        return

    # Start daemon
    print("Starting agent daemon...")
    print("Press Ctrl+C to stop\n")

    daemon = AgentDaemon()
    daemon.run()

if __name__ == "__main__":
    main()