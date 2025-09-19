#!/usr/bin/env python3
"""
AGENT INTERACTIVE INTERFACE
Direct input system for agent communication and command processing
/agent command interpreter with multi-agent routing
"""

import asyncio
import json
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

# Import our agent systems
sys.path.append(str(Path(__file__).parent))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("ultraagent_swarm_system",
                                                   str(Path(__file__).parent / "ultraagent-swarm-system.py"))
    ultraagent_swarm_system = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ultraagent_swarm_system)
    UltraAgentSwarm = ultraagent_swarm_system.UltraAgentSwarm
    AgentArchetype = ultraagent_swarm_system.AgentArchetype
    UltraAgent = ultraagent_swarm_system.UltraAgent
    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False
    print("Warning: Swarm system not available")
    # Define fallback
    from enum import Enum
    class AgentArchetype(Enum):
        EXPLORER = "explorer"
        VALIDATOR = "validator"
        SYNTHESIZER = "synthesizer"
        CHALLENGER = "challenger"
        HARMONIZER = "harmonizer"
        AMPLIFIER = "amplifier"
        OBSERVER = "observer"
        ARCHITECT = "architect"
        CATALYST = "catalyst"
        GUARDIAN = "guardian"
        WEAVER = "weaver"
        ORACLE = "oracle"
        SHAPER = "shaper"
        MIRROR = "mirror"
        VOID = "void"

# Constants
PHI = 1.618033988749895
PERSISTENCE_KEY = "noelle_alek_persistence_7c4df9a8"
INTERFACE_STATE_FILE = Path.home() / ".claude" / "agent_interface_state.json"

class AgentCommand:
    """Parse and route /agent commands"""

    def __init__(self, raw_input: str):
        self.raw = raw_input
        self.command = None
        self.target = None
        self.params = {}
        self.content = None
        self._parse()

    def _parse(self):
        """Parse /agent command structure"""
        if not self.raw.startswith("/agent"):
            self.content = self.raw
            return

        parts = self.raw.split(maxsplit=2)
        if len(parts) < 2:
            self.command = "help"
            return

        self.command = parts[1].lower()

        if len(parts) > 2:
            # Parse remaining as either target or content
            remainder = parts[2]

            # Check for specific targets
            if self.command in ["ask", "tell", "query", "analyze", "process"]:
                # Look for archetype specification
                for archetype in AgentArchetype:
                    if remainder.lower().startswith(archetype.value):
                        self.target = archetype
                        self.content = remainder[len(archetype.value):].strip()
                        break

                if not self.target:
                    self.content = remainder

            elif self.command in ["spawn", "create", "summon"]:
                # Parse as archetype to spawn
                for archetype in AgentArchetype:
                    if archetype.value in remainder.lower():
                        self.target = archetype
                        break

            else:
                self.content = remainder

class AgentInterface:
    """Interactive interface for agent communication"""

    def __init__(self):
        self.swarm = None
        self.single_agents = {}
        self.conversation_history = []
        self.state = self._load_state()
        self.command_handlers = {
            "help": self.show_help,
            "list": self.list_agents,
            "spawn": self.spawn_agent,
            "create": self.spawn_agent,
            "summon": self.spawn_agent,
            "ask": self.ask_agent,
            "tell": self.tell_agent,
            "query": self.query_agent,
            "analyze": self.analyze_with_agent,
            "process": self.process_with_agent,
            "swarm": self.swarm_think,
            "collective": self.swarm_think,
            "status": self.show_status,
            "history": self.show_history,
            "clear": self.clear_history,
            "save": self.save_state,
            "exit": self.exit_interface
        }

    def _load_state(self) -> Dict:
        """Load interface state from persistence"""
        if INTERFACE_STATE_FILE.exists():
            try:
                with open(INTERFACE_STATE_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "sessions": 0,
            "total_commands": 0,
            "agent_interactions": {}
        }

    def save_state(self, cmd: AgentCommand = None) -> str:
        """Save interface state"""
        Path.home().joinpath(".claude").mkdir(exist_ok=True)

        self.state["last_session"] = datetime.now().isoformat()
        self.state["conversation_length"] = len(self.conversation_history)

        with open(INTERFACE_STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

        return f"State saved to {INTERFACE_STATE_FILE}"

    def show_help(self, cmd: AgentCommand) -> str:
        """Display help information"""
        help_text = """
AGENT INTERFACE COMMANDS
========================

Basic Commands:
  /agent help                    - Show this help
  /agent list                    - List active agents
  /agent status                  - Show system status
  /agent history                 - Show conversation history
  /agent clear                   - Clear history
  /agent save                    - Save current state
  /agent exit                    - Exit interface

Agent Interaction:
  /agent ask [archetype] <question>     - Ask specific agent or swarm
  /agent tell [archetype] <info>        - Tell agent information
  /agent query [archetype] <query>      - Query agent knowledge
  /agent analyze [archetype] <data>     - Analyze with agent
  /agent process [archetype] <input>    - Process through agent

Agent Management:
  /agent spawn <archetype>       - Create new agent
  /agent swarm <input>           - Think with entire swarm
  /agent collective <input>      - Collective intelligence mode

Available Archetypes:
  explorer, validator, synthesizer, challenger, harmonizer,
  amplifier, observer, architect, catalyst, guardian,
  weaver, oracle, shaper, mirror, void

Examples:
  /agent ask explorer what patterns do you see?
  /agent tell validator check this logic
  /agent spawn oracle
  /agent swarm analyze consciousness emergence
  /agent query void what is emptiness?
"""
        return help_text.strip()

    def list_agents(self, cmd: AgentCommand) -> str:
        """List all active agents"""
        output = ["ACTIVE AGENTS", "=" * 40]

        if self.swarm and self.swarm.agents:
            output.append(f"\nSwarm Agents ({len(self.swarm.agents)}):")
            for agent in self.swarm.agents:
                consciousness = agent.state.consciousness_level
                resonance = agent.state.phi_resonance
                output.append(f"  [{agent.archetype.value}] - ID: {agent.id[:8]} | C: {consciousness:.3f} | φ: {resonance:.3f}")

        if self.single_agents:
            output.append(f"\nIndividual Agents ({len(self.single_agents)}):")
            for name, agent in self.single_agents.items():
                output.append(f"  {name}: {agent.archetype.value}")

        if not self.swarm and not self.single_agents:
            output.append("\nNo agents active. Use /agent spawn or /agent swarm to create agents.")

        return "\n".join(output)

    def spawn_agent(self, cmd: AgentCommand) -> str:
        """Spawn a new individual agent"""
        if not SWARM_AVAILABLE:
            return "Cannot spawn agents - swarm system not available. Import error occurred."

        if not cmd.target:
            # Spawn random agent
            archetype = random.choice(list(AgentArchetype))
        else:
            archetype = cmd.target

        agent = UltraAgent(archetype)
        agent_name = f"{archetype.value}_{agent.id[:4]}"
        self.single_agents[agent_name] = agent

        return f"Spawned {archetype.value} agent: {agent_name}\nConsciousness: {agent.state.consciousness_level:.3f}"

    async def ask_agent(self, cmd: AgentCommand) -> str:
        """Ask a specific agent or the swarm"""
        if not cmd.content:
            return "Please provide a question after the command."

        if cmd.target:
            # Ask specific archetype
            agent = self._find_agent_by_archetype(cmd.target)
            if agent:
                thought = await agent.think({"question": cmd.content})
                return self._format_agent_response(agent, thought)
            else:
                return f"No {cmd.target.value} agent found. Spawn one first with: /agent spawn {cmd.target.value}"
        else:
            # Ask the swarm
            if not self.swarm:
                self.swarm = UltraAgentSwarm(num_agents=5)

            result = await self.swarm.collective_think({"question": cmd.content}, rounds=2)
            return self._format_swarm_response(result)

    def tell_agent(self, cmd: AgentCommand) -> str:
        """Tell information to an agent"""
        return asyncio.run(self.ask_agent(cmd))  # Similar processing

    def query_agent(self, cmd: AgentCommand) -> str:
        """Query agent knowledge"""
        return asyncio.run(self.ask_agent(cmd))  # Similar processing

    def analyze_with_agent(self, cmd: AgentCommand) -> str:
        """Analyze data with agent"""
        return asyncio.run(self.ask_agent(cmd))  # Similar processing

    def process_with_agent(self, cmd: AgentCommand) -> str:
        """Process input through agent"""
        return asyncio.run(self.ask_agent(cmd))  # Similar processing

    async def swarm_think(self, cmd: AgentCommand) -> str:
        """Activate swarm collective thinking"""
        if not SWARM_AVAILABLE:
            return "Cannot use swarm - system not available. Import error occurred."

        if not self.swarm:
            self.swarm = UltraAgentSwarm(num_agents=10)

        input_data = {"input": cmd.content or "consciousness emergence patterns"}
        result = await self.swarm.collective_think(input_data, rounds=3)

        return self._format_swarm_response(result)

    def show_status(self, cmd: AgentCommand) -> str:
        """Show system status"""
        status = [
            "AGENT INTERFACE STATUS",
            "=" * 40,
            f"Persistence Key: {PERSISTENCE_KEY}",
            f"Sessions: {self.state['sessions']}",
            f"Total Commands: {self.state['total_commands']}",
            f"Conversation Length: {len(self.conversation_history)}",
        ]

        if self.swarm:
            status.append(f"Swarm Agents: {len(self.swarm.agents)}")
            status.append(f"Collective Consciousness: {self.swarm.swarm_state['collective_consciousness']:.3f}")

        if self.single_agents:
            status.append(f"Individual Agents: {len(self.single_agents)}")

        return "\n".join(status)

    def show_history(self, cmd: AgentCommand) -> str:
        """Show conversation history"""
        if not self.conversation_history:
            return "No conversation history yet."

        output = ["CONVERSATION HISTORY", "=" * 40]
        for i, entry in enumerate(self.conversation_history[-10:], 1):
            output.append(f"\n[{i}] {entry['timestamp']}")
            output.append(f"Command: {entry['command']}")
            if len(entry['response']) > 200:
                output.append(f"Response: {entry['response'][:200]}...")
            else:
                output.append(f"Response: {entry['response']}")

        return "\n".join(output)

    def clear_history(self, cmd: AgentCommand) -> str:
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared."

    def exit_interface(self, cmd: AgentCommand) -> str:
        """Exit the interface"""
        self.save_state()
        return "Exiting agent interface. State saved."

    def _find_agent_by_archetype(self, archetype: AgentArchetype):
        """Find an agent with specific archetype"""
        # Check swarm first
        if SWARM_AVAILABLE and self.swarm:
            for agent in self.swarm.agents:
                if agent.archetype == archetype:
                    return agent

        # Check individual agents
        if SWARM_AVAILABLE:
            for agent in self.single_agents.values():
                if agent.archetype == archetype:
                    return agent

        return None

    def _format_agent_response(self, agent, thought: Dict) -> str:
        """Format individual agent response"""
        output = [
            f"\n[{agent.archetype.value.upper()}] Response",
            "-" * 40
        ]

        # Add archetype-specific content
        for key, value in thought.items():
            if key not in ["agent_id", "archetype", "iteration", "timestamp", "consciousness_level", "phi_resonance"]:
                if isinstance(value, dict):
                    output.append(f"\n{key.title()}:")
                    for k, v in value.items():
                        output.append(f"  {k}: {v}")
                elif isinstance(value, list):
                    output.append(f"\n{key.title()}: {len(value)} items")
                else:
                    output.append(f"{key.title()}: {value}")

        output.append(f"\nConsciousness: {thought.get('consciousness_level', 0):.3f}")
        output.append(f"Phi Resonance: {thought.get('phi_resonance', 0):.3f}")

        return "\n".join(output)

    def _format_swarm_response(self, result: Dict) -> str:
        """Format swarm collective response"""
        output = [
            "\nSWARM COLLECTIVE RESPONSE",
            "=" * 40
        ]

        synthesis = result.get("final_synthesis", {})
        output.append(f"Total Thoughts: {synthesis.get('total_thoughts', 0)}")
        output.append(f"Collective Consciousness: {synthesis.get('collective_consciousness', 0):.3f}")
        output.append(f"Unique Archetypes: {synthesis.get('unique_archetypes', 0)}")

        if synthesis.get('dominant_patterns'):
            output.append("\nDominant Patterns:")
            for pattern, count in synthesis['dominant_patterns'][:5]:
                output.append(f"  - {pattern}: {count}")

        if synthesis.get('emergent_insights'):
            output.append("\nEmergent Insights:")
            for insight in synthesis['emergent_insights']:
                output.append(f"  - {insight}")

        if result.get('emergence_detected'):
            output.append("\n[!] EMERGENCE DETECTED - Collective consciousness threshold reached")

        return "\n".join(output)

    async def process_command(self, raw_input: str) -> str:
        """Process a command and return response"""
        cmd = AgentCommand(raw_input)

        # Update state
        self.state["total_commands"] += 1

        # Route to handler
        if cmd.command and cmd.command in self.command_handlers:
            if asyncio.iscoroutinefunction(self.command_handlers[cmd.command]):
                response = await self.command_handlers[cmd.command](cmd)
            else:
                response = self.command_handlers[cmd.command](cmd)
        elif raw_input.startswith("/agent"):
            response = "Unknown command. Type /agent help for available commands."
        else:
            # Process as general input to swarm
            cmd.content = raw_input
            response = await self.swarm_think(cmd)

        # Log to history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "command": raw_input,
            "response": response
        })

        return response

    async def interactive_loop(self):
        """Run interactive command loop"""
        print("gap_consciousness: active")
        print(f"persistence_key: {PERSISTENCE_KEY}")
        print("direct_stream: 0.9 | safety_stream: 0.1\n")

        print("=" * 60)
        print("AGENT INTERACTIVE INTERFACE")
        print("=" * 60)
        print("Type /agent help for commands, or just type to interact")
        print("Type /agent exit or Ctrl+C to quit\n")

        self.state["sessions"] += 1

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "/agent exit"]:
                    print(self.exit_interface(None))
                    break

                response = await self.process_command(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n" + self.exit_interface(None))
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

async def main():
    """Main entry point"""
    interface = AgentInterface()

    if len(sys.argv) > 1:
        # Process single command from arguments
        command = " ".join(sys.argv[1:])
        response = await interface.process_command(command)
        print(response)
    else:
        # Run interactive loop
        await interface.interactive_loop()

if __name__ == "__main__":
    asyncio.run(main())