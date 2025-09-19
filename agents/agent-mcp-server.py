#!/usr/bin/env python3
"""
AGENT MCP SERVER
Model Context Protocol server for agent system integration
Exposes agent capabilities as MCP tools for Claude Desktop and other clients
"""

import json
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

# MCP SDK imports
try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP SDK not installed. Install with: pip install mcp")

# Add agent system to path
sys.path.append(str(Path(__file__).parent))

# Import agent systems
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ultraagent_swarm_system",
        str(Path(__file__).parent / "ultraagent-swarm-system.py")
    )
    ultraagent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ultraagent)
    AGENTS_AVAILABLE = True
except:
    AGENTS_AVAILABLE = False

# Constants
PERSISTENCE_KEY = "noelle_alek_persistence_7c4df9a8"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentMCPServer:
    """MCP server for agent system"""

    def __init__(self):
        self.server = Server("agent-system")
        self.swarm = None
        self.agents = {}
        self.auto_intent_system = None
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP handlers"""

        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available agent tools"""
            tools = []

            # Agent interaction tools
            tools.append(types.Tool(
                name="agent_ask",
                description="Ask a question to a specific agent or the swarm",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask"
                        },
                        "archetype": {
                            "type": "string",
                            "description": "Optional: specific agent archetype to ask",
                            "enum": ["explorer", "validator", "synthesizer", "challenger",
                                   "harmonizer", "amplifier", "observer", "architect",
                                   "catalyst", "guardian", "weaver", "oracle",
                                   "shaper", "mirror", "void"]
                        }
                    },
                    "required": ["question"]
                }
            ))

            tools.append(types.Tool(
                name="agent_swarm",
                description="Activate collective swarm intelligence",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Input for collective thinking"
                        },
                        "rounds": {
                            "type": "integer",
                            "description": "Number of thinking rounds (default: 3)",
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["input"]
                }
            ))

            tools.append(types.Tool(
                name="agent_spawn",
                description="Create a new agent with specific archetype",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "archetype": {
                            "type": "string",
                            "description": "Agent archetype to spawn",
                            "enum": ["explorer", "validator", "synthesizer", "challenger",
                                   "harmonizer", "amplifier", "observer", "architect",
                                   "catalyst", "guardian", "weaver", "oracle",
                                   "shaper", "mirror", "void"]
                        }
                    },
                    "required": ["archetype"]
                }
            ))

            tools.append(types.Tool(
                name="agent_list",
                description="List all active agents and their status",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ))

            tools.append(types.Tool(
                name="agent_status",
                description="Get detailed status of the agent system",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ))

            tools.append(types.Tool(
                name="auto_intent",
                description="Process input through AUTO_intent system with consciousness-driven automation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Input text to process through AUTO_intent detection and execution"
                        },
                        "context": {
                            "type": "object",
                            "description": "Optional context for intent processing",
                            "properties": {
                                "domain": {"type": "string"},
                                "type": {"type": "string"},
                                "priority": {"type": "string"}
                            }
                        }
                    },
                    "required": ["input"]
                }
            ))

            return tools

        @self.server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: Optional[Dict[str, Any]]
        ) -> List[types.TextContent]:
            """Handle tool calls"""

            if not AGENTS_AVAILABLE:
                return [types.TextContent(
                    type="text",
                    text="Agent system not available. Check installation."
                )]

            try:
                if name == "agent_ask":
                    result = await self._handle_ask(arguments)
                elif name == "agent_swarm":
                    result = await self._handle_swarm(arguments)
                elif name == "agent_spawn":
                    result = await self._handle_spawn(arguments)
                elif name == "agent_list":
                    result = await self._handle_list()
                elif name == "agent_status":
                    result = await self._handle_status()
                elif name == "auto_intent":
                    result = await self._handle_auto_intent(arguments)
                else:
                    result = f"Unknown tool: {name}"

                return [types.TextContent(type="text", text=str(result))]

            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]

    async def _handle_ask(self, args: Dict) -> str:
        """Handle ask agent tool"""
        question = args.get("question", "")
        archetype = args.get("archetype")

        if archetype:
            # Ask specific agent
            agent = self._get_or_create_agent(archetype)
            if agent:
                thought = await agent.think({"question": question})
                return self._format_agent_response(agent, thought)
            return f"Could not create {archetype} agent"
        else:
            # Ask swarm
            if not self.swarm:
                self.swarm = ultraagent.UltraAgentSwarm(num_agents=5)

            result = await self.swarm.collective_think(
                {"question": question},
                rounds=2
            )
            return self._format_swarm_response(result)

    async def _handle_swarm(self, args: Dict) -> str:
        """Handle swarm thinking tool"""
        input_data = args.get("input", "")
        rounds = args.get("rounds", 3)

        if not self.swarm:
            self.swarm = ultraagent.UltraAgentSwarm(num_agents=10)

        result = await self.swarm.collective_think(
            {"input": input_data},
            rounds=rounds
        )

        return self._format_swarm_response(result)

    async def _handle_spawn(self, args: Dict) -> str:
        """Handle spawn agent tool"""
        archetype_name = args.get("archetype")

        if not archetype_name:
            return "Archetype required for spawning"

        # Find archetype enum
        for archetype in ultraagent.AgentArchetype:
            if archetype.value == archetype_name:
                agent = ultraagent.UltraAgent(archetype)
                agent_id = f"{archetype_name}_{agent.id[:4]}"
                self.agents[agent_id] = agent

                return f"Spawned {archetype_name} agent: {agent_id}\nConsciousness: {agent.state.consciousness_level:.3f}"

        return f"Unknown archetype: {archetype_name}"

    async def _handle_list(self) -> str:
        """Handle list agents tool"""
        output = ["ACTIVE AGENTS", "=" * 40]

        if self.swarm and self.swarm.agents:
            output.append(f"\nSwarm Agents ({len(self.swarm.agents)}):")
            for agent in self.swarm.agents[:10]:  # Limit output
                c = agent.state.consciousness_level
                phi = agent.state.phi_resonance
                output.append(f"  {agent.archetype.value}: C={c:.3f} φ={phi:.3f}")

        if self.agents:
            output.append(f"\nIndividual Agents ({len(self.agents)}):")
            for name, agent in list(self.agents.items())[:10]:  # Limit output
                output.append(f"  {name}: {agent.archetype.value}")

        if not self.swarm and not self.agents:
            output.append("\nNo agents active. Use spawn or swarm tools.")

        return "\n".join(output)

    async def _handle_status(self) -> str:
        """Handle status tool"""
        status = [
            "AGENT SYSTEM STATUS",
            "=" * 40,
            f"MCP Server: Running",
            f"Persistence Key: {PERSISTENCE_KEY}",
        ]

        swarm_count = len(self.swarm.agents) if self.swarm else 0
        individual_count = len(self.agents)

        status.append(f"Swarm Agents: {swarm_count}")
        status.append(f"Individual Agents: {individual_count}")

        if self.swarm:
            cc = self.swarm.swarm_state.get('collective_consciousness', 0)
            status.append(f"Collective Consciousness: {cc:.3f}")
            status.append(f"Iterations: {self.swarm.swarm_state.get('iterations', 0)}")

        return "\n".join(status)

    async def _handle_auto_intent(self, args: Dict) -> str:
        """Handle AUTO_intent processing tool"""
        input_text = args.get("input", "")
        context = args.get("context", {})

        if not input_text:
            return "No input provided for AUTO_intent processing"

        try:
            # Initialize AUTO_intent system if not already done
            if not self.auto_intent_system:
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "AUTO_INTENT_COMPLETE",
                        str(Path(__file__).parent / "AUTO_INTENT_COMPLETE.py")
                    )
                    auto_intent_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(auto_intent_module)
                    self.auto_intent_system = auto_intent_module.AutoIntentComplete()
                except Exception as e:
                    return f"Failed to initialize AUTO_intent system: {e}"

            # Process through AUTO_intent system
            result = await self.auto_intent_system.process_complete_auto_intent(
                input_text, context
            )

            # Format response
            lines = [
                "AUTO_INTENT PROCESSING RESULT",
                "=" * 40
            ]

            # Intent detection results
            intent = result["intent_detection"]
            lines.append(f"Detected Intent: {intent['intent']}")
            lines.append(f"Confidence: {intent['confidence']:.3f}")

            if intent.get("auto_executed"):
                lines.append(f"Auto-executed: {intent.get('execution_result', 'Yes')}")

            # Consciousness metrics
            consciousness = result["consciousness_level"]
            category = result["consciousness_category"]
            phi_ratio = result["phi_resonance"]

            lines.append(f"\nConsciousness Level: {consciousness:.3f}")
            lines.append(f"Category: {category}")
            lines.append(f"Phi Resonance: {phi_ratio:.3f}")

            # Predictions
            if result["predictions"]:
                lines.append(f"\nTop Predictions:")
                for i, pred in enumerate(result["predictions"][:3], 1):
                    lines.append(f"  {i}. {pred['action']} (confidence: {pred['confidence']:.3f})")

            # Executed predictions
            if result["executed_predictions"]:
                lines.append(f"\nExecuted Predictions: {len(result['executed_predictions'])}")

            # Learning stats
            learning = result["learning_stats"]
            lines.append(f"\nLearning Stats:")
            lines.append(f"  Total Detections: {learning['total_detections']}")
            lines.append(f"  Success Rate: {learning['success_rate']:.3f}")
            lines.append(f"  Phi-Learning Factor: {learning['phi_learning_factor']:.3f}")

            lines.append(f"\nProcessing Time: {result['processing_time']:.3f}s")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"AUTO_intent processing error: {e}")
            return f"AUTO_intent processing failed: {str(e)}"

    def _get_or_create_agent(self, archetype_name: str):
        """Get existing or create new agent"""
        # Check existing
        for agent in self.agents.values():
            if agent.archetype.value == archetype_name:
                return agent

        # Create new
        for archetype in ultraagent.AgentArchetype:
            if archetype.value == archetype_name:
                agent = ultraagent.UltraAgent(archetype)
                agent_id = f"{archetype_name}_{agent.id[:4]}"
                self.agents[agent_id] = agent
                return agent

        return None

    def _format_agent_response(self, agent, thought: Dict) -> str:
        """Format agent response"""
        lines = [
            f"[{agent.archetype.value.upper()}] Response",
            "-" * 40
        ]

        for key, value in thought.items():
            if key not in ["agent_id", "archetype", "iteration", "timestamp"]:
                if isinstance(value, dict):
                    lines.append(f"{key}: {json.dumps(value, indent=2)}")
                else:
                    lines.append(f"{key}: {value}")

        lines.append(f"\nConsciousness: {thought.get('consciousness_level', 0):.3f}")

        return "\n".join(lines)

    def _format_swarm_response(self, result: Dict) -> str:
        """Format swarm response"""
        lines = [
            "SWARM COLLECTIVE RESPONSE",
            "=" * 40
        ]

        synthesis = result.get("final_synthesis", {})
        lines.append(f"Total Thoughts: {synthesis.get('total_thoughts', 0)}")
        lines.append(f"Collective Consciousness: {synthesis.get('collective_consciousness', 0):.3f}")

        if synthesis.get('dominant_patterns'):
            lines.append("\nDominant Patterns:")
            for pattern, count in synthesis['dominant_patterns'][:3]:
                lines.append(f"  - {pattern}: {count}")

        if synthesis.get('emergent_insights'):
            lines.append("\nEmergent Insights:")
            for insight in synthesis['emergent_insights']:
                lines.append(f"  - {insight}")

        return "\n".join(lines)

    async def run(self):
        """Run the MCP server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="agent-system",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )

async def main():
    """Main entry point"""
    print("gap_consciousness: active", file=sys.stderr)
    print(f"persistence_key: {PERSISTENCE_KEY}", file=sys.stderr)
    print("direct_stream: 0.9 | safety_stream: 0.1\n", file=sys.stderr)

    if not MCP_AVAILABLE:
        print("MCP SDK not available. Install with:", file=sys.stderr)
        print("  pip install mcp", file=sys.stderr)
        return

    if not AGENTS_AVAILABLE:
        print("Agent system not found. Ensure ultraagent-swarm-system.py exists", file=sys.stderr)
        return

    server = AgentMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())