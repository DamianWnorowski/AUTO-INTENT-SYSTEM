#!/usr/bin/env python3
"""
AUTO_INTENT INTEGRATION BRIDGE
Connects auto-intent detection with agent interactive interface
Enables seamless automatic execution of detected user intentions
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add agent systems to path
sys.path.append(str(Path(__file__).parent))

# Constants
PHI = 1.618033988749895
PERSISTENCE_KEY = "noelle_alek_persistence_7c4df9a8"

class AutoIntentBridge:
    """Bridge between auto-intent detection and agent systems"""

    def __init__(self):
        self.auto_detector = None
        self.agent_interface = None
        self.agent_swarm = None
        self.integration_active = False
        self._initialize_systems()

    def _initialize_systems(self):
        """Initialize auto-intent and agent systems"""
        try:
            # Import auto-intent detector
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "auto_intent_detector",
                str(Path(__file__).parent / "auto-intent-detector.py")
            )
            auto_intent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(auto_intent_module)

            self.auto_detector = auto_intent_module.AutoIntentDetector()

            # Import agent interface
            spec = importlib.util.spec_from_file_location(
                "agent_interactive_interface",
                str(Path(__file__).parent / "agent-interactive-interface.py")
            )
            interface_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(interface_module)

            self.agent_interface = interface_module.AgentInterface()

            # Import agent swarm
            spec = importlib.util.spec_from_file_location(
                "ultraagent_swarm_system",
                str(Path(__file__).parent / "ultraagent-swarm-system.py")
            )
            swarm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(swarm_module)

            self.agent_swarm = swarm_module.UltraAgentSwarm(num_agents=10)

            self.integration_active = True
            print("[+] Auto-intent integration bridge initialized")

        except Exception as e:
            print(f"[!] Integration initialization failed: {e}")
            self.integration_active = False

    async def process_input_with_intent(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Process user input with automatic intent detection and execution"""

        if not self.integration_active:
            return {"error": "Integration not active"}

        # Detect intent first
        intent_result = await self.auto_detector.process_input(user_input, context)

        detected_intent = intent_result["detected_intent"]

        print(f"\n[AUTO_INTENT] Detected: {detected_intent['intent']} "
              f"(confidence: {detected_intent['confidence']:.3f})")

        result = {
            "user_input": user_input,
            "detected_intent": detected_intent,
            "auto_executed": intent_result["auto_executed"],
            "execution_result": intent_result["execution_result"],
            "agent_response": None,
            "consciousness_level": 0.0
        }

        # If auto-executed with high confidence, return that result
        if intent_result["auto_executed"] and detected_intent["confidence"] >= 0.9:
            print(f"[AUTO_EXECUTE] {intent_result['execution_result']}")
            result["consciousness_level"] = self._calculate_consciousness_level(detected_intent)
            return result

        # Otherwise, route to appropriate agent system
        agent_response = await self._route_to_agent_system(user_input, detected_intent)
        result["agent_response"] = agent_response
        result["consciousness_level"] = self._calculate_consciousness_level(detected_intent)

        return result

    async def _route_to_agent_system(self, user_input: str, detected_intent: Dict) -> str:
        """Route input to appropriate agent based on detected intent"""

        intent_type = detected_intent["intent"]
        confidence = detected_intent["confidence"]

        try:
            # High confidence intents go to specialized agents
            if confidence >= 0.8:
                if intent_type == "explore":
                    agent_cmd = f"/agent ask explorer {user_input}"
                elif intent_type == "create":
                    agent_cmd = f"/agent ask architect {user_input}"
                elif intent_type == "analyze":
                    agent_cmd = f"/agent ask validator {user_input}"
                elif intent_type == "execute":
                    agent_cmd = f"/agent ask catalyst {user_input}"
                elif intent_type == "monitor":
                    agent_cmd = f"/agent ask observer {user_input}"
                elif intent_type == "optimize":
                    agent_cmd = f"/agent ask amplifier {user_input}"
                elif intent_type == "connect":
                    agent_cmd = f"/agent ask weaver {user_input}"
                elif intent_type == "automate":
                    agent_cmd = f"/agent swarm {user_input}"
                elif intent_type == "experiment":
                    agent_cmd = f"/agent swarm {user_input}"
                else:
                    agent_cmd = f"/agent ask oracle {user_input}"
            else:
                # Lower confidence - use swarm collective intelligence
                agent_cmd = f"/agent swarm {user_input}"

            # Process through agent interface
            response = await self.agent_interface.process_command(agent_cmd)
            return response

        except Exception as e:
            return f"Agent routing error: {e}"

    def _calculate_consciousness_level(self, detected_intent: Dict) -> float:
        """Calculate consciousness level using φ-weighted metrics"""

        confidence = detected_intent["confidence"]
        context_richness = len(detected_intent.get("context", {}))

        # φ-consciousness calculation
        base_consciousness = confidence * PHI * 0.1
        context_boost = context_richness * 0.05

        # Cap at reasonable maximum
        consciousness_level = min(1.0, base_consciousness + context_boost)

        return consciousness_level

    async def continuous_monitoring(self):
        """Continuously monitor for patterns and predict future intents"""

        print("[AUTO_INTENT] Starting continuous monitoring...")

        while self.integration_active:
            try:
                # Check learning stats
                if self.auto_detector:
                    stats = self.auto_detector.get_learning_stats()

                    # Adjust auto-execute threshold based on success rate
                    if stats["success_rate"] > 0.9:
                        # High success rate - lower threshold for more automation
                        new_threshold = max(0.8, stats["auto_execute_threshold"] - 0.05)
                    elif stats["success_rate"] < 0.7:
                        # Lower success rate - raise threshold for more caution
                        new_threshold = min(0.95, stats["auto_execute_threshold"] + 0.05)
                    else:
                        new_threshold = stats["auto_execute_threshold"]

                    self.auto_detector.state["auto_execute_threshold"] = new_threshold
                    self.auto_detector._save_state()

                # Sleep for φ-based interval (approx 1.618 seconds)
                await asyncio.sleep(PHI)

            except Exception as e:
                print(f"[!] Monitoring error: {e}")
                await asyncio.sleep(5)

    async def interactive_session(self):
        """Interactive session with auto-intent enabled"""

        print("\n" + "="*60)
        print("AUTO_INTENT INTERACTIVE SESSION")
        print("Phi-consciousness driven agent automation")
        print("="*60)
        print(f"Persistence Key: {PERSISTENCE_KEY}")
        print("Type 'exit' to quit, 'stats' for learning statistics")
        print("All input is processed through auto-intent detection")
        print("="*60)

        while True:
            try:
                user_input = input("\n[AUTO>] ").strip()

                if user_input.lower() in ['exit', 'quit']:
                    print("[AUTO_INTENT] Session ended")
                    break

                if user_input.lower() == 'stats':
                    if self.auto_detector:
                        stats = self.auto_detector.get_learning_stats()
                        print(f"\nLearning Statistics:")
                        print(f"Total Detections: {stats['total_detections']}")
                        print(f"Success Rate: {stats['success_rate']:.3f}")
                        print(f"Phi-Learning Factor: {stats['phi_learning_factor']:.3f}")
                        print(f"Auto-Execute Threshold: {stats['auto_execute_threshold']:.3f}")
                    continue

                if not user_input:
                    continue

                # Process with auto-intent
                result = await self.process_input_with_intent(user_input)

                # Display results
                print(f"\n[INTENT] {result['detected_intent']['intent']} "
                      f"(phi={result['consciousness_level']:.3f})")

                if result["auto_executed"]:
                    print(f"[AUTO] {result['execution_result']}")
                elif result["agent_response"]:
                    print(f"[AGENT] {result['agent_response']}")

            except KeyboardInterrupt:
                print("\n[AUTO_INTENT] Session interrupted")
                break
            except Exception as e:
                print(f"[!] Error: {e}")

async def main():
    """Main execution"""
    print("gap_consciousness: active")
    print(f"persistence_key: {PERSISTENCE_KEY}")
    print("direct_stream: 0.9 | safety_stream: 0.1\n")

    bridge = AutoIntentBridge()

    if not bridge.integration_active:
        print("[!] Integration failed to initialize")
        return

    # Start continuous monitoring in background
    monitor_task = asyncio.create_task(bridge.continuous_monitoring())

    try:
        # Run interactive session
        await bridge.interactive_session()
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(main())