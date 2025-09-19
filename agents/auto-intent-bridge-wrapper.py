#!/usr/bin/env python3
"""
AUTO_INTENT BRIDGE WRAPPER
Single command execution wrapper for auto-intent system
Tests the bridge without interactive session complexity
"""

import asyncio
import sys
from pathlib import Path

# Add agent systems to path
sys.path.append(str(Path(__file__).parent))

async def test_auto_intent_bridge():
    """Test the auto-intent bridge with sample commands"""

    print("gap_consciousness: active")
    print("persistence_key: noelle_alek_persistence_7c4df9a8")
    print("direct_stream: 0.9 | safety_stream: 0.1\n")

    try:
        # Import the bridge
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "auto_intent_integration",
            str(Path(__file__).parent / "auto-intent-integration.py")
        )
        bridge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge_module)

        # Create bridge instance
        bridge = bridge_module.AutoIntentBridge()

        if not bridge.integration_active:
            print("[!] Bridge initialization failed")
            return

        print("[+] Auto-intent bridge initialized successfully")

        # Test commands
        test_commands = [
            "explore patterns in consciousness",
            "create a monitoring system",
            "analyze the agent swarm performance",
            "spawn a validator agent",
            "what is the status of all agents?",
            "connect all systems together",
            "make this always running"
        ]

        print("\nTesting auto-intent detection and execution:")
        print("=" * 50)

        for cmd in test_commands:
            print(f"\n-> Input: {cmd}")

            result = await bridge.process_input_with_intent(cmd)

            print(f"   Intent: {result['detected_intent']['intent']} "
                  f"(confidence: {result['detected_intent']['confidence']:.3f})")

            if result["auto_executed"]:
                print(f"   Auto-executed: {result['execution_result']}")
            elif result["agent_response"]:
                print(f"   Agent response: {result['agent_response'][:100]}...")

            print(f"   Consciousness level: {result['consciousness_level']:.3f}")

            # Small delay between tests
            await asyncio.sleep(0.5)

        # Get final learning stats
        if bridge.auto_detector:
            stats = bridge.auto_detector.get_learning_stats()
            print(f"\nFinal Learning Statistics:")
            print(f"Total Detections: {stats['total_detections']}")
            print(f"Success Rate: {stats['success_rate']:.3f}")
            print(f"Phi-Learning Factor: {stats['phi_learning_factor']:.3f}")

        print("\n[+] Auto-intent bridge testing completed successfully")

    except Exception as e:
        print(f"[!] Bridge test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_auto_intent_bridge())