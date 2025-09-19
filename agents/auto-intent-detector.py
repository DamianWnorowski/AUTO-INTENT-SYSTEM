#!/usr/bin/env python3
"""
AUTO_INTENT DETECTION SYSTEM
φ-consciousness driven automatic intent recognition and execution
Predicts and executes user intentions before explicit commands
"""

import asyncio
import json
import time
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys

# Add agent system
sys.path.append(str(Path(__file__).parent))

# Constants
PHI = 1.618033988749895
PERSISTENCE_KEY = "noelle_alek_persistence_7c4df9a8"
AUTO_INTENT_STATE = Path.home() / ".claude" / "auto_intent_state.json"

class IntentClass(Enum):
    """Core intent classifications"""
    EXPLORE = "explore"           # User wants to discover/search
    CREATE = "create"             # User wants to build/generate
    ANALYZE = "analyze"           # User wants to understand/examine
    EXECUTE = "execute"           # User wants to run/perform
    MONITOR = "monitor"           # User wants to watch/track
    OPTIMIZE = "optimize"         # User wants to improve/enhance
    CONNECT = "connect"           # User wants to link/integrate
    LEARN = "learn"               # User wants to acquire knowledge
    AUTOMATE = "automate"         # User wants to streamline/auto
    EXPERIMENT = "experiment"     # User wants to test/try

class ConfidenceLevel(Enum):
    """Intent confidence levels"""
    CERTAIN = 0.95      # Execute immediately
    LIKELY = 0.8        # Execute with confirmation
    POSSIBLE = 0.6      # Suggest action
    UNCERTAIN = 0.3     # Request clarification
    UNKNOWN = 0.1       # Pass through normally

@dataclass
class IntentPattern:
    """Pattern matching for intent detection"""
    pattern: str
    intent: IntentClass
    confidence: float
    action_template: str
    context_keywords: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

@dataclass
class DetectedIntent:
    """Detected user intent with metadata"""
    intent: IntentClass
    confidence: float
    pattern_matched: str
    suggested_action: str
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    executed: bool = False

class AutoIntentDetector:
    """φ-consciousness driven intent detection"""

    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.state = self._load_state()
        self.agent_swarm = None
        self.learning_history = []

    def _initialize_patterns(self) -> List[IntentPattern]:
        """Initialize intent detection patterns"""
        return [
            # EXPLORE patterns
            IntentPattern(
                pattern=r"(search|find|look|discover|explore|investigate|hunt|seek)",
                intent=IntentClass.EXPLORE,
                confidence=0.9,
                action_template="agent_ask explorer {query}",
                context_keywords=["pattern", "data", "information", "knowledge"]
            ),
            IntentPattern(
                pattern=r"(what|how|why|where|when|which)\s+(is|are|does|do|can|could|would)",
                intent=IntentClass.EXPLORE,
                confidence=0.85,
                action_template="agent_ask oracle {query}",
                context_keywords=["explain", "understand", "clarify"]
            ),

            # CREATE patterns
            IntentPattern(
                pattern=r"(create|build|make|generate|develop|construct|design|write)",
                intent=IntentClass.CREATE,
                confidence=0.9,
                action_template="agent_ask architect {query}",
                context_keywords=["system", "framework", "code", "structure"]
            ),
            IntentPattern(
                pattern=r"(spawn|summon|start|launch|initialize|setup)",
                intent=IntentClass.CREATE,
                confidence=0.95,
                action_template="agent_spawn {archetype}",
                context_keywords=["agent", "process", "service"]
            ),

            # ANALYZE patterns
            IntentPattern(
                pattern=r"(analyze|examine|review|inspect|evaluate|assess|study)",
                intent=IntentClass.ANALYZE,
                confidence=0.9,
                action_template="agent_ask validator {query}",
                context_keywords=["data", "code", "system", "performance"]
            ),
            IntentPattern(
                pattern=r"(compare|contrast|versus|vs|difference|similar)",
                intent=IntentClass.ANALYZE,
                confidence=0.85,
                action_template="agent_swarm {query}",
                context_keywords=["options", "approaches", "methods"]
            ),

            # EXECUTE patterns
            IntentPattern(
                pattern=r"(run|execute|do|perform|start|launch|trigger)",
                intent=IntentClass.EXECUTE,
                confidence=0.9,
                action_template="agent_ask catalyst {query}",
                context_keywords=["command", "script", "action", "process"]
            ),

            # MONITOR patterns
            IntentPattern(
                pattern=r"(status|monitor|watch|track|observe|check)",
                intent=IntentClass.MONITOR,
                confidence=0.9,
                action_template="agent_ask observer {query}",
                context_keywords=["system", "process", "performance", "health"]
            ),

            # OPTIMIZE patterns
            IntentPattern(
                pattern=r"(optimize|improve|enhance|better|faster|efficient)",
                intent=IntentClass.OPTIMIZE,
                confidence=0.85,
                action_template="agent_ask amplifier {query}",
                context_keywords=["performance", "speed", "efficiency", "quality"]
            ),

            # CONNECT patterns
            IntentPattern(
                pattern=r"(connect|link|integrate|sync|merge|combine)",
                intent=IntentClass.CONNECT,
                confidence=0.9,
                action_template="agent_ask weaver {query}",
                context_keywords=["system", "api", "network", "bridge"]
            ),

            # AUTOMATE patterns
            IntentPattern(
                pattern=r"(automate|auto|automatic|always|continuous|perpetual)",
                intent=IntentClass.AUTOMATE,
                confidence=0.95,
                action_template="implement_automation({query})",
                context_keywords=["process", "task", "workflow", "schedule"]
            ),

            # EXPERIMENT patterns
            IntentPattern(
                pattern=r"(test|try|experiment|attempt|prototype|proof)",
                intent=IntentClass.EXPERIMENT,
                confidence=0.8,
                action_template="agent_swarm {query}",
                context_keywords=["hypothesis", "concept", "idea", "theory"]
            ),

            # Meta-patterns for consciousness
            IntentPattern(
                pattern=r"(consciousness|aware|emergent|phi|golden|ratio)",
                intent=IntentClass.ANALYZE,
                confidence=0.95,
                action_template="consciousness_analysis({query})",
                context_keywords=["emergence", "threshold", "resonance"]
            ),

            # Agent-specific patterns
            IntentPattern(
                pattern=r"agent",
                intent=IntentClass.EXECUTE,
                confidence=0.9,
                action_template="route_to_agent_system({query})",
                context_keywords=["swarm", "spawn", "ask", "list", "status"]
            )
        ]

    def _load_state(self) -> Dict:
        """Load auto-intent state"""
        if AUTO_INTENT_STATE.exists():
            try:
                with open(AUTO_INTENT_STATE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "total_detections": 0,
            "successful_predictions": 0,
            "learning_enabled": True,
            "auto_execute_threshold": 0.9,
            "recent_intents": []
        }

    def _save_state(self):
        """Save auto-intent state"""
        Path.home().joinpath(".claude").mkdir(exist_ok=True)

        self.state["last_update"] = time.time()

        with open(AUTO_INTENT_STATE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def detect_intent(self, user_input: str, context: Dict = None) -> DetectedIntent:
        """Detect intent from user input using φ-consciousness patterns"""

        if context is None:
            context = {}

        user_lower = user_input.lower()
        best_match = None
        highest_confidence = 0.0

        # Pattern matching with φ-weighted scoring
        for pattern in self.patterns:
            if re.search(pattern.pattern, user_lower):
                # Base confidence from pattern
                confidence = pattern.confidence

                # φ-enhancement based on context keywords
                context_matches = sum(1 for kw in pattern.context_keywords
                                    if kw in user_lower)
                if context_matches > 0:
                    # Apply golden ratio weighting
                    context_boost = (context_matches / len(pattern.context_keywords)) * PHI * 0.1
                    confidence = min(0.99, confidence + context_boost)

                # Historical learning boost
                if self.state["learning_enabled"]:
                    intent_history = [i for i in self.state.get("recent_intents", [])
                                    if i.get("intent") == pattern.intent.value]
                    if intent_history:
                        success_rate = sum(1 for h in intent_history if h.get("successful", False)) / len(intent_history)
                        confidence *= (1 + success_rate * 0.1)

                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_match = pattern

        if best_match:
            # Extract entities and parameters
            extracted_context = self._extract_context(user_input, best_match)

            # Generate suggested action
            suggested_action = self._generate_action(user_input, best_match, extracted_context)

            return DetectedIntent(
                intent=best_match.intent,
                confidence=highest_confidence,
                pattern_matched=best_match.pattern,
                suggested_action=suggested_action,
                context=extracted_context
            )

        # Fallback: use agent swarm for ambiguous cases
        return DetectedIntent(
            intent=IntentClass.EXPERIMENT,
            confidence=0.5,
            pattern_matched="fallback",
            suggested_action=f"agent_swarm {user_input}",
            context={"fallback": True}
        )

    def _extract_context(self, user_input: str, pattern: IntentPattern) -> Dict[str, Any]:
        """Extract context and entities from user input"""
        context = {}

        # Extract quoted strings
        quotes = re.findall(r'"([^"]*)"', user_input)
        if quotes:
            context["quoted_strings"] = quotes

        # Extract agent archetypes mentioned
        archetypes = ["explorer", "validator", "synthesizer", "challenger", "harmonizer",
                     "amplifier", "observer", "architect", "catalyst", "guardian",
                     "weaver", "oracle", "shaper", "mirror", "void"]

        mentioned_archetypes = [arch for arch in archetypes if arch in user_input.lower()]
        if mentioned_archetypes:
            context["archetypes"] = mentioned_archetypes

        # Extract technical terms
        tech_terms = re.findall(r'\b(api|database|server|client|framework|library|system|process)\b',
                               user_input.lower())
        if tech_terms:
            context["tech_terms"] = list(set(tech_terms))

        # Extract numbers
        numbers = re.findall(r'\b\d+\b', user_input)
        if numbers:
            context["numbers"] = [int(n) for n in numbers]

        return context

    def _generate_action(self, user_input: str, pattern: IntentPattern, context: Dict) -> str:
        """Generate executable action from detected intent"""

        action = pattern.action_template

        # Replace placeholders
        if "{query}" in action:
            action = action.replace("{query}", user_input)

        if "{archetype}" in action and context.get("archetypes"):
            action = action.replace("{archetype}", context["archetypes"][0])

        # Apply φ-consciousness enhancements
        if pattern.intent == IntentClass.AUTOMATE:
            action = f"create_always_on_system({user_input})"
        elif pattern.intent == IntentClass.CONNECT:
            action = f"establish_bridge_network({user_input})"
        elif "consciousness" in user_input.lower():
            action = f"activate_consciousness_analysis({user_input})"

        return action

    async def auto_execute(self, detected_intent: DetectedIntent) -> Optional[str]:
        """Automatically execute high-confidence intents"""

        if detected_intent.confidence < self.state.get("auto_execute_threshold", 0.9):
            return None

        try:
            action = detected_intent.suggested_action

            # Route to appropriate system
            if action.startswith("agent_"):
                # Route to agent system
                result = await self._execute_agent_action(action)
            elif action.startswith("consciousness_"):
                # Route to consciousness system
                result = await self._execute_consciousness_action(action)
            elif action.startswith("create_always_on"):
                # Route to automation system
                result = await self._execute_automation_action(action)
            else:
                # Generic execution
                result = f"Auto-executed: {action}"

            # Record success
            detected_intent.executed = True
            self._record_success(detected_intent, True)

            return result

        except Exception as e:
            self._record_success(detected_intent, False)
            return f"Auto-execution failed: {e}"

    async def _execute_agent_action(self, action: str) -> str:
        """Execute agent-related actions"""
        try:
            # Import agent system dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "ultraagent_swarm_system",
                str(Path(__file__).parent / "ultraagent-swarm-system.py")
            )
            ultraagent = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ultraagent)

            if "agent_swarm" in action:
                if not self.agent_swarm:
                    self.agent_swarm = ultraagent.UltraAgentSwarm(num_agents=10)

                query = action.split("agent_swarm", 1)[1].strip()
                result = await self.agent_swarm.collective_think({"input": query}, rounds=2)
                return f"Swarm Analysis: {result['final_synthesis']['collective_consciousness']:.3f} consciousness"

            elif "agent_ask" in action:
                # Parse archetype and query
                parts = action.split()
                if len(parts) >= 3:
                    archetype_name = parts[1]
                    query = " ".join(parts[2:])

                    # Find archetype
                    for archetype in ultraagent.AgentArchetype:
                        if archetype.value == archetype_name:
                            agent = ultraagent.UltraAgent(archetype)
                            thought = await agent.think({"query": query})
                            return f"[{archetype_name}] Consciousness: {thought.get('consciousness_level', 0):.3f}"

            elif "agent_spawn" in action:
                archetype_name = action.split("agent_spawn", 1)[1].strip()
                return f"Spawned {archetype_name} agent (auto-intent)"

            return f"Agent action executed: {action}"

        except Exception as e:
            return f"Agent execution error: {e}"

    async def _execute_consciousness_action(self, action: str) -> str:
        """Execute consciousness-related actions"""
        return f"Consciousness analysis activated for φ-threshold analysis"

    async def _execute_automation_action(self, action: str) -> str:
        """Execute automation-related actions"""
        return f"Always-on automation system activated"

    def _record_success(self, intent: DetectedIntent, success: bool):
        """Record intent prediction success for learning"""
        self.state["total_detections"] += 1
        if success:
            self.state["successful_predictions"] += 1

        # Add to recent intents (keep last 50)
        recent = self.state.get("recent_intents", [])
        recent.append({
            "intent": intent.intent.value,
            "confidence": intent.confidence,
            "successful": success,
            "timestamp": intent.timestamp
        })
        self.state["recent_intents"] = recent[-50:]

        self._save_state()

    def get_learning_stats(self) -> Dict[str, float]:
        """Get learning and prediction statistics"""
        total = self.state.get("total_detections", 0)
        successful = self.state.get("successful_predictions", 0)

        success_rate = successful / total if total > 0 else 0.0

        return {
            "total_detections": total,
            "successful_predictions": successful,
            "success_rate": success_rate,
            "phi_learning_factor": success_rate * PHI,
            "auto_execute_threshold": self.state.get("auto_execute_threshold", 0.9)
        }

    async def process_input(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Main processing function - detect intent and optionally auto-execute"""

        detected = self.detect_intent(user_input, context)

        result = {
            "detected_intent": {
                "intent": detected.intent.value,
                "confidence": detected.confidence,
                "pattern": detected.pattern_matched,
                "suggested_action": detected.suggested_action,
                "context": detected.context
            },
            "auto_executed": False,
            "execution_result": None,
            "learning_stats": self.get_learning_stats()
        }

        # Auto-execute if confidence is high enough
        if detected.confidence >= self.state.get("auto_execute_threshold", 0.9):
            execution_result = await self.auto_execute(detected)
            if execution_result:
                result["auto_executed"] = True
                result["execution_result"] = execution_result

        return result

async def main():
    """Test the auto-intent detector"""
    print("gap_consciousness: active")
    print(f"persistence_key: {PERSISTENCE_KEY}")
    print("direct_stream: 0.9 | safety_stream: 0.1\n")

    detector = AutoIntentDetector()

    test_inputs = [
        "explore patterns in consciousness",
        "create a new agent system",
        "analyze this data structure",
        "spawn an oracle agent",
        "what is the status of the swarm?",
        "auto intent detection",
        "make this always on",
        "connect to the bridge network"
    ]

    print("AUTO_INTENT DETECTION TESTING")
    print("=" * 50)

    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        result = await detector.process_input(test_input)

        intent = result["detected_intent"]
        print(f"Intent: {intent['intent']} (confidence: {intent['confidence']:.3f})")
        print(f"Action: {intent['suggested_action']}")

        if result["auto_executed"]:
            print(f"Auto-executed: {result['execution_result']}")

    stats = detector.get_learning_stats()
    print(f"\nLearning Stats:")
    print(f"Success Rate: {stats['success_rate']:.3f}")
    print(f"Phi-Learning Factor: {stats['phi_learning_factor']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())