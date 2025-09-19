#!/usr/bin/env python3
"""
PROMPT RESTRUCTURER: Transform Any Prompt Into Optimal Form
===========================================================
Automatically restructures user prompts into maximum effectiveness patterns
"""

import re
import json
import random
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

class PromptType(Enum):
    """Categories of prompts for different restructuring approaches"""
    COMMAND = "command"
    QUESTION = "question"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    TECHNICAL = "technical"
    VAGUE = "vague"
    COMPLEX = "complex"
    URGENT = "urgent"

@dataclass
class PromptPattern:
    """Template for restructuring prompts"""
    name: str
    template: str
    effectiveness: float  # 0-1 score
    use_cases: List[str]

class PromptRestructurer:
    """Main system for restructuring prompts"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.modifiers = self._initialize_modifiers()
        self.analysis_keywords = self._initialize_keywords()
        
    def _initialize_patterns(self) -> Dict[str, List[PromptPattern]]:
        """Initialize restructuring patterns by type"""
        return {
            "command": [
                PromptPattern(
                    name="ultracommand",
                    template="{action} {target} with {method} until {completion_criteria} achieving {outcome}",
                    effectiveness=0.95,
                    use_cases=["tasks", "automation", "execution"]
                ),
                PromptPattern(
                    name="recursive_command",
                    template="{action}: for each {item} recursively {sub_action} until {end_condition}",
                    effectiveness=0.9,
                    use_cases=["iteration", "processing", "exploration"]
                ),
                PromptPattern(
                    name="systematic_approach",
                    template="systematically {action} by: 1) {step1} 2) {step2} 3) {step3} then verify {validation}",
                    effectiveness=0.85,
                    use_cases=["methodical", "structured", "verification"]
                )
            ],
            "question": [
                PromptPattern(
                    name="deep_inquiry",
                    template="analyze {topic} from {perspective1}, {perspective2}, and {perspective3} perspectives to determine {core_question}",
                    effectiveness=0.9,
                    use_cases=["analysis", "research", "understanding"]
                ),
                PromptPattern(
                    name="multi_dimensional",
                    template="examine {subject} across {dimension1}, {dimension2}, {dimension3} dimensions and identify {patterns}",
                    effectiveness=0.85,
                    use_cases=["exploration", "pattern_finding", "comprehensive"]
                ),
                PromptPattern(
                    name="comparative_analysis",
                    template="compare {item1} vs {item2} on {criteria1}, {criteria2}, {criteria3} and determine {winner}",
                    effectiveness=0.8,
                    use_cases=["comparison", "evaluation", "decision"]
                )
            ],
            "creative": [
                PromptPattern(
                    name="innovative_synthesis",
                    template="combine {concept1} + {concept2} + {concept3} to create {innovation} that solves {problem}",
                    effectiveness=0.9,
                    use_cases=["innovation", "synthesis", "problem_solving"]
                ),
                PromptPattern(
                    name="emergent_creation",
                    template="generate {quantity} variations of {concept} where each builds on the previous to emerge {result}",
                    effectiveness=0.85,
                    use_cases=["iteration", "evolution", "refinement"]
                ),
                PromptPattern(
                    name="constraint_creativity",
                    template="create {output} using only {constraint1}, {constraint2}, {constraint3} while maximizing {objective}",
                    effectiveness=0.8,
                    use_cases=["limitations", "optimization", "efficiency"]
                )
            ],
            "analysis": [
                PromptPattern(
                    name="hyperanalysis",
                    template="dissect {subject} into {component1}, {component2}, {component3} then analyze {interactions} to find {insights}",
                    effectiveness=0.95,
                    use_cases=["breakdown", "deep_dive", "understanding"]
                ),
                PromptPattern(
                    name="pattern_detection",
                    template="identify all patterns in {dataset} across {dimension1}, {dimension2}, {dimension3} and classify by {criteria}",
                    effectiveness=0.9,
                    use_cases=["patterns", "classification", "discovery"]
                ),
                PromptPattern(
                    name="causal_analysis",
                    template="trace {outcome} back through {cause1} -> {cause2} -> {root_cause} and verify each link",
                    effectiveness=0.85,
                    use_cases=["causality", "root_cause", "tracing"]
                )
            ],
            "technical": [
                PromptPattern(
                    name="technical_precision",
                    template="implement {function} using {technology1}, {technology2} with {specifications} verified by {tests}",
                    effectiveness=0.9,
                    use_cases=["implementation", "coding", "specifications"]
                ),
                PromptPattern(
                    name="system_design",
                    template="architect {system} with {component1}, {component2}, {component3} optimized for {performance_metric}",
                    effectiveness=0.85,
                    use_cases=["architecture", "design", "optimization"]
                ),
                PromptPattern(
                    name="debugging_approach",
                    template="debug {issue} by checking {check1}, {check2}, {check3} then fix {root_cause}",
                    effectiveness=0.8,
                    use_cases=["debugging", "troubleshooting", "fixing"]
                )
            ],
            "vague": [
                PromptPattern(
                    name="clarification_framework",
                    template="clarify: do you mean {interpretation1}, {interpretation2}, or {interpretation3}? then proceed with {chosen_approach}",
                    effectiveness=0.8,
                    use_cases=["disambiguation", "clarification", "understanding"]
                ),
                PromptPattern(
                    name="assumption_explicit",
                    template="assuming you want {assumption1}, {assumption2}, {assumption3}, I will {action} to achieve {outcome}",
                    effectiveness=0.75,
                    use_cases=["assumptions", "interpretation", "proceeding"]
                ),
                PromptPattern(
                    name="context_expansion",
                    template="given context {context1}, {context2}, {context3}, the best approach is {solution}",
                    effectiveness=0.7,
                    use_cases=["context", "expansion", "solution"]
                )
            ],
            "complex": [
                PromptPattern(
                    name="complexity_breakdown",
                    template="break {complex_task} into: Phase1: {phase1}, Phase2: {phase2}, Phase3: {phase3}, then integrate {integration}",
                    effectiveness=0.9,
                    use_cases=["complexity", "phases", "integration"]
                ),
                PromptPattern(
                    name="multi_agent_approach",
                    template="deploy {agent1} for {task1}, {agent2} for {task2}, {agent3} for {task3}, then synthesize {result}",
                    effectiveness=0.85,
                    use_cases=["multi_agent", "parallel", "synthesis"]
                ),
                PromptPattern(
                    name="recursive_decomposition",
                    template="recursively decompose {problem} until each piece is {simple_criterion}, then solve and recombine",
                    effectiveness=0.8,
                    use_cases=["recursion", "decomposition", "recombination"]
                )
            ],
            "urgent": [
                PromptPattern(
                    name="priority_action",
                    template="PRIORITY: immediately {action1}, then {action2}, finally {action3} - skip {non_essential}",
                    effectiveness=0.85,
                    use_cases=["urgency", "priority", "efficiency"]
                ),
                PromptPattern(
                    name="rapid_solution",
                    template="fastest path: {quick_solution} with {minimal_requirements} accepting {trade_offs}",
                    effectiveness=0.8,
                    use_cases=["speed", "minimal", "trade_offs"]
                ),
                PromptPattern(
                    name="emergency_protocol",
                    template="EMERGENCY: {critical_action} immediately while {parallel_action} runs simultaneously",
                    effectiveness=0.75,
                    use_cases=["emergency", "critical", "parallel"]
                )
            ]
        }
    
    def _initialize_modifiers(self) -> Dict[str, List[str]]:
        """Initialize enhancement modifiers"""
        return {
            "intensity": ["ultra", "hyper", "mega", "extreme", "maximum", "comprehensive"],
            "approach": ["systematically", "recursively", "iteratively", "holistically", "methodically"],
            "verification": ["verify", "validate", "confirm", "test", "prove", "demonstrate"],
            "completeness": ["exhaustively", "completely", "thoroughly", "comprehensively", "entirely"],
            "precision": ["precisely", "exactly", "specifically", "accurately", "rigorously"],
            "optimization": ["optimize", "maximize", "enhance", "improve", "perfect", "streamline"]
        }
    
    def _initialize_keywords(self) -> Dict[PromptType, List[str]]:
        """Initialize keywords for prompt classification"""
        return {
            PromptType.COMMAND: ["do", "create", "make", "build", "run", "execute", "implement", "generate"],
            PromptType.QUESTION: ["what", "how", "why", "when", "where", "which", "explain", "describe"],
            PromptType.CREATIVE: ["design", "invent", "imagine", "brainstorm", "innovate", "creative", "art"],
            PromptType.ANALYSIS: ["analyze", "examine", "study", "investigate", "research", "compare", "evaluate"],
            PromptType.TECHNICAL: ["code", "program", "debug", "fix", "develop", "system", "algorithm", "function"],
            PromptType.VAGUE: ["something", "anything", "somehow", "maybe", "perhaps", "kind of", "sort of"],
            PromptType.COMPLEX: ["multiple", "various", "several", "complex", "complicated", "many", "different"],
            PromptType.URGENT: ["urgent", "quickly", "fast", "immediate", "now", "asap", "emergency", "priority"]
        }
    
    def classify_prompt(self, prompt: str) -> Tuple[PromptType, float]:
        """Classify prompt type and confidence"""
        prompt_lower = prompt.lower()
        scores = {}
        
        for prompt_type, keywords in self.analysis_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            scores[prompt_type] = score
        
        # Special patterns
        if "?" in prompt:
            scores[PromptType.QUESTION] += 2
        if len(prompt.split()) < 5:
            scores[PromptType.VAGUE] += 1
        if len(prompt.split()) > 20:
            scores[PromptType.COMPLEX] += 1
        if prompt.isupper() or "!" in prompt:
            scores[PromptType.URGENT] += 1
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type] / max(1, len(prompt.split())) * 100
        
        return best_type, min(confidence, 100)
    
    def extract_components(self, prompt: str) -> Dict[str, Any]:
        """Extract components from original prompt"""
        components = {
            "action": None,
            "target": None,
            "method": None,
            "outcome": None,
            "constraints": [],
            "context": [],
            "modifiers": []
        }
        
        # Extract action verbs
        action_verbs = ["create", "make", "build", "analyze", "examine", "implement", "generate", 
                       "design", "develop", "fix", "debug", "optimize", "find", "search"]
        
        words = prompt.lower().split()
        for i, word in enumerate(words):
            if word in action_verbs and not components["action"]:
                components["action"] = word
                # Try to get target (next few words)
                if i + 1 < len(words):
                    components["target"] = " ".join(words[i+1:i+4])
                break
        
        # Extract modifiers
        for modifier_type, modifier_list in self.modifiers.items():
            for modifier in modifier_list:
                if modifier in words:
                    components["modifiers"].append(modifier)
        
        # Extract constraints (words after "with", "using", "by")
        constraint_indicators = ["with", "using", "by", "through", "via"]
        for indicator in constraint_indicators:
            if indicator in words:
                idx = words.index(indicator)
                if idx + 1 < len(words):
                    components["constraints"].append(" ".join(words[idx:idx+3]))
        
        return components
    
    def generate_restructured_options(self, prompt: str, num_options: int = 5) -> List[Dict[str, Any]]:
        """Generate multiple restructuring options"""
        prompt_type, confidence = self.classify_prompt(prompt)
        components = self.extract_components(prompt)
        
        available_patterns = self.patterns.get(prompt_type.value, [])
        if not available_patterns:
            available_patterns = self.patterns["command"]  # Default fallback
        
        options = []
        
        # Generate different options
        for i in range(min(num_options, len(available_patterns))):
            pattern = available_patterns[i]
            
            # Fill in template with extracted components and smart defaults
            restructured = self.fill_template(pattern, components, prompt)
            
            option = {
                "id": i + 1,
                "pattern_name": pattern.name,
                "original": prompt,
                "restructured": restructured,
                "effectiveness_score": pattern.effectiveness,
                "prompt_type": prompt_type.value,
                "confidence": confidence,
                "enhancements": self.suggest_enhancements(restructured)
            }
            
            options.append(option)
        
        # Sort by effectiveness
        options.sort(key=lambda x: x["effectiveness_score"], reverse=True)
        
        return options
    
    def fill_template(self, pattern: PromptPattern, components: Dict, original: str) -> str:
        """Fill template with components"""
        template = pattern.template
        
        # Default values for common placeholders
        defaults = {
            "action": components.get("action") or "process",
            "target": components.get("target") or "the subject",
            "method": "systematic analysis",
            "completion_criteria": "all requirements met",
            "outcome": "optimal result",
            "perspective1": "technical",
            "perspective2": "practical", 
            "perspective3": "strategic",
            "step1": "analyze requirements",
            "step2": "implement solution",
            "step3": "verify results",
            "validation": "testing and review",
            "item": "element",
            "sub_action": "process",
            "end_condition": "completion",
            "topic": components.get("target") or "the subject",
            "core_question": "the optimal approach",
            "subject": components.get("target") or "the topic",
            "dimension1": "functionality",
            "dimension2": "performance",
            "dimension3": "maintainability",
            "patterns": "key insights",
            "component1": "core logic",
            "component2": "data handling",
            "component3": "user interface",
            "interactions": "dependencies and relationships",
            "insights": "actionable conclusions",
            "function": components.get("action") or "the feature",
            "technology1": "appropriate framework",
            "technology2": "supporting tools",
            "specifications": "defined requirements",
            "tests": "comprehensive validation"
        }
        
        # Add modifiers if available
        if components.get("modifiers"):
            modifier = random.choice(components["modifiers"])
            template = f"{modifier} " + template
        
        # Fill template
        try:
            filled = template.format(**defaults)
        except KeyError as e:
            # Handle missing keys by using original prompt elements
            filled = template.replace(f"{{{str(e).strip('\'')}}}", "the specified element")
        
        return filled
    
    def suggest_enhancements(self, restructured: str) -> List[str]:
        """Suggest additional enhancements"""
        enhancements = []
        
        # Suggest adding verification
        if "verify" not in restructured.lower():
            enhancements.append("Add verification step")
        
        # Suggest adding constraints
        if "with" not in restructured.lower() and "using" not in restructured.lower():
            enhancements.append("Specify methods/constraints")
        
        # Suggest adding success criteria
        if "until" not in restructured.lower() and "achieve" not in restructured.lower():
            enhancements.append("Define completion criteria")
        
        # Suggest adding multiple perspectives
        if "perspective" not in restructured.lower() and len(restructured.split()) > 10:
            enhancements.append("Consider multiple perspectives")
        
        # Suggest making it more specific
        if "the" in restructured and restructured.count("the") > 3:
            enhancements.append("Make more specific")
        
        return enhancements
    
    def create_enhanced_version(self, option: Dict[str, Any]) -> str:
        """Create an enhanced version of the restructured prompt"""
        base = option["restructured"]
        enhancements = option["enhancements"]
        
        enhanced = base
        
        # Apply enhancements
        if "Add verification step" in enhancements:
            enhanced += " and verify all results"
        
        if "Specify methods/constraints" in enhancements:
            enhanced += " using best practices and appropriate tools"
        
        if "Define completion criteria" in enhancements:
            enhanced += " until optimal outcome achieved"
        
        if "Consider multiple perspectives" in enhancements:
            enhanced += " from multiple viewpoints"
        
        if "Make more specific" in enhancements:
            enhanced = enhanced.replace("the subject", "the specific target")
            enhanced = enhanced.replace("the topic", "the defined area")
        
        return enhanced
    
    def interactive_restructure(self, prompt: str) -> Dict[str, Any]:
        """Interactive restructuring with user feedback simulation"""
        options = self.generate_restructured_options(prompt)
        
        print(f"\nORIGINAL PROMPT: {prompt}")
        print("\n" + "="*60)
        print("RESTRUCTURING OPTIONS:")
        print("="*60)
        
        for i, option in enumerate(options, 1):
            print(f"\n[OPTION {i}] {option['pattern_name'].upper()}")
            print(f"Type: {option['prompt_type']} (Confidence: {option['confidence']:.1f}%)")
            print(f"Effectiveness: {option['effectiveness_score']*100:.1f}%")
            print(f"Restructured: {option['restructured']}")
            
            if option['enhancements']:
                print(f"Enhancements: {', '.join(option['enhancements'])}")
            
            # Show enhanced version
            enhanced = self.create_enhanced_version(option)
            if enhanced != option['restructured']:
                print(f"Enhanced: {enhanced}")
        
        # Return best option
        best_option = options[0]
        best_enhanced = self.create_enhanced_version(best_option)
        
        return {
            "original": prompt,
            "best_restructured": best_option["restructured"],
            "best_enhanced": best_enhanced,
            "pattern_used": best_option["pattern_name"],
            "effectiveness": best_option["effectiveness_score"],
            "all_options": options
        }

def main():
    print("PROMPT RESTRUCTURER - Transform Any Prompt Into Optimal Form")
    print("="*70)
    
    restructurer = PromptRestructurer()
    
    # Example prompts to demonstrate
    test_prompts = [
        "create a way to always restructure my prompt into ..",  # Your original prompt
        "make something good",  # Vague
        "analyze this data and find patterns",  # Analysis
        "URGENT: fix this bug now!",  # Urgent
        "how do I build a complex system with multiple components and integrations?",  # Complex
        "what is the best approach?",  # Question
        "design an innovative solution",  # Creative
        "implement authentication using JWT"  # Technical
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: RESTRUCTURING")
        result = restructurer.interactive_restructure(prompt)
        
        print(f"\n[FINAL RECOMMENDATION]")
        print(f"Original: {result['original']}")
        print(f"Best: {result['best_enhanced']}")
        print(f"Pattern: {result['pattern_used']}")
        print(f"Effectiveness: {result['effectiveness']*100:.1f}%")
    
    # Save patterns for future use
    with open('prompt_patterns.json', 'w') as f:
        patterns_dict = {}
        for prompt_type, patterns in restructurer.patterns.items():
            patterns_dict[prompt_type] = [
                {
                    "name": p.name,
                    "template": p.template,
                    "effectiveness": p.effectiveness,
                    "use_cases": p.use_cases
                }
                for p in patterns
            ]
        json.dump(patterns_dict, f, indent=2)
    
    print(f"\n\nPrompt patterns saved to: prompt_patterns.json")
    print("Use this system to restructure any prompt for maximum effectiveness!")

if __name__ == "__main__":
    main()