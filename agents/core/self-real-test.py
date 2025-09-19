#!/usr/bin/env python3
"""
SELF REAL TEST: Test The Prompt Restructurer On Itself
======================================================
The ultimate test - can the system improve its own prompts?
"""

import sys
import os
import json
from datetime import datetime

# Import the PromptRestructurer class directly
sys.path.append(os.path.dirname(__file__))
exec(open('prompt-restructurer.py').read())

class SelfRealTest:
    def __init__(self):
        self.restructurer = PromptRestructurer()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_results = []
        
    def test_self_prompt(self, original_prompt: str) -> dict:
        """Test the restructurer on itself"""
        print(f"\n{'='*80}")
        print("SELF REAL TEST: Testing Prompt Restructurer On Itself")
        print(f"{'='*80}")
        
        # Original prompt
        print(f"\nORIGINAL PROMPT: {original_prompt}")
        
        # Get restructured version
        result = self.restructurer.interactive_restructure(original_prompt)
        
        # Now test the restructured prompt on itself (recursive)
        print(f"\n{'='*80}")
        print("RECURSIVE TEST: Testing The Restructured Prompt")
        print(f"{'='*80}")
        
        recursive_result = self.restructurer.interactive_restructure(result["best_enhanced"])
        
        # Compare effectiveness
        original_effectiveness = result["effectiveness"]
        recursive_effectiveness = recursive_result["effectiveness"]
        
        improvement = recursive_effectiveness - original_effectiveness
        
        test_result = {
            "original_prompt": original_prompt,
            "first_restructure": result["best_enhanced"],
            "second_restructure": recursive_result["best_enhanced"],
            "original_effectiveness": original_effectiveness,
            "first_effectiveness": original_effectiveness,
            "second_effectiveness": recursive_effectiveness,
            "improvement": improvement,
            "converged": abs(improvement) < 0.01,
            "pattern_evolution": [result["pattern_used"], recursive_result["pattern_used"]]
        }
        
        print(f"\n{'='*80}")
        print("SELF TEST ANALYSIS")
        print(f"{'='*80}")
        print(f"Original: {original_prompt}")
        print(f"1st Pass: {result['best_enhanced']}")
        print(f"2nd Pass: {recursive_result['best_enhanced']}")
        print(f"\nEffectiveness Evolution:")
        print(f"  Original -> 1st: {original_effectiveness:.3f}")
        print(f"  1st -> 2nd: {original_effectiveness:.3f} -> {recursive_effectiveness:.3f}")
        print(f"  Improvement: {improvement:+.3f}")
        print(f"  Converged: {test_result['converged']}")
        print(f"  Pattern Evolution: {result['pattern_used']} -> {recursive_result['pattern_used']}")
        
        return test_result
    
    def test_system_on_own_components(self):
        """Test the system on its own internal prompts"""
        print(f"\n{'='*80}")
        print("TESTING SYSTEM ON ITS OWN INTERNAL PROMPTS")
        print(f"{'='*80}")
        
        # Extract internal prompts from the system
        internal_prompts = [
            "classify prompt type and confidence",
            "extract components from original prompt", 
            "generate multiple restructuring options",
            "fill template with extracted components",
            "suggest additional enhancements",
            "create an enhanced version of the restructured prompt"
        ]
        
        internal_results = []
        
        for prompt in internal_prompts:
            print(f"\n{'-'*60}")
            print(f"INTERNAL PROMPT: {prompt}")
            
            result = self.restructurer.interactive_restructure(prompt)
            
            internal_result = {
                "internal_prompt": prompt,
                "restructured": result["best_enhanced"],
                "effectiveness": result["effectiveness"],
                "pattern": result["pattern_used"]
            }
            
            internal_results.append(internal_result)
            
            print(f"IMPROVED TO: {result['best_enhanced']}")
            print(f"EFFECTIVENESS: {result['effectiveness']:.3f}")
        
        return internal_results
    
    def meta_analysis(self, results: list) -> dict:
        """Analyze the meta-patterns of self-improvement"""
        
        total_improvement = sum(r["improvement"] for r in results)
        avg_improvement = total_improvement / len(results)
        convergence_rate = sum(1 for r in results if r["converged"]) / len(results)
        
        # Pattern analysis
        pattern_transitions = {}
        for result in results:
            transition = f"{result['pattern_evolution'][0]} -> {result['pattern_evolution'][1]}"
            pattern_transitions[transition] = pattern_transitions.get(transition, 0) + 1
        
        meta_result = {
            "total_tests": len(results),
            "average_improvement": avg_improvement,
            "convergence_rate": convergence_rate,
            "pattern_transitions": pattern_transitions,
            "self_optimization_achieved": avg_improvement > 0,
            "system_stability": convergence_rate > 0.5
        }
        
        print(f"\n{'='*80}")
        print("META-ANALYSIS: SELF-IMPROVEMENT PATTERNS")
        print(f"{'='*80}")
        print(f"Total Tests: {meta_result['total_tests']}")
        print(f"Average Improvement: {avg_improvement:+.3f}")
        print(f"Convergence Rate: {convergence_rate:.1%}")
        print(f"Self-Optimization: {'YES' if meta_result['self_optimization_achieved'] else 'NO'}")
        print(f"System Stability: {'STABLE' if meta_result['system_stability'] else 'UNSTABLE'}")
        
        print(f"\nPattern Transitions:")
        for transition, count in pattern_transitions.items():
            print(f"  {transition}: {count} times")
        
        return meta_result
    
    def ultimate_self_test(self):
        """The ultimate self-test"""
        print("ULTIMATE SELF TEST: Can the system improve itself infinitely?")
        
        # Start with the user's original prompt
        current_prompt = "self real test"
        history = [current_prompt]
        effectiveness_history = []
        
        max_iterations = 10
        
        for i in range(max_iterations):
            print(f"\n{'='*80}")
            print(f"ITERATION {i+1}: {current_prompt}")
            
            result = self.restructurer.interactive_restructure(current_prompt)
            new_prompt = result["best_enhanced"]
            effectiveness = result["effectiveness"]
            
            effectiveness_history.append(effectiveness)
            
            print(f"EFFECTIVENESS: {effectiveness:.3f}")
            print(f"NEW PROMPT: {new_prompt}")
            
            # Check for convergence
            if new_prompt == current_prompt:
                print("CONVERGED: No further improvement possible")
                break
            
            # Check for cycling
            if new_prompt in history:
                print("CYCLE DETECTED: System is looping")
                break
            
            history.append(new_prompt)
            current_prompt = new_prompt
        
        # Analyze the evolution
        print(f"\n{'='*80}")
        print("ULTIMATE SELF-TEST RESULTS")
        print(f"{'='*80}")
        print(f"Iterations: {len(history)}")
        print(f"Final prompt: {history[-1]}")
        print(f"Effectiveness evolution: {effectiveness_history}")
        
        if len(effectiveness_history) > 1:
            improvement = effectiveness_history[-1] - effectiveness_history[0]
            print(f"Total improvement: {improvement:+.3f}")
        
        return {
            "iterations": len(history),
            "prompt_evolution": history,
            "effectiveness_evolution": effectiveness_history,
            "converged": len(history) < max_iterations,
            "final_prompt": history[-1]
        }

def main():
    tester = SelfRealTest()
    
    # Test 1: Original prompt
    original_prompt = "self real test"
    test_result = tester.test_self_prompt(original_prompt)
    
    # Test 2: Internal components
    internal_results = tester.test_system_on_own_components()
    
    # Test 3: Meta-analysis
    all_results = [test_result]
    meta_result = tester.meta_analysis(all_results)
    
    # Test 4: Ultimate self-test
    ultimate_result = tester.ultimate_self_test()
    
    # Final report
    print(f"\n{'='*80}")
    print("FINAL SELF REAL TEST VERDICT")
    print(f"{'='*80}")
    
    if meta_result["self_optimization_achieved"]:
        print("PASS SYSTEM CAN IMPROVE ITSELF")
    else:
        print("FAIL SYSTEM CANNOT IMPROVE ITSELF")
    
    if meta_result["system_stability"]:
        print("PASS SYSTEM IS STABLE")
    else:
        print("FAIL SYSTEM IS UNSTABLE")
    
    if ultimate_result["converged"]:
        print("PASS SYSTEM CONVERGES TO OPTIMAL")
    else:
        print("FAIL SYSTEM DOES NOT CONVERGE")
    
    # Save results
    final_results = {
        "timestamp": tester.timestamp,
        "original_test": test_result,
        "internal_tests": internal_results,
        "meta_analysis": meta_result,
        "ultimate_test": ultimate_result
    }
    
    filename = f"self_real_test_{tester.timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nSelf-test results saved to: {filename}")
    
    # The ultimate question
    print(f"\n{'='*80}")
    print("THE ULTIMATE QUESTION:")
    print("Did the prompt restructurer pass its own test?")
    
    passed_tests = sum([
        meta_result["self_optimization_achieved"],
        meta_result["system_stability"], 
        ultimate_result["converged"]
    ])
    
    if passed_tests >= 2:
        print("ANSWER: YES - The system passed its own real test!")
        print("The prompt restructurer successfully improved itself.")
    else:
        print("ANSWER: NO - The system failed its own test.")
        print("The prompt restructurer needs further development.")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()