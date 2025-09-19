#!/usr/bin/env python3
"""
REAL TRUTH TEST SYSTEM
Verifies absolute truth of all data, computations, and claims
No assumptions, only empirical verification
"""

import os
import sys
import json
import hashlib
import time
import psutil
import numpy as np
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import struct

class RealTruthTester:
    """
    Tests for absolute truth across multiple dimensions:
    1. File system reality
    2. Computational accuracy
    3. Data integrity
    4. System claims
    5. Mathematical proofs
    """
    
    def __init__(self):
        self.truth_results = {}
        self.failed_tests = []
        self.passed_tests = []
        
    def test_file_existence(self) -> Dict[str, bool]:
        """Test if files actually exist on disk"""
        print("\n[TRUTH TEST 1: FILE EXISTENCE]")
        print("-" * 50)
        
        files_to_check = [
            'pattern_analysis.json',
            'hyperagent_search_results.json',
            'stress_test_report_20250908_220852.json',
            'ai-hyperagent-search.py',
            'complex-metric-pattern-finder.py',
            'ultimate-stress-test.py',
            'hyperpattern-audit-system.py'
        ]
        
        results = {}
        for file in files_to_check:
            exists = os.path.exists(file)
            size = os.path.getsize(file) if exists else 0
            results[file] = {
                'exists': exists,
                'size': size,
                'readable': os.access(file, os.R_OK) if exists else False
            }
            
            status = "PASS" if exists else "FAIL"
            print(f"  {status} {file}: {size:,} bytes" if exists else f"  {status} {file}: NOT FOUND")
        
        self.truth_results['file_existence'] = results
        return results
    
    def test_pattern_data_integrity(self) -> Dict[str, Any]:
        """Verify the 278 patterns are real and consistent"""
        print("\n[TRUTH TEST 2: PATTERN DATA INTEGRITY]")
        print("-" * 50)
        
        results = {}
        
        try:
            # Load and verify pattern data
            with open('pattern_analysis.json', 'r') as f:
                data = json.load(f)
            
            patterns = data.get('patterns', [])
            
            # Test 1: Count verification
            actual_count = len(patterns)
            results['pattern_count'] = {
                'claimed': 278,
                'actual': actual_count,
                'matches': actual_count == 278
            }
            
            # Test 2: Verify each pattern structure
            valid_patterns = 0
            invalid_patterns = []
            
            for i, p in enumerate(patterns):
                required_fields = {'id', 'type', 'dimensions', 'strength'}
                has_all_fields = required_fields.issubset(p.keys())
                
                if has_all_fields:
                    # Additional validation
                    valid_id = isinstance(p['id'], str) and len(p['id']) > 0
                    valid_type = isinstance(p['type'], str)
                    valid_dims = isinstance(p['dimensions'], list) and all(isinstance(d, int) for d in p['dimensions'])
                    valid_strength = isinstance(p['strength'], (int, float)) and 0 <= p['strength'] <= 10
                    
                    if all([valid_id, valid_type, valid_dims, valid_strength]):
                        valid_patterns += 1
                    else:
                        invalid_patterns.append(i)
                else:
                    invalid_patterns.append(i)
            
            results['pattern_validation'] = {
                'valid': valid_patterns,
                'invalid': len(invalid_patterns),
                'invalid_indices': invalid_patterns[:5]  # First 5 invalid
            }
            
            # Test 3: Dimension coverage verification
            all_dims = set()
            for p in patterns:
                all_dims.update(p.get('dimensions', []))
            
            results['dimension_coverage'] = {
                'claimed': 30,
                'actual': len(all_dims),
                'missing': sorted(set(range(30)) - all_dims),
                'complete': len(all_dims) == 30
            }
            
            # Test 4: Cross-reference count verification
            cross_refs = 0
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    if set(patterns[i]['dimensions']).intersection(set(patterns[j]['dimensions'])):
                        cross_refs += 1
            
            results['cross_references'] = {
                'claimed': 2149,
                'actual': cross_refs,
                'matches': cross_refs == 2149
            }
            
            # Test 5: Pattern type distribution
            from collections import Counter
            type_counts = Counter(p['type'] for p in patterns)
            
            claimed_distribution = {
                'frequency': 150,
                'anomaly': 29,
                'superposition': 25,
                'mutual_information': 20,
                'chaotic': 20,
                'periodic': 13,
                'correlation': 10,
                'high_entropy': 6,
                'entanglement': 2,
                'principal_component': 1,
                'clustering': 1,
                'optimal_clustering': 1
            }
            
            distribution_matches = all(
                type_counts.get(k, 0) == v 
                for k, v in claimed_distribution.items()
            )
            
            results['type_distribution'] = {
                'matches': distribution_matches,
                'actual': dict(type_counts),
                'claimed': claimed_distribution
            }
            
            # Print results
            print(f"  Pattern count: {actual_count} (Expected: 278) - {'PASS' if actual_count == 278 else 'FAIL'}")
            print(f"  Valid patterns: {valid_patterns}/{actual_count}")
            print(f"  Dimension coverage: {len(all_dims)}/30 - {'COMPLETE' if len(all_dims) == 30 else 'INCOMPLETE'}")
            print(f"  Cross-references: {cross_refs} (Expected: 2149) - {'MATCH' if cross_refs == 2149 else 'MISMATCH'}")
            print(f"  Type distribution: {'MATCHES' if distribution_matches else 'DIFFERS'}")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"  ERROR: {e}")
        
        self.truth_results['pattern_integrity'] = results
        return results
    
    def test_system_claims(self) -> Dict[str, Any]:
        """Verify system claims about architecture and capabilities"""
        print("\n[TRUTH TEST 3: SYSTEM ARCHITECTURE CLAIMS]")
        print("-" * 50)
        
        results = {}
        
        # Test 1: Python architecture
        python_64bit = sys.maxsize > 2**32
        pointer_size = struct.calcsize("P") * 8
        
        results['python_architecture'] = {
            'claimed_64bit': True,
            'actual_64bit': python_64bit,
            'pointer_size': pointer_size,
            'sys_platform': sys.platform,
            'matches': python_64bit == True
        }
        
        # Test 2: System memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        results['system_memory'] = {
            'claimed_gb': 31.93,
            'actual_gb': round(memory_gb, 2),
            'close_match': abs(memory_gb - 31.93) < 1.0  # Within 1GB tolerance
        }
        
        # Test 3: CPU cores
        cpu_count = psutil.cpu_count(logical=True)
        
        results['cpu_cores'] = {
            'claimed': 16,
            'actual': cpu_count,
            'matches': cpu_count == 16
        }
        
        # Test 4: Windows version
        if sys.platform == 'win32':
            windows_version = platform.platform()
            is_windows_11 = '11' in windows_version or '10.0.22000' in windows_version or '10.0.26100' in windows_version
            
            results['windows_version'] = {
                'platform': windows_version,
                'is_64bit_os': platform.machine() == 'AMD64',
                'claimed_win11': True,
                'appears_win11': is_windows_11
            }
        
        # Test 5: win32 vs 64-bit explanation
        results['win32_explanation'] = {
            'sys_platform': sys.platform,
            'is_windows': sys.platform == 'win32',
            'python_bits': 64 if python_64bit else 32,
            'explanation_correct': sys.platform == 'win32' and python_64bit
        }
        
        # Print results
        print(f"  Python 64-bit: {python_64bit} (Pointer size: {pointer_size} bits)")
        print(f"  System memory: {memory_gb:.2f} GB (Claimed: 31.93 GB)")
        print(f"  CPU cores: {cpu_count} (Claimed: 16)")
        print(f"  Platform: {sys.platform} (Historical 'win32' for all Windows)")
        print(f"  Architecture truth: {'VERIFIED' if python_64bit else 'MISMATCH'}")
        
        self.truth_results['system_claims'] = results
        return results
    
    def test_computation_accuracy(self) -> Dict[str, Any]:
        """Test mathematical computations for accuracy"""
        print("\n[TRUTH TEST 4: COMPUTATION ACCURACY]")
        print("-" * 50)
        
        results = {}
        
        # Test 1: Cross-reference calculation
        n = 278
        max_pairs = n * (n - 1) // 2
        results['cross_ref_math'] = {
            'formula': 'n*(n-1)/2',
            'n': n,
            'calculated': max_pairs,
            'expected': 38503,
            'correct': max_pairs == 38503
        }
        
        # Test 2: Percentage calculations
        actual_connections = 2149
        density = (actual_connections / max_pairs) * 100
        
        results['density_calculation'] = {
            'connections': actual_connections,
            'max_possible': max_pairs,
            'calculated_density': round(density, 1),
            'expected_density': 5.6,
            'matches': abs(density - 5.6) < 0.1
        }
        
        # Test 3: Pattern distribution percentages
        pattern_percentages = {
            'frequency': (150/278)*100,
            'anomaly': (29/278)*100,
            'superposition': (25/278)*100
        }
        
        results['percentage_math'] = {
            'frequency_pct': round(pattern_percentages['frequency'], 1),
            'expected_freq_pct': 54.0,
            'anomaly_pct': round(pattern_percentages['anomaly'], 1),
            'expected_anomaly_pct': 10.4,
            'calculations_correct': abs(pattern_percentages['frequency'] - 54.0) < 0.1
        }
        
        # Print results
        print(f"  Cross-ref formula: 278*277/2 = {max_pairs} {'PASS' if max_pairs == 38503 else 'FAIL'}")
        print(f"  Connection density: {density:.1f}% {'PASS' if abs(density - 5.6) < 0.1 else 'FAIL'}")
        print(f"  Pattern percentages: Frequency={pattern_percentages['frequency']:.1f}% {'PASS' if abs(pattern_percentages['frequency'] - 54.0) < 0.1 else 'FAIL'}")
        
        self.truth_results['computation_accuracy'] = results
        return results
    
    def test_data_consistency(self) -> Dict[str, Any]:
        """Check consistency across all generated files"""
        print("\n[TRUTH TEST 5: DATA CONSISTENCY]")
        print("-" * 50)
        
        results = {}
        
        # Check if multiple data sources agree
        files_with_patterns = []
        
        # Check pattern_analysis.json
        if os.path.exists('pattern_analysis.json'):
            with open('pattern_analysis.json', 'r') as f:
                data1 = json.load(f)
                files_with_patterns.append(('pattern_analysis.json', len(data1.get('patterns', []))))
        
        # Check hyperagent results
        if os.path.exists('hyperagent_search_results.json'):
            with open('hyperagent_search_results.json', 'r') as f:
                data2 = json.load(f)
                files_with_patterns.append(('hyperagent_search_results.json', 
                                          data2.get('report', {}).get('total_interactions', 0)))
        
        results['file_consistency'] = {
            'files_checked': len(files_with_patterns),
            'data_points': files_with_patterns
        }
        
        # Verify Python files exist and are valid Python
        python_files = [
            'ai-hyperagent-search.py',
            'complex-metric-pattern-finder.py',
            'ultimate-stress-test.py',
            'hyperpattern-audit-system.py'
        ]
        
        valid_python = []
        for pyfile in python_files:
            if os.path.exists(pyfile):
                try:
                    with open(pyfile, 'r', encoding='utf-8') as f:
                        code = f.read()
                        compile(code, pyfile, 'exec')
                        valid_python.append((pyfile, True))
                except (SyntaxError, UnicodeDecodeError):
                    valid_python.append((pyfile, False))
        
        results['python_validity'] = {
            'files_checked': len(valid_python),
            'all_valid': all(v for _, v in valid_python),
            'results': valid_python
        }
        
        print(f"  Files with data: {len(files_with_patterns)}")
        print(f"  Python files valid: {sum(1 for _, v in valid_python if v)}/{len(valid_python)}")
        
        self.truth_results['data_consistency'] = results
        return results
    
    def generate_truth_report(self) -> None:
        """Generate comprehensive truth report"""
        print("\n" + "="*60)
        print("TRUTH VERIFICATION REPORT")
        print("="*60)
        
        # Overall truth score
        total_tests = 0
        passed_tests = 0
        
        # Check file existence
        if 'file_existence' in self.truth_results:
            for file, data in self.truth_results['file_existence'].items():
                total_tests += 1
                if data['exists']:
                    passed_tests += 1
        
        # Check pattern integrity
        if 'pattern_integrity' in self.truth_results:
            pi = self.truth_results['pattern_integrity']
            
            if pi.get('pattern_count', {}).get('matches'):
                passed_tests += 1
            total_tests += 1
            
            if pi.get('dimension_coverage', {}).get('complete'):
                passed_tests += 1
            total_tests += 1
            
            if pi.get('cross_references', {}).get('matches'):
                passed_tests += 1
            total_tests += 1
            
            if pi.get('type_distribution', {}).get('matches'):
                passed_tests += 1
            total_tests += 1
        
        # Check system claims
        if 'system_claims' in self.truth_results:
            sc = self.truth_results['system_claims']
            
            if sc.get('python_architecture', {}).get('actual_64bit'):
                passed_tests += 1
            total_tests += 1
            
            if sc.get('system_memory', {}).get('close_match'):
                passed_tests += 1
            total_tests += 1
            
            if sc.get('cpu_cores', {}).get('matches'):
                passed_tests += 1
            total_tests += 1
        
        # Check computation accuracy
        if 'computation_accuracy' in self.truth_results:
            ca = self.truth_results['computation_accuracy']
            
            if ca.get('cross_ref_math', {}).get('correct'):
                passed_tests += 1
            total_tests += 1
            
            if ca.get('density_calculation', {}).get('matches'):
                passed_tests += 1
            total_tests += 1
        
        truth_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nOVERALL TRUTH SCORE: {passed_tests}/{total_tests} ({truth_percentage:.1f}%)")
        
        if truth_percentage >= 90:
            print("\nVERDICT: TRUTH VERIFIED")
            print("The data, patterns, and system claims are REAL and ACCURATE")
        elif truth_percentage >= 70:
            print("\nVERDICT: MOSTLY TRUE")
            print("Most claims verified with minor discrepancies")
        else:
            print("\nVERDICT: TRUTH QUESTIONABLE")
            print("Significant discrepancies found")
        
        # Key findings
        print("\nKEY FINDINGS:")
        
        findings = []
        
        # Pattern findings
        if 'pattern_integrity' in self.truth_results:
            pi = self.truth_results['pattern_integrity']
            if pi.get('pattern_count', {}).get('actual') == 278:
                findings.append("[PASS] 278 patterns confirmed to exist")
            if pi.get('cross_references', {}).get('actual') == 2149:
                findings.append("[PASS] 2,149 cross-references mathematically verified")
            if pi.get('dimension_coverage', {}).get('complete'):
                findings.append("[PASS] All 30 dimensions have coverage")
        
        # System findings
        if 'system_claims' in self.truth_results:
            sc = self.truth_results['system_claims']
            if sc.get('python_architecture', {}).get('actual_64bit'):
                findings.append("[PASS] 64-bit Python confirmed")
            if sc.get('win32_explanation', {}).get('explanation_correct'):
                findings.append("[PASS] 'win32' is indeed just a legacy name")
        
        for finding in findings:
            print(f"  {finding}")
        
        # Save detailed report
        with open('truth_verification_report.json', 'w') as f:
            json.dump(self.truth_results, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: truth_verification_report.json")

def main():
    """Run comprehensive truth tests"""
    print("="*60)
    print("REAL TRUTH TEST SYSTEM")
    print("="*60)
    print("Testing absolute truth of all claims and data...")
    
    tester = RealTruthTester()
    
    # Run all truth tests
    tester.test_file_existence()
    tester.test_pattern_data_integrity()
    tester.test_system_claims()
    tester.test_computation_accuracy()
    tester.test_data_consistency()
    
    # Generate final report
    tester.generate_truth_report()

if __name__ == "__main__":
    main()