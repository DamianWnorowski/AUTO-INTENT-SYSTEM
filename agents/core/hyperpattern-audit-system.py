#!/usr/bin/env python3
"""
HyperPattern Audit System
Validates and audits the 278 discovered patterns across 30 dimensions
Ensures pattern integrity, cross-references relationships, and identifies meta-patterns
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib
import networkx as nx
from scipy import stats
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AuditResult:
    """Represents an audit finding"""
    pattern_id: str
    check_type: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    confidence: float
    message: str
    metadata: Dict = field(default_factory=dict)

class HyperPatternAuditor:
    """
    Comprehensive audit system for discovered patterns
    Validates: integrity, consistency, relationships, and meta-patterns
    """
    
    def __init__(self, pattern_file: str = "pattern_analysis.json"):
        self.pattern_file = pattern_file
        self.patterns = []
        self.audit_results = []
        self.meta_patterns = []
        self.pattern_graph = nx.Graph()
        self.validation_stats = {}
        
        # Load patterns
        self.load_patterns()
        
    def load_patterns(self):
        """Load patterns from JSON file"""
        try:
            with open(self.pattern_file, 'r') as f:
                data = json.load(f)
                self.patterns = data.get('patterns', [])
                self.pattern_library = data.get('pattern_library', {})
                print(f"Loaded {len(self.patterns)} patterns from {self.pattern_file}")
        except FileNotFoundError:
            print(f"Pattern file {self.pattern_file} not found. Using sample data.")
            self.generate_sample_patterns()
    
    def generate_sample_patterns(self):
        """Generate sample patterns matching the discovered 278 patterns"""
        pattern_distribution = {
            'frequency': 150,
            'anomaly': 29,
            'superposition': 25,
            'chaotic': 20,
            'mutual_information': 20,
            'periodic': 13,
            'correlation': 10,
            'high_entropy': 6,
            'entanglement': 2,
            'principal_component': 1,
            'clustering': 1,
            'optimal_clustering': 1
        }
        
        self.patterns = []
        pattern_id = 0
        
        for ptype, count in pattern_distribution.items():
            for i in range(count):
                pattern = {
                    'id': hashlib.md5(f"{ptype}_{i}".encode()).hexdigest()[:8],
                    'type': ptype,
                    'dimensions': self._generate_dimensions(ptype),
                    'strength': np.random.beta(5, 2),  # Skewed towards higher values
                    'frequency': np.random.exponential(2) if 'frequency' in ptype else 0,
                    'metadata': self._generate_metadata(ptype, i)
                }
                self.patterns.append(pattern)
                pattern_id += 1
        
        print(f"Generated {len(self.patterns)} sample patterns")
    
    def _generate_dimensions(self, ptype: str) -> List[int]:
        """Generate appropriate dimensions for pattern type"""
        if ptype in ['clustering', 'optimal_clustering']:
            return list(range(30))  # All 30 dimensions
        elif ptype == 'entanglement':
            return list(np.random.choice(30, size=np.random.randint(3, 6), replace=False))
        elif ptype in ['correlation', 'mutual_information']:
            dims = np.random.choice(30, size=2, replace=False)
            return [int(dims[0]), int(dims[1])]
        else:
            return [int(np.random.randint(0, 30))]
    
    def _generate_metadata(self, ptype: str, index: int) -> Dict:
        """Generate appropriate metadata for pattern type"""
        metadata = {}
        
        if ptype == 'mutual_information':
            metadata['mutual_information'] = 1.5 + np.random.random() * 0.5
            metadata['features'] = [index % 30, (index + 1) % 30]
        elif ptype == 'frequency':
            metadata['frequency_hz'] = np.random.uniform(0.01, 0.5)
            metadata['power'] = np.random.exponential(1000)
            metadata['harmonic_order'] = index + 1
        elif ptype == 'chaotic':
            metadata['lyapunov_exponent'] = np.random.uniform(0.01, 0.5)
            metadata['chaotic'] = True
        elif ptype == 'superposition':
            metadata['n_states'] = np.random.randint(2, 6)
            metadata['state_positions'] = list(np.random.randn(metadata['n_states']))
        elif ptype == 'entanglement':
            metadata['group_size'] = np.random.randint(3, 6)
            metadata['avg_correlation'] = 0.8 + np.random.random() * 0.15
        
        return metadata
    
    def audit_pattern_integrity(self):
        """Check integrity of each pattern"""
        print("\n[1/5] AUDITING PATTERN INTEGRITY...")
        print("-" * 50)
        
        for pattern in self.patterns:
            # Check required fields
            required_fields = ['id', 'type', 'dimensions', 'strength']
            for field in required_fields:
                if field not in pattern:
                    self.audit_results.append(AuditResult(
                        pattern_id=pattern.get('id', 'UNKNOWN'),
                        check_type='integrity',
                        status='FAIL',
                        confidence=1.0,
                        message=f"Missing required field: {field}"
                    ))
            
            # Validate strength range
            strength = pattern.get('strength', 0)
            if not 0 <= strength <= 10:
                self.audit_results.append(AuditResult(
                    pattern_id=pattern['id'],
                    check_type='integrity',
                    status='WARNING',
                    confidence=0.8,
                    message=f"Strength {strength:.3f} outside normal range [0, 10]"
                ))
            
            # Validate dimensions
            dims = pattern.get('dimensions', [])
            if any(d < 0 or d >= 30 for d in dims):
                self.audit_results.append(AuditResult(
                    pattern_id=pattern['id'],
                    check_type='integrity',
                    status='FAIL',
                    confidence=1.0,
                    message=f"Invalid dimensions: {dims}"
                ))
            
            # Pattern-specific validation
            self._validate_pattern_specific(pattern)
        
        integrity_pass = sum(1 for r in self.audit_results if r.check_type == 'integrity' and r.status == 'PASS')
        integrity_total = sum(1 for r in self.audit_results if r.check_type == 'integrity')
        
        print(f"Integrity Check: {len(self.patterns) - integrity_total + integrity_pass}/{len(self.patterns)} patterns valid")
    
    def _validate_pattern_specific(self, pattern: Dict):
        """Validate pattern based on its type"""
        ptype = pattern.get('type')
        metadata = pattern.get('metadata', {})
        
        if ptype == 'mutual_information':
            mi = metadata.get('mutual_information', 0)
            if mi > 0:  # Valid MI
                self.audit_results.append(AuditResult(
                    pattern_id=pattern['id'],
                    check_type='integrity',
                    status='PASS',
                    confidence=1.0,
                    message=f"Valid mutual information: {mi:.3f} bits"
                ))
        
        elif ptype == 'chaotic':
            lyapunov = metadata.get('lyapunov_exponent', 0)
            if lyapunov > 0:  # Positive Lyapunov indicates chaos
                self.audit_results.append(AuditResult(
                    pattern_id=pattern['id'],
                    check_type='integrity',
                    status='PASS',
                    confidence=1.0,
                    message=f"Valid chaotic pattern: λ = {lyapunov:.3f}"
                ))
        
        elif ptype == 'entanglement':
            group_size = metadata.get('group_size', 0)
            avg_corr = metadata.get('avg_correlation', 0)
            if group_size >= 3 and avg_corr > 0.8:
                self.audit_results.append(AuditResult(
                    pattern_id=pattern['id'],
                    check_type='integrity',
                    status='PASS',
                    confidence=0.95,
                    message=f"Valid entanglement: {group_size} features, r={avg_corr:.3f}"
                ))
    
    def audit_cross_references(self):
        """Check relationships between patterns"""
        print("\n[2/5] AUDITING CROSS-REFERENCES...")
        print("-" * 50)
        
        # Build pattern graph
        for i, p1 in enumerate(self.patterns):
            for j, p2 in enumerate(self.patterns[i+1:], i+1):
                # Check dimension overlap
                dims1 = set(p1.get('dimensions', []))
                dims2 = set(p2.get('dimensions', []))
                overlap = dims1.intersection(dims2)
                
                if overlap:
                    weight = len(overlap) / max(len(dims1), len(dims2))
                    self.pattern_graph.add_edge(p1['id'], p2['id'], weight=weight)
        
        # Analyze graph properties
        if len(self.pattern_graph) > 0:
            components = list(nx.connected_components(self.pattern_graph))
            
            self.audit_results.append(AuditResult(
                pattern_id='GLOBAL',
                check_type='cross_reference',
                status='PASS',
                confidence=0.9,
                message=f"Pattern network has {len(components)} connected components",
                metadata={'components': len(components), 'nodes': len(self.pattern_graph)}
            ))
            
            # Check for critical nodes (hubs)
            if self.pattern_graph.edges():
                centrality = nx.degree_centrality(self.pattern_graph)
                top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for hub_id, centrality_score in top_hubs:
                    if centrality_score > 0.1:  # Significant hub
                        self.audit_results.append(AuditResult(
                            pattern_id=hub_id,
                            check_type='cross_reference',
                            status='PASS',
                            confidence=0.85,
                            message=f"Hub pattern with centrality {centrality_score:.3f}",
                            metadata={'centrality': centrality_score}
                        ))
        
        print(f"Cross-reference network: {len(self.pattern_graph)} nodes, {len(self.pattern_graph.edges())} edges")
    
    def audit_statistical_consistency(self):
        """Check statistical consistency of patterns"""
        print("\n[3/5] AUDITING STATISTICAL CONSISTENCY...")
        print("-" * 50)
        
        # Group patterns by type
        type_groups = defaultdict(list)
        for p in self.patterns:
            type_groups[p['type']].append(p)
        
        for ptype, patterns in type_groups.items():
            strengths = [p['strength'] for p in patterns]
            
            if len(strengths) > 3:
                # Check for outliers using IQR
                q1, q3 = np.percentile(strengths, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = [p for p in patterns if p['strength'] < lower_bound or p['strength'] > upper_bound]
                
                if outliers:
                    for outlier in outliers:
                        self.audit_results.append(AuditResult(
                            pattern_id=outlier['id'],
                            check_type='statistical',
                            status='WARNING',
                            confidence=0.7,
                            message=f"Statistical outlier in {ptype} patterns (strength={outlier['strength']:.3f})"
                        ))
                
                # Check distribution
                if len(strengths) > 10:
                    _, p_value = stats.normaltest(strengths)
                    distribution = "normal" if p_value > 0.05 else "non-normal"
                    
                    self.audit_results.append(AuditResult(
                        pattern_id=f"{ptype}_GROUP",
                        check_type='statistical',
                        status='PASS',
                        confidence=0.8,
                        message=f"{ptype} patterns show {distribution} distribution (p={p_value:.3f})",
                        metadata={'p_value': p_value, 'n_patterns': len(patterns)}
                    ))
        
        print(f"Statistical analysis complete for {len(type_groups)} pattern types")
    
    def audit_dimensional_coverage(self):
        """Check coverage across 30 dimensions"""
        print("\n[4/5] AUDITING DIMENSIONAL COVERAGE...")
        print("-" * 50)
        
        # Track dimension usage
        dimension_coverage = Counter()
        dimension_patterns = defaultdict(list)
        
        for pattern in self.patterns:
            for dim in pattern.get('dimensions', []):
                dimension_coverage[dim] += 1
                dimension_patterns[dim].append(pattern['id'])
        
        # Check for uncovered dimensions
        all_dims = set(range(30))
        covered_dims = set(dimension_coverage.keys())
        uncovered = all_dims - covered_dims
        
        if uncovered:
            self.audit_results.append(AuditResult(
                pattern_id='DIMENSIONAL',
                check_type='coverage',
                status='WARNING',
                confidence=0.9,
                message=f"{len(uncovered)} dimensions have no patterns: {sorted(uncovered)}",
                metadata={'uncovered_dims': list(uncovered)}
            ))
        else:
            self.audit_results.append(AuditResult(
                pattern_id='DIMENSIONAL',
                check_type='coverage',
                status='PASS',
                confidence=1.0,
                message="All 30 dimensions have pattern coverage"
            ))
        
        # Check for over-represented dimensions
        if dimension_coverage:
            max_coverage = max(dimension_coverage.values())
            avg_coverage = np.mean(list(dimension_coverage.values()))
            
            over_represented = [dim for dim, count in dimension_coverage.items() 
                              if count > avg_coverage + 2 * np.std(list(dimension_coverage.values()))]
            
            if over_represented:
                self.audit_results.append(AuditResult(
                    pattern_id='DIMENSIONAL',
                    check_type='coverage',
                    status='WARNING',
                    confidence=0.75,
                    message=f"Dimensions {over_represented} are over-represented",
                    metadata={'over_represented': over_represented}
                ))
        
        coverage_percent = len(covered_dims) / 30 * 100
        print(f"Dimensional coverage: {coverage_percent:.1f}% ({len(covered_dims)}/30 dimensions)")
    
    def identify_meta_patterns(self):
        """Identify patterns of patterns (meta-patterns)"""
        print("\n[5/5] IDENTIFYING META-PATTERNS...")
        print("-" * 50)
        
        # Meta-pattern 1: Co-occurring pattern types
        co_occurrences = defaultdict(int)
        for i, p1 in enumerate(self.patterns):
            for p2 in self.patterns[i+1:]:
                if set(p1['dimensions']).intersection(set(p2['dimensions'])):
                    key = tuple(sorted([p1['type'], p2['type']]))
                    co_occurrences[key] += 1
        
        top_cooccur = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)[:5]
        for (type1, type2), count in top_cooccur:
            if count > 5:  # Significant co-occurrence
                self.meta_patterns.append({
                    'type': 'co_occurrence',
                    'patterns': [type1, type2],
                    'count': count,
                    'significance': count / len(self.patterns)
                })
        
        # Meta-pattern 2: Pattern cascades (one pattern leading to another)
        strength_correlations = defaultdict(list)
        for ptype in set(p['type'] for p in self.patterns):
            type_patterns = [p for p in self.patterns if p['type'] == ptype]
            if len(type_patterns) > 5:
                avg_strength = np.mean([p['strength'] for p in type_patterns])
                strength_correlations[ptype] = avg_strength
        
        # Find strength hierarchy
        if strength_correlations:
            strength_hierarchy = sorted(strength_correlations.items(), key=lambda x: x[1], reverse=True)
            
            self.meta_patterns.append({
                'type': 'strength_hierarchy',
                'hierarchy': [t for t, s in strength_hierarchy],
                'strengths': [s for t, s in strength_hierarchy],
                'significance': 'ordered_by_average_strength'
            })
        
        # Meta-pattern 3: Dimensional clustering
        if len(self.patterns) > 10:
            dim_vectors = []
            for p in self.patterns[:100]:  # Limit for performance
                vec = np.zeros(30)
                for d in p['dimensions']:
                    vec[d] = p['strength']
                dim_vectors.append(vec)
            
            if len(dim_vectors) > 3:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=min(5, len(dim_vectors)//2), random_state=42)
                clusters = kmeans.fit_predict(dim_vectors)
                
                self.meta_patterns.append({
                    'type': 'dimensional_clusters',
                    'n_clusters': len(set(clusters)),
                    'cluster_sizes': dict(Counter(clusters)),
                    'significance': 'patterns_group_in_dimensional_space'
                })
        
        print(f"Identified {len(self.meta_patterns)} meta-patterns")
    
    def generate_audit_report(self):
        """Generate comprehensive audit report"""
        print("\n" + "="*60)
        print("HYPERPATTERN AUDIT REPORT")
        print("="*60)
        
        # Summary statistics
        total_patterns = len(self.patterns)
        pass_count = sum(1 for r in self.audit_results if r.status == 'PASS')
        fail_count = sum(1 for r in self.audit_results if r.status == 'FAIL')
        warning_count = sum(1 for r in self.audit_results if r.status == 'WARNING')
        
        print(f"\n📊 AUDIT SUMMARY")
        print(f"Total Patterns Audited: {total_patterns}")
        print(f"Audit Checks Performed: {len(self.audit_results)}")
        print(f"  ✅ PASS: {pass_count}")
        print(f"  ❌ FAIL: {fail_count}")
        print(f"  ⚠️  WARNING: {warning_count}")
        
        # Pattern type distribution
        type_dist = Counter(p['type'] for p in self.patterns)
        print(f"\n📈 PATTERN DISTRIBUTION (Total: {total_patterns})")
        for ptype, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * (count // 3)
            print(f"  {ptype:20s}: {count:3d} {bar}")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS")
        
        # Strongest patterns
        strongest = sorted(self.patterns, key=lambda x: x.get('strength', 0), reverse=True)[:3]
        print(f"\nStrongest Patterns:")
        for i, p in enumerate(strongest, 1):
            print(f"  {i}. {p['type']} (ID: {p['id'][:8]}) - Strength: {p['strength']:.3f}")
        
        # Most complex patterns
        most_complex = sorted(self.patterns, key=lambda x: len(x.get('dimensions', [])), reverse=True)[:3]
        print(f"\nMost Complex Patterns (Multi-dimensional):")
        for i, p in enumerate(most_complex, 1):
            print(f"  {i}. {p['type']} - {len(p['dimensions'])} dimensions")
        
        # Meta-patterns
        if self.meta_patterns:
            print(f"\n🔮 META-PATTERNS DISCOVERED")
            for mp in self.meta_patterns[:3]:
                if mp['type'] == 'co_occurrence':
                    print(f"  • Co-occurrence: {mp['patterns'][0]} ↔ {mp['patterns'][1]} ({mp['count']} times)")
                elif mp['type'] == 'strength_hierarchy':
                    print(f"  • Strength Hierarchy: {' > '.join(mp['hierarchy'][:5])}")
                elif mp['type'] == 'dimensional_clusters':
                    print(f"  • Dimensional Clusters: {mp['n_clusters']} distinct groups found")
        
        # Critical warnings
        critical_warnings = [r for r in self.audit_results if r.status == 'FAIL']
        if critical_warnings:
            print(f"\n⚠️  CRITICAL ISSUES ({len(critical_warnings)})")
            for warning in critical_warnings[:5]:
                print(f"  • {warning.pattern_id}: {warning.message}")
        
        # Network analysis
        if self.pattern_graph and self.pattern_graph.edges():
            print(f"\n🌐 PATTERN NETWORK ANALYSIS")
            print(f"  Nodes (patterns): {len(self.pattern_graph)}")
            print(f"  Edges (relationships): {len(self.pattern_graph.edges())}")
            if nx.is_connected(self.pattern_graph):
                print(f"  Network is CONNECTED")
                diameter = nx.diameter(self.pattern_graph)
                print(f"  Network diameter: {diameter}")
            else:
                components = list(nx.connected_components(self.pattern_graph))
                print(f"  Network has {len(components)} components")
                largest = max(components, key=len)
                print(f"  Largest component: {len(largest)} patterns")
        
        # Save detailed report
        self.save_audit_results()
        
        print(f"\n💾 Detailed audit saved to: hyperpattern_audit_report.json")
    
    def save_audit_results(self):
        """Save audit results to file"""
        report = {
            'summary': {
                'total_patterns': len(self.patterns),
                'audit_checks': len(self.audit_results),
                'pass': sum(1 for r in self.audit_results if r.status == 'PASS'),
                'fail': sum(1 for r in self.audit_results if r.status == 'FAIL'),
                'warning': sum(1 for r in self.audit_results if r.status == 'WARNING')
            },
            'pattern_distribution': dict(Counter(p['type'] for p in self.patterns)),
            'audit_results': [
                {
                    'pattern_id': r.pattern_id,
                    'check_type': r.check_type,
                    'status': r.status,
                    'confidence': r.confidence,
                    'message': r.message,
                    'metadata': r.metadata
                }
                for r in self.audit_results
            ],
            'meta_patterns': self.meta_patterns,
            'network_stats': {
                'nodes': len(self.pattern_graph),
                'edges': len(self.pattern_graph.edges()),
                'connected': nx.is_connected(self.pattern_graph) if self.pattern_graph else False
            }
        }
        
        with open('hyperpattern_audit_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def visualize_audit_results(self):
        """Create visualization of audit results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Pattern type distribution
        type_counts = Counter(p['type'] for p in self.patterns)
        axes[0, 0].bar(range(len(type_counts)), list(type_counts.values()))
        axes[0, 0].set_xticks(range(len(type_counts)))
        axes[0, 0].set_xticklabels(list(type_counts.keys()), rotation=45, ha='right')
        axes[0, 0].set_title('Pattern Type Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Strength distribution
        strengths = [p['strength'] for p in self.patterns]
        axes[0, 1].hist(strengths, bins=20, edgecolor='black')
        axes[0, 1].set_title('Pattern Strength Distribution')
        axes[0, 1].set_xlabel('Strength')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Dimensional coverage heatmap
        dim_matrix = np.zeros((len(self.patterns[:50]), 30))  # Limit to 50 patterns for visibility
        for i, p in enumerate(self.patterns[:50]):
            for d in p['dimensions']:
                dim_matrix[i, d] = p['strength']
        
        im = axes[1, 0].imshow(dim_matrix.T, aspect='auto', cmap='YlOrRd')
        axes[1, 0].set_title('Dimensional Coverage (First 50 Patterns)')
        axes[1, 0].set_xlabel('Pattern Index')
        axes[1, 0].set_ylabel('Dimension')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Audit results pie chart
        status_counts = Counter(r.status for r in self.audit_results)
        colors = {'PASS': 'green', 'FAIL': 'red', 'WARNING': 'orange'}
        axes[1, 1].pie(status_counts.values(), labels=status_counts.keys(), 
                      colors=[colors.get(s, 'gray') for s in status_counts.keys()],
                      autopct='%1.1f%%')
        axes[1, 1].set_title('Audit Results Distribution')
        
        plt.tight_layout()
        plt.savefig('hyperpattern_audit_visualization.png', dpi=150, bbox_inches='tight')
        print("\n📊 Visualization saved to: hyperpattern_audit_visualization.png")

def main():
    """Main execution"""
    print("="*60)
    print("HYPERPATTERN AUDIT SYSTEM")
    print("="*60)
    print("\nAuditing 278 patterns across 30 dimensions...")
    
    # Initialize auditor
    auditor = HyperPatternAuditor()
    
    # Run comprehensive audit
    auditor.audit_pattern_integrity()
    auditor.audit_cross_references()
    auditor.audit_statistical_consistency()
    auditor.audit_dimensional_coverage()
    auditor.identify_meta_patterns()
    
    # Generate report
    auditor.generate_audit_report()
    
    # Create visualizations
    try:
        auditor.visualize_audit_results()
    except Exception as e:
        print(f"\nVisualization error: {e}")
    
    print("\n" + "="*60)
    print("AUDIT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()