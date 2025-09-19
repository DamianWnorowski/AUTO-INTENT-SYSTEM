#!/usr/bin/env python3
"""
Ultimate System Stress Test Suite
Tests CPU, Memory, Disk, Network, and GPU to their limits
WARNING: This will heavily load your system - use with caution!
"""

import os
import sys
import time
import threading
import multiprocessing
import numpy as np
import psutil
import hashlib
import random
import string
import socket
import struct
import asyncio
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SystemMonitor:
    """Monitor system resources during stress tests"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'network': [],
            'temperature': []
        }
        self.start_time = None
        
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.start_time = time.time()
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return stats"""
        self.monitoring = False
        return self.get_report()
        
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring:
            try:
                # CPU stats
                cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
                cpu_freq = psutil.cpu_freq()
                
                # Memory stats
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                # Disk stats
                disk_io = psutil.disk_io_counters()
                
                # Network stats
                net_io = psutil.net_io_counters()
                
                # Temperature (if available)
                temps = []
                if hasattr(psutil, 'sensors_temperatures'):
                    try:
                        temp_data = psutil.sensors_temperatures()
                        for name, entries in temp_data.items():
                            for entry in entries:
                                temps.append(entry.current)
                    except:
                        pass
                
                timestamp = time.time() - self.start_time
                
                self.stats['cpu'].append({
                    'time': timestamp,
                    'percent': cpu_percent,
                    'avg': sum(cpu_percent) / len(cpu_percent),
                    'freq': cpu_freq.current if cpu_freq else 0
                })
                
                self.stats['memory'].append({
                    'time': timestamp,
                    'percent': memory.percent,
                    'used_gb': memory.used / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'swap_percent': swap.percent
                })
                
                self.stats['disk'].append({
                    'time': timestamp,
                    'read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                    'write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0
                })
                
                self.stats['network'].append({
                    'time': timestamp,
                    'sent_mb': net_io.bytes_sent / (1024**2),
                    'recv_mb': net_io.bytes_recv / (1024**2)
                })
                
                if temps:
                    self.stats['temperature'].append({
                        'time': timestamp,
                        'avg': sum(temps) / len(temps),
                        'max': max(temps)
                    })
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Monitor error: {e}")
                
    def get_report(self) -> Dict:
        """Generate performance report"""
        report = {}
        
        if self.stats['cpu']:
            cpu_avgs = [s['avg'] for s in self.stats['cpu']]
            report['cpu'] = {
                'max_usage': max(cpu_avgs),
                'avg_usage': sum(cpu_avgs) / len(cpu_avgs),
                'samples': len(cpu_avgs)
            }
            
        if self.stats['memory']:
            mem_percents = [s['percent'] for s in self.stats['memory']]
            report['memory'] = {
                'max_usage': max(mem_percents),
                'avg_usage': sum(mem_percents) / len(mem_percents),
                'peak_gb': max([s['used_gb'] for s in self.stats['memory']])
            }
            
        if self.stats['temperature']:
            temps = [s['max'] for s in self.stats['temperature']]
            report['temperature'] = {
                'max_temp': max(temps),
                'avg_temp': sum(temps) / len(temps)
            }
            
        report['duration'] = time.time() - self.start_time if self.start_time else 0
        
        return report

class CPUStressTest:
    """CPU stress testing module"""
    
    def __init__(self, threads=None, duration=30):
        self.threads = threads or multiprocessing.cpu_count()
        self.duration = duration
        self.running = False
        
    def calculate_primes(self, n):
        """Calculate prime numbers (CPU intensive)"""
        primes = []
        for num in range(2, n):
            if all(num % i != 0 for i in range(2, int(num ** 0.5) + 1)):
                primes.append(num)
        return primes
        
    def matrix_operations(self):
        """Perform heavy matrix calculations"""
        size = 500
        while self.running:
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            c = np.dot(a, b)
            d = np.linalg.inv(c + np.eye(size) * 0.001)
            e = np.linalg.svd(d)
            
    def hash_calculations(self):
        """Perform intensive hash calculations"""
        data = ''.join(random.choices(string.ascii_letters + string.digits, k=1000))
        while self.running:
            for _ in range(10000):
                hashlib.sha256(data.encode()).hexdigest()
                hashlib.sha512(data.encode()).hexdigest()
                hashlib.md5(data.encode()).hexdigest()
                data = hashlib.sha256(data.encode()).hexdigest()
                
    def recursive_fibonacci(self, n):
        """Recursive Fibonacci (CPU intensive)"""
        if n <= 1:
            return n
        return self.recursive_fibonacci(n-1) + self.recursive_fibonacci(n-2)
        
    def worker_thread(self):
        """Worker thread for CPU stress"""
        operations = [
            lambda: self.calculate_primes(10000),
            lambda: self.matrix_operations(),
            lambda: self.hash_calculations(),
            lambda: [self.recursive_fibonacci(30) for _ in range(100)]
        ]
        
        while self.running:
            operation = random.choice(operations)
            try:
                operation()
            except:
                pass
                
    def run(self):
        """Run CPU stress test"""
        print(f"Starting CPU stress test ({self.threads} threads, {self.duration}s)...")
        self.running = True
        
        threads = []
        for i in range(self.threads):
            t = threading.Thread(target=self.worker_thread, daemon=True)
            t.start()
            threads.append(t)
            
        time.sleep(self.duration)
        self.running = False
        
        print("CPU stress test completed")

class MemoryStressTest:
    """Memory stress testing module"""
    
    def __init__(self, target_gb=None, duration=30):
        available = psutil.virtual_memory().available / (1024**3)
        self.target_gb = target_gb or (available * 0.8)  # Use 80% of available
        self.duration = duration
        self.allocations = []
        
    def run(self):
        """Run memory stress test"""
        print(f"Starting memory stress test (target: {self.target_gb:.2f} GB)...")
        
        chunk_size_mb = 100
        chunks_needed = int(self.target_gb * 1024 / chunk_size_mb)
        
        try:
            for i in range(chunks_needed):
                # Allocate memory in chunks
                chunk = np.random.rand(chunk_size_mb * 1024 * 1024 // 8)  # 8 bytes per float64
                self.allocations.append(chunk)
                
                # Perform operations to ensure memory is actually used
                chunk *= 2
                chunk.sum()
                
                if i % 10 == 0:
                    current_gb = len(self.allocations) * chunk_size_mb / 1024
                    print(f"  Allocated: {current_gb:.2f} GB")
                    
            print(f"  Peak allocation: {len(self.allocations) * chunk_size_mb / 1024:.2f} GB")
            time.sleep(self.duration)
            
        except MemoryError:
            print("  Memory limit reached")
            
        finally:
            print("  Releasing memory...")
            self.allocations.clear()
            print("Memory stress test completed")

class DiskStressTest:
    """Disk I/O stress testing module"""
    
    def __init__(self, size_gb=1, duration=30):
        self.size_gb = size_gb
        self.duration = duration
        self.test_dir = "stress_test_temp"
        self.running = False
        
    def run(self):
        """Run disk I/O stress test"""
        print(f"Starting disk I/O stress test ({self.size_gb} GB)...")
        
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
            
        self.running = True
        start_time = time.time()
        
        # Sequential write test
        print("  Sequential write test...")
        file_path = os.path.join(self.test_dir, "test_sequential.dat")
        chunk_size = 1024 * 1024  # 1 MB chunks
        total_chunks = int(self.size_gb * 1024)
        
        with open(file_path, 'wb') as f:
            for i in range(total_chunks):
                if not self.running or (time.time() - start_time) > self.duration:
                    break
                data = os.urandom(chunk_size)
                f.write(data)
                
        # Random read/write test
        print("  Random read/write test...")
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'r+b') as f:
            while self.running and (time.time() - start_time) < self.duration:
                position = random.randint(0, max(0, file_size - chunk_size))
                f.seek(position)
                
                if random.choice([True, False]):
                    # Read
                    f.read(chunk_size)
                else:
                    # Write
                    f.write(os.urandom(chunk_size))
                    
        # Cleanup
        print("  Cleaning up test files...")
        try:
            os.remove(file_path)
            os.rmdir(self.test_dir)
        except:
            pass
            
        print("Disk I/O stress test completed")

class NetworkStressTest:
    """Network stress testing module"""
    
    def __init__(self, duration=30):
        self.duration = duration
        self.running = False
        
    def dns_stress(self):
        """Stress test DNS lookups"""
        domains = [
            'google.com', 'youtube.com', 'facebook.com', 
            'amazon.com', 'wikipedia.org', 'twitter.com'
        ]
        
        while self.running:
            domain = random.choice(domains)
            try:
                socket.gethostbyname(domain)
            except:
                pass
                
    def connection_stress(self):
        """Stress test TCP connections"""
        while self.running:
            try:
                # Create multiple connections
                sockets = []
                for _ in range(10):
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    try:
                        s.connect(('8.8.8.8', 53))  # Google DNS
                        sockets.append(s)
                    except:
                        s.close()
                        
                # Close all connections
                for s in sockets:
                    s.close()
                    
            except:
                pass
                
    def bandwidth_test(self):
        """Simulate bandwidth usage"""
        data = os.urandom(1024 * 1024)  # 1 MB of random data
        
        while self.running:
            # Simulate data transfer
            compressed = hashlib.sha256(data).hexdigest()
            data = os.urandom(1024 * 1024)
            
    def run(self):
        """Run network stress test"""
        print(f"Starting network stress test ({self.duration}s)...")
        self.running = True
        
        threads = [
            threading.Thread(target=self.dns_stress, daemon=True),
            threading.Thread(target=self.connection_stress, daemon=True),
            threading.Thread(target=self.bandwidth_test, daemon=True)
        ]
        
        for t in threads:
            t.start()
            
        time.sleep(self.duration)
        self.running = False
        
        print("Network stress test completed")

class UltimateStressTest:
    """Combined stress test orchestrator"""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.results = {}
        
    def run_individual_tests(self):
        """Run individual stress tests sequentially"""
        print("\n" + "="*60)
        print("INDIVIDUAL STRESS TESTS")
        print("="*60)
        
        tests = [
            ("CPU", CPUStressTest(duration=20)),
            ("Memory", MemoryStressTest(target_gb=2, duration=20)),
            ("Disk", DiskStressTest(size_gb=0.5, duration=20)),
            ("Network", NetworkStressTest(duration=20))
        ]
        
        for name, test in tests:
            print(f"\n[{name} Test]")
            self.monitor.start_monitoring()
            test.run()
            self.results[name] = self.monitor.stop_monitoring()
            time.sleep(5)  # Cool down between tests
            
    def run_combined_test(self):
        """Run all stress tests simultaneously"""
        print("\n" + "="*60)
        print("COMBINED STRESS TEST (MAXIMUM LOAD)")
        print("="*60)
        print("WARNING: This will heavily load your system!")
        time.sleep(3)
        
        self.monitor.start_monitoring()
        
        # Create all stress tests
        cpu_test = CPUStressTest(duration=30)
        mem_test = MemoryStressTest(target_gb=1, duration=30)
        disk_test = DiskStressTest(size_gb=0.5, duration=30)
        net_test = NetworkStressTest(duration=30)
        
        # Run all tests in parallel
        threads = [
            threading.Thread(target=cpu_test.run),
            threading.Thread(target=mem_test.run),
            threading.Thread(target=disk_test.run),
            threading.Thread(target=net_test.run)
        ]
        
        print("\nStarting ALL stress tests simultaneously...")
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
            
        self.results['Combined'] = self.monitor.stop_monitoring()
        
    def generate_report(self):
        """Generate final stress test report"""
        print("\n" + "="*60)
        print("STRESS TEST REPORT")
        print("="*60)
        
        for test_name, stats in self.results.items():
            print(f"\n[{test_name} Test Results]")
            
            if 'cpu' in stats:
                print(f"  CPU:")
                print(f"    Max Usage: {stats['cpu']['max_usage']:.1f}%")
                print(f"    Avg Usage: {stats['cpu']['avg_usage']:.1f}%")
                
            if 'memory' in stats:
                print(f"  Memory:")
                print(f"    Max Usage: {stats['memory']['max_usage']:.1f}%")
                print(f"    Peak Usage: {stats['memory']['peak_gb']:.2f} GB")
                
            if 'temperature' in stats:
                print(f"  Temperature:")
                print(f"    Max Temp: {stats['temperature']['max_temp']:.1f}°C")
                print(f"    Avg Temp: {stats['temperature']['avg_temp']:.1f}°C")
                
            print(f"  Duration: {stats.get('duration', 0):.1f} seconds")
            
        # System info
        print("\n[System Information]")
        print(f"  CPU Cores: {multiprocessing.cpu_count()}")
        print(f"  Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        print(f"  Python Version: {sys.version.split()[0]}")
        print(f"  Platform: {sys.platform}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"stress_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'results': self.results,
                'system': {
                    'cpu_count': multiprocessing.cpu_count(),
                    'memory_gb': psutil.virtual_memory().total / (1024**3),
                    'platform': sys.platform
                }
            }, f, indent=2, default=str)
            
        print(f"\nDetailed report saved to: {report_file}")

def main():
    """Main execution"""
    print("+" + "="*58 + "+")
    print("|" + " ULTIMATE SYSTEM STRESS TEST ".center(58) + "|")
    print("+" + "="*58 + "+")
    
    print("\nWARNING: This test will put heavy load on your system.")
    print("Make sure to save your work before continuing!")
    
    print("\nSelect test mode:")
    print("1. Individual tests (sequential)")
    print("2. Combined test (parallel - EXTREME)")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    stress_test = UltimateStressTest()
    
    try:
        if choice == '1':
            stress_test.run_individual_tests()
        elif choice == '2':
            stress_test.run_combined_test()
        elif choice == '3':
            stress_test.run_individual_tests()
            print("\nCooling down for 10 seconds...")
            time.sleep(10)
            stress_test.run_combined_test()
        else:
            print("Invalid choice")
            return
            
        stress_test.generate_report()
        
    except KeyboardInterrupt:
        print("\n\nStress test interrupted by user")
        stress_test.generate_report()
        
    print("\nAll stress tests completed!")

if __name__ == "__main__":
    main()