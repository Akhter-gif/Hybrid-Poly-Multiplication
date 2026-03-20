"""
Unit tests for all algorithms
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
from typing import List, Tuple

from core.polynomial import Polynomial
from core.karatsuba import karatsuba_multiply, test_karatsuba
from core.toomcook import toomcook_multiply, test_toomcook
from core.ntt import ntt_multiply, naive_multiply, test_ntt
from core.hybrid import hybrid_multiply, test_hybrid

def test_consistency():
    """Test if all algorithms produce same results"""
    print("\n🔍 Testing algorithm consistency...")
    
    q = 3329
    test_sizes = [8, 16, 32, 64, 128]
    
    for size in test_sizes:
        # Generate random polynomials
        A = [random.randint(0, q-1) for _ in range(size)]
        B = [random.randint(0, q-1) for _ in range(size)]
        
        # Get reference result (naive)
        expected = naive_multiply(A, B, q)
        
        # Test Karatsuba (if size appropriate)
        if size <= 64:
            karat_result = karatsuba_multiply(A, B, q)
            min_len = min(len(expected), len(karat_result))
            assert expected[:min_len] == karat_result[:min_len], \
                   f"Karatsuba mismatch for size {size}"
        
        # Test Toom-Cook (if size appropriate)
        if 32 <= size <= 256:
            toom_result = toomcook_multiply(A, B, q)
            min_len = min(len(expected), len(toom_result))
            assert expected[:min_len] == toom_result[:min_len], \
                   f"Toom-Cook mismatch for size {size}"
        
        # Test NTT
        ntt_result = ntt_multiply(A, B, q)
        min_len = min(len(expected), len(ntt_result))
        assert expected[:min_len] == ntt_result[:min_len], \
               f"NTT mismatch for size {size}"
        
        # Test hybrid
        hybrid_result = hybrid_multiply(A, B, q)
        min_len = min(len(expected), len(hybrid_result["result"]))
        assert expected[:min_len] == hybrid_result["result"][:min_len], \
               f"Hybrid mismatch for size {size}"
        
        print(f"  ✓ Size {size:3d} passed")
    
    print("✅ All consistency tests passed!")

def test_ring_reduction():
    """Test modulo (x^n + 1) reduction"""
    print("\n🔍 Testing ring reduction...")
    
    q = 3329
    n = 4
    
    # Test x^4 should become -1 mod (x^4 + 1)
    poly = Polynomial([0, 0, 0, 0, 1], q, n)
    expected = [q-1, 0, 0, 0]  # -1 mod q
    assert poly.coeffs == expected, \
           f"Ring reduction failed: {poly.coeffs} != {expected}"
    
    # Test (x^5 + x^4) reduction
    poly = Polynomial([0, 0, 0, 0, 1, 1], q, n)
    # x^4 becomes -1, x^5 becomes -x
    expected = [q-1, q-1, 0, 0]  # -1 - x
    assert poly.coeffs == expected, \
           f"Ring reduction failed: {poly.coeffs} != {expected}"
    
    print("✅ Ring reduction tests passed!")

def test_rlwe_operations():
    """Test RLWE-specific operations"""
    print("\n🔍 Testing RLWE operations...")
    
    q = 3329
    n = 256
    
    # Create random polynomials
    a = Polynomial.random(n-1, q)
    s = Polynomial.random(n-1, q)
    e = Polynomial([random.randint(-3, 3) for _ in range(n)], q)
    
    # Compute b = a*s + e mod (x^n + 1)
    a_s = a * s
    a_s.mod_ring(n)
    b = a_s + e
    
    # Verify degree
    assert len(b) <= n, f"RLWE result too large: {len(b)} > {n}"
    
    print("✅ RLWE operations tests passed!")

def test_performance():
    """Performance benchmark"""
    print("\n📊 Running performance benchmarks...")
    
    q = 3329
    sizes = [16, 32, 64, 128, 256, 512]
    
    print("\nSize   Karatsuba  Toom-Cook  NTT        Hybrid     Fastest")
    print("-" * 65)
    
    for size in sizes:
        A = [i % q for i in range(size)]
        B = [(i * 2) % q for i in range(size)]
        
        times = {}
        
        # Test each algorithm
        for algo in ["karatsuba", "toomcook", "ntt", "hybrid"]:
            try:
                if algo == "hybrid":
                    result = hybrid_multiply(A, B, q, profile=False)
                    times[algo] = result["time_ms"]
                else:
                    start = time.time()
                    if algo == "karatsuba":
                        karatsuba_multiply(A, B, q)
                    elif algo == "toomcook":
                        toomcook_multiply(A, B, q)
                    else:  # ntt
                        ntt_multiply(A, B, q)
                    end = time.time()
                    times[algo] = (end - start) * 1000
            except Exception:
                times[algo] = None
        
        # Find fastest
        valid_times = {k: v for k, v in times.items() if v is not None}
        fastest = min(valid_times, key=valid_times.get) if valid_times else "N/A"
        
        # Print results
        print(f"{size:4d}  ", end="")
        for algo in ["karatsuba", "toomcook", "ntt", "hybrid"]:
            if times[algo] is not None:
                print(f"{times[algo]:8.2f}ms ", end="")
            else:
                print(f"{'N/A':8} ", end="")
        print(f"{fastest:10}")

def run_all_tests():
    """Run all test suites"""
    print("=" * 50)
    print("🧪 Running Hybrid Polynomial Multiplication Tests")
    print("=" * 50)
    
    # Run individual algorithm tests
    test_karatsuba()
    test_toomcook()
    test_ntt()
    test_hybrid()
    
    # Run integration tests
    test_consistency()
    test_ring_reduction()
    test_rlwe_operations()
    test_performance()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    run_all_tests()