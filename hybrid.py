"""
Hybrid controller for dynamic algorithm selection
Selects optimal algorithm based on polynomial size
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import Dict, List, Optional, Tuple, Any
from config import T1, T2, ENABLE_PROFILING, LOG_FILE
from utils.profiler import log_performance
from core.karatsuba import karatsuba_multiply
from core.toomcook import toomcook_multiply
from core.ntt import ntt_multiply

# Algorithm metadata
ALGORITHM_INFO = {
    "karatsuba": {
        "name": "Karatsuba",
        "complexity": "O(n^1.585)",
        "best_for": "Small polynomials (n < 64)",
        "description": "Divide and conquer with 3 multiplications"
    },
    "toomcook": {
        "name": "Toom-Cook 3-way",
        "complexity": "O(n^1.465)",
        "best_for": "Medium polynomials (64 <= n < 256)",
        "description": "Evaluation at 5 points with interpolation"
    },
    "ntt": {
        "name": "Number Theoretic Transform",
        "complexity": "O(n log n)",
        "best_for": "Large polynomials (n >= 256)",
        "description": "FFT in finite field with exact arithmetic"
    }
}

def hybrid_multiply(
    A: List[int], 
    B: List[int], 
    q: int, 
    force_algorithm: Optional[str] = None,
    profile: bool = ENABLE_PROFILING
) -> Dict[str, Any]:
    """
    Hybrid polynomial multiplication with dynamic algorithm selection
    
    The controller automatically selects the best algorithm based on:
    - Input polynomial size (n)
    - Configurable thresholds (T1, T2)
    - Optional forced algorithm selection for testing
    
    Args:
        A: First polynomial coefficients
        B: Second polynomial coefficients
        q: Modulus
        force_algorithm: Force specific algorithm (karatsuba, toomcook, ntt)
        profile: Whether to log performance metrics
    
    Returns:
        Dictionary containing:
        - algorithm: Name of algorithm used
        - result: Product polynomial coefficients
        - time_ms: Execution time in milliseconds
        - input_size: Size of input polynomials
        - output_size: Size of output polynomial
        - threshold_info: Information about thresholds (optional)
    """
    # Ensure A and B are same length
    n = max(len(A), len(B))
    if len(A) < n:
        A = A + [0] * (n - len(A))
    if len(B) < n:
        B = B + [0] * (n - len(B))
    
    start_time = time.time()
    threshold_info = {}
    
    # Algorithm selection logic
    if force_algorithm:
        # Forced algorithm (for testing/comparison)
        algo_key = force_algorithm.lower()
        if algo_key == "karatsuba":
            result = karatsuba_multiply(A, B, q)
            algorithm = "Karatsuba (forced)"
        elif algo_key == "toomcook":
            result = toomcook_multiply(A, B, q)
            algorithm = "Toom-Cook (forced)"
        elif algo_key == "ntt":
            result = ntt_multiply(A, B, q)
            algorithm = "NTT (forced)"
        else:
            # Invalid force, use auto
            return hybrid_multiply(A, B, q, force_algorithm=None, profile=profile)
    
    else:
        # Dynamic selection based on thresholds
        threshold_info = {
            "T1": T1,
            "T2": T2,
            "input_size": n,
            "selected_region": ""
        }
        
        if n < T1:
            # Small polynomials: Karatsuba
            result = karatsuba_multiply(A, B, q)
            algorithm = f"Karatsuba (n={n} < {T1})"
            threshold_info["selected_region"] = "small"
            
        elif n < T2:
            # Medium polynomials: Toom-Cook
            result = toomcook_multiply(A, B, q)
            algorithm = f"Toom-Cook ({T1} <= n={n} < {T2})"
            threshold_info["selected_region"] = "medium"
            
        else:
            # Large polynomials: NTT
            result = ntt_multiply(A, B, q)
            algorithm = f"NTT (n={n} >= {T2})"
            threshold_info["selected_region"] = "large"
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Log performance if enabled
    if profile:
        log_performance(algorithm, n, execution_time, len(result))
    
    # Prepare response
    response = {
        "algorithm": algorithm,
        "result": result,
        "time_ms": round(execution_time, 3),
        "input_size": n,
        "output_size": len(result)
    }
    
    # Add threshold info if available
    if threshold_info:
        response["threshold_info"] = threshold_info
    
    return response

def get_algorithm_info() -> Dict[str, Any]:
    """Get information about available algorithms"""
    return {
        "algorithms": ALGORITHM_INFO,
        "current_thresholds": {
            "T1": T1,
            "T2": T2,
            "karatsuba_max": T1,
            "toomcook_min": T1,
            "toomcook_max": T2,
            "ntt_min": T2
        },
        "description": "Dynamic algorithm selection based on polynomial size"
    }

def benchmark_algorithms(
    sizes: List[int], 
    q: int = 3329,
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark all algorithms on different polynomial sizes
    
    Args:
        sizes: List of polynomial sizes to test
        q: Modulus
        iterations: Number of iterations per test
    
    Returns:
        Benchmark results dictionary
    """
    results = {}
    
    for size in sizes:
        # Generate test polynomials
        A = [i % q for i in range(size)]
        B = [(i * 2) % q for i in range(size)]
        
        size_results = {}
        
        for algo in ["karatsuba", "toomcook", "ntt"]:
            times = []
            successes = 0
            
            for _ in range(iterations):
                try:
                    result = hybrid_multiply(A, B, q, force_algorithm=algo, profile=False)
                    times.append(result["time_ms"])
                    successes += 1
                except Exception as e:
                    # Algorithm may not support this size
                    pass
            
            if times:
                size_results[algo] = {
                    "avg_time_ms": sum(times) / len(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "success_rate": successes / iterations
                }
            else:
                size_results[algo] = {
                    "avg_time_ms": None,
                    "min_time_ms": None,
                    "max_time_ms": None,
                    "success_rate": 0
                }
        
        # Determine fastest algorithm for this size
        valid_times = {k: v["avg_time_ms"] for k, v in size_results.items() 
                      if v["avg_time_ms"] is not None}
        if valid_times:
            fastest = min(valid_times, key=valid_times.get)
            size_results["fastest"] = fastest
            
            # Check if selection matches thresholds
            expected = "karatsuba" if size < T1 else "toomcook" if size < T2 else "ntt"
            size_results["threshold_match"] = (fastest == expected)
        
        results[size] = size_results
    
    return results

def compare_algorithms(
    A: List[int], 
    B: List[int], 
    q: int
) -> Dict[str, Any]:
    """
    Compare all algorithms on the same input
    Useful for testing and visualization
    
    Args:
        A: First polynomial
        B: Second polynomial
        q: Modulus
    
    Returns:
        Comparison results
    """
    results = {}
    
    for algo in ["karatsuba", "toomcook", "ntt"]:
        try:
            result = hybrid_multiply(A, B, q, force_algorithm=algo, profile=False)
            results[algo] = {
                "time_ms": result["time_ms"],
                "output_size": result["output_size"],
                "success": True
            }
        except Exception as e:
            results[algo] = {
                "error": str(e),
                "success": False
            }
    
    # Add hybrid result
    hybrid_result = hybrid_multiply(A, B, q, profile=False)
    results["hybrid"] = {
        "time_ms": hybrid_result["time_ms"],
        "output_size": hybrid_result["output_size"],
        "algorithm": hybrid_result["algorithm"],
        "success": True
    }
    
    # Find fastest successful algorithm
    successful = {k: v["time_ms"] for k, v in results.items() 
                 if v.get("success", False) and k != "hybrid"}
    if successful:
        results["fastest_algorithm"] = min(successful, key=successful.get)
    
    return results


# Test function
def test_hybrid():
    """Test hybrid controller"""
    q = 3329
    
    # Test different sizes
    test_sizes = [16, 32, 64, 128, 256, 512]
    
    for size in test_sizes:
        A = [i % 17 for i in range(size)]
        B = [i % 19 for i in range(size)]
        
        result = hybrid_multiply(A, B, q)
        
        print(f"Size {size}: {result['algorithm']} - {result['time_ms']:.2f}ms")
        
        # Verify correctness
        from core.ntt import naive_multiply
        expected = naive_multiply(A, B, q)
        min_len = min(len(expected), len(result["result"]))
        
        assert result["result"][:min_len] == expected[:min_len], \
               f"Hybrid failed for size {size}"
    
    print("✅ Hybrid controller tests passed!")


if __name__ == "__main__":
    test_hybrid()