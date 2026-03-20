"""
Karatsuba algorithm for fast polynomial multiplication
Optimized for small polynomials (n < 64)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List, Optional
from utils.modular import mod_multiply
from config import KARATSUBA_BASE_CASE

def karatsuba_multiply(
    A: List[int], 
    B: List[int], 
    q: int,
    base_case_size: int = KARATSUBA_BASE_CASE
) -> List[int]:
    """
    Karatsuba polynomial multiplication
    
    Algorithm: Divide and conquer with 3 multiplications instead of 4
    Complexity: O(n^1.585)
    
    Args:
        A: First polynomial coefficients
        B: Second polynomial coefficients
        q: Modulus
        base_case_size: Switch to naive multiplication below this size
    
    Returns:
        Product polynomial coefficients
    """
    n = max(len(A), len(B))
    
    # Make copies and ensure same length
    A = A.copy()
    B = B.copy()
    
    # Pad to same length (power of 2 for balanced split)
    while len(A) < n:
        A.append(0)
    while len(B) < n:
        B.append(0)
    
    # Base case: naive multiplication for small polynomials
    if n <= base_case_size:
        return _naive_multiply(A, B, q)
    
    # Handle odd lengths by padding to even
    if n % 2 != 0:
        A.append(0)
        B.append(0)
        n += 1
    
    mid = n // 2
    
    # Split polynomials into low and high parts
    A_low = A[:mid]
    A_high = A[mid:]
    B_low = B[:mid]
    B_high = B[mid:]
    
    # Recursive calls
    # z0 = low * low
    z0 = karatsuba_multiply(A_low, B_low, q, base_case_size)
    
    # z2 = high * high
    z2 = karatsuba_multiply(A_high, B_high, q, base_case_size)
    
    # (A_low + A_high) * (B_low + B_high)
    A_sum = _poly_add(A_low, A_high, q)
    B_sum = _poly_add(B_low, B_high, q)
    z1 = karatsuba_multiply(A_sum, B_sum, q, base_case_size)
    
    # z1 = z1 - z0 - z2
    z1 = _poly_sub(_poly_sub(z1, z0, q), z2, q)
    
    # Combine results
    result = [0] * (2 * n - 1)
    
    # Add z0 to lower part
    for i in range(len(z0)):
        result[i] = (result[i] + z0[i]) % q
    
    # Add z1 to middle part
    for i in range(len(z1)):
        result[i + mid] = (result[i + mid] + z1[i]) % q
    
    # Add z2 to higher part
    for i in range(len(z2)):
        result[i + 2 * mid] = (result[i + 2 * mid] + z2[i]) % q
    
    # Trim trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    return result

def _naive_multiply(A: List[int], B: List[int], q: int) -> List[int]:
    """
    Naive O(n²) polynomial multiplication
    Used as base case for Karatsuba
    """
    n = len(A)
    m = len(B)
    result = [0] * (n + m - 1)
    
    for i in range(n):
        if A[i] == 0:
            continue
        for j in range(m):
            if B[j] == 0:
                continue
            result[i + j] = (result[i + j] + mod_multiply(A[i], B[j], q)) % q
    
    return result

def _poly_add(A: List[int], B: List[int], q: int) -> List[int]:
    """Add two polynomials"""
    max_len = max(len(A), len(B))
    result = [0] * max_len
    
    for i in range(max_len):
        a_val = A[i] if i < len(A) else 0
        b_val = B[i] if i < len(B) else 0
        result[i] = (a_val + b_val) % q
    
    return result

def _poly_sub(A: List[int], B: List[int], q: int) -> List[int]:
    """Subtract two polynomials"""
    max_len = max(len(A), len(B))
    result = [0] * max_len
    
    for i in range(max_len):
        a_val = A[i] if i < len(A) else 0
        b_val = B[i] if i < len(B) else 0
        result[i] = (a_val - b_val) % q
    
    return result

def karatsuba_multiply_optimized(
    A: List[int], 
    B: List[int], 
    q: int
) -> List[int]:
    """
    Optimized Karatsuba with memory reuse
    Uses in-place operations for better performance
    """
    # This is a placeholder for more optimized version
    # For production, implement with memory pools
    return karatsuba_multiply(A, B, q)


# Test function
def test_karatsuba():
    """Test Karatsuba implementation"""
    q = 3329
    
    test_cases = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 0, 1], [1, 0, 1]),
        ([1], [1]),
        ([1, 1], [1, 1]),
        ([i % 17 for i in range(32)], [i % 19 for i in range(32)])
    ]
    
    for A, B in test_cases:
        # Get Karatsuba result
        karat_result = karatsuba_multiply(A, B, q)
        
        # Get naive result for comparison
        naive_result = _naive_multiply(A, B, q)
        
        # Compare
        assert karat_result == naive_result, f"Failed for {len(A)}"
    
    print("✅ Karatsuba tests passed!")


if __name__ == "__main__":
    test_karatsuba()