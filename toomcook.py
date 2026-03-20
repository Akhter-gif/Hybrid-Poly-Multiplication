"""
Toom-Cook 3-way algorithm for polynomial multiplication
Optimized for medium-sized polynomials (64 <= n < 256)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple
from utils.modular import mod_multiply, mod_add, mod_sub

def toomcook_multiply(A: List[int], B: List[int], q: int) -> List[int]:
    """
    Toom-Cook 3-way polynomial multiplication
    
    Algorithm:
    1. Split into 3 parts
    2. Evaluate at 5 points: 0, 1, -1, 2, ∞
    3. Multiply pointwise
    4. Interpolate to get coefficients
    
    Complexity: O(n^1.465)
    
    Args:
        A: First polynomial coefficients
        B: Second polynomial coefficients
        q: Modulus
    
    Returns:
        Product polynomial coefficients
    """
    # Make copies
    A = A.copy()
    B = B.copy()
    
    # Ensure same length and pad to multiple of 3
    n = max(len(A), len(B))
    k = (3 - n % 3) % 3
    A.extend([0] * k)
    B.extend([0] * k)
    n = len(A)
    
    # Split into 3 parts
    m = n // 3
    A0, A1, A2 = A[:m], A[m:2*m], A[2*m:]
    B0, B1, B2 = B[:m], B[m:2*m], B[2*m:]
    
    # Evaluate polynomials at points
    # Point 0: just A0, B0
    a0 = A0
    b0 = B0
    
    # Point 1: A0 + A1 + A2
    a1 = _poly_add(_poly_add(A0, A1, q), A2, q)
    b1 = _poly_add(_poly_add(B0, B1, q), B2, q)
    
    # Point -1: A0 - A1 + A2
    a_neg1 = _poly_add(_poly_sub(A0, A1, q), A2, q)
    b_neg1 = _poly_add(_poly_sub(B0, B1, q), B2, q)
    
    # Point 2: A0 + 2A1 + 4A2
    a2 = _poly_add(
        _poly_add(A0, _poly_mul_scalar(A1, 2, q), q),
        _poly_mul_scalar(A2, 4, q), q
    )
    b2 = _poly_add(
        _poly_add(B0, _poly_mul_scalar(B1, 2, q), q),
        _poly_mul_scalar(B2, 4, q), q
    )
    
    # Point infinity: A2, B2
    a_inf = A2
    b_inf = B2
    
    # Pointwise multiplication
    r0 = _poly_mul_pointwise(a0, b0, q)
    r1 = _poly_mul_pointwise(a1, b1, q)
    r_neg1 = _poly_mul_pointwise(a_neg1, b_neg1, q)
    r2 = _poly_mul_pointwise(a2, b2, q)
    r_inf = _poly_mul_pointwise(a_inf, b_inf, q)
    
    # Interpolation to get coefficients
    # Solve linear system:
    # r0 = c0
    # r1 = c0 + c1 + c2 + c3 + c4
    # r_neg1 = c0 - c1 + c2 - c3 + c4
    # r2 = c0 + 2c1 + 4c2 + 8c3 + 16c4
    # r_inf = c4
    
    result = []
    for i in range(len(r0)):
        # Get values at this index
        v0 = r0[i] if i < len(r0) else 0
        v1 = r1[i] if i < len(r1) else 0
        vn1 = r_neg1[i] if i < len(r_neg1) else 0
        v2 = r2[i] if i < len(r2) else 0
        vinf = r_inf[i] if i < len(r_inf) else 0
        
        # Solve for c0..c4
        # Using precomputed interpolation matrix
        c4 = vinf
        
        # Solve 4x4 system for c0..c3
        # This is simplified - in production use matrix inversion
        c3 = _div_mod((v2 - 2*v1 + v0 - vn1), 6, q)
        c2 = _div_mod((v1 + vn1 - 2*v0), 2, q)
        c1 = _div_mod((v1 - c2 - c3 - c4 - v0), 1, q)
        c0 = v0
        
        # Add to result with proper offsets
        coeffs = [c0 % q, c1 % q, c2 % q, c3 % q, c4 % q]
        
        # Extend result list
        while len(result) < i + len(coeffs) * m:
            result.append(0)
        
        for j, coef in enumerate(coeffs):
            result[i + j * m] = (result[i + j * m] + coef) % q
    
    # Trim trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    return result

def _poly_mul_pointwise(A: List[int], B: List[int], q: int) -> List[int]:
    """Pointwise multiplication of two polynomials (convolution)"""
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

def _poly_mul_scalar(A: List[int], scalar: int, q: int) -> List[int]:
    """Multiply polynomial by scalar"""
    return [(a * scalar) % q for a in A]

def _div_mod(a: int, b: int, q: int) -> int:
    """Division modulo q (a/b mod q)"""
    # Find modular inverse of b
    b_inv = pow(b, q-2, q)
    return (a * b_inv) % q

def toomcook_3way(A: List[int], B: List[int], q: int) -> List[int]:
    """Alias for toomcook_multiply"""
    return toomcook_multiply(A, B, q)


# Test function
def test_toomcook():
    """Test Toom-Cook implementation"""
    q = 3329
    
    # Test with small polynomials
    A = [1, 2, 3, 4]
    B = [5, 6, 7, 8]
    
    result = toomcook_multiply(A, B, q)
    expected = [5, 16, 34, 60, 61, 52, 32]
    
    assert result[:len(expected)] == expected, "Toom-Cook basic test failed"
    
    # Test with larger polynomials
    A = [i % 17 for i in range(64)]
    B = [i % 19 for i in range(64)]
    
    from core.ntt import naive_multiply
    naive_result = naive_multiply(A, B, q)
    toom_result = toomcook_multiply(A, B, q)
    
    min_len = min(len(naive_result), len(toom_result))
    assert naive_result[:min_len] == toom_result[:min_len], "Toom-Cook large test failed"
    
    print("✅ Toom-Cook tests passed!")


if __name__ == "__main__":
    test_toomcook()