"""
Number Theoretic Transform (NTT) for fast polynomial multiplication
Optimized for large polynomials (n >= 256)
Uses integer arithmetic for exact results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from typing import List, Optional, Tuple
from utils.modular import mod_multiply, mod_add, mod_sub
from config import NTT_PARAMS

def ntt_multiply(A: List[int], B: List[int], q: int, n: Optional[int] = None) -> List[int]:
    """
    Multiply polynomials using Number Theoretic Transform
    
    NTT is the integer version of FFT that works in finite fields
    Provides O(n log n) complexity with exact integer results
    
    Args:
        A: First polynomial coefficients
        B: Second polynomial coefficients
        q: Modulus (must be prime with suitable root of unity)
        n: Transform size (default: next power of 2 >= len(A)+len(B)-1)
    
    Returns:
        Product polynomial coefficients
    """
    # Determine transform size
    if n is None:
        n = 1
        while n < len(A) + len(B) - 1:
            n *= 2
    
    # Pad arrays to size n
    A_pad = A + [0] * (n - len(A))
    B_pad = B + [0] * (n - len(B))
    
    # Find primitive root of unity
    root = find_primitive_root(n, q)
    if root is None:
        # Fallback to naive multiplication
        return naive_multiply(A, B, q)
    
    # Compute forward NTT
    A_ntt = ntt_transform(A_pad, root, n, q)
    B_ntt = ntt_transform(B_pad, root, n, q)
    
    # Pointwise multiplication
    C_ntt = [mod_multiply(A_ntt[i], B_ntt[i], q) for i in range(n)]
    
    # Compute inverse NTT
    result = inverse_ntt_transform(C_ntt, root, n, q)
    
    # Trim trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    return result

def find_primitive_root(n: int, q: int) -> Optional[int]:
    """
    Find primitive n-th root of unity modulo q
    
    A number g is a primitive n-th root of unity if:
    1. g^n ≡ 1 (mod q)
    2. g^k ≠ 1 for any 1 ≤ k < n
    
    Args:
        n: Transform size (power of 2)
        q: Prime modulus
    
    Returns:
        Primitive root or None if not found
    """
    # Check if q is prime (simplified for demo)
    # In production, use proper primality test
    
    # For Kyber parameters
    if q == 3329:
        if n == 256:
            return 17
        elif n == 512:
            return 3
        elif n == 1024:
            return 3
    
    # For other parameters, search for root
    # Check if n divides q-1 (necessary condition)
    if (q - 1) % n != 0:
        return None
    
    # Find primitive root of the field
    g = find_field_primitive_root(q)
    if g is None:
        return None
    
    # Compute g^((q-1)/n)
    root = pow(g, (q - 1) // n, q)
    
    # Verify it's primitive
    if pow(root, n // 2, q) != 1:
        return root
    
    return None

def find_field_primitive_root(q: int) -> Optional[int]:
    """
    Find primitive root of the finite field F_q
    """
    # For prime q, find g such that order is q-1
    # This is simplified - in production use factorization
    for g in range(2, q):
        if pow(g, q - 1, q) != 1:
            continue
        
        # Check if it's primitive
        is_primitive = True
        for factor in [2, 3, 5, 7, 11, 13, 17, 19]:  # Small prime factors
            if (q - 1) % factor == 0:
                if pow(g, (q - 1) // factor, q) == 1:
                    is_primitive = False
                    break
        
        if is_primitive:
            return g
    
    return None

def ntt_transform(a: List[int], root: int, n: int, q: int) -> List[int]:
    """
    Forward NTT transform (iterative Cooley-Tukey)
    
    Args:
        a: Input polynomial coefficients (length n)
        root: Primitive n-th root of unity
        n: Transform size
        q: Modulus
    
    Returns:
        NTT-transformed coefficients
    """
    result = a.copy()
    
    # Bit-reversal reordering
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        
        if i < j:
            result[i], result[j] = result[j], result[i]
    
    # Iterative NTT
    length = 2
    while length <= n:
        # Compute root for this length
        wlen = pow(root, n // length, q)
        
        for i in range(0, n, length):
            w = 1
            half = length // 2
            
            for j in range(i, i + half):
                u = result[j]
                v = mod_multiply(result[j + half], w, q)
                
                result[j] = mod_add(u, v, q)
                result[j + half] = mod_sub(u, v, q)
                
                w = mod_multiply(w, wlen, q)
        
        length <<= 1
    
    return result

def inverse_ntt_transform(a: List[int], root: int, n: int, q: int) -> List[int]:
    """
    Inverse NTT transform
    
    Args:
        a: NTT-transformed coefficients
        root: Primitive n-th root of unity
        n: Transform size
        q: Modulus
    
    Returns:
        Original polynomial coefficients
    """
    # Find inverse of root
    root_inv = pow(root, q - 2, q)
    
    # Apply NTT with inverse root
    result = ntt_transform(a, root_inv, n, q)
    
    # Multiply by n^(-1) mod q
    n_inv = pow(n, q - 2, q)
    result = [mod_multiply(x, n_inv, q) for x in result]
    
    return result

def naive_multiply(A: List[int], B: List[int], q: int) -> List[int]:
    """
    Naive O(n²) polynomial multiplication
    Used as fallback when NTT not available
    """
    n, m = len(A), len(B)
    result = [0] * (n + m - 1)
    
    for i in range(n):
        if A[i] == 0:
            continue
        for j in range(m):
            if B[j] == 0:
                continue
            result[i + j] = (result[i + j] + mod_multiply(A[i], B[j], q)) % q
    
    return result

def ntt_convolution(A: List[int], B: List[int], q: int) -> List[int]:
    """Alias for ntt_multiply"""
    return ntt_multiply(A, B, q)

def is_power_of_two(n: int) -> bool:
    """Check if n is power of 2"""
    return (n & (n - 1)) == 0 and n > 0

def next_power_of_two(n: int) -> int:
    """Return next power of 2 >= n"""
    return 1 << (n - 1).bit_length()


# Optimized NTT with precomputed roots
class NTTProcessor:
    """
    NTT processor with precomputed roots for repeated use
    Useful when multiplying many polynomials of same size
    """
    
    def __init__(self, n: int, q: int, root: int):
        """
        Initialize NTT processor
        
        Args:
            n: Transform size
            q: Modulus
            root: Primitive root of unity
        """
        self.n = n
        self.q = q
        self.root = root
        self.root_inv = pow(root, q - 2, q)
        self.n_inv = pow(n, q - 2, q)
        
        # Precompute roots for each stage
        self.roots = []
        self.roots_inv = []
        
        length = 2
        while length <= n:
            wlen = pow(root, n // length, q)
            wlen_inv = pow(wlen, q - 2, q)
            
            # Precompute all powers for this length
            powers = [1]
            powers_inv = [1]
            for i in range(1, length // 2):
                powers.append(mod_multiply(powers[-1], wlen, q))
                powers_inv.append(mod_multiply(powers_inv[-1], wlen_inv, q))
            
            self.roots.append(powers)
            self.roots_inv.append(powers_inv)
            length <<= 1
    
    def forward(self, a: List[int]) -> List[int]:
        """Forward NTT using precomputed roots"""
        n = self.n
        q = self.q
        result = a.copy()
        
        # Bit-reversal
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                result[i], result[j] = result[j], result[i]
        
        # NTT with precomputed roots
        stage = 0
        length = 2
        while length <= n:
            roots = self.roots[stage]
            half = length // 2
            
            for i in range(0, n, length):
                for j in range(half):
                    w = roots[j]
                    u = result[i + j]
                    v = mod_multiply(result[i + j + half], w, q)
                    
                    result[i + j] = mod_add(u, v, q)
                    result[i + j + half] = mod_sub(u, v, q)
            
            length <<= 1
            stage += 1
        
        return result
    
    def inverse(self, a: List[int]) -> List[int]:
        """Inverse NTT using precomputed roots"""
        n = self.n
        q = self.q
        result = a.copy()
        
        # Bit-reversal
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                result[i], result[j] = result[j], result[i]
        
        # Inverse NTT with precomputed roots
        stage = 0
        length = 2
        while length <= n:
            roots_inv = self.roots_inv[stage]
            half = length // 2
            
            for i in range(0, n, length):
                for j in range(half):
                    w = roots_inv[j]
                    u = result[i + j]
                    v = mod_multiply(result[i + j + half], w, q)
                    
                    result[i + j] = mod_add(u, v, q)
                    result[i + j + half] = mod_sub(u, v, q)
            
            length <<= 1
            stage += 1
        
        # Multiply by n^(-1)
        result = [mod_multiply(x, self.n_inv, q) for x in result]
        return result


# Test function
def test_ntt():
    """Test NTT implementation"""
    q = 3329
    
    # Test with small polynomials
    A = [1, 2, 3, 4]
    B = [5, 6, 7, 8]
    
    result = ntt_multiply(A, B, q)
    expected = [5, 16, 34, 60, 61, 52, 32]
    
    assert result[:len(expected)] == expected, "NTT basic test failed"
    
    # Test with larger polynomials
    A = [i % 17 for i in range(128)]
    B = [i % 19 for i in range(128)]
    
    naive_result = naive_multiply(A, B, q)
    ntt_result = ntt_multiply(A, B, q)
    
    min_len = min(len(naive_result), len(ntt_result))
    assert naive_result[:min_len] == ntt_result[:min_len], "NTT large test failed"
    
    print("✅ NTT tests passed!")


if __name__ == "__main__":
    test_ntt()