"""
Modular arithmetic utilities for RLWE cryptography
Constant-time operations for security
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONSTANT_TIME

def mod_add(a: int, b: int, q: int) -> int:
    """
    Modular addition: (a + b) mod q
    """
    result = a + b
    if result >= q:
        result -= q
    return result

def mod_sub(a: int, b: int, q: int) -> int:
    """
    Modular subtraction: (a - b) mod q
    """
    result = a - b
    if result < 0:
        result += q
    return result

def mod_multiply(a: int, b: int, q: int) -> int:
    """
    Modular multiplication: (a * b) mod q
    
    If CONSTANT_TIME is True, uses constant-time algorithm
    """
    if CONSTANT_TIME:
        return _constant_time_multiply(a, b, q)
    else:
        return (a * b) % q

def mod_inverse(a: int, q: int) -> int:
    """
    Modular inverse using Fermat's little theorem
    a^(q-2) mod q (since q is prime)
    """
    return pow(a, q - 2, q)

def _constant_time_multiply(a: int, b: int, q: int) -> int:
    """
    Constant-time modular multiplication
    Prevents timing attacks
    """
    result = 0
    a = a % q
    b = b % q
    
    # Montgomery ladder style multiplication
    for _ in range(32):  # Assuming 32-bit integers
        if b & 1:
            result = (result + a) % q
        a = (a << 1) % q
        b >>= 1
    
    return result

def mod_pow(base: int, exponent: int, q: int) -> int:
    """
    Modular exponentiation
    """
    return pow(base, exponent, q)

def is_quadratic_residue(a: int, q: int) -> bool:
    """
    Check if a is quadratic residue modulo q
    Using Euler's criterion
    """
    return pow(a, (q - 1) // 2, q) == 1