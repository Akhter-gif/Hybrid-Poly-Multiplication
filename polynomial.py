"""
Polynomial class for operations in Zq[x]/(x^n + 1)
RLWE-based cryptography with modular arithmetic
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
from typing import List, Optional, Union
from utils.modular import mod_add, mod_sub, mod_multiply

class Polynomial:
    """
    Polynomial class for RLWE operations
    
    Represents polynomial in ring Zq[x]/(x^n + 1)
    with coefficients modulo q
    
    Attributes:
        coeffs: List of coefficients [a0, a1, a2, ...]
        q: Modulus (prime number)
        n: Degree of ring (optional)
    """
    
    def __init__(self, coeffs: List[int], q: int = 3329, n: Optional[int] = None):
        """
        Initialize polynomial
        
        Args:
            coeffs: List of coefficients
            q: Modulus (default: 3329 for Kyber)
            n: Optional degree for ring reduction
        """
        self.q = q
        # Apply modulo q to all coefficients
        self.coeffs = [int(c % q) for c in coeffs]
        
        # Remove trailing zeros
        while len(self.coeffs) > 1 and self.coeffs[-1] == 0:
            self.coeffs.pop()
        
        # Apply ring reduction if degree specified
        if n is not None:
            self.mod_ring(n)
    
    def __len__(self) -> int:
        """Return number of coefficients"""
        return len(self.coeffs)
    
    def __getitem__(self, idx: int) -> int:
        """Get coefficient at index (0 if out of bounds)"""
        if idx < 0:
            idx = len(self.coeffs) + idx
        return self.coeffs[idx] if 0 <= idx < len(self.coeffs) else 0
    
    def __setitem__(self, idx: int, value: int):
        """Set coefficient at index"""
        if idx < 0:
            idx = len(self.coeffs) + idx
        if 0 <= idx < len(self.coeffs):
            self.coeffs[idx] = value % self.q
        else:
            # Extend list if index out of bounds
            while len(self.coeffs) <= idx:
                self.coeffs.append(0)
            self.coeffs[idx] = value % self.q
    
    def degree(self) -> int:
        """Return degree of polynomial"""
        return len(self.coeffs) - 1
    
    def is_zero(self) -> bool:
        """Check if polynomial is zero"""
        return all(c == 0 for c in self.coeffs)
    
    def copy(self) -> 'Polynomial':
        """Create deep copy of polynomial"""
        return Polynomial(self.coeffs.copy(), self.q)
    
    def mod_ring(self, n: int) -> 'Polynomial':
        """
        Reduce polynomial modulo (x^n + 1)
        
        For ring Zq[x]/(x^n + 1):
        - Coefficients beyond degree n-1 wrap around
        - Odd wraps are subtracted (because x^n ≡ -1)
        
        Args:
            n: Ring degree
            
        Returns:
            Self with reduced coefficients
        """
        if len(self.coeffs) <= n:
            return self
        
        result = [0] * n
        
        for i, val in enumerate(self.coeffs):
            idx = i % n
            # Number of wraps: i // n
            # If wraps odd: subtract, if even: add
            if (i // n) % 2 == 0:
                result[idx] = (result[idx] + val) % self.q
            else:
                result[idx] = (result[idx] - val) % self.q
        
        self.coeffs = result
        return self
    
    def __add__(self, other: Union['Polynomial', int]) -> 'Polynomial':
        """
        Polynomial addition
        
        Args:
            other: Another polynomial or integer
            
        Returns:
            New polynomial (self + other)
        """
        if isinstance(other, int):
            # Add constant
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] = (new_coeffs[0] + other) % self.q
            return Polynomial(new_coeffs, self.q)
        
        elif isinstance(other, Polynomial):
            # Add polynomial
            max_len = max(len(self), len(other))
            result = [0] * max_len
            
            for i in range(max_len):
                val = (self[i] + other[i]) % self.q
                result[i] = val
            
            return Polynomial(result, self.q)
        
        return NotImplemented
    
    def __sub__(self, other: Union['Polynomial', int]) -> 'Polynomial':
        """
        Polynomial subtraction
        
        Args:
            other: Another polynomial or integer
            
        Returns:
            New polynomial (self - other)
        """
        if isinstance(other, int):
            # Subtract constant
            new_coeffs = self.coeffs.copy()
            new_coeffs[0] = (new_coeffs[0] - other) % self.q
            return Polynomial(new_coeffs, self.q)
        
        elif isinstance(other, Polynomial):
            # Subtract polynomial
            max_len = max(len(self), len(other))
            result = [0] * max_len
            
            for i in range(max_len):
                val = (self[i] - other[i]) % self.q
                result[i] = val
            
            return Polynomial(result, self.q)
        
        return NotImplemented
    
    def __mul__(self, other: Union['Polynomial', int]) -> 'Polynomial':
        """
        Polynomial multiplication (uses hybrid algorithm)
        
        Args:
            other: Another polynomial or integer
            
        Returns:
            New polynomial (self * other)
        """
        if isinstance(other, int):
            # Scalar multiplication
            new_coeffs = [(c * other) % self.q for c in self.coeffs]
            return Polynomial(new_coeffs, self.q)
        
        elif isinstance(other, Polynomial):
            # Polynomial multiplication using hybrid
            from core.hybrid import hybrid_multiply
            result_dict = hybrid_multiply(
                self.coeffs, 
                other.coeffs, 
                self.q
            )
            return Polynomial(result_dict["result"], self.q)
        
        return NotImplemented
    
    def __neg__(self) -> 'Polynomial':
        """Negate polynomial"""
        return Polynomial([(-c) % self.q for c in self.coeffs], self.q)
    
    def __eq__(self, other) -> bool:
        """Check equality with another polynomial"""
        if not isinstance(other, Polynomial):
            return False
        if self.q != other.q:
            return False
        return self.coeffs == other.coeffs
    
    def __str__(self) -> str:
        """String representation"""
        if self.is_zero():
            return "0"
        
        terms = []
        for i, coeff in enumerate(self.coeffs):
            if coeff == 0:
                continue
            
            if i == 0:
                terms.append(str(coeff))
            elif i == 1:
                if coeff == 1:
                    terms.append("x")
                elif coeff == -1:
                    terms.append("-x")
                else:
                    terms.append(f"{coeff}x")
            else:
                if coeff == 1:
                    terms.append(f"x^{i}")
                elif coeff == -1:
                    terms.append(f"-x^{i}")
                else:
                    terms.append(f"{coeff}x^{i}")
        
        return " + ".join(terms).replace("+ -", "- ")
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"Polynomial(coeffs={self.coeffs}, q={self.q})"
    
    def to_list(self) -> List[int]:
        """Export coefficients as list"""
        return self.coeffs.copy()
    
    @classmethod
    def random(cls, degree: int, q: int = 3329) -> 'Polynomial':
        """
        Generate random polynomial
        
        Args:
            degree: Polynomial degree
            q: Modulus
            
        Returns:
            Random polynomial
        """
        import random
        coeffs = [random.randint(0, q-1) for _ in range(degree + 1)]
        return cls(coeffs, q)
    
    @classmethod
    def zero(cls, q: int = 3329) -> 'Polynomial':
        """Create zero polynomial"""
        return cls([0], q)
    
    @classmethod
    def one(cls, q: int = 3329) -> 'Polynomial':
        """Create constant polynomial 1"""
        return cls([1], q)
    
    @classmethod
    def monomial(cls, degree: int, coeff: int = 1, q: int = 3329) -> 'Polynomial':
        """Create monomial coeff * x^degree"""
        coeffs = [0] * (degree + 1)
        coeffs[degree] = coeff % q
        return cls(coeffs, q)


# Utility functions for polynomial operations
def poly_add(a: List[int], b: List[int], q: int) -> List[int]:
    """Add two polynomials (list version)"""
    max_len = max(len(a), len(b))
    result = [0] * max_len
    for i in range(max_len):
        val = ((a[i] if i < len(a) else 0) + 
               (b[i] if i < len(b) else 0)) % q
        result[i] = val
    return result

def poly_sub(a: List[int], b: List[int], q: int) -> List[int]:
    """Subtract two polynomials (list version)"""
    max_len = max(len(a), len(b))
    result = [0] * max_len
    for i in range(max_len):
        val = ((a[i] if i < len(a) else 0) - 
               (b[i] if i < len(b) else 0)) % q
        result[i] = val
    return result

def poly_mul_scalar(a: List[int], scalar: int, q: int) -> List[int]:
    """Multiply polynomial by scalar"""
    return [(c * scalar) % q for c in a]

def poly_trim(a: List[int]) -> List[int]:
    """Remove trailing zeros"""
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a