"""
Core module for Hybrid Polynomial Multiplication Framework
Contains algorithm implementations and polynomial operations
"""

from .polynomial import Polynomial
from .karatsuba import karatsuba_multiply
from .toomcook import toomcook_multiply
from .ntt import ntt_multiply, naive_multiply, find_primitive_root
from .hybrid import hybrid_multiply

__all__ = [
    'Polynomial',
    'karatsuba_multiply',
    'toomcook_multiply', 
    'ntt_multiply',
    'naive_multiply',
    'find_primitive_root',
    'hybrid_multiply'
]

__version__ = '1.0.0'
__author__ = 'Hybrid Polynomial Multiplication Team'