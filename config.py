"""
Configuration file for Hybrid Polynomial Multiplication Framework
RLWE-Based Cryptography with dynamic algorithm selection
"""

# ============================================
# THRESHOLD CONFIGURATION
# ============================================

# Algorithm selection thresholds
# Karatsuba: n < T1
# Toom-Cook: T1 <= n < T2  
# NTT/FFT: n >= T2

T1 = 64      # Karatsuba threshold (small inputs)
T2 = 256     # Toom-Cook threshold (medium inputs)
             # NTT for large inputs (n >= 256)

# ============================================
# CRYPTOGRAPHIC PARAMETERS
# ============================================

# Default modulus for Kyber-like RLWE
# q = 3329 is used in Kyber (NIST PQC standard)
DEFAULT_MODULUS = 3329

# Supported polynomial degrees
# Must be powers of 2 for NTT
SUPPORTED_DEGREES = [64, 128, 256, 512, 1024]

# NTT parameters for different sizes
# root = primitive n-th root of unity modulo q
NTT_PARAMS = {
    256: {
        "root": 17,      # 17^128 ≡ 1 mod 3329
        "q": 3329,
        "inv_root": 1964  # inverse of 17 mod 3329
    },
    512: {
        "root": 3,        # 3^256 ≡ 1 mod 3329  
        "q": 3329,
        "inv_root": 1110  # inverse of 3 mod 3329
    },
    1024: {
        "root": 3,        # 3^512 ≡ 1 mod 3329
        "q": 3329,
        "inv_root": 1110
    }
}

# ============================================
# PERFORMANCE CONFIGURATION
# ============================================

# Enable/disable performance profiling
ENABLE_PROFILING = True

# Log file for performance metrics
LOG_FILE = "performance_logs.txt"

# Benchmark configurations
BENCHMARK_SIZES = [16, 32, 64, 128, 256, 512, 1024]
BENCHMARK_ITERATIONS = 5  # Run each test multiple times for average

# ============================================
# ALGORITHM-SPECIFIC CONFIGURATION
# ============================================

# Karatsuba
KARATSUBA_BASE_CASE = 32  # Switch to naive multiplication below this size

# Toom-Cook
TOOMCOOK_SPLIT = 3  # 3-way Toom-Cook

# NTT
NTT_ZERO_PADDING = True  # Pad to next power of 2

# ============================================
# SECURITY CONFIGURATION
# ============================================

# Constant-time operations (for production crypto)
CONSTANT_TIME = False  # Set to True for production

# Memory wiping (clear sensitive data)
WIPE_MEMORY = False  # Set to True for production

# ============================================
# API CONFIGURATION
# ============================================

API_TITLE = "Hybrid Polynomial Multiplication API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
RLWE-based polynomial multiplication with dynamic algorithm selection.
Supports Karatsuba, Toom-Cook, and NTT algorithms.
"""

# CORS settings (for frontend)
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000",
    "http://localhost:5500",  # Live server
    "http://127.0.0.1:5500",
    "*"  # Allow all for development (restrict in production)
]

# ============================================
# LOGGING CONFIGURATION
# ============================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "file": {
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": "api.log",
        },
        "console": {
            "formatter": "default",
            "class": "logging.StreamHandler",
        }
    },
    "root": {
        "handlers": ["file", "console"],
        "level": "INFO"
    }
}

# ============================================
# VALIDATION FUNCTIONS
# ============================================

def validate_degree(n: int) -> bool:
    """Check if polynomial degree is supported"""
    return n in SUPPORTED_DEGREES or n <= max(SUPPORTED_DEGREES)

def validate_modulus(q: int) -> bool:
    """Check if modulus is valid for RLWE"""
    # q should be prime for NTT
    # 3329 is prime
    prime_moduli = [3329, 12289, 40961]
    return q in prime_moduli or q > 0

# ============================================
# DEFAULT TEST VECTORS
# ============================================

TEST_VECTORS = {
    "small": {
        "A": [1, 2, 3, 4],
        "B": [5, 6, 7, 8],
        "q": 3329,
        "expected": [5, 16, 34, 60, 61, 52, 32]
    },
    "medium": {
        "A": [i % 17 for i in range(64)],
        "B": [i % 19 for i in range(64)],
        "q": 3329
    },
    "large": {
        "A": [i % 31 for i in range(256)],
        "B": [i % 37 for i in range(256)],
        "q": 3329
    }
}

# ============================================
# SYSTEM INFO
# ============================================

SYSTEM_INFO = {
    "name": "Hybrid Polynomial Multiplication Framework",
    "purpose": "RLWE-Based Post-Quantum Cryptography",
    "algorithms": ["Karatsuba", "Toom-Cook", "NTT"],
    "features": [
        "Dynamic algorithm selection",
        "Configurable thresholds",
        "Performance profiling",
        "RLWE simulation",
        "Web interface",
        "Benchmarking"
    ]
}

# Export all configurations
__all__ = [
    'T1', 'T2',
    'DEFAULT_MODULUS',
    'SUPPORTED_DEGREES',
    'NTT_PARAMS',
    'ENABLE_PROFILING',
    'LOG_FILE',
    'BENCHMARK_SIZES',
    'BENCHMARK_ITERATIONS',
    'KARATSUBA_BASE_CASE',
    'TOOMCOOK_SPLIT',
    'NTT_ZERO_PADDING',
    'CONSTANT_TIME',
    'WIPE_MEMORY',
    'API_TITLE',
    'API_VERSION',
    'API_DESCRIPTION',
    'ALLOWED_ORIGINS',
    'LOGGING_CONFIG',
    'TEST_VECTORS',
    'SYSTEM_INFO',
    'validate_degree',
    'validate_modulus'
]