"""
Hybrid Polynomial Multiplication API
Main FastAPI application for RLWE-based polynomial multiplication
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import time
import logging
import logging.config
import numpy as np
from enum import Enum

# Import configuration
from config import (
    DEFAULT_MODULUS, ALLOWED_ORIGINS, API_TITLE, 
    API_VERSION, API_DESCRIPTION, ENABLE_PROFILING,
    TEST_VECTORS, SYSTEM_INFO, LOGGING_CONFIG
)

# Import core modules
from core.hybrid import hybrid_multiply
from core.polynomial import Polynomial

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# CORS MIDDLEWARE
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# ENUMS FOR ALGORITHM SELECTION
# ============================================

class AlgorithmChoice(str, Enum):
    """Available algorithms for polynomial multiplication"""
    AUTO = "auto"
    KARATSUBA = "karatsuba"
    TOOMCOOK = "toomcook"
    NTT = "ntt"

# ============================================
# PYDANTIC MODELS (Request/Response)
# ============================================

class PolyInput(BaseModel):
    """
    Input model for polynomial multiplication
    """
    A: List[int] = Field(..., description="First polynomial coefficients", example=[1, 2, 3, 4])
    B: List[int] = Field(..., description="Second polynomial coefficients", example=[5, 6, 7, 8])
    q: Optional[int] = Field(DEFAULT_MODULUS, description="Modulus (prime number)", ge=2, le=10000)
    algorithm: Optional[AlgorithmChoice] = Field(AlgorithmChoice.AUTO, description="Algorithm choice (auto for hybrid)")
    reduce_ring: Optional[bool] = Field(True, description="Apply (x^n + 1) reduction")
    
    @validator('A')
    def validate_A(cls, v):
        if not v:
            raise ValueError('Polynomial A cannot be empty')
        if len(v) > 4096:
            raise ValueError('Polynomial degree too large (max 4096)')
        return v
    
    @validator('B')
    def validate_B(cls, v):
        if not v:
            raise ValueError('Polynomial B cannot be empty')
        if len(v) > 4096:
            raise ValueError('Polynomial degree too large (max 4096)')
        return v
    
    @validator('q')
    def validate_q(cls, v):
        # Check if q is prime (simplified for demo)
        if v < 2:
            raise ValueError('Modulus must be >= 2')
        return v

class PolyResponse(BaseModel):
    """
    Response model for polynomial multiplication
    """
    algorithm: str = Field(..., description="Algorithm used")
    result: List[int] = Field(..., description="Result polynomial coefficients")
    time_ms: float = Field(..., description="Execution time in milliseconds")
    input_size: int = Field(..., description="Input polynomial degree")
    output_size: int = Field(..., description="Output polynomial degree")
    reduction_applied: bool = Field(..., description="Whether ring reduction was applied")

class RLWEInput(BaseModel):
    """
    Input model for RLWE sample generation
    """
    n: int = Field(..., description="Polynomial degree", ge=1, le=1024)
    q: Optional[int] = Field(DEFAULT_MODULUS, description="Modulus")
    seed: Optional[int] = Field(None, description="Random seed (for reproducibility)")

class RLWEResponse(BaseModel):
    """
    Response model for RLWE sample
    """
    a: List[int] = Field(..., description="Public polynomial a(x)")
    b: List[int] = Field(..., description="RLWE sample b = a*s + e")
    s: List[int] = Field(..., description="Secret polynomial s(x)")
    e: List[int] = Field(..., description="Error polynomial e(x)")
    n: int = Field(..., description="Degree")
    q: int = Field(..., description="Modulus")

class BenchmarkResult(BaseModel):
    """
    Benchmark result model
    """
    size: int
    karatsuba_time: float
    toomcook_time: float
    ntt_time: float
    fastest: str

class PerformanceStats(BaseModel):
    """Performance statistics model"""
    karatsuba: Dict[str, Any]
    toomcook: Dict[str, Any]
    ntt: Dict[str, Any]

# ============================================
# MIDDLEWARE
# ============================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log request details
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# ============================================
# HEALTH CHECK ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚀 Hybrid Polynomial Multiplication API",
        "version": API_VERSION,
        "status": "operational",
        "documentation": "/docs",
        "algorithms": ["Karatsuba", "Toom-Cook", "NTT"],
        "thresholds": {
            "karatsuba": "n < 64",
            "toomcook": "64 <= n < 256", 
            "ntt": "n >= 256"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "modules": ["core", "utils", "api"]
    }

@app.get("/info")
async def system_info():
    """Get system information"""
    return SYSTEM_INFO

# ============================================
# MAIN MULTIPLICATION ENDPOINT
# ============================================

@app.post("/multiply", response_model=PolyResponse)
async def multiply_polynomials(data: PolyInput):
    """
    Multiply two polynomials using hybrid algorithm selection
    
    - **A**: First polynomial coefficients
    - **B**: Second polynomial coefficients  
    - **q**: Modulus (default: 3329)
    - **algorithm**: Force specific algorithm (karatsuba, toomcook, ntt, auto)
    - **reduce_ring**: Apply (x^n + 1) reduction
    
    Returns:
    - **algorithm**: Algorithm used
    - **result**: Result polynomial
    - **time_ms**: Execution time
    - **input_size**: Input degree
    - **output_size**: Output degree
    """
    try:
        logger.info(f"Multiplication request: len(A)={len(data.A)}, len(B)={len(data.B)}")
        
        # Pad polynomials to same length
        max_len = max(len(data.A), len(data.B))
        A_padded = data.A + [0] * (max_len - len(data.A))
        B_padded = data.B + [0] * (max_len - len(data.B))
        
        # Perform multiplication
        start_time = time.time()
        
        # Convert algorithm enum to string
        algo_str = None if data.algorithm == AlgorithmChoice.AUTO else data.algorithm.value
        
        result_dict = hybrid_multiply(
            A_padded, 
            B_padded, 
            data.q, 
            force_algorithm=algo_str
        )
        
        # Apply ring reduction if requested
        if data.reduce_ring:
            poly = Polynomial(result_dict["result"], data.q, max_len)
            result_dict["result"] = poly.coeffs
        
        end_time = time.time()
        
        # Prepare response
        response = PolyResponse(
            algorithm=result_dict["algorithm"],
            result=result_dict["result"],
            time_ms=(end_time - start_time) * 1000,
            input_size=max_len,
            output_size=len(result_dict["result"]),
            reduction_applied=data.reduce_ring
        )
        
        logger.info(f"Multiplication completed: {response.algorithm} - {response.time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Multiplication failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# RLWE ENDPOINTS
# ============================================

@app.post("/rlwe/sample", response_model=RLWEResponse)
async def generate_rlwe_sample(data: RLWEInput):
    """
    Generate RLWE sample: b = a * s + e (mod q, x^n + 1)
    
    - **n**: Polynomial degree
    - **q**: Modulus
    - **seed**: Random seed (optional)
    
    Returns a, b, s, e polynomials
    """
    try:
        import random
        
        logger.info(f"RLWE sample request: n={data.n}, q={data.q}")
        
        # Set seed if provided
        if data.seed:
            random.seed(data.seed)
            np.random.seed(data.seed)
        
        # Generate random polynomials
        a = [random.randint(0, data.q-1) for _ in range(data.n)]
        s = [random.randint(0, data.q-1) for _ in range(data.n)]
        e = [random.randint(-3, 3) for _ in range(data.n)]  # Small error
        
        # Convert negative errors to modulo q
        e = [x % data.q for x in e]
        
        # Compute a * s
        mult_result = hybrid_multiply(a, s, data.q)
        a_times_s = mult_result["result"]
        
        # Create polynomial objects for ring reduction
        a_times_s_poly = Polynomial(a_times_s, data.q, data.n)
        e_poly = Polynomial(e, data.q, data.n)
        
        # b = a*s + e (mod x^n + 1)
        b_poly = a_times_s_poly + e_poly
        
        logger.info(f"RLWE sample generated successfully")
        
        return RLWEResponse(
            a=a,
            b=b_poly.coeffs,
            s=s,
            e=e,
            n=data.n,
            q=data.q
        )
        
    except Exception as e:
        logger.error(f"RLWE sample failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# BENCHMARK ENDPOINTS
# ============================================

@app.get("/benchmark", response_model=List[BenchmarkResult])
async def run_benchmark():
    """
    Run performance benchmarks for all algorithms
    Tests polynomial sizes: 16, 32, 64, 128, 256, 512, 1024
    """
    try:
        logger.info("Starting benchmark")
        
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        results = []
        
        for size in sizes:
            # Generate test polynomials
            A = [i % DEFAULT_MODULUS for i in range(size)]
            B = [(i * 2) % DEFAULT_MODULUS for i in range(size)]
            
            benchmark_result = {
                "size": size,
                "karatsuba_time": 0,
                "toomcook_time": 0,
                "ntt_time": 0
            }
            
            # Test each algorithm
            for algo in ["karatsuba", "toomcook", "ntt"]:
                try:
                    start = time.time()
                    hybrid_multiply(A, B, DEFAULT_MODULUS, algo)
                    end = time.time()
                    time_ms = (end - start) * 1000
                    benchmark_result[f"{algo}_time"] = round(time_ms, 3)
                except Exception as e:
                    logger.warning(f"Benchmark failed for {algo} size {size}: {str(e)}")
                    benchmark_result[f"{algo}_time"] = -1
            
            # Find fastest algorithm
            times = {
                "karatsuba": benchmark_result["karatsuba_time"],
                "toomcook": benchmark_result["toomcook_time"], 
                "ntt": benchmark_result["ntt_time"]
            }
            
            # Filter out -1 values
            valid_times = {k: v for k, v in times.items() if v > 0}
            if valid_times:
                fastest = min(valid_times, key=valid_times.get)
            else:
                fastest = "unknown"
            
            benchmark_result["fastest"] = fastest
            
            results.append(BenchmarkResult(**benchmark_result))
            
        logger.info(f"Benchmark completed: {len(results)} sizes tested")
        return results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark/size/{size}")
async def benchmark_size(size: int):
    """
    Benchmark specific polynomial size
    """
    try:
        if size > 2048:
            raise HTTPException(status_code=400, detail="Size too large (max 2048)")
        
        # Generate test polynomials
        A = [i % DEFAULT_MODULUS for i in range(size)]
        B = [(i * 3) % DEFAULT_MODULUS for i in range(size)]
        
        results = {}
        
        for algo in ["karatsuba", "toomcook", "ntt"]:
            try:
                start = time.time()
                result = hybrid_multiply(A, B, DEFAULT_MODULUS, algo)
                end = time.time()
                
                results[algo] = {
                    "time_ms": (end - start) * 1000,
                    "output_size": len(result["result"]),
                    "success": True
                }
            except Exception as e:
                results[algo] = {
                    "error": str(e),
                    "success": False
                }
        
        return {
            "size": size,
            "modulus": DEFAULT_MODULUS,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# PERFORMANCE STATS ENDPOINT
# ============================================

@app.get("/performance/stats", response_model=PerformanceStats)
async def get_performance_stats():
    """Get performance statistics from logs"""
    try:
        from utils.profiler import get_performance_stats
        stats = get_performance_stats()
        
        # Format stats for response
        formatted_stats = {}
        for algo in ["karatsuba", "toomcook", "ntt"]:
            if algo in stats:
                formatted_stats[algo] = stats[algo]
            else:
                formatted_stats[algo] = {"count": 0, "total_time": 0, "avg_time": 0}
        
        return PerformanceStats(**formatted_stats)
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# TEST VECTOR ENDPOINTS
# ============================================

@app.get("/test-vectors")
async def get_test_vectors():
    """Get predefined test vectors"""
    return TEST_VECTORS

@app.post("/verify")
async def verify_implementation(data: PolyInput):
    """
    Verify multiplication result against naive implementation
    Useful for testing correctness
    """
    try:
        from core.ntt import naive_multiply
        
        # Get result from hybrid
        hybrid_result = await multiply_polynomials(data)
        
        # Get result from naive
        naive_result = naive_multiply(data.A, data.B, data.q)
        
        # Compare
        min_len = min(len(hybrid_result.result), len(naive_result))
        match = hybrid_result.result[:min_len] == naive_result[:min_len]
        
        return {
            "match": match,
            "hybrid_result": hybrid_result.result[:min_len],
            "naive_result": naive_result[:min_len],
            "difference": [hybrid_result.result[i] - naive_result[i] for i in range(min_len)] if not match else []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# THRESHOLD MANAGEMENT
# ============================================

@app.get("/thresholds")
async def get_thresholds():
    """Get current algorithm selection thresholds"""
    from config import T1, T2
    
    return {
        "karatsuba_max": T1,
        "toomcook_min": T1,
        "toomcook_max": T2,
        "ntt_min": T2,
        "description": {
            "karatsuba": f"Used when n < {T1}",
            "toomcook": f"Used when {T1} <= n < {T2}",
            "ntt": f"Used when n >= {T2}"
        }
    }

@app.post("/thresholds/update")
async def update_thresholds(t1: int = None, t2: int = None):
    """
    Update algorithm selection thresholds
    Warning: This changes the hybrid decision logic
    """
    import config
    
    if t1:
        config.T1 = t1
    if t2:
        config.T2 = t2
    
    logger.info(f"Thresholds updated: T1={config.T1}, T2={config.T2}")
    
    return {
        "message": "Thresholds updated",
        "new_values": {
            "T1": config.T1,
            "T2": config.T2
        }
    }

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An error occurred",
            "path": request.url.path
        }
    )

# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🚀 Starting Hybrid Polynomial Multiplication API")
    print("=" * 50)
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"📊 Alternative docs: http://localhost:8000/redoc")
    print(f"🏥 Health check: http://localhost:8000/health")
    print("=" * 50)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )