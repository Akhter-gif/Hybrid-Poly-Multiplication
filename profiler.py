"""
Performance profiling utilities
Logs and analyzes algorithm performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from config import LOG_FILE

def log_performance(
    algorithm: str,
    input_size: int,
    time_ms: float,
    output_size: int,
    metadata: Optional[Dict] = None
) -> None:
    """
    Log performance metrics to file
    
    Args:
        algorithm: Name of algorithm used
        input_size: Size of input polynomials
        time_ms: Execution time in milliseconds
        output_size: Size of output polynomial
        metadata: Additional metadata
    """
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "algorithm": algorithm,
        "input_size": input_size,
        "time_ms": round(time_ms, 3),
        "output_size": output_size
    }
    
    if metadata:
        log_entry.update(metadata)
    
    # Append to log file
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def get_performance_stats(
    limit: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Get performance statistics from log file
    
    Args:
        limit: Maximum number of entries to analyze
    
    Returns:
        Statistics dictionary
    """
    if not os.path.exists(LOG_FILE):
        return {}
    
    stats = {
        "karatsuba": {"count": 0, "total_time": 0, "total_input": 0, "times": []},
        "toomcook": {"count": 0, "total_time": 0, "total_input": 0, "times": []},
        "ntt": {"count": 0, "total_time": 0, "total_input": 0, "times": []}
    }
    
    with open(LOG_FILE, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            
            try:
                entry = json.loads(line.strip())
                algo = entry["algorithm"].lower()
                
                # Match algorithm name
                for key in stats.keys():
                    if key in algo:
                        stats[key]["count"] += 1
                        stats[key]["total_time"] += entry["time_ms"]
                        stats[key]["total_input"] += entry["input_size"]
                        stats[key]["times"].append(entry["time_ms"])
                        break
                        
            except (json.JSONDecodeError, KeyError):
                continue
    
    # Calculate averages
    for algo in stats:
        if stats[algo]["count"] > 0:
            stats[algo]["avg_time"] = stats[algo]["total_time"] / stats[algo]["count"]
            stats[algo]["avg_input"] = stats[algo]["total_input"] / stats[algo]["count"]
            stats[algo]["min_time"] = min(stats[algo]["times"])
            stats[algo]["max_time"] = max(stats[algo]["times"])
    
    return stats

def clear_logs() -> None:
    """Clear all performance logs"""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

def export_logs(format: str = "json") -> str:
    """
    Export logs in specified format
    
    Args:
        format: "json" or "csv"
    
    Returns:
        Formatted string
    """
    if not os.path.exists(LOG_FILE):
        return ""
    
    with open(LOG_FILE, 'r') as f:
        entries = [json.loads(line.strip()) for line in f]
    
    if format == "json":
        return json.dumps(entries, indent=2)
    elif format == "csv":
        if not entries:
            return ""
        
        # Create CSV header
        header = list(entries[0].keys())
        csv_lines = [",".join(header)]
        
        for entry in entries:
            csv_lines.append(",".join(str(entry.get(h, "")) for h in header))
        
        return "\n".join(csv_lines)
    
    return ""