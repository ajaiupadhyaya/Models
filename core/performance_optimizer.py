"""
Performance Optimization Utilities
Caching, parallel processing, and efficiency improvements
"""

import functools
import hashlib
import pickle
import time
from typing import Callable, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SmartCache:
    """
    Intelligent caching system with TTL and size limits.
    """
    
    def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 3600, max_size_mb: int = 500):
        """
        Initialize cache.
        
        Args:
            cache_dir: Cache directory
            default_ttl: Default time-to-live in seconds
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size_mb = max_size_mb
        self._cache_metadata = {}
    
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function and arguments."""
        key_data = f"{func_name}_{args}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, cache_key: str) -> Optional[Any]:
        """
        Get cached value.
        
        Args:
            cache_key: Cache key
        
        Returns:
            Cached value or None
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            # Check metadata
            if cache_key in self._cache_metadata:
                metadata = self._cache_metadata[cache_key]
                if time.time() > metadata['expires_at']:
                    cache_path.unlink()
                    del self._cache_metadata[cache_key]
                    return None
            
            # Load cached data
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def set(self, cache_key: str, value: Any, ttl: Optional[int] = None):
        """
        Set cached value.
        
        Args:
            cache_key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        if ttl is None:
            ttl = self.default_ttl
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            self._cache_metadata[cache_key] = {
                'expires_at': time.time() + ttl,
                'size': cache_path.stat().st_size
            }
            
            # Check cache size
            self._enforce_size_limit()
        
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _enforce_size_limit(self):
        """Enforce maximum cache size."""
        total_size = sum(
            self._get_cache_path(key).stat().st_size 
            for key in self._cache_metadata.keys()
            if self._get_cache_path(key).exists()
        )
        
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Remove oldest entries
            sorted_keys = sorted(
                self._cache_metadata.items(),
                key=lambda x: x[1]['expires_at']
            )
            
            for key, _ in sorted_keys:
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
                    del self._cache_metadata[key]
                
                total_size = sum(
                    self._get_cache_path(k).stat().st_size 
                    for k in self._cache_metadata.keys()
                    if self._get_cache_path(k).exists()
                )
                
                if total_size <= max_size_bytes:
                    break
    
    def clear(self):
        """Clear all cache."""
        for cache_path in self.cache_dir.glob("*.pkl"):
            cache_path.unlink()
        self._cache_metadata.clear()


# Global cache instance
_global_cache = SmartCache()


def cached(ttl: int = 3600, cache_key_prefix: str = ""):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds
        cache_key_prefix: Prefix for cache key
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = _global_cache._get_cache_key(
                f"{cache_key_prefix}{func.__name__}",
                *args,
                **kwargs
            )
            
            # Try to get from cache
            cached_value = _global_cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Compute and cache
            result = func(*args, **kwargs)
            _global_cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


def parallel_process(data: list, func: Callable, n_jobs: int = 4) -> list:
    """
    Process data in parallel.
    
    Args:
        data: List of data items
        func: Function to apply
        n_jobs: Number of parallel jobs
    
    Returns:
        List of results
    """
    try:
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
        
        # Use ThreadPoolExecutor for I/O bound tasks
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(func, data))
        
        return results
    
    except Exception as e:
        logger.warning(f"Parallel processing failed: {e}, falling back to sequential")
        return [func(item) for item in data]


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage.
    
    Args:
        df: DataFrame to optimize
    
    Returns:
        Optimized DataFrame
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    return df
