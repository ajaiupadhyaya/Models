"""
Data caching system for performance optimization.
Intelligent caching with TTL and automatic refresh.
"""

import os
import pickle
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Callable, Any, Dict
import warnings
warnings.filterwarnings('ignore')


class DataCache:
    """
    Intelligent data caching with TTL and automatic invalidation.
    """
    
    def __init__(self, cache_dir: str = 'data/cache', default_ttl: int = 3600):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory for cache files
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function and arguments."""
        key_string = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _is_cache_valid(self, cache_path: str, ttl: int) -> bool:
        """Check if cache is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age = (datetime.now() - file_time).total_seconds()
        
        return age < ttl
    
    def get(self, cache_key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        Get data from cache.
        
        Args:
            cache_key: Cache key
            ttl: Time-to-live in seconds (uses default if None)
        
        Returns:
            Cached data or None if not found/invalid
        """
        if ttl is None:
            ttl = self.default_ttl
        
        cache_path = self._get_cache_path(cache_key)
        
        if not self._is_cache_valid(cache_path, ttl):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def set(self, cache_key: str, data: Any):
        """
        Store data in cache.
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def cached(self, ttl: Optional[int] = None):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time-to-live in seconds
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                cache_key = self._get_cache_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key, ttl)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(cache_key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def clear(self, pattern: Optional[str] = None):
        """
        Clear cache files.
        
        Args:
            pattern: Optional pattern to match (clears all if None)
        """
        if pattern is None:
            # Clear all
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        else:
            # Clear matching files
            for filename in os.listdir(self.cache_dir):
                if pattern in filename and filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
    
    def get_cache_info(self) -> Dict:
        """
        Get information about cache.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        
        total_size = 0
        for filename in cache_files:
            filepath = os.path.join(self.cache_dir, filename)
            total_size += os.path.getsize(filepath)
        
        return {
            'num_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': self.cache_dir
        }


# Global cache instance
_cache = DataCache()


def cached(ttl: Optional[int] = None):
    """Convenience decorator for caching."""
    return _cache.cached(ttl)
