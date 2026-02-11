"""
Predictive Cache and Prefetch System
Intelligent caching with machine learning-based prefetching strategy.

Features:
- Predictive access pattern analysis
- ML-based prefetch recommendations
- Distributed cache invalidation
- Compression for large datasets
- Smart TTL management
- Cache hit rate optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
import threading
import hashlib
from collections import defaultdict, deque
import logging
from enum import Enum
import zlib

logger = logging.getLogger(__name__)


class CachePriority(Enum):
    """Cache priority levels."""
    CRITICAL = 1      # Must keep in cache
    HIGH = 2          # Keep in fast cache
    MEDIUM = 3        # Standard cache
    LOW = 4           # Can be evicted
    EPHEMERAL = 5     # Temporary, will be evicted


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    ttl: int  # Time-to-live in seconds
    hit_count: int = 0
    priority: CachePriority = CachePriority.MEDIUM
    size_bytes: int = 0
    access_times: deque = field(default_factory=lambda: deque(maxlen=100))
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl
    
    def record_access(self):
        """Record access time for pattern analysis."""
        self.hit_count += 1
        self.access_times.append(datetime.now())
    
    def get_access_frequency(self, window_seconds: int = 3600) -> float:
        """Get access frequency in accesses per second."""
        if not self.access_times:
            return 0.0
        
        now = datetime.now()
        recent_accesses = sum(
            1 for t in self.access_times 
            if (now - t).total_seconds() < window_seconds
        )
        return recent_accesses / window_seconds if window_seconds > 0 else 0.0


class AccessPatternAnalyzer:
    """Analyzes access patterns to predict future access."""
    
    def __init__(self, history_window: int = 1000):
        """
        Initialize analyzer.
        
        Args:
            history_window: Number of accesses to analyze
        """
        self.history_window = history_window
        self.access_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.temporal_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.lock = threading.Lock()
    
    def record_access(self, key: str, related_keys: Optional[List[str]] = None):
        """
        Record a key access with optional related keys.
        
        Args:
            key: Accessed key
            related_keys: Keys typically accessed together
        """
        with self.lock:
            # Record temporal pattern
            self.temporal_patterns[key].append(datetime.now())
            
            # Build access graph
            if related_keys:
                for related_key in related_keys:
                    self.access_graph[key][related_key] += 1
    
    def predict_next_accesses(self, key: str, count: int = 5) -> List[Tuple[str, float]]:
        """
        Predict next keys likely to be accessed.
        
        Args:
            key: Current key accessed
            count: Number of predictions
        
        Returns:
            List of (key, confidence) tuples
        """
        if key not in self.access_graph:
            return []
        
        # Get related keys with their access counts
        related = self.access_graph[key]
        if not related:
            return []
        
        # Normalize to confidence scores
        total = sum(related.values())
        predictions = [
            (k, v / total) 
            for k, v in sorted(related.items(), key=lambda x: -x[1])[:count]
        ]
        
        return predictions
    
    def get_access_velocity(self, key: str) -> float:
        """Get access velocity (accesses per minute)."""
        if key not in self.temporal_patterns:
            return 0.0
        
        times = list(self.temporal_patterns[key])
        if len(times) < 2:
            return 0.0
        
        age_seconds = (datetime.now() - times[0]).total_seconds()
        if age_seconds == 0:
            return 0.0
        
        return len(times) / (age_seconds / 60)


class PredictiveCache:
    """
    Intelligent cache with predictive prefetching and ML-based optimization.
    """
    
    def __init__(self, 
                 max_memory_mb: int = 1000,
                 enable_compression: bool = True,
                 enable_prefetch: bool = True):
        """
        Initialize predictive cache.
        
        Args:
            max_memory_mb: Maximum memory in MB
            enable_compression: Enable compression for large values
            enable_prefetch: Enable predictive prefetching
        """
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_compression = enable_compression
        self.enable_prefetch = enable_prefetch
        
        # Storage
        self.cache: Dict[str, CacheEntry] = {}
        self.prefetch_queue: deque = deque(maxlen=100)
        
        # Analysis
        self.analyzer = AccessPatternAnalyzer()
        
        # State tracking
        self.total_hits = 0
        self.total_misses = 0
        self.current_memory_bytes = 0
        self.lock = threading.RLock()
        
        logger.info(f"Predictive cache initialized (max: {max_memory_mb}MB)")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with pattern tracking.
        
        Args:
            key: Cache key
            default: Default value if not found or expired
        
        Returns:
            Cached value or default
        """
        with self.lock:
            if key not in self.cache:
                self.total_misses += 1
                return default
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._evict(key)
                self.total_misses += 1
                return default
            
            # Record access
            entry.record_access()
            self.total_hits += 1
            
            # Record pattern
            self.analyzer.record_access(key)
            
            # Trigger prefetch if enabled
            if self.enable_prefetch:
                predictions = self.analyzer.predict_next_accesses(key, count=3)
                for predicted_key, _ in predictions:
                    if predicted_key not in self.cache:
                        self.prefetch_queue.append(predicted_key)
            
            # Decompress if needed
            value = entry.value
            if entry.compressed and isinstance(value, bytes):
                try:
                    value = pickle.loads(zlib.decompress(value))
                except Exception as e:
                    logger.error(f"Decompression error for key {key}: {e}")
            
            return value
    
    def set(self, key: str, value: Any, ttl: int = 3600, 
            priority: CachePriority = CachePriority.MEDIUM,
            related_keys: Optional[List[str]] = None):
        """
        Set value in cache with TTL and priority.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            priority: Priority level
            related_keys: Keys typically accessed together
        """
        with self.lock:
            # Remove old entry if exists
            if key in self.cache:
                self._evict(key)
            
            # Serialize and optionally compress
            serialized_value = value
            compressed = False
            
            # Check if compression would help
            if self.enable_compression:
                try:
                    pickled = pickle.dumps(value)
                    if isinstance(value, (pd.DataFrame, dict, list)) and len(pickled) > 1000:
                        compressed_data = zlib.compress(pickled)
                        if len(compressed_data) < len(pickled) * 0.8:  # >20% compression
                            serialized_value = compressed_data
                            compressed = True
                except Exception as e:
                    logger.debug(f"Compression error: {e}")
            
            # Calculate size
            try:
                if isinstance(serialized_value, bytes):
                    size = len(serialized_value)
                else:
                    size = len(pickle.dumps(serialized_value))
            except:
                size = 0
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=serialized_value,
                timestamp=datetime.now(),
                ttl=ttl,
                priority=priority,
                size_bytes=size,
                compressed=compressed
            )
            
            # Check if we need to evict
            while self.current_memory_bytes + size > self.max_memory_bytes and self.cache:
                self._evict_lru()
            
            self.cache[key] = entry
            self.current_memory_bytes += size
            
            # Record access pattern
            if related_keys:
                self.analyzer.record_access(key, related_keys)
            
            logger.debug(f"Cached {key} ({size} bytes, TTL: {ttl}s)")
    
    def _evict(self, key: str):
        """Evict specific key from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory_bytes -= entry.size_bytes
            del self.cache[key]
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Score: (priority_value * recency_weight + hit_count * hit_weight) / access_frequency
        def score(entry: CacheEntry) -> float:
            age_minutes = (datetime.now() - entry.timestamp).total_seconds() / 60
            recency = max(0.1, 1.0 / (1 + age_minutes))
            hit_boost = min(entry.hit_count / 100, 1.0)
            priority_weight = entry.priority.value  # Higher = lower priority
            
            return priority_weight * recency * (1 + hit_boost)
        
        # Find entry with lowest score
        key_to_evict = max(
            self.cache.items(),
            key=lambda x: score(x[1])
        )[0]
        
        self._evict(key_to_evict)
    
    def get_prefetch_candidates(self, count: int = 10) -> List[str]:
        """
        Get keys that should be prefetched.
        
        Args:
            count: Number of candidates
        
        Returns:
            List of keys to prefetch
        """
        candidates = []
        with self.lock:
            # From prefetch queue
            for _ in range(min(count, len(self.prefetch_queue))):
                candidates.append(self.prefetch_queue.popleft())
        
        return candidates
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.total_hits + self.total_misses
            hit_rate = self.total_hits / total_requests if total_requests > 0 else 0
            
            return {
                'total_items': len(self.cache),
                'memory_mb': self.current_memory_bytes / 1024 / 1024,
                'memory_percent': (self.current_memory_bytes / self.max_memory_bytes) * 100,
                'hit_rate': hit_rate,
                'total_hits': self.total_hits,
                'total_misses': self.total_misses,
                'prefetch_queue_size': len(self.prefetch_queue),
            }
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            self.current_memory_bytes = 0
            self.total_hits = 0
            self.total_misses = 0
            logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        with self.lock:
            expired_keys = [
                k for k, v in self.cache.items() 
                if v.is_expired()
            ]
            
            for key in expired_keys:
                self._evict(key)
            
            return len(expired_keys)


class SmartDataPrefetcher:
    """
    Intelligent prefetcher that predicts data access patterns.
    """
    
    def __init__(self, cache: PredictiveCache):
        """
        Initialize prefetcher.
        
        Args:
            cache: PredictiveCache instance to manage
        """
        self.cache = cache
        self.prefetch_functions: Dict[str, callable] = {}
        self.is_running = False
        self.prefetch_thread: Optional[threading.Thread] = None
    
    def register_prefetch_function(self, key_pattern: str, fetch_function: callable):
        """
        Register a function to prefetch data for a key pattern.
        
        Args:
            key_pattern: Pattern for keys (e.g., "stock:*")
            fetch_function: Async function that fetches data
        """
        self.key_pattern = key_pattern
        self.prefetch_functions[key_pattern] = fetch_function
    
    def start_prefetching(self):
        """Start background prefetching."""
        if self.is_running:
            return
        
        self.is_running = True
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True
        )
        self.prefetch_thread.start()
        logger.info("Prefetching started")
    
    def stop_prefetching(self):
        """Stop background prefetching."""
        self.is_running = False
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5)
        logger.info("Prefetching stopped")
    
    def _prefetch_loop(self):
        """Background prefetch loop."""
        while self.is_running:
            try:
                candidates = self.cache.get_prefetch_candidates(count=5)
                
                for key in candidates:
                    for pattern, fetch_func in self.prefetch_functions.items():
                        if self._matches_pattern(key, pattern):
                            try:
                                data = fetch_func(key)
                                self.cache.set(
                                    key, data,
                                    ttl=3600,
                                    priority=CachePriority.MEDIUM
                                )
                            except Exception as e:
                                logger.error(f"Prefetch error for {key}: {e}")
                
                # Check and cleanup expired entries periodically
                self.cache.cleanup_expired()
                
                # Sleep before next cycle
                import time
                time.sleep(5)
            
            except Exception as e:
                logger.error(f"Prefetch loop error: {e}")
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern."""
        if '*' in pattern:
            import fnmatch
            return fnmatch.fnmatch(key, pattern)
        return key == pattern
