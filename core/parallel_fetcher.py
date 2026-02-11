"""
Parallel Data Fetching System
High-performance data fetching with concurrent requests and smart scheduling.

Features:
- Concurrent batch data fetching
- Smart request scheduling
- Duplicate request detection
- Rate limiting with token bucket
- Adaptive retry strategies
- Request prioritization
"""

import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import queue
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class DataRequest:
    """Represents a data fetch request."""
    request_id: str
    symbol: str
    data_type: str  # 'price', 'economic', 'intraday'
    priority: RequestPriority = RequestPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    completed: bool = False
    
    def __lt__(self, other: 'DataRequest') -> bool:
        """Enable priority queue sorting."""
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_second: float = 10):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
        """
        self.requests_per_second = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            Time waited in seconds
        """
        with self.lock:
            wait_time = 0.0
            
            while self.tokens < tokens:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.requests_per_second,
                    self.tokens + elapsed * self.requests_per_second
                )
                self.last_update = now
                
                if self.tokens < tokens:
                    wait_time += 0.01
                    time.sleep(0.01)
            
            self.tokens -= tokens
            return wait_time
    
    def reset(self):
        """Reset rate limiter."""
        with self.lock:
            self.tokens = self.requests_per_second
            self.last_update = time.time()


class RequestDeduplicator:
    """Detects and deduplicates identical concurrent requests."""
    
    def __init__(self, window_seconds: int = 60):
        """
        Initialize deduplicator.
        
        Args:
            window_seconds: Time window for deduplication
        """
        self.window = window_seconds
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
        self.lock = threading.Lock()
    
    def get_cache_key(self, symbol: str, data_type: str) -> str:
        """Generate cache key."""
        return f"{symbol}:{data_type}"
    
    def register_pending(self, key: str, future: asyncio.Future) -> bool:
        """
        Register a pending request.
        
        Returns:
            True if new request, False if already pending
        """
        with self.lock:
            if key in self.pending_requests:
                return False
            self.pending_requests[key] = future
            return True
    
    def complete_pending(self, key: str, result: Any):
        """Mark request as complete."""
        with self.lock:
            if key in self.pending_requests:
                del self.pending_requests[key]
            self.request_cache[key] = (result, datetime.now())
    
    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached result if not expired."""
        with self.lock:
            if key in self.request_cache:
                result, timestamp = self.request_cache[key]
                if (datetime.now() - timestamp).total_seconds() < self.window:
                    return result
            return None


class ParallelDataFetcher:
    """
    High-performance parallel data fetching system.
    """
    
    def __init__(self,
                 max_workers: int = 10,
                 requests_per_second: float = 10,
                 cache_ttl: int = 300):
        """
        Initialize parallel fetcher.
        
        Args:
            max_workers: Maximum concurrent threads
            requests_per_second: Rate limit
            cache_ttl: Cache TTL in seconds
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.rate_limiter = RateLimiter(requests_per_second)
        self.deduplicator = RequestDeduplicator(window_seconds=cache_ttl)
        
        # Request queue
        self.request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.active_requests: Dict[str, DataRequest] = {}
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        
        self.is_running = False
        self.lock = threading.RLock()
    
    def submit_request(self, 
                      symbol: str,
                      data_type: str = 'price',
                      priority: RequestPriority = RequestPriority.NORMAL,
                      callback: Optional[Callable] = None) -> str:
        """
        Submit a data fetch request.
        
        Args:
            symbol: Data symbol
            data_type: Type of data to fetch
            priority: Request priority
            callback: Callback function when data available
        
        Returns:
            Request ID
        """
        request_id = f"{symbol}:{data_type}:{datetime.now().timestamp()}"
        
        request = DataRequest(
            request_id=request_id,
            symbol=symbol,
            data_type=data_type,
            priority=priority,
            callback=callback
        )
        
        with self.lock:
            self.request_queue.put((priority.value, request))
            self.active_requests[request_id] = request
            self.total_requests += 1
        
        return request_id
    
    def submit_batch(self,
                    symbols: List[str],
                    data_type: str = 'price',
                    priority: RequestPriority = RequestPriority.NORMAL) -> List[str]:
        """
        Submit multiple requests in batch.
        
        Args:
            symbols: List of symbols
            data_type: Type of data
            priority: Priority level
        
        Returns:
            List of request IDs
        """
        request_ids = []
        for symbol in symbols:
            request_id = self.submit_request(symbol, data_type, priority)
            request_ids.append(request_id)
        return request_ids
    
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Get result of a request.
        
        Args:
            request_id: Request ID
            timeout: Max wait time in seconds
        
        Returns:
            Result data or None if not ready
        """
        with self.lock:
            if request_id not in self.active_requests:
                return None
            
            request = self.active_requests[request_id]
            
            if timeout:
                start_time = time.time()
                while not request.completed:
                    if time.time() - start_time > timeout:
                        return None
                    time.sleep(0.01)
            
            return request.result if request.completed else None
    
    def start_processing(self):
        """Start the request processor thread."""
        if self.is_running:
            return
        
        self.is_running = True
        processor_thread = threading.Thread(
            target=self._process_queue,
            daemon=True
        )
        processor_thread.start()
        logger.info("Parallel data fetcher started")
    
    def stop_processing(self):
        """Stop the request processor."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("Parallel data fetcher stopped")
    
    def _process_queue(self):
        """Main request processing loop."""
        while self.is_running:
            try:
                # Get next request
                priority, request = self.request_queue.get(timeout=1)
                
                # Check cache
                cache_key = self.deduplicator.get_cache_key(request.symbol, request.data_type)
                cached_result = self.deduplicator.get_cached(cache_key)
                
                if cached_result is not None:
                    request.result = cached_result
                    request.completed = True
                    self.cache_hits += 1
                    if request.callback:
                        request.callback(request)
                    continue
                
                # Apply rate limiting
                self.rate_limiter.acquire()
                
                # Submit to executor
                future = self.executor.submit(
                    self._fetch_data,
                    request
                )
                
                # Process result
                try:
                    result = future.result(timeout=30)
                    request.result = result
                    request.completed = True
                    self.successful_requests += 1
                    
                    # Cache result
                    self.deduplicator.complete_pending(cache_key, result)
                    
                except Exception as e:
                    request.error = str(e)
                    request.retry_count += 1
                    
                    if request.retry_count < request.max_retries:
                        # Re-queue for retry
                        request.priority = RequestPriority(request.priority.value + 1)
                        self.request_queue.put((request.priority.value, request))
                    else:
                        request.completed = True
                        self.failed_requests += 1
                
                if request.callback:
                    request.callback(request)
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in request processor: {e}")
    
    def _fetch_data(self, request: DataRequest) -> Any:
        """Fetch actual data based on request type."""
        try:
            if request.data_type == 'price':
                data = yf.download(
                    request.symbol,
                    period='1y',
                    progress=False
                )
                return data
            
            elif request.data_type == 'intraday':
                data = yf.download(
                    request.symbol,
                    period='1d',
                    interval='1m',
                    progress=False
                )
                return data
            
            else:
                raise ValueError(f"Unknown data type: {request.data_type}")
        
        except Exception as e:
            logger.error(f"Error fetching {request.symbol}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics."""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'cache_hits': self.cache_hits,
            'success_rate': (
                self.successful_requests / self.total_requests
                if self.total_requests > 0 else 0
            ),
            'active_requests': len(self.active_requests),
            'queue_size': self.request_queue.qsize(),
        }


def create_parallel_fetcher(symbols: List[str],
                           max_workers: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple symbols in parallel efficiently.
    
    Args:
        symbols: List of symbols
        max_workers: Max concurrent fetches
    
    Returns:
        Dictionary of symbol -> DataFrame
    """
    fetcher = ParallelDataFetcher(max_workers=max_workers)
    fetcher.start_processing()
    
    try:
        # Submit all requests
        request_ids = fetcher.submit_batch(symbols, priority=RequestPriority.HIGH)
        
        # Collect results
        results = {}
        for symbol, request_id in zip(symbols, request_ids):
            data = fetcher.get_result(request_id, timeout=60)
            if data is not None:
                results[symbol] = data
        
        return results
    
    finally:
        fetcher.stop_processing()
