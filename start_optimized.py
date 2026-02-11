"""
Optimized Platform Startup
Launches the quantitative trading platform with all advanced features.

Features enabled:
- Real-time data streaming
- Intelligent caching with predictive prefetch
- Advanced signal generation with market regime detection
- Live trading with risk guardrails
- Parallel data fetching
- Production-ready configuration
- Real-time monitoring and alerting
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'platform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import optimized modules
from core.config_manager import load_config, get_config_manager
from core.realtime_streamer import RealTimeStreamer, create_default_streamer
from core.predictive_cache import PredictiveCache, SmartDataPrefetcher
from core.signal_generator import AdvancedSignalGenerator, MarketRegimeDetector
from core.live_trading import LiveTradingEngine, RiskManager
from core.parallel_fetcher import ParallelDataFetcher, RequestPriority
from core.data_fetcher import DataFetcher


class QuantPlatformOptimized:
    """
    Optimized quantitative trading platform with all advanced features.
    """
    
    def __init__(self, environment: str = 'development'):
        """
        Initialize optimized platform.
        
        Args:
            environment: 'development', 'staging', or 'production'
        """
        logger.info("="*60)
        logger.info("Initializing Optimized Quantitative Trading Platform")
        logger.info("="*60)
        
        # Load configuration
        config_manager = get_config_manager()
        self.config = config_manager.load(environment)
        
        # Validate configuration
        is_valid, errors = config_manager.validate(self.config)
        if not is_valid:
            logger.error(f"Configuration validation failed: {errors}")
            raise ValueError("Invalid configuration")
        
        logger.info(f"Configuration loaded: {self.config.environment.value}")
        logger.info(f"Performance profile: {self.config.performance_profile.value}")
        
        # Initialize components
        self._init_data_systems()
        self._init_analytics_systems()
        self._init_trading_systems()
        
        logger.info("Platform initialization complete")
    
    def _init_data_systems(self):
        """Initialize data fetching and caching systems."""
        logger.info("Initializing data systems...")
        
        # Real-time streamer
        self.streamer = create_default_streamer(
            symbols=self.config.portfolio.watchlist
        )
        logger.info(f"Real-time streamer initialized for {len(self.config.portfolio.watchlist)} symbols")
        
        # Predictive cache
        self.cache = PredictiveCache(
            max_memory_mb=self.config.data.cache_max_mb,
            enable_compression=self.config.data.enable_compression,
            enable_prefetch=self.config.data.enable_prefetch
        )
        logger.info(f"Predictive cache initialized ({self.config.data.cache_max_mb}MB)")
        
        # Smart prefetcher
        self.prefetcher = SmartDataPrefetcher(self.cache)
        if self.config.data.enable_prefetch:
            self.prefetcher.start_prefetching()
        logger.info("Smart prefetcher initialized")
        
        # Parallel data fetcher
        self.parallel_fetcher = ParallelDataFetcher(
            max_workers=self.config.data.parallel_workers,
            requests_per_second=self.config.data.rate_limit_per_second,
            cache_ttl=self.config.data.cache_ttl
        )
        self.parallel_fetcher.start_processing()
        logger.info(f"Parallel fetcher initialized ({self.config.data.parallel_workers} workers)")
        
        # Traditional data fetcher
        self.data_fetcher = DataFetcher()
        logger.info("Data fetcher initialized")
    
    def _init_analytics_systems(self):
        """Initialize analytics and signal generation."""
        logger.info("Initializing analytics systems...")
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(lookback=50)
        logger.info("Market regime detector initialized")
        
        # Advanced signal generator
        self.signal_generator = AdvancedSignalGenerator(lookback=50)
        logger.info("Advanced signal generator initialized")
    
    def _init_trading_systems(self):
        """Initialize trading and risk management."""
        logger.info("Initializing trading systems...")
        
        # Risk manager
        self.risk_manager = RiskManager(
            initial_capital=self.config.portfolio.initial_capital,
            max_position_size=self.config.trading.max_position_size_pct,
            max_daily_loss=self.config.trading.max_daily_loss_pct,
            max_correlation=self.config.trading.max_correlation,
            max_sector_exposure=self.config.trading.max_sector_exposure_pct
        )
        logger.info(f"Risk manager initialized (capital: ${self.config.portfolio.initial_capital:,.0f})")
        
        # Live trading engine
        self.trading_engine = LiveTradingEngine(
            risk_manager=self.risk_manager,
            trading_hours_only=self.config.trading.hours_only
        )
        
        if self.config.trading.enabled:
            logger.info("LIVE TRADING ENABLED - Risk guardrails active")
        else:
            logger.info("Live trading disabled - running in analysis mode")
    
    def start_real_time_streaming(self):
        """Start real-time data streaming."""
        logger.info("Starting real-time data streaming...")
        self.streamer.start_streaming(data_source='yahoo')
        logger.info("Real-time streaming active")
    
    def stop_real_time_streaming(self):
        """Stop real-time data streaming."""
        logger.info("Stopping real-time data streaming...")
        self.streamer.stop_streaming()
        logger.info("Real-time streaming stopped")
    
    def fetch_watchlist_data(self):
        """Fetch initial data for watchlist."""
        logger.info("Fetching initial watchlist data...")
        
        request_ids = self.parallel_fetcher.submit_batch(
            symbols=self.config.portfolio.watchlist,
            data_type='price',
            priority=RequestPriority.HIGH
        )
        
        logger.info(f"Submitted {len(request_ids)} data requests")
        
        results = {}
        for symbol, request_id in zip(self.config.portfolio.watchlist, request_ids):
            data = self.parallel_fetcher.get_result(request_id, timeout=30)
            if data is not None:
                results[symbol] = data
        
        logger.info(f"Successfully fetched {len(results)}/{len(self.config.portfolio.watchlist)} symbols")
        
        return results
    
    def generate_signals(self, price_data):
        """Generate trading signals for watchlist."""
        logger.info("Generating advanced trading signals...")
        
        all_signals = []
        
        for symbol, df in price_data.items():
            if df.empty:
                continue
            
            df.attrs['symbol'] = symbol
            signals = self.signal_generator.generate_signals(df)
            all_signals.extend(signals)
            
            if signals:
                for signal in signals:
                    logger.info(
                        f"Signal: {signal.symbol} {signal.signal_type.value} "
                        f"(strength: {signal.strength:.2f}, confidence: {signal.confidence:.2f})"
                    )
        
        logger.info(f"Generated {len(all_signals)} trading signals")
        
        return all_signals
    
    def get_system_status(self) -> dict:
        """Get current system status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.config.environment.value,
            'performance_profile': self.config.performance_profile.value,
            'data': {
                'streaming': self.streamer.is_streaming,
                'cache_stats': self.cache.get_stats(),
                'fetcher_stats': self.parallel_fetcher.get_stats(),
                'market_info': self.streamer.get_market_info()
            },
            'trading': {
                'enabled': self.config.trading.enabled,
                'portfolio_metrics': self.risk_manager.get_portfolio_metrics().__dict__,
                'risk_alerts': self.trading_engine.check_circuit_breaker()
            },
            'config': {
                'watchlist_size': len(self.config.portfolio.watchlist),
                'initial_capital': self.config.portfolio.initial_capital
            }
        }
        
        return status
    
    def shutdown(self):
        """Gracefully shutdown platform."""
        logger.info("Shutting down platform...")
        
        # Stop streaming
        if self.streamer.is_streaming:
            self.stop_real_time_streaming()
        
        # Stop prefetcher
        if self.prefetcher.is_running:
            self.prefetcher.stop_prefetching()
        
        # Stop parallel fetcher
        self.parallel_fetcher.stop_processing()
        
        logger.info("Platform shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Optimized Quantitative Trading Platform'
    )
    parser.add_argument(
        '--environment',
        choices=['development', 'staging', 'production'],
        default='development',
        help='Deployment environment'
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Start real-time data streaming'
    )
    parser.add_argument(
        '--fetch-watchlist',
        action='store_true',
        help='Fetch initial watchlist data'
    )
    parser.add_argument(
        '--generate-signals',
        action='store_true',
        help='Generate trading signals'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize platform
        platform = QuantPlatformOptimized(environment=args.environment)
        
        # Execute requested operations
        if args.fetch_watchlist:
            price_data = platform.fetch_watchlist_data()
            
            if args.generate_signals:
                signals = platform.generate_signals(price_data)
        
        if args.stream:
            platform.start_real_time_streaming()
            logger.info("Platform running with real-time streaming...")
            try:
                import time
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                platform.shutdown()
        
        if args.status:
            status = platform.get_system_status()
            import json
            print(json.dumps(status, indent=2, default=str))
        
        if not (args.stream or args.fetch_watchlist or args.status):
            # Just show status by default
            status = platform.get_system_status()
            import json
            print(json.dumps(status, indent=2, default=str))
    
    except Exception as e:
        logger.error(f"Platform error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
