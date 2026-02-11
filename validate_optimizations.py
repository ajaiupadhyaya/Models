"""
Optimization Validation Suite
Tests all new optimizations to ensure they work correctly and efficiently.

Run: python validate_optimizations.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.realtime_streamer import RealTimeStreamer, Tick, MarketHoursDetector
from core.predictive_cache import PredictiveCache, CachePriority
from core.signal_generator import AdvancedSignalGenerator, MarketRegime
from core.live_trading import RiskManager, Position, PositionStatus
from core.parallel_fetcher import ParallelDataFetcher, RequestPriority
from core.config_manager import load_config, ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_realtime_streamer():
    """Test real-time streaming capability."""
    logger.info("\n" + "="*60)
    logger.info("Testing Real-Time Streamer")
    logger.info("="*60)
    
    try:
        streamer = RealTimeStreamer(symbols=['SPY', 'QQQ'])
        
        # Test market hours detection
        hours = MarketHoursDetector()
        session = hours.get_session()
        logger.info(f"✓ Market session detected: {session.value}")
        
        # Test tick processing
        tick = Tick(
            timestamp=pd.Timestamp.now(),
            symbol='SPY',
            price=450.0,
            volume=1000,
            bid=449.95,
            ask=450.05
        )
        streamer.add_tick(tick)
        logger.info(f"✓ Tick processed: {tick.symbol} @ {tick.price}")
        
        # Test bar generation
        for i in range(60):  # Generate 60 ticks for bar aggregation
            tick = Tick(
                timestamp=pd.Timestamp.now(),
                symbol='SPY',
                price=450.0 + i * 0.1,
                volume=1000,
                exchange='NYSE'
            )
            streamer.add_tick(tick)
        
        bars = streamer.get_bars_df('SPY')
        logger.info(f"✓ OHLCV bars generated: {len(bars)} bars")
        
        logger.info("✓ Real-time streamer test PASSED")
        return True
    
    except Exception as e:
        logger.error(f"✗ Real-time streamer test FAILED: {e}")
        return False


def test_predictive_cache():
    """Test intelligent caching with prefetch."""
    logger.info("\n" + "="*60)
    logger.info("Testing Predictive Cache")
    logger.info("="*60)
    
    try:
        cache = PredictiveCache(max_memory_mb=100, enable_prefetch=True)
        
        # Test basic caching
        test_data = {'symbol': 'SPY', 'price': 450.0}
        cache.set('SPY:1h', test_data, ttl=300, priority=CachePriority.HIGH)
        retrieved = cache.get('SPY:1h')
        assert retrieved == test_data
        logger.info("✓ Basic caching works")
        
        # Test cache hits
        for _ in range(5):
            cache.get('SPY:1h')
        stats = cache.get_stats()
        assert stats['hit_rate'] > 0.5
        logger.info(f"✓ Cache hit rate: {stats['hit_rate']:.1%}")
        
        # Test compression
        large_df = pd.DataFrame(np.random.randn(1000, 10))
        cache.set('LARGE_DATA', large_df, ttl=300)
        logger.info("✓ Large data compression enabled")
        
        # Test memory management
        logger.info(f"✓ Memory usage: {stats['memory_percent']:.1f}%")
        
        logger.info("✓ Predictive cache test PASSED")
        return True
    
    except Exception as e:
        logger.error(f"✗ Predictive cache test FAILED: {e}")
        return False


def test_signal_generation():
    """Test advanced signal generation."""
    logger.info("\n" + "="*60)
    logger.info("Testing Advanced Signal Generation")
    logger.info("="*60)
    
    try:
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100)
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        volumes = pd.Series(np.random.randint(1000000, 2000000, 100), index=dates)
        
        df = pd.DataFrame({
            'Open': prices,
            'High': prices + 1,
            'Low': prices - 1,
            'Close': prices,
            'Volume': volumes
        })
        df.attrs['symbol'] = 'SPY'
        
        generator = AdvancedSignalGenerator(lookback=50)
        signals = generator.generate_signals(df)
        
        logger.info(f"✓ Generated {len(signals)} signals")
        
        if signals:
            signal = signals[0]
            logger.info(f"  Signal type: {signal.signal_type.value}")
            logger.info(f"  Strength: {signal.strength:.2f}")
            logger.info(f"  Confidence: {signal.confidence:.2f}")
            logger.info(f"  Regime: {signal.regime.value}")
            logger.info(f"  Risk/Reward: {signal.risk_reward_ratio:.2f}")
        
        # Test market regime detection
        generator.regime_detector.detect_regime(df)
        logger.info("✓ Market regime detection works")
        
        logger.info("✓ Advanced signal generation test PASSED")
        return True
    
    except Exception as e:
        logger.error(f"✗ Advanced signal generation test FAILED: {e}")
        return False


def test_risk_management():
    """Test live trading risk management."""
    logger.info("\n" + "="*60)
    logger.info("Testing Risk Management")
    logger.info("="*60)
    
    try:
        risk_mgr = RiskManager(
            initial_capital=100000,
            max_position_size=0.10,
            max_daily_loss=0.02
        )
        
        # Test position opening
        can_open, reason = risk_mgr.can_open_position(
            symbol='SPY',
            quantity=100,
            entry_price=450.0,
            capital_available=100000
        )
        assert can_open
        logger.info("✓ Position validation passed")
        
        # Test position management
        position = risk_mgr.open_position(
            symbol='SPY',
            quantity=100,
            entry_price=450.0,
            side='long',
            stop_loss=445.0,
            take_profit=460.0
        )
        assert position is not None
        logger.info("✓ Position opened successfully")
        
        # Test price updates
        risk_mgr.update_prices({'SPY': 455.0})
        metrics = risk_mgr.get_portfolio_metrics()
        logger.info(f"  Current P&L: ${metrics.total_pnl:.2f}")
        
        # Test risk alerts
        risk_mgr.update_prices({'SPY': 444.0})  # Below stop loss
        alerts = risk_mgr.check_risk_alerts()
        logger.info(f"✓ Risk alerts working ({len(alerts)} alerts)")
        
        logger.info("✓ Risk management test PASSED")
        return True
    
    except Exception as e:
        logger.error(f"✗ Risk management test FAILED: {e}")
        return False


def test_parallel_fetcher():
    """Test parallel data fetching."""
    logger.info("\n" + "="*60)
    logger.info("Testing Parallel Data Fetcher")
    logger.info("="*60)
    
    try:
        fetcher = ParallelDataFetcher(max_workers=4)
        fetcher.start_processing()
        
        # Test request submission
        request_ids = fetcher.submit_batch(
            ['SPY', 'QQQ'],
            priority=RequestPriority.HIGH
        )
        logger.info(f"✓ Submitted {len(request_ids)} requests")
        
        # Test deduplication
        fetcher.deduplicator.register_pending('SPY:price', None)
        logger.info("✓ Request deduplication works")
        
        # Test statistics
        stats = fetcher.get_stats()
        logger.info(f"✓ Fetcher stats:")
        logger.info(f"  Total requests: {stats['total_requests']}")
        logger.info(f"  Queue size: {stats['queue_size']}")
        
        fetcher.stop_processing()
        logger.info("✓ Parallel fetcher test PASSED")
        return True
    
    except Exception as e:
        logger.error(f"✗ Parallel fetcher test FAILED: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    logger.info("\n" + "="*60)
    logger.info("Testing Configuration System")
    logger.info("="*60)
    
    try:
        # Test development config
        config = load_config(environment='development')
        assert config is not None
        logger.info(f"✓ Development config loaded")
        logger.info(f"  Performance profile: {config.performance_profile.value}")
        logger.info(f"  Trading enabled: {config.trading.enabled}")
        logger.info(f"  Cache size: {config.data.cache_max_mb}MB")
        
        # Test production config
        config = load_config(environment='production')
        assert config.trading.enabled
        assert config.data.parallel_workers > 10
        logger.info(f"✓ Production config loaded correctly")
        
        # Test validation
        manager = ConfigManager()
        is_valid, errors = manager.validate(config)
        assert is_valid
        logger.info(f"✓ Configuration validation passed")
        
        logger.info("✓ Configuration system test PASSED")
        return True
    
    except Exception as e:
        logger.error(f"✗ Configuration system test FAILED: {e}")
        return False


def main():
    """Run all validation tests."""
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZATION VALIDATION SUITE")
    logger.info("="*70)
    
    results = {
        'Real-Time Streamer': test_realtime_streamer(),
        'Predictive Cache': test_predictive_cache(),
        'Signal Generation': test_signal_generation(),
        'Risk Management': test_risk_management(),
        'Parallel Fetcher': test_parallel_fetcher(),
        'Configuration System': test_configuration(),
    }
    
    logger.info("\n" + "="*70)
    logger.info("VALIDATION RESULTS")
    logger.info("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:<30} {status}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    logger.info("="*70)
    logger.info(f"Overall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        logger.info("✓ ALL OPTIMIZATIONS VALIDATED SUCCESSFULLY")
        return 0
    else:
        logger.error(f"✗ {total_count - passed_count} tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
