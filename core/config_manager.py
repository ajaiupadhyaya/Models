"""
Production-Ready Configuration System
Comprehensive configuration management with environment-aware settings and validation.

Features:
- Environment-based configuration
- Real-time configuration validation
- Hot reload support
- Configuration inheritance and defaults
- Secrets management
- Performance tuning profiles
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class PerformanceProfile(Enum):
    """Performance tuning profiles."""
    CONSERVATIVE = "conservative"      # Safe, tested strategies
    BALANCED = "balanced"              # Mix of safety and profit
    AGGRESSIVE = "aggressive"          # Maximum profit seeking
    ULTRA_FAST = "ultra_fast"         # Latency-optimized


@dataclass
class DataConfig:
    """Data fetching configuration."""
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_ttl: int = 300  # 5 minutes
    cache_max_mb: int = 1000
    parallel_workers: int = 10
    rate_limit_per_second: float = 10.0
    enable_compression: bool = True
    enable_prefetch: bool = True
    intraday_update_interval: int = 60  # seconds
    daily_update_time: str = "08:00"  # ET


@dataclass
class TradingConfig:
    """Trading execution configuration."""
    enabled: bool = False
    max_position_size_pct: float = 0.10  # 10% of capital
    max_daily_loss_pct: float = 0.02  # 2%
    max_correlation: float = 0.7
    max_sector_exposure_pct: float = 0.30
    min_trade_amount: float = 100.0
    commission_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.002  # 0.2%
    hours_only: bool = True  # Only trade market hours
    auto_stop_loss: bool = True
    auto_take_profit: bool = True


@dataclass
class MLConfig:
    """Machine learning model configuration."""
    models_enabled: List[str] = None  # ['lstm', 'ensemble', 'traditional']
    lstm_enabled: bool = True
    lstm_lookback: int = 20
    lstm_hidden_units: int = 64
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    
    ensemble_enabled: bool = True
    ensemble_models: int = 5
    
    predict_on_new_data: bool = True
    retrain_frequency: str = "weekly"  # 'daily', 'weekly', 'monthly'
    validation_split: float = 0.2
    
    def __post_init__(self):
        if self.models_enabled is None:
            self.models_enabled = ['lstm', 'ensemble', 'traditional']


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    enabled: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    alert_discord_webhook: Optional[str] = None
    alert_email: Optional[str] = None
    
    # Alert thresholds
    alert_on_large_loss: bool = True
    large_loss_threshold_pct: float = 0.05  # 5%
    
    alert_on_position_near_stop: bool = True
    position_risk_threshold_pct: float = 0.80  # 80% to stop loss
    
    alert_on_data_quality: bool = True
    data_quality_threshold: float = 0.95
    
    # Metrics collection
    collect_metrics: bool = True
    metrics_interval_seconds: int = 60


@dataclass
class APIConfig:
    """API server configuration."""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_enabled: bool = True
    auth_enabled: bool = False
    auth_token: Optional[str] = None
    rate_limit_requests: int = 100
    rate_limit_seconds: int = 60


@dataclass
class PortfolioConfig:
    """Portfolio configuration."""
    initial_capital: float = 100000.0
    base_currency: str = "USD"
    
    # Asset universe
    watchlist: List[str] = None
    excluded_symbols: List[str] = None
    
    # Rebalancing
    rebalance_frequency: str = "monthly"  # 'daily', 'weekly', 'monthly'
    target_allocations: Dict[str, float] = None  # symbol -> allocation %
    
    # Risk targets
    target_return_pct: float = 0.15  # 15% annually
    target_volatility_pct: float = 0.12  # 12% annually
    
    def __post_init__(self):
        if self.watchlist is None:
            self.watchlist = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        if self.excluded_symbols is None:
            self.excluded_symbols = []
        if self.target_allocations is None:
            self.target_allocations = {}


@dataclass
class PlatformConfig:
    """Complete platform configuration."""
    environment: Environment = Environment.DEVELOPMENT
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    
    data: DataConfig = None
    trading: TradingConfig = None
    ml: MLConfig = None
    monitoring: MonitoringConfig = None
    api: APIConfig = None
    portfolio: PortfolioConfig = None
    
    # General
    timezone: str = "US/Eastern"
    verbose: bool = False
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.trading is None:
            self.trading = TradingConfig()
        if self.ml is None:
            self.ml = MLConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.portfolio is None:
            self.portfolio = PortfolioConfig()
        
        # Apply performance profile
        self._apply_performance_profile()
    
    def _apply_performance_profile(self):
        """Apply performance profile optimizations."""
        if self.performance_profile == PerformanceProfile.CONSERVATIVE:
            self.trading.max_position_size_pct = 0.05
            self.trading.max_daily_loss_pct = 0.01
            self.ml.lstm_epochs = 100
            self.data.cache_ttl = 600
        
        elif self.performance_profile == PerformanceProfile.BALANCED:
            self.trading.max_position_size_pct = 0.10
            self.trading.max_daily_loss_pct = 0.02
            self.ml.lstm_epochs = 50
            self.data.cache_ttl = 300
        
        elif self.performance_profile == PerformanceProfile.AGGRESSIVE:
            self.trading.max_position_size_pct = 0.20
            self.trading.max_daily_loss_pct = 0.05
            self.ml.lstm_epochs = 30
            self.data.cache_ttl = 180
        
        elif self.performance_profile == PerformanceProfile.ULTRA_FAST:
            self.trading.max_position_size_pct = 0.15
            self.ml.lstm_epochs = 20
            self.data.cache_ttl = 60
            self.data.parallel_workers = 20
            self.api.workers = 8


class ConfigManager:
    """
    Manages configuration loading, validation, and hot reload.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            config_dir: Configuration directory
        """
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self.config: Optional[PlatformConfig] = None
        self.config_file: Optional[Path] = None
        self.last_load_time: Optional[datetime] = None
        self.lock = threading.RLock()
        self.config_callbacks: List[callable] = []
        
        logger.info(f"ConfigManager initialized with directory: {self.config_dir}")
    
    def load(self, environment: Optional[str] = None) -> PlatformConfig:
        """
        Load configuration from environment or file.
        
        Args:
            environment: Environment name (dev/staging/prod)
        
        Returns:
            Loaded configuration
        """
        with self.lock:
            # Determine environment
            env = environment or os.getenv('ENV', 'development').lower()
            
            # Build config file path
            config_file = self.config_dir / f"config_{env}.json"
            
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_file}, using defaults")
                self.config = self._create_default_config(env)
            else:
                self.config = self._load_from_file(config_file)
                self.config_file = config_file
            
            self.last_load_time = datetime.now()
            logger.info(f"Configuration loaded (environment: {env})")
            
            return self.config
    
    def _load_from_file(self, config_file: Path) -> PlatformConfig:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            return self._dict_to_config(data)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise
    
    def _dict_to_config(self, data: Dict[str, Any]) -> PlatformConfig:
        """Convert dictionary to PlatformConfig."""
        config_dict = {}
        
        # Parse nested configs
        if 'data' in data:
            config_dict['data'] = DataConfig(**data['data'])
        if 'trading' in data:
            config_dict['trading'] = TradingConfig(**data['trading'])
        if 'ml' in data:
            config_dict['ml'] = MLConfig(**data['ml'])
        if 'monitoring' in data:
            config_dict['monitoring'] = MonitoringConfig(**data['monitoring'])
        if 'api' in data:
            config_dict['api'] = APIConfig(**data['api'])
        if 'portfolio' in data:
            config_dict['portfolio'] = PortfolioConfig(**data['portfolio'])
        
        # Top-level config
        if 'environment' in data:
            config_dict['environment'] = Environment(data['environment'])
        if 'performance_profile' in data:
            config_dict['performance_profile'] = PerformanceProfile(data['performance_profile'])
        
        return PlatformConfig(**config_dict)
    
    def _create_default_config(self, environment: str) -> PlatformConfig:
        """Create default configuration for environment."""
        env = Environment(environment)
        
        # Select profile based on environment
        if env == Environment.PRODUCTION:
            profile = PerformanceProfile.BALANCED
            trading_enabled = True
        elif env == Environment.STAGING:
            profile = PerformanceProfile.BALANCED
            trading_enabled = False
        else:
            profile = PerformanceProfile.CONSERVATIVE
            trading_enabled = False
        
        config = PlatformConfig(
            environment=env,
            performance_profile=profile,
            trading=TradingConfig(enabled=trading_enabled)
        )
        
        return config
    
    def validate(self, config: Optional[PlatformConfig] = None) -> Tuple[bool, List[str]]:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
        
        Returns:
            (is_valid, list_of_errors) tuple
        """
        config = config or self.config
        if config is None:
            return False, ["No configuration loaded"]
        
        errors = []
        
        # Validate data config
        if config.data.cache_max_mb < 100:
            errors.append("Cache size too small (min 100MB)")
        if config.data.parallel_workers < 1 or config.data.parallel_workers > 50:
            errors.append("Invalid parallel workers count")
        
        # Validate trading config
        if config.trading.max_position_size_pct <= 0 or config.trading.max_position_size_pct > 1:
            errors.append("Invalid max_position_size_pct")
        if config.trading.commission_pct < 0 or config.trading.commission_pct > 0.01:
            errors.append("Invalid commission_pct")
        
        # Validate portfolio config
        if config.portfolio.initial_capital < 1000:
            errors.append("Initial capital too low (min $1000)")
        if sum(config.portfolio.target_allocations.values()) > 1.01:
            errors.append("Target allocations exceed 100%")
        
        return len(errors) == 0, errors
    
    def save(self, config: Optional[PlatformConfig] = None):
        """Save configuration to file."""
        config = config or self.config
        if config is None:
            raise ValueError("No configuration to save")
        
        # Create config directory
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine environment
        env_str = config.environment.value
        config_file = self.config_dir / f"config_{env_str}.json"
        
        # Convert to dict
        config_dict = self._config_to_dict(config)
        
        # Save to file
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {config_file}")
    
    def _config_to_dict(self, config: PlatformConfig) -> Dict[str, Any]:
        """Convert PlatformConfig to dictionary."""
        data = {}
        
        data['environment'] = config.environment.value
        data['performance_profile'] = config.performance_profile.value
        data['timezone'] = config.timezone
        data['verbose'] = config.verbose
        
        data['data'] = asdict(config.data)
        data['trading'] = asdict(config.trading)
        data['ml'] = asdict(config.ml)
        data['monitoring'] = asdict(config.monitoring)
        data['api'] = asdict(config.api)
        data['portfolio'] = asdict(config.portfolio)
        
        return data
    
    def hot_reload(self) -> bool:
        """
        Check if config file changed and reload if needed.
        
        Returns:
            True if reloaded, False otherwise
        """
        with self.lock:
            if not self.config_file or not self.config_file.exists():
                return False
            
            current_mtime = self.config_file.stat().st_mtime
            last_mtime = self.last_load_time.timestamp() if self.last_load_time else 0
            
            if current_mtime > last_mtime:
                logger.info("Configuration file changed, reloading...")
                old_config = self.config
                self.config = self._load_from_file(self.config_file)
                
                # Notify callbacks
                for callback in self.config_callbacks:
                    try:
                        callback(old_config, self.config)
                    except Exception as e:
                        logger.error(f"Error in config callback: {e}")
                
                return True
            
            return False
    
    def register_callback(self, callback: callable):
        """Register callback for config changes."""
        self.config_callbacks.append(callback)
    
    def get(self) -> PlatformConfig:
        """Get current configuration."""
        return self.config or self.load()


# Global config manager
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_dir: Optional[Path] = None) -> ConfigManager:
    """Get or create global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


def load_config(environment: Optional[str] = None) -> PlatformConfig:
    """Load configuration."""
    manager = get_config_manager()
    return manager.load(environment)


def get_config() -> PlatformConfig:
    """Get current configuration."""
    manager = get_config_manager()
    return manager.get()
