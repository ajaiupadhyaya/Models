"""
Automation Orchestrator
Central coordinator for all automated processes
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.pipeline.data_scheduler import DataScheduler, UpdateFrequency
from core.pipeline.data_monitor import DataQualityMonitor
from core.data_fetcher import DataFetcher
from core.backtesting import BacktestEngine
from core.paper_trading import PaperTradingEngine

logger = logging.getLogger(__name__)


class ProcessStatus(Enum):
    """Process status enumeration."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class ProcessMetrics:
    """Process execution metrics."""
    process_id: str
    name: str
    status: ProcessStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


class AutomationOrchestrator:
    """
    Central orchestrator for all automated processes.
    Manages data pipelines, ML training, trading, and monitoring.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize orchestrator.
        
        Args:
            config_path: Path to configuration file
        
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist and can't be created
        """
        self.config_path = config_path or project_root / "automation" / "config.json"
        self.config = self._load_config()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components with error handling
        try:
            max_workers = self.config.get('max_workers', 4)
            if not isinstance(max_workers, int) or max_workers < 1:
                logger.warning(f"Invalid max_workers {max_workers}, using default 4")
                max_workers = 4
            
            self.data_scheduler = DataScheduler(max_workers=max_workers)
            self.data_monitor = DataQualityMonitor()
            self.data_fetcher = DataFetcher()
        except Exception as e:
            logger.error(f"Error initializing orchestrator components: {e}")
            raise
        
        # Process tracking
        self.processes: Dict[str, ProcessMetrics] = {}
        self.is_running = False
        
        # Setup logging
        self._setup_logging()
    
    def _validate_config(self):
        """Validate configuration file structure and values."""
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Validate required keys
        required_keys = []
        for key in required_keys:
            if key not in self.config:
                logger.warning(f"Missing optional config key: {key}")
        
        # Validate max_workers
        if 'max_workers' in self.config:
            max_workers = self.config['max_workers']
            if not isinstance(max_workers, int) or max_workers < 1 or max_workers > 32:
                raise ValueError(f"max_workers must be between 1 and 32, got {max_workers}")
        
        logger.info("Automation Orchestrator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        # Default configuration
        default_config = {
            "max_workers": 4,
            "data_update_frequency": "daily",
            "ml_retrain_frequency": "weekly",
            "trading_enabled": False,
            "monitoring_enabled": True,
            "alert_thresholds": {
                "data_quality": 0.95,
                "model_accuracy": 0.70,
                "portfolio_drawdown": 0.10
            },
            "symbols": ["SPY", "QQQ", "DIA"],
            "economic_indicators": ["UNRATE", "GDP", "CPIAUCSL", "FEDFUNDS"]
        }
        
        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def register_process(self, process_id: str, name: str) -> ProcessMetrics:
        """
        Register a new process.
        
        Args:
            process_id: Unique process identifier
            name: Process name
        
        Returns:
            ProcessMetrics instance
        """
        metrics = ProcessMetrics(
            process_id=process_id,
            name=name,
            status=ProcessStatus.IDLE
        )
        self.processes[process_id] = metrics
        logger.info(f"Registered process: {name} ({process_id})")
        return metrics
    
    def start_data_pipeline(self):
        """Start automated data pipeline."""
        process_id = "data_pipeline"
        self.register_process(process_id, "Data Pipeline")
        
        # Create data update jobs
        symbols = self.config.get('symbols', ['SPY'])
        
        from core.pipeline.data_scheduler import UpdateJobBuilder
        
        # Stock data updates
        for symbol in symbols:
            job = UpdateJobBuilder.stock_data_update(
                tickers=[symbol],
                frequency=UpdateFrequency.DAILY
            )
            self.data_scheduler.add_job(job)
        
        # Economic data updates
        indicators = self.config.get('economic_indicators', [])
        if indicators:
            job = UpdateJobBuilder.economic_data_update(
                indicators=indicators,
                frequency=UpdateFrequency.WEEKLY
            )
            self.data_scheduler.add_job(job)
        
        self.data_scheduler.start()
        self.processes[process_id].status = ProcessStatus.RUNNING
        logger.info("Data pipeline started")
    
    def start_ml_training_pipeline(self):
        """Start automated ML model training."""
        process_id = "ml_training"
        self.register_process(process_id, "ML Training Pipeline")
        
        def train_models():
            """Train all ML models."""
            try:
                from models.ml.advanced_trading import LSTMPredictor, EnsemblePredictor
                from models.ml.forecasting import TimeSeriesForecaster
                
                symbols = self.config.get('symbols', ['SPY'])
                results = {}
                
                for symbol in symbols:
                    # Fetch data
                    data = self.data_fetcher.get_stock_data(symbol, period='2y')
                    
                    if len(data) < 100:
                        logger.warning(f"Insufficient data for {symbol}")
                        continue
                    
                    # Train LSTM
                    try:
                        lstm = LSTMPredictor(lookback_window=20, hidden_units=64)
                        lstm.train(data, epochs=10, batch_size=32)
                        results[f"{symbol}_lstm"] = "trained"
                    except Exception as e:
                        logger.error(f"LSTM training failed for {symbol}: {e}")
                    
                    # Train Ensemble
                    try:
                        ensemble = EnsemblePredictor()
                        ensemble.train(data)
                        results[f"{symbol}_ensemble"] = "trained"
                    except Exception as e:
                        logger.error(f"Ensemble training failed for {symbol}: {e}")
                
                return results
                
            except Exception as e:
                logger.error(f"ML training pipeline error: {e}")
                return {'error': str(e)}
        
        # Schedule ML training
        frequency = self.config.get('ml_retrain_frequency', 'weekly')
        freq_map = {
            'daily': UpdateFrequency.DAILY,
            'weekly': UpdateFrequency.WEEKLY,
            'monthly': UpdateFrequency.MONTHLY
        }
        
        job = DataScheduler.UpdateJob(
            job_id=process_id,
            name="ML Model Training",
            function=train_models,
            frequency=freq_map.get(frequency, UpdateFrequency.WEEKLY),
            time_of_day="02:00"  # Train during off-hours
        )
        
        self.data_scheduler.add_job(job)
        self.processes[process_id].status = ProcessStatus.RUNNING
        logger.info("ML training pipeline started")
    
    def start_monitoring(self):
        """Start automated monitoring and alerting."""
        process_id = "monitoring"
        self.register_process(process_id, "Monitoring & Alerts")
        
        def check_system_health():
            """Check system health and generate alerts."""
            alerts = []
            
            # Check data quality
            scheduler_status = self.data_scheduler.get_status()
            for job_status in scheduler_status.get('jobs', []):
                if job_status.get('error_count', 0) > 5:
                    alerts.append(f"Job {job_status.get('name')} has {job_status.get('error_count')} errors")
            
            # Check process health
            for process_id, metrics in self.processes.items():
                if metrics.status == ProcessStatus.FAILED:
                    alerts.append(f"Process {metrics.name} has failed")
            
            if alerts:
                logger.warning(f"System alerts: {alerts}")
                # Here you could send notifications (email, Slack, etc.)
            
            return {'alerts': alerts, 'timestamp': datetime.now()}
        
        # Schedule health checks
        job = DataScheduler.UpdateJob(
            job_id=process_id,
            name="System Health Check",
            function=check_system_health,
            frequency=UpdateFrequency.INTRADAY,
            time_of_day=None
        )
        
        self.data_scheduler.add_job(job)
        self.processes[process_id].status = ProcessStatus.RUNNING
        logger.info("Monitoring started")
    
    def start_trading_automation(self):
        """Start automated trading (if enabled)."""
        if not self.config.get('trading_enabled', False):
            logger.info("Trading automation disabled in config")
            return
        
        process_id = "trading"
        self.register_process(process_id, "Trading Automation")
        
        # Trading automation would be implemented here
        # This is a placeholder for the actual trading logic
        logger.info("Trading automation started (placeholder)")
        self.processes[process_id].status = ProcessStatus.RUNNING
    
    def start_all(self):
        """Start all automated processes."""
        logger.info("Starting all automation processes...")
        
        self.is_running = True
        
        # Start all pipelines
        self.start_data_pipeline()
        self.start_ml_training_pipeline()
        self.start_monitoring()
        
        if self.config.get('trading_enabled', False):
            self.start_trading_automation()
        
        logger.info("All automation processes started")
    
    def stop_all(self):
        """Stop all automated processes."""
        logger.info("Stopping all automation processes...")
        
        self.data_scheduler.stop()
        self.is_running = False
        
        for process_id, metrics in self.processes.items():
            metrics.status = ProcessStatus.PAUSED
        
        logger.info("All automation processes stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get orchestrator status.
        
        Returns:
            Status dictionary
        """
        return {
            'is_running': self.is_running,
            'processes': {pid: asdict(metrics) for pid, metrics in self.processes.items()},
            'scheduler_status': self.data_scheduler.get_status(),
            'config': self.config
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration.
        
        Args:
            updates: Configuration updates
        """
        self.config.update(updates)
        
        # Save to file
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Configuration updated: {updates}")


if __name__ == "__main__":
    # Example usage
    orchestrator = AutomationOrchestrator()
    orchestrator.start_all()
    
    try:
        # Keep running
        import time
        while True:
            status = orchestrator.get_status()
            print(f"\nStatus: {status['is_running']}")
            print(f"Processes: {len(status['processes'])}")
            time.sleep(60)
    except KeyboardInterrupt:
        orchestrator.stop_all()
