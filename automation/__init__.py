"""
Comprehensive Automation Framework
Production-ready automation for trading, analysis, and model management
"""

from .orchestrator import AutomationOrchestrator
from .ml_pipeline import MLPipeline
from .trading_automation import TradingAutomation
from .data_pipeline import DataPipeline

__all__ = [
    'AutomationOrchestrator',
    'MLPipeline',
    'TradingAutomation',
    'DataPipeline'
]
