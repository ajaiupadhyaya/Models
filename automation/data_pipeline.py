"""
Automated Data Pipeline
Real-time data fetching, validation, and storage
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_fetcher import DataFetcher
from core.data_cache import DataCache
from core.pipeline.data_monitor import DataQualityMonitor, DataValidator

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Automated data pipeline with validation and quality monitoring.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data pipeline.
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir or project_root / "data" / "pipeline"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_fetcher = DataFetcher()
        self.data_cache = DataCache(cache_dir=str(self.data_dir / "cache"))
        self.quality_monitor = DataQualityMonitor()
        self.validator = DataValidator()
        
        # Setup OHLC validation
        self.validator.add_rule('High', 'high_ge_open', lambda x: x >= 0)
        self.validator.add_rule('Low', 'low_le_high', lambda x: x >= 0)
        
        logger.info(f"Data pipeline initialized: {self.data_dir}")
    
    def fetch_and_validate_stock(self, 
                                 symbol: str, 
                                 period: str = "1y",
                                 validate: bool = True) -> Dict[str, Any]:
        """
        Fetch stock data and validate quality.
        
        Args:
            symbol: Stock symbol
            period: Data period
            validate: Whether to validate data
        
        Returns:
            Dictionary with data and validation results
        """
        try:
            # Fetch data
            logger.info(f"Fetching data for {symbol}...")
            data = self.data_fetcher.get_stock_data(symbol, period=period)
            
            if data.empty:
                return {
                    'symbol': symbol,
                    'success': False,
                    'error': 'No data returned'
                }
            
            # Validate
            validation_result = None
            if validate:
                validation_result = self.validator.validate_ohlc(data)
                
                if not validation_result.get('valid', False):
                    logger.warning(f"Data validation issues for {symbol}: {validation_result.get('issues')}")
            
            # Quality metrics
            quality_metrics = self.quality_monitor.evaluate_quality(data, f"{symbol}_stock")
            
            # Save data
            data_file = self.data_dir / f"{symbol}_{datetime.now().strftime('%Y%m%d')}.parquet"
            data.to_parquet(data_file, compression='snappy')
            
            return {
                'symbol': symbol,
                'success': True,
                'rows': len(data),
                'date_range': {
                    'start': str(data.index[0]),
                    'end': str(data.index[-1])
                },
                'latest_price': float(data['Close'].iloc[-1]),
                'validation': validation_result,
                'quality_metrics': {
                    'completeness': quality_metrics.completeness,
                    'validity': quality_metrics.validity,
                    'timeliness': quality_metrics.timeliness,
                    'overall_accuracy': quality_metrics.accuracy
                },
                'data_file': str(data_file)
            }
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e)
            }
    
    def fetch_and_validate_economic(self,
                                    indicators: List[str],
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch economic data and validate.
        
        Args:
            indicators: List of FRED series IDs
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary with data and validation results
        """
        results = {}
        
        for indicator in indicators:
            try:
                logger.info(f"Fetching economic indicator: {indicator}...")
                data = self.data_fetcher.get_economic_indicator(
                    indicator, 
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data.empty:
                    results[indicator] = {
                        'success': False,
                        'error': 'No data returned'
                    }
                    continue
                
                # Quality metrics
                df = pd.DataFrame({'value': data})
                quality_metrics = self.quality_monitor.evaluate_quality(df, indicator)
                
                # Save data
                data_file = self.data_dir / f"{indicator}_{datetime.now().strftime('%Y%m%d')}.parquet"
                df.to_parquet(data_file, compression='snappy')
                
                results[indicator] = {
                    'success': True,
                    'rows': len(data),
                    'latest_value': float(data.iloc[-1]),
                    'latest_date': str(data.index[-1]),
                    'quality_metrics': {
                        'completeness': quality_metrics.completeness,
                        'validity': quality_metrics.validity,
                        'timeliness': quality_metrics.timeliness
                    },
                    'data_file': str(data_file)
                }
                
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {e}")
                results[indicator] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def batch_fetch_stocks(self, 
                          symbols: List[str],
                          period: str = "1y") -> Dict[str, Any]:
        """
        Batch fetch multiple stocks.
        
        Args:
            symbols: List of stock symbols
            period: Data period
        
        Returns:
            Dictionary with results for all symbols
        """
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.fetch_and_validate_stock(symbol, period)
        
        # Summary
        successful = sum(1 for r in results.values() if r.get('success'))
        total = len(symbols)
        
        return {
            'summary': {
                'total': total,
                'successful': successful,
                'failed': total - successful
            },
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Get comprehensive data quality report.
        
        Returns:
            Quality report dictionary
        """
        return self.quality_monitor.get_quality_report()
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Clean up old data files.
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        deleted_count = 0
        for file in self.data_dir.glob("*.parquet"):
            if file.stat().st_mtime < cutoff_date.timestamp():
                file.unlink()
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old data files")
        return deleted_count


if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline()
    
    # Fetch stock data
    result = pipeline.fetch_and_validate_stock("AAPL", period="6mo")
    print(f"AAPL data: {result}")
    
    # Fetch economic data
    econ_result = pipeline.fetch_and_validate_economic(["UNRATE", "GDP"])
    print(f"Economic data: {econ_result}")
