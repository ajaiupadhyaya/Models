"""
Data Quality Monitoring and Validation
Monitor data integrity and detect issues
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    completeness: float  # % non-null values
    validity: float      # % valid values
    consistency: float   # % consistent values
    timeliness: float    # How recent data is
    accuracy: float      # Estimated accuracy


class DataValidator:
    """
    Validates data quality and integrity.
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = logging.getLogger("DataValidator")
        self.validation_rules = {}
    
    def add_rule(self,
                column: str,
                rule_name: str,
                validation_func) -> bool:
        """
        Add a validation rule.
        
        Args:
            column: Column to validate
            rule_name: Name of rule
            validation_func: Function that returns bool
        
        Returns:
            True if added
        """
        if column not in self.validation_rules:
            self.validation_rules[column] = {}
        
        self.validation_rules[column][rule_name] = validation_func
        return True
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict:
        """
        Validate a dataframe.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns_validated': {},
            'issues': [],
            'valid': True
        }
        
        # Check for nulls
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                pct = count / len(df) * 100
                results['issues'].append(f"Column {col}: {count} null values ({pct:.1f}%)")
                results['valid'] = False
        
        # Check data types
        for col in df.columns:
            col_type = str(df[col].dtype)
            results['columns_validated'][col] = {
                'type': col_type,
                'non_null': df[col].notna().sum(),
                'null_pct': df[col].isnull().sum() / len(df) * 100
            }
        
        # Run custom rules
        for col, rules in self.validation_rules.items():
            if col in df.columns:
                for rule_name, rule_func in rules.items():
                    try:
                        valid = rule_func(df[col])
                        if not valid:
                            results['issues'].append(f"Column {col}: Rule '{rule_name}' failed")
                            results['valid'] = False
                    except Exception as e:
                        results['issues'].append(f"Column {col}: Rule '{rule_name}' error: {str(e)}")
        
        return results
    
    @staticmethod
    def validate_ohlc(df: pd.DataFrame) -> Dict:
        """
        Validate OHLC data.
        
        Args:
            df: OHLC dataframe
        
        Returns:
            Validation results
        """
        issues = []
        
        # Check required columns
        required = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")
            return {'valid': False, 'issues': issues}
        
        # Check High >= Open, High >= Close, High >= Low
        bad_high = (df['High'] < df['Open']) | (df['High'] < df['Close']) | (df['High'] < df['Low'])
        if bad_high.any():
            issues.append(f"High price violation in {bad_high.sum()} rows")
        
        # Check Low <= Open, Low <= Close, Low <= High
        bad_low = (df['Low'] > df['Open']) | (df['Low'] > df['Close']) | (df['Low'] > df['High'])
        if bad_low.any():
            issues.append(f"Low price violation in {bad_low.sum()} rows")
        
        # Check for negative prices
        if (df[required] < 0).any().any():
            issues.append("Negative prices found")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'rows_checked': len(df)
        }
    
    @staticmethod
    def validate_returns(returns: pd.Series) -> Dict:
        """
        Validate return series.
        
        Args:
            returns: Return series
        
        Returns:
            Validation results
        """
        issues = []
        
        # Check for extreme values
        if (returns > 1).any() or (returns < -1).any():
            extreme = ((returns > 1) | (returns < -1)).sum()
            issues.append(f"Extreme values found in {extreme} rows (>100% return)")
        
        # Check for NaN
        nan_count = returns.isnull().sum()
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values in returns")
        
        # Check for infinite
        if np.isinf(returns).any():
            issues.append("Infinite values found")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'mean_return': returns.mean(),
            'std_return': returns.std()
        }


class DataQualityMonitor:
    """
    Monitor data quality over time.
    """
    
    def __init__(self):
        """Initialize monitor."""
        self.logger = logging.getLogger("DataQualityMonitor")
        self.quality_history = []
        self.alerts = []
    
    def evaluate_quality(self,
                        df: pd.DataFrame,
                        dataset_name: str) -> DataQualityMetrics:
        """
        Evaluate data quality.
        
        Args:
            df: DataFrame to evaluate
            dataset_name: Name of dataset
        
        Returns:
            DataQualityMetrics
        """
        # Completeness: % non-null values
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        # Validity: % numeric values are in reasonable range
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        valid_count = 0
        total_numeric = 0
        for col in numeric_cols:
            # Check for reasonable values (not NaN, Inf, or extreme outliers)
            valid = (~df[col].isnull()) & (~np.isinf(df[col]))
            valid_count += valid.sum()
            total_numeric += len(df)
        
        validity = (valid_count / total_numeric * 100) if total_numeric > 0 else 0
        
        # Consistency: check for duplicates, gaps
        consistency = 100 - (df.duplicated().sum() / len(df) * 100)
        
        # Timeliness: check how recent data is (days old)
        if 'Date' in df.columns or isinstance(df.index, pd.DatetimeIndex):
            last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['Date'].iloc[-1]
            days_old = (datetime.now() - last_date).days
            timeliness = max(0, 100 - days_old * 10)  # Decreases with age
        else:
            timeliness = 50  # Unknown
        
        # Accuracy: based on data validation
        accuracy = (completeness + validity) / 2
        
        metrics = DataQualityMetrics(
            completeness=completeness,
            validity=validity,
            consistency=consistency,
            timeliness=timeliness,
            accuracy=accuracy
        )
        
        # Store in history
        self.quality_history.append({
            'timestamp': datetime.now(),
            'dataset': dataset_name,
            'metrics': metrics
        })
        
        # Check for issues
        if completeness < 95:
            self.alerts.append(f"{dataset_name}: Completeness {completeness:.1f}% < 95%")
        if validity < 95:
            self.alerts.append(f"{dataset_name}: Validity {validity:.1f}% < 95%")
        if timeliness < 50:
            self.alerts.append(f"{dataset_name}: Data is stale (timeliness {timeliness:.1f}%)")
        
        return metrics
    
    def get_quality_report(self,
                          dataset_name: str = None) -> Dict:
        """
        Get quality report.
        
        Args:
            dataset_name: Optional dataset name filter
        
        Returns:
            Quality report dictionary
        """
        if dataset_name:
            history = [h for h in self.quality_history if h['dataset'] == dataset_name]
        else:
            history = self.quality_history
        
        if not history:
            return {'error': 'No quality data available'}
        
        # Get latest metrics
        latest = history[-1]['metrics']
        
        # Calculate trends
        if len(history) > 1:
            previous = history[-2]['metrics']
            trends = {
                'completeness_trend': latest.completeness - previous.completeness,
                'validity_trend': latest.validity - previous.validity,
                'consistency_trend': latest.consistency - previous.consistency
            }
        else:
            trends = {}
        
        return {
            'dataset': dataset_name or 'All',
            'latest_metrics': {
                'completeness': latest.completeness,
                'validity': latest.validity,
                'consistency': latest.consistency,
                'timeliness': latest.timeliness,
                'overall_accuracy': latest.accuracy
            },
            'trends': trends,
            'alerts': self.alerts,
            'last_updated': history[-1]['timestamp']
        }
    
    def data_profile(self, df: pd.DataFrame) -> Dict:
        """
        Generate data profile.
        
        Args:
            df: DataFrame to profile
        
        Returns:
            Data profile dictionary
        """
        profile = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'columns': {},
            'missing_values': {},
            'duplicates': df.duplicated().sum()
        }
        
        for col in df.columns:
            profile['columns'][col] = {
                'dtype': str(df[col].dtype),
                'non_null': df[col].notna().sum(),
                'null_pct': df[col].isnull().sum() / len(df) * 100
            }
            
            if df[col].dtype in ['float64', 'int64']:
                profile['columns'][col].update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                })
            
            profile['missing_values'][col] = df[col].isnull().sum()
        
        return profile
