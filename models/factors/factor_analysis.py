"""
Factor Analysis wrapper using alphalens for systematic alpha research.
Phase 2 - Awesome Quant Integration
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    import alphalens as al
    ALPHALENS_AVAILABLE = True
except ImportError:
    ALPHALENS_AVAILABLE = False
    warnings.warn("alphalens not available. Install with: pip install alphalens-reloaded")


class FactorAnalysis:
    """
    Wrapper for alphalens factor analysis.
    Provides standardized interface for systematic alpha research.
    """
    
    def __init__(
        self,
        factor_data: pd.Series,
        prices: pd.DataFrame,
        quantiles: int = 5,
        periods: List[int] = [1, 5, 10]
    ):
        """
        Initialize factor analysis.
        
        Args:
            factor_data: Factor values (MultiIndex with date and asset)
            prices: Price data (columns = assets, index = dates)
            quantiles: Number of quantile buckets for analysis
            periods: Forward return periods to analyze
        """
        if not ALPHALENS_AVAILABLE:
            raise RuntimeError("alphalens-reloaded required. Install: pip install alphalens-reloaded")
        
        self.factor_data = factor_data
        self.prices = prices
        self.quantiles = quantiles
        self.periods = periods
        self.merged_data = None
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for alphalens analysis.
        
        Returns:
            Merged DataFrame with factor values and forward returns
        """
        # Get forward returns
        self.merged_data = al.utils.get_clean_factor_and_forward_returns(
            factor=self.factor_data,
            prices=self.prices,
            quantiles=self.quantiles,
            periods=self.periods,
            max_loss=0.35  # Allow up to 35% data loss
        )
        
        return self.merged_data
    
    def information_coefficient(self) -> Dict[str, Any]:
        """
        Calculate Information Coefficient (IC).
        
        IC measures correlation between factor values and forward returns.
        High absolute IC indicates predictive power.
        
        Returns:
            Dictionary with IC statistics
        """
        if self.merged_data is None:
            self.prepare_data()
        
        ic = al.performance.factor_information_coefficient(self.merged_data)
        ic_summary = al.performance.factor_information_coefficient(self.merged_data).describe()
        
        return {
            'ic_mean': {f'{p}D': float(ic_summary[f'{p}D']['mean']) for p in self.periods},
            'ic_std': {f'{p}D': float(ic_summary[f'{p}D']['std']) for p in self.periods},
            'ic_ir': {
                f'{p}D': float(ic_summary[f'{p}D']['mean'] / ic_summary[f'{p}D']['std'])
                for p in self.periods
            },
            'ic_series': ic.to_dict()
        }
    
    def quantile_returns(self) -> Dict[str, Any]:
        """
        Calculate mean returns by factor quantile.
        
        Returns:
            Dictionary with quantile return statistics
        """
        if self.merged_data is None:
            self.prepare_data()
        
        quantile_rets = al.performance.mean_return_by_quantile(
            self.merged_data,
            by_date=False,
            by_group=False
        )
        
        results = {}
        for period in self.periods:
            period_key = f'{period}D'
            if period_key in quantile_rets[0].columns:
                quantile_data = quantile_rets[0][period_key].to_dict()
                results[period_key] = {
                    'by_quantile': quantile_data,
                    'spread': float(quantile_data[self.quantiles] - quantile_data[1])
                }
        
        return results
    
    def turnover_analysis(self) -> Dict[str, float]:
        """
        Calculate factor turnover (stability).
        
        Low turnover indicates stable factor rankings.
        High turnover may increase transaction costs.
        
        Returns:
            Dictionary with turnover statistics
        """
        if self.merged_data is None:
            self.prepare_data()
        
        turnover = al.performance.factor_autocorrelation(self.merged_data)
        
        return {
            'autocorr_1d': float(turnover.iloc[0]) if len(turnover) > 0 else 0.0,
            'autocorr_5d': float(turnover.iloc[4]) if len(turnover) > 4 else 0.0,
            'mean_autocorr': float(turnover.mean())
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive factor analysis summary.
        
        Returns:
            Dictionary with all factor statistics
        """
        if self.merged_data is None:
            self.prepare_data()
        
        ic_stats = self.information_coefficient()
        quantile_rets = self.quantile_returns()
        turnover = self.turnover_analysis()
        
        return {
            'information_coefficient': ic_stats,
            'quantile_returns': quantile_rets,
            'turnover': turnover,
            'num_observations': len(self.merged_data),
            'num_assets': len(self.merged_data.index.get_level_values(1).unique()),
            'date_range': {
                'start': str(self.merged_data.index.get_level_values(0).min()),
                'end': str(self.merged_data.index.get_level_values(0).max())
            }
        }


class SimpleFactorAnalysis:
    """
    Simplified factor analysis without alphalens dependency.
    Provides basic IC and quantile analysis.
    """
    
    @staticmethod
    def calculate_ic(
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        method: str = 'spearman'
    ) -> pd.Series:
        """
        Calculate Information Coefficient.
        
        Args:
            factor_values: Factor values (rows = dates, columns = assets)
            forward_returns: Forward returns (same structure)
            method: 'spearman' or 'pearson' correlation
        
        Returns:
            Series of IC values per date
        """
        ic_values = []
        
        for date in factor_values.index:
            if date not in forward_returns.index:
                continue
            
            factors = factor_values.loc[date].dropna()
            returns = forward_returns.loc[date].dropna()
            
            # Get common assets
            common = factors.index.intersection(returns.index)
            if len(common) < 10:
                ic_values.append(np.nan)
                continue
            
            # Calculate correlation
            if method == 'spearman':
                from scipy.stats import spearmanr
                corr, _ = spearmanr(factors[common], returns[common])
            else:
                corr = np.corrcoef(factors[common], returns[common])[0, 1]
            
            ic_values.append(corr)
        
        return pd.Series(ic_values, index=factor_values.index, name='IC')
    
    @staticmethod
    def quantile_analysis(
        factor_values: pd.Series,
        returns: pd.Series,
        n_quantiles: int = 5
    ) -> Dict[int, float]:
        """
        Analyze returns by factor quantile.
        
        Args:
            factor_values: Factor values
            returns: Corresponding returns
            n_quantiles: Number of quantile buckets
        
        Returns:
            Dictionary mapping quantile number to mean return
        """
        # Align data
        common_idx = factor_values.index.intersection(returns.index)
        factors = factor_values.loc[common_idx]
        rets = returns.loc[common_idx]
        
        # Remove NaNs
        valid = factors.notna() & rets.notna()
        factors = factors[valid]
        rets = rets[valid]
        
        if len(factors) == 0:
            return {}
        
        # Assign quantiles
        quantiles = pd.qcut(factors, q=n_quantiles, labels=False, duplicates='drop') + 1
        
        # Calculate mean return per quantile
        quantile_returns = {}
        for q in range(1, n_quantiles + 1):
            mask = quantiles == q
            if mask.sum() > 0:
                quantile_returns[q] = float(rets[mask].mean())
        
        return quantile_returns
