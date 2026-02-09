"""
Multi-Factor Model Framework for systematic factor analysis.
Phase 2 - Awesome Quant Integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MultiFactorModel:
    """
    Multi-factor model for asset returns decomposition.
    Supports Fama-French factors and custom factor models.
    """
    
    def __init__(self, returns: pd.Series, factors: pd.DataFrame):
        """
        Initialize multi-factor model.
        
        Args:
            returns: Asset returns (single column)
            factors: Factor returns (columns = factors like SMB, HML, MOM, etc.)
        """
        self.returns = returns
        self.factors = factors
        self.model = None
        self.results = None
    
    def fit(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Fit multi-factor model using OLS regression.
        
        Returns:
            Statsmodels regression results
        """
        # Align indices
        common_idx = self.returns.index.intersection(self.factors.index)
        y = self.returns.loc[common_idx]
        X = self.factors.loc[common_idx]
        
        # Add constant for alpha (intercept)
        X = sm.add_constant(X)
        
        # Fit OLS
        self.model = sm.OLS(y, X)
        self.results = self.model.fit()
        
        return self.results
    
    def get_alpha(self) -> Tuple[float, float]:
        """
        Get alpha (excess return) and its statistical significance.
        
        Returns:
            Tuple of (alpha, p_value)
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        alpha = float(self.results.params['const'])
        p_value = float(self.results.pvalues['const'])
        
        return alpha, p_value
    
    def get_factor_exposures(self) -> Dict[str, float]:
        """
        Get factor loadings (betas).
        
        Returns:
            Dictionary mapping factor names to beta coefficients
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Exclude constant
        exposures = {
            name: float(coef)
            for name, coef in self.results.params.items()
            if name != 'const'
        }
        
        return exposures
    
    def factor_attribution(self) -> Dict[str, float]:
        """
        Attribute returns to each factor.
        
        Returns:
            Dictionary mapping factors to their contribution to returns
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        exposures = self.get_factor_exposures()
        attribution = {}
        
        for factor_name, beta in exposures.items():
            factor_mean_return = float(self.factors[factor_name].mean())
            attribution[factor_name] = beta * factor_mean_return
        
        alpha, _ = self.get_alpha()
        attribution['alpha'] = alpha
        
        return attribution
    
    def residual_analysis(self) -> Dict[str, float]:
        """
        Analyze residuals for model diagnostics.
        
        Returns:
            Dictionary of residual statistics
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        residuals = self.results.resid
        
        # Durbin-Watson statistic for autocorrelation
        dw = float(sm.stats.durbin_watson(residuals))
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue, _, _ = sm.stats.jarque_bera(residuals)
        
        return {
            'residual_std': float(residuals.std()),
            'residual_mean': float(residuals.mean()),
            'residual_skew': float(residuals.skew()),
            'residual_kurtosis': float(residuals.kurtosis()),
            'durbin_watson': dw,
            'jarque_bera_stat': float(jb_stat),
            'jarque_bera_pvalue': float(jb_pvalue),
            'r_squared': float(self.results.rsquared),
            'adj_r_squared': float(self.results.rsquared_adj)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary with all model statistics
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        alpha, alpha_pval = self.get_alpha()
        exposures = self.get_factor_exposures()
        attribution = self.factor_attribution()
        residuals = self.residual_analysis()
        
        return {
            'alpha': alpha,
            'alpha_pvalue': alpha_pval,
            'alpha_significant': alpha_pval < 0.05,
            'factor_exposures': exposures,
            'factor_attribution': attribution,
            'residual_analysis': residuals,
            'r_squared': residuals['r_squared'],
            'adj_r_squared': residuals['adj_r_squared']
        }


class FactorConstructor:
    """
    Construct common factor portfolios (SMB, HML, MOM, etc.).
    """
    
    @staticmethod
    def construct_size_factor(
        returns: pd.DataFrame,
        market_caps: pd.DataFrame,
        percentile: float = 0.5
    ) -> pd.Series:
        """
        Construct Size factor (SMB - Small Minus Big).
        
        Args:
            returns: Asset returns (columns = assets)
            market_caps: Market capitalization (columns = assets)
            percentile: Cutoff percentile for small/big split
        
        Returns:
            SMB factor returns
        """
        # Align data
        common_idx = returns.index.intersection(market_caps.index)
        common_cols = returns.columns.intersection(market_caps.columns)
        
        returns_aligned = returns.loc[common_idx, common_cols]
        caps_aligned = market_caps.loc[common_idx, common_cols]
        
        # Calculate median market cap each period
        smb_returns = []
        
        for date in common_idx:
            caps_date = caps_aligned.loc[date]
            returns_date = returns_aligned.loc[date]
            
            # Split by size
            median_cap = caps_date.quantile(percentile)
            small_mask = caps_date <= median_cap
            big_mask = caps_date > median_cap
            
            # Equal-weighted portfolios
            small_return = returns_date[small_mask].mean()
            big_return = returns_date[big_mask].mean()
            
            smb_returns.append(small_return - big_return)
        
        return pd.Series(smb_returns, index=common_idx, name='SMB')
    
    @staticmethod
    def construct_value_factor(
        returns: pd.DataFrame,
        book_to_market: pd.DataFrame,
        percentile: float = 0.5
    ) -> pd.Series:
        """
        Construct Value factor (HML - High Minus Low book-to-market).
        
        Args:
            returns: Asset returns
            book_to_market: Book-to-market ratios
            percentile: Cutoff percentile for high/low split
        
        Returns:
            HML factor returns
        """
        common_idx = returns.index.intersection(book_to_market.index)
        common_cols = returns.columns.intersection(book_to_market.columns)
        
        returns_aligned = returns.loc[common_idx, common_cols]
        btm_aligned = book_to_market.loc[common_idx, common_cols]
        
        hml_returns = []
        
        for date in common_idx:
            btm_date = btm_aligned.loc[date]
            returns_date = returns_aligned.loc[date]
            
            # Split by book-to-market
            median_btm = btm_date.quantile(percentile)
            high_mask = btm_date >= median_btm
            low_mask = btm_date < median_btm
            
            # Equal-weighted portfolios
            value_return = returns_date[high_mask].mean()
            growth_return = returns_date[low_mask].mean()
            
            hml_returns.append(value_return - growth_return)
        
        return pd.Series(hml_returns, index=common_idx, name='HML')
    
    @staticmethod
    def construct_momentum_factor(
        returns: pd.DataFrame,
        lookback: int = 252,
        skip: int = 21
    ) -> pd.Series:
        """
        Construct Momentum factor (WML - Winners Minus Losers).
        
        Args:
            returns: Asset returns
            lookback: Lookback period for momentum calculation (252 = 1 year)
            skip: Skip recent period to avoid reversal (21 = 1 month)
        
        Returns:
            MOM factor returns
        """
        # Calculate cumulative returns over lookback period (excluding skip)
        cum_returns = returns.rolling(window=lookback).apply(
            lambda x: (1 + x[:-skip]).prod() - 1, raw=False
        )
        
        mom_returns = []
        
        for date in returns.index:
            if date not in cum_returns.index:
                continue
            
            mom_scores = cum_returns.loc[date]
            returns_date = returns.loc[date]
            
            # Split by momentum (top 30% vs bottom 30%)
            if mom_scores.notna().sum() < 10:
                mom_returns.append(0)
                continue
            
            top_30 = mom_scores.quantile(0.70)
            bottom_30 = mom_scores.quantile(0.30)
            
            winners = returns_date[mom_scores >= top_30].mean()
            losers = returns_date[mom_scores <= bottom_30].mean()
            
            mom_returns.append(winners - losers)
        
        return pd.Series(mom_returns, index=returns.index, name='MOM')
