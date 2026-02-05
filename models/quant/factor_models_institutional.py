"""
Institutional-Grade Factor Models
Fama-French, APT, Style Factors, Risk Factor Models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.diagnostic import het_white
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from models.quant.institutional_grade import FamaFrenchFactorModel


class APTModel:
    """
    Arbitrage Pricing Theory (APT) Model.
    Multi-factor model without requiring market portfolio.
    """
    
    def __init__(self, n_factors: int = 5):
        """
        Initialize APT model.
        
        Args:
            n_factors: Number of factors
        """
        self.n_factors = n_factors
        self.factor_loadings = None
        self.factor_returns = None
        self.residuals = None
    
    def fit(self, returns: pd.Series, factor_returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit APT model using OLS.
        
        Args:
            returns: Asset returns
            factor_returns: Factor returns DataFrame
        
        Returns:
            Model fit statistics
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required for APT")
        
        # Align data
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()
        y = aligned[returns.name]
        X = aligned[factor_returns.columns[:self.n_factors]]
        
        # OLS regression
        model = OLS(y, X).fit()
        
        self.factor_loadings = model.params
        self.factor_returns = X
        self.residuals = model.resid
        
        # White test for heteroskedasticity
        try:
            white_stat, white_pvalue, _, _ = het_white(model.resid, model.model.exog)
        except:
            white_stat, white_pvalue = np.nan, np.nan
        
        return {
            'factor_loadings': {k: float(v) for k, v in self.factor_loadings.items()},
            'r_squared': float(model.rsquared),
            'adjusted_r_squared': float(model.rsquared_adj),
            'f_statistic': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue),
            'white_test_stat': float(white_stat) if not np.isnan(white_stat) else None,
            'white_test_pvalue': float(white_pvalue) if not np.isnan(white_pvalue) else None,
            'residual_std': float(self.residuals.std()),
            't_stats': {k: float(v) for k, v in model.tvalues.items()},
            'p_values': {k: float(v) for k, v in model.pvalues.items()}
        }
    
    def predict(self, factor_returns: pd.DataFrame) -> pd.Series:
        """Predict returns from factor returns."""
        if self.factor_loadings is None:
            raise ValueError("Model must be fitted first")
        
        factors = factor_returns[self.factor_loadings.index]
        predicted = (factors * self.factor_loadings).sum(axis=1)
        return predicted


class StyleFactorModel:
    """
    Style Factor Model (Value, Growth, Size, Momentum, Quality, etc.).
    """
    
    def __init__(self):
        """Initialize style factor model."""
        self.factors = {}
        self.loadings = {}
    
    def calculate_value_factor(self, 
                              pe_ratio: float,
                              pb_ratio: float,
                              ev_ebitda: float) -> float:
        """
        Calculate value factor score.
        
        Args:
            pe_ratio: P/E ratio
            pb_ratio: P/B ratio
            ev_ebitda: EV/EBITDA
        
        Returns:
            Value factor score
        """
        # Lower multiples = higher value score
        pe_score = 1.0 / (pe_ratio + 1e-6)
        pb_score = 1.0 / (pb_ratio + 1e-6)
        ev_score = 1.0 / (ev_ebitda + 1e-6)
        
        # Normalize and combine
        value_score = (pe_score + pb_score + ev_score) / 3
        return value_score
    
    def calculate_momentum_factor(self, returns: pd.Series, periods: List[int] = [1, 3, 6, 12]) -> float:
        """
        Calculate momentum factor score.
        
        Args:
            returns: Returns series
            periods: Lookback periods in months
        
        Returns:
            Momentum factor score
        """
        momentum_scores = []
        for period in periods:
            if len(returns) >= period:
                cumulative_return = (1 + returns.tail(period)).prod() - 1
                momentum_scores.append(cumulative_return)
        
        return np.mean(momentum_scores) if momentum_scores else 0.0
    
    def calculate_quality_factor(self,
                                roe: float,
                                roa: float,
                                debt_to_equity: float,
                                current_ratio: float) -> float:
        """
        Calculate quality factor score.
        
        Args:
            roe: Return on equity
            roa: Return on assets
            debt_to_equity: Debt to equity ratio
            current_ratio: Current ratio
        
        Returns:
            Quality factor score
        """
        # Higher ROE/ROA = better
        profitability_score = (roe + roa) / 2
        
        # Lower debt = better
        leverage_score = 1.0 / (debt_to_equity + 1e-6)
        
        # Higher current ratio = better liquidity
        liquidity_score = current_ratio
        
        # Combine (normalized)
        quality_score = (profitability_score * 0.5 + 
                        leverage_score * 0.3 + 
                        liquidity_score * 0.2)
        
        return quality_score
    
    def calculate_size_factor(self, market_cap: float) -> float:
        """
        Calculate size factor (log market cap).
        
        Args:
            market_cap: Market capitalization
        
        Returns:
            Size factor score
        """
        return np.log(market_cap + 1e-6)


class RiskFactorModel:
    """
    Risk Factor Model for portfolio risk decomposition.
    """
    
    def __init__(self):
        """Initialize risk factor model."""
        self.factor_covariance = None
        self.factor_loadings = None
    
    def decompose_risk(self,
                      portfolio_weights: pd.Series,
                      factor_loadings: pd.DataFrame,
                      factor_covariance: pd.DataFrame,
                      specific_risk: pd.Series) -> Dict[str, float]:
        """
        Decompose portfolio risk into factor and specific risk.
        
        Args:
            portfolio_weights: Portfolio weights
            factor_loadings: Factor loadings matrix (assets x factors)
            factor_covariance: Factor covariance matrix
            specific_risk: Specific (idiosyncratic) risk
        
        Returns:
            Risk decomposition
        """
        # Factor risk
        portfolio_factor_loadings = portfolio_weights @ factor_loadings
        factor_risk = np.sqrt(portfolio_factor_loadings @ factor_covariance @ portfolio_factor_loadings.T)
        
        # Specific risk
        specific_risk_portfolio = np.sqrt((portfolio_weights ** 2 * specific_risk ** 2).sum())
        
        # Total risk
        total_risk = np.sqrt(factor_risk ** 2 + specific_risk_portfolio ** 2)
        
        # Factor contributions
        factor_contributions = {}
        for factor in factor_loadings.columns:
            factor_loading = portfolio_factor_loadings[factor]
            factor_var = factor_covariance.loc[factor, factor]
            contribution = (factor_loading ** 2 * factor_var) / (total_risk ** 2)
            factor_contributions[factor] = float(contribution)
        
        return {
            'total_risk': float(total_risk),
            'factor_risk': float(factor_risk),
            'specific_risk': float(specific_risk_portfolio),
            'factor_risk_pct': float(factor_risk / total_risk * 100),
            'specific_risk_pct': float(specific_risk_portfolio / total_risk * 100),
            'factor_contributions': factor_contributions
        }
    
    def calculate_tracking_error(self,
                                portfolio_returns: pd.Series,
                                benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate tracking error and its decomposition.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
        
        Returns:
            Tracking error statistics
        """
        active_returns = portfolio_returns - benchmark_returns
        
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        return {
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio),
            'active_return': float(active_returns.mean() * 252),
            'active_volatility': float(active_returns.std() * np.sqrt(252))
        }
