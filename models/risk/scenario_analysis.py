"""
Scenario Analysis and Risk Metrics
Advanced scenario analysis and systemic risk measures
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ScenarioAnalysisFull:
    """
    Full scenario analysis with multiple dimensions.
    """
    
    def __init__(self, returns_df: pd.DataFrame):
        """
        Initialize with returns dataframe.
        
        Args:
            returns_df: DataFrame with asset returns (columns = assets, rows = time periods)
        """
        self.returns_df = returns_df
        self.assets = returns_df.columns.tolist()
        self.n_assets = len(self.assets)
        self.n_periods = len(returns_df)
    
    def tail_risk_analysis(self, confidence_level: float = 0.95) -> Dict:
        """
        Analyze tail risk (extreme losses).
        
        Args:
            confidence_level: Confidence level for tail (e.g., 0.95 for 5% tail)
        
        Returns:
            Tail risk metrics
        """
        tail_threshold = 1 - confidence_level
        results = {}
        
        for asset in self.assets:
            returns = self.returns_df[asset]
            
            # Identify tail returns
            tail_returns = returns[returns <= returns.quantile(tail_threshold)]
            
            results[asset] = {
                'tail_mean': tail_returns.mean(),
                'tail_std': tail_returns.std(),
                'tail_skew': tail_returns.skew(),
                'tail_count': len(tail_returns),
                'max_loss': returns.min(),
                'tail_var': returns.quantile(tail_threshold),
                'expected_shortfall': tail_returns.mean() if len(tail_returns) > 0 else 0
            }
        
        return results
    
    def correlation_breakdown(self, window: int = 60, down_market_threshold: float = -0.01) -> Dict:
        """
        Analyze correlation breakdown in stress.
        
        Args:
            window: Rolling window size
            down_market_threshold: Return threshold for 'down market'
        
        Returns:
            Normal vs stress correlations
        """
        results = {
            'normal_correlation': None,
            'stress_correlation': None,
            'correlation_increase': {}
        }
        
        # Calculate normal periods correlation
        normal_mask = self.returns_df.mean(axis=1) > down_market_threshold
        if normal_mask.sum() > 5:
            normal_corr = self.returns_df[normal_mask].corr()
            results['normal_correlation'] = normal_corr
        
        # Calculate stress periods correlation
        stress_mask = self.returns_df.mean(axis=1) <= down_market_threshold
        if stress_mask.sum() > 5:
            stress_corr = self.returns_df[stress_mask].corr()
            results['stress_correlation'] = stress_corr
        
        # Calculate changes
        if results['normal_correlation'] is not None and results['stress_correlation'] is not None:
            for i, asset1 in enumerate(self.assets):
                for j, asset2 in enumerate(self.assets):
                    if i < j:
                        pair = f"{asset1}-{asset2}"
                        normal = results['normal_correlation'].iloc[i, j]
                        stress = results['stress_correlation'].iloc[i, j]
                        results['correlation_increase'][pair] = stress - normal
        
        return results
    
    def expected_shortfall(self, confidence_level: float = 0.95) -> Dict:
        """
        Calculate Expected Shortfall (CVaR).
        
        Args:
            confidence_level: Confidence level
        
        Returns:
            Expected shortfall for each asset
        """
        results = {}
        threshold = 1 - confidence_level
        
        for asset in self.assets:
            returns = self.returns_df[asset]
            var = returns.quantile(threshold)
            cvar = returns[returns <= var].mean()
            results[asset] = {
                'var': var,
                'cvar': cvar,
                'tail_ratio': cvar / var if var != 0 else 0
            }
        
        return results
    
    def coherent_risk_measure(self) -> Dict:
        """
        Calculate coherent risk measures.
        
        Returns:
            Coherent risk metrics
        """
        results = {}
        
        for asset in self.assets:
            returns = self.returns_df[asset]
            
            # Value at Risk (95%)
            var_95 = returns.quantile(0.05)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Expected Shortfall weighted by severity
            tail = returns[returns <= var_95]
            severity_weighted_es = (tail * abs(tail)).sum() / abs(tail).sum() if len(tail) > 0 else 0
            
            # Marginal VaR
            marginal_var = returns.std() * stats.norm.ppf(0.05)
            
            results[asset] = {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'severity_weighted_es': severity_weighted_es,
                'marginal_var': marginal_var
            }
        
        return results


class SystemicRiskMeasures:
    """
    Calculate systemic risk measures (CoVaR, MES, etc.).
    """
    
    def __init__(self, returns_df: pd.DataFrame):
        """
        Initialize with returns dataframe.
        
        Args:
            returns_df: DataFrame with asset returns
        """
        self.returns_df = returns_df
        self.assets = returns_df.columns.tolist()
    
    def marginal_expected_shortfall(self, asset: str, confidence_level: float = 0.95) -> Dict:
        """
        Calculate Marginal Expected Shortfall.
        
        Args:
            asset: Asset to analyze
            confidence_level: Confidence level
        
        Returns:
            MES for each other asset
        """
        if asset not in self.returns_df.columns:
            return None
        
        results = {}
        threshold = 1 - confidence_level
        
        # Identify stress periods (bottom tail of asset)
        asset_var = self.returns_df[asset].quantile(threshold)
        stress_periods = self.returns_df[self.returns_df[asset] <= asset_var].index
        
        for other_asset in self.assets:
            if other_asset != asset:
                # Expected shortfall of other asset during stress
                stress_returns = self.returns_df.loc[stress_periods, other_asset]
                mes = stress_returns.mean()
                results[other_asset] = {
                    'mes': mes,
                    'stress_periods': len(stress_periods),
                    'avg_stress_return': stress_returns.mean()
                }
        
        return results
    
    def conditional_value_at_risk(self, asset: str, confidence_level: float = 0.95) -> Dict:
        """
        Calculate CoVaR (Conditional VaR).
        
        Args:
            asset: Asset to analyze
            confidence_level: Confidence level
        
        Returns:
            CoVaR for portfolio given asset stress
        """
        if asset not in self.returns_df.columns:
            return None
        
        # Calculate VaR when asset is in stress
        threshold = 1 - confidence_level
        asset_var = self.returns_df[asset].quantile(threshold)
        stress_periods = self.returns_df[self.returns_df[asset] <= asset_var].index
        
        results = {
            'asset': asset,
            'asset_var': asset_var,
            'stress_periods': len(stress_periods),
            'covar_estimates': {}
        }
        
        # For each other asset
        for other_asset in self.assets:
            if other_asset != asset:
                stress_returns = self.returns_df.loc[stress_periods, other_asset]
                covar = stress_returns.quantile(threshold)
                results['covar_estimates'][other_asset] = covar
        
        # Portfolio CoVaR
        portfolio_stress = self.returns_df.loc[stress_periods].mean(axis=1)
        portfolio_covar = portfolio_stress.quantile(threshold)
        results['portfolio_covar'] = portfolio_covar
        
        return results
    
    def systemic_importance(self) -> Dict:
        """
        Rank assets by systemic importance.
        
        Returns:
            Systemic importance scores
        """
        scores = {}
        
        for asset in self.assets:
            # Calculate impact when this asset is in stress
            mes_results = self.marginal_expected_shortfall(asset)
            
            # Average impact on other assets
            impacts = [v['mes'] for v in mes_results.values()]
            avg_impact = np.mean(impacts) if impacts else 0
            std_impact = np.std(impacts) if impacts else 0
            
            # Systemic score: magnitude + consistency
            systemic_score = abs(avg_impact) * (1 + std_impact)
            
            scores[asset] = {
                'systemic_score': systemic_score,
                'avg_impact_on_others': avg_impact,
                'impact_std': std_impact
            }
        
        # Rank by systemic score
        ranked = sorted(scores.items(), key=lambda x: abs(x[1]['systemic_score']), reverse=True)
        
        return {
            'scores': scores,
            'ranking': [asset for asset, _ in ranked]
        }


class CorrelationDynamics:
    """
    Model time-varying correlations and correlation breakdown.
    """
    
    def __init__(self, returns_df: pd.DataFrame, window: int = 60):
        """
        Initialize correlation dynamics analysis.
        
        Args:
            returns_df: DataFrame with returns
            window: Rolling window size
        """
        self.returns_df = returns_df
        self.window = window
        self.rolling_correlations = None
    
    def calculate_rolling_correlations(self) -> pd.DataFrame:
        """
        Calculate rolling correlations.
        
        Returns:
            Time series of correlations
        """
        # This will be a list of correlation matrices over time
        correlations = []
        
        for i in range(len(self.returns_df) - self.window):
            window_returns = self.returns_df.iloc[i:i+self.window]
            corr = window_returns.corr()
            correlations.append(corr)
        
        self.rolling_correlations = correlations
        return correlations
    
    def correlation_clustering(self) -> Dict:
        """
        Identify correlation clustering and structure.
        
        Returns:
            Correlation structure analysis
        """
        if self.rolling_correlations is None:
            self.calculate_rolling_correlations()
        
        results = {}
        
        # Calculate average correlation and volatility
        corrs = []
        for corr_matrix in self.rolling_correlations:
            # Get upper triangle (exclude diagonal)
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            corrs.extend(corr_matrix.values[mask])
        
        results['mean_correlation'] = np.mean(corrs)
        results['std_correlation'] = np.std(corrs)
        results['min_correlation'] = np.min(corrs)
        results['max_correlation'] = np.max(corrs)
        
        # Identify periods of high correlation clustering
        high_corr_periods = []
        for i, corr_matrix in enumerate(self.rolling_correlations):
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            period_corrs = corr_matrix.values[mask]
            if np.mean(period_corrs) > np.percentile(corrs, 75):
                high_corr_periods.append(i)
        
        results['high_clustering_periods'] = high_corr_periods
        
        return results
    
    def correlation_breakdown_metric(self, threshold_pct: float = 25) -> Dict:
        """
        Measure correlation breakdown during stress.
        
        Args:
            threshold_pct: Percentile threshold for 'down markets'
        
        Returns:
            Breakdown analysis
        """
        if self.rolling_correlations is None:
            self.calculate_rolling_correlations()
        
        # Identify down market periods
        portfolio_returns = self.returns_df.mean(axis=1)
        down_threshold = np.percentile(portfolio_returns, threshold_pct)
        down_periods = portfolio_returns <= down_threshold
        
        results = {}
        
        # For each pair, compare correlation in down vs normal markets
        assets = self.returns_df.columns.tolist()
        
        normal_correlations = []
        down_correlations = []
        
        # Overall normal period correlation
        normal_corr = self.returns_df[~down_periods].corr()
        down_corr = self.returns_df[down_periods].corr()
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    pair = f"{asset1}-{asset2}"
                    normal_c = normal_corr.iloc[i, j]
                    down_c = down_corr.iloc[i, j]
                    breakdown = down_c - normal_c
                    
                    results[pair] = {
                        'normal_correlation': normal_c,
                        'down_correlation': down_c,
                        'breakdown': breakdown,
                        'correlation_increase_pct': (breakdown / normal_c * 100) if normal_c != 0 else 0
                    }
        
        return results
