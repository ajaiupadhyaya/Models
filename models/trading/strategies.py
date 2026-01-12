"""
Trading Strategy Implementations
Momentum, Mean Reversion, Pairs Trading, Factor Investing
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MomentumStrategy:
    """
    Momentum trading strategy.
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 holding_period: int = 5):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_period: Period for calculating momentum
            holding_period: Period to hold position
        """
        self.lookback_period = lookback_period
        self.holding_period = holding_period
    
    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate trading signals based on momentum.
        
        Args:
            prices: Price series
        
        Returns:
            Signal series (1 for long, -1 for short, 0 for neutral)
        """
        returns = prices.pct_change(self.lookback_period)
        signals = pd.Series(0, index=prices.index)
        
        # Long when momentum is positive
        signals[returns > 0] = 1
        # Short when momentum is negative
        signals[returns < 0] = -1
        
        return signals
    
    def backtest(self, prices: pd.Series, 
                initial_capital: float = 100000) -> Dict:
        """
        Backtest momentum strategy.
        
        Args:
            prices: Price series
            initial_capital: Initial capital
        
        Returns:
            Dictionary with backtest results
        """
        signals = self.generate_signals(prices)
        returns = prices.pct_change()
        
        strategy_returns = signals.shift(1) * returns
        strategy_returns = strategy_returns.fillna(0)
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        return {
            'signals': signals,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'portfolio_value': portfolio_value,
            'total_return': cumulative_returns.iloc[-1] - 1,
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        }


class MeanReversionStrategy:
    """
    Mean reversion trading strategy using Bollinger Bands.
    """
    
    def __init__(self,
                 lookback_period: int = 20,
                 num_std: float = 2.0):
        """
        Initialize mean reversion strategy.
        
        Args:
            lookback_period: Period for moving average
            num_std: Number of standard deviations for bands
        """
        self.lookback_period = lookback_period
        self.num_std = num_std
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
        
        Returns:
            DataFrame with upper, middle, lower bands
        """
        middle = prices.rolling(window=self.lookback_period).mean()
        std = prices.rolling(window=self.lookback_period).std()
        
        upper = middle + (std * self.num_std)
        lower = middle - (std * self.num_std)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        }, index=prices.index)
    
    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate mean reversion signals.
        
        Args:
            prices: Price series
        
        Returns:
            Signal series
        """
        bands = self.calculate_bollinger_bands(prices)
        signals = pd.Series(0, index=prices.index)
        
        # Long when price touches lower band
        signals[prices <= bands['lower']] = 1
        # Short when price touches upper band
        signals[prices >= bands['upper']] = -1
        
        return signals
    
    def backtest(self, prices: pd.Series,
                initial_capital: float = 100000) -> Dict:
        """
        Backtest mean reversion strategy.
        
        Args:
            prices: Price series
            initial_capital: Initial capital
        
        Returns:
            Dictionary with backtest results
        """
        signals = self.generate_signals(prices)
        returns = prices.pct_change()
        
        strategy_returns = signals.shift(1) * returns
        strategy_returns = strategy_returns.fillna(0)
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        bands = self.calculate_bollinger_bands(prices)
        
        return {
            'signals': signals,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'portfolio_value': portfolio_value,
            'bollinger_bands': bands,
            'total_return': cumulative_returns.iloc[-1] - 1,
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        }


class PairsTradingStrategy:
    """
    Pairs trading strategy using cointegration.
    """
    
    def __init__(self,
                 lookback_period: int = 60,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5):
        """
        Initialize pairs trading strategy.
        
        Args:
            lookback_period: Period for calculating spread
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def calculate_spread(self, 
                        asset1: pd.Series,
                        asset2: pd.Series,
                        hedge_ratio: Optional[float] = None) -> pd.Series:
        """
        Calculate spread between two assets.
        
        Args:
            asset1: First asset price series
            asset2: Second asset price series
            hedge_ratio: Optional hedge ratio (calculated if None)
        
        Returns:
            Spread series
        """
        if hedge_ratio is None:
            # Calculate hedge ratio using linear regression
            from sklearn.linear_model import LinearRegression
            common_index = asset1.index.intersection(asset2.index)
            X = asset1.loc[common_index].values.reshape(-1, 1)
            y = asset2.loc[common_index].values
            
            model = LinearRegression()
            model.fit(X, y)
            hedge_ratio = model.coef_[0]
        
        spread = asset1 - hedge_ratio * asset2
        return spread
    
    def generate_signals(self,
                        asset1: pd.Series,
                        asset2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Generate pairs trading signals.
        
        Args:
            asset1: First asset price series
            asset2: Second asset price series
        
        Returns:
            Tuple of (signals_asset1, signals_asset2)
        """
        spread = self.calculate_spread(asset1, asset2)
        z_scores = (spread - spread.rolling(self.lookback_period).mean()) / \
                   spread.rolling(self.lookback_period).std()
        
        signals1 = pd.Series(0, index=asset1.index)
        signals2 = pd.Series(0, index=asset2.index)
        
        # Long spread when z-score is low (short asset1, long asset2)
        signals1[z_scores < -self.entry_threshold] = -1
        signals2[z_scores < -self.entry_threshold] = 1
        
        # Short spread when z-score is high (long asset1, short asset2)
        signals1[z_scores > self.entry_threshold] = 1
        signals2[z_scores > self.entry_threshold] = -1
        
        # Exit when z-score returns to mean
        signals1[(z_scores.abs() < self.exit_threshold) & 
                (signals1.shift(1) != 0)] = 0
        signals2[(z_scores.abs() < self.exit_threshold) & 
                (signals2.shift(1) != 0)] = 0
        
        return signals1, signals2
    
    def backtest(self,
                asset1: pd.Series,
                asset2: pd.Series,
                initial_capital: float = 100000) -> Dict:
        """
        Backtest pairs trading strategy.
        
        Args:
            asset1: First asset price series
            asset2: Second asset price series
            initial_capital: Initial capital
        
        Returns:
            Dictionary with backtest results
        """
        signals1, signals2 = self.generate_signals(asset1, asset2)
        returns1 = asset1.pct_change()
        returns2 = asset2.pct_change()
        
        strategy_returns = (signals1.shift(1) * returns1 + 
                          signals2.shift(1) * returns2) / 2
        strategy_returns = strategy_returns.fillna(0)
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        spread = self.calculate_spread(asset1, asset2)
        
        return {
            'signals_asset1': signals1,
            'signals_asset2': signals2,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'portfolio_value': portfolio_value,
            'spread': spread,
            'total_return': cumulative_returns.iloc[-1] - 1,
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        }


class FactorStrategy:
    """
    Factor-based investing strategy.
    """
    
    def __init__(self, factors: List[str]):
        """
        Initialize factor strategy.
        
        Args:
            factors: List of factor names (e.g., ['momentum', 'value', 'quality'])
        """
        self.factors = factors
    
    def calculate_factor_scores(self,
                               returns_df: pd.DataFrame,
                               factor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate composite factor scores.
        
        Args:
            returns_df: Returns DataFrame
            factor_data: Dictionary of factor name to factor values DataFrame
        
        Returns:
            DataFrame with factor scores
        """
        scores = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        
        for asset in returns_df.columns:
            asset_scores = []
            for factor in self.factors:
                if factor in factor_data and asset in factor_data[factor].columns:
                    factor_values = factor_data[factor][asset]
                    # Normalize factor values
                    normalized = (factor_values - factor_values.mean()) / factor_values.std()
                    asset_scores.append(normalized)
            
            if asset_scores:
                # Equal-weighted composite score
                composite = pd.concat(asset_scores, axis=1).mean(axis=1)
                scores[asset] = composite
        
        return scores
    
    def generate_signals(self,
                        factor_scores: pd.DataFrame,
                        top_n: int = 10) -> pd.DataFrame:
        """
        Generate signals based on factor scores.
        
        Args:
            factor_scores: Factor scores DataFrame
            top_n: Number of top assets to select
        
        Returns:
            Signals DataFrame
        """
        signals = pd.DataFrame(0, index=factor_scores.index, columns=factor_scores.columns)
        
        for date in factor_scores.index:
            scores = factor_scores.loc[date].dropna()
            if len(scores) >= top_n:
                top_assets = scores.nlargest(top_n).index
                signals.loc[date, top_assets] = 1
        
        return signals
