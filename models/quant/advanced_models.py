"""
Advanced Quantitative Models
Factor models, regime detection, and sophisticated quant techniques
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.regime_switching import MarkovRegression
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class FactorModel:
    """
    Multi-factor model for asset returns.
    Implements Fama-French style factor models.
    """
    
    def __init__(self, n_factors: int = 3):
        """
        Initialize factor model.
        
        Args:
            n_factors: Number of factors to extract
        """
        self.n_factors = n_factors
        self.factors = None
        self.factor_loadings = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_factors)
    
    def fit(self, returns: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Fit factor model using PCA.
        
        Args:
            returns: DataFrame of asset returns (assets x time)
        
        Returns:
            Dictionary with factors and loadings
        """
        # Standardize returns
        returns_scaled = self.scaler.fit_transform(returns.T).T
        
        # Extract factors using PCA
        self.factors = self.pca.fit_transform(returns_scaled.T)
        self.factor_loadings = self.pca.components_.T
        
        # Calculate factor returns
        factor_returns = pd.DataFrame(
            self.factors,
            index=returns.columns,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        return {
            'factors': self.factors,
            'loadings': self.factor_loadings,
            'factor_returns': factor_returns,
            'explained_variance': self.pca.explained_variance_ratio_
        }
    
    def predict(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Predict asset returns from factor returns.
        
        Args:
            factor_returns: Factor returns (time x factors)
        
        Returns:
            Predicted asset returns
        """
        if self.factor_loadings is None:
            raise ValueError("Model must be fitted first")
        
        predicted = factor_returns @ self.factor_loadings.T
        return pd.DataFrame(
            predicted,
            columns=[f'Asset_{i+1}' for i in range(self.factor_loadings.shape[0])]
        )


class RegimeDetector:
    """
    Market regime detection using Hidden Markov Models and clustering.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
        """
        self.n_regimes = n_regimes
        self.regimes = None
        self.regime_characteristics = {}
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    
    def detect_regimes(self, returns: pd.Series, method: str = "kmeans") -> pd.Series:
        """
        Detect market regimes.
        
        Args:
            returns: Time series of returns
            method: Detection method ("kmeans" or "hmm")
        
        Returns:
            Series of regime labels
        """
        if method == "kmeans":
            return self._detect_kmeans(returns)
        elif method == "hmm" and HAS_STATSMODELS:
            return self._detect_hmm(returns)
        else:
            return self._detect_kmeans(returns)
    
    def _detect_kmeans(self, returns: pd.Series) -> pd.Series:
        """Detect regimes using K-means clustering."""
        # Features: returns, volatility, trend
        returns_rolling = returns.rolling(20).mean()
        volatility = returns.rolling(20).std()
        
        features = pd.DataFrame({
            'returns': returns_rolling,
            'volatility': volatility,
            'trend': returns.rolling(5).mean()
        }).fillna(method='bfill').fillna(0)
        
        # Cluster
        labels = self.kmeans.fit_predict(features.values)
        self.regimes = pd.Series(labels, index=returns.index)
        
        # Characterize regimes
        self._characterize_regimes(returns)
        
        return self.regimes
    
    def _detect_hmm(self, returns: pd.Series) -> pd.Series:
        """Detect regimes using Hidden Markov Model."""
        try:
            model = MarkovRegression(returns.values, k_regimes=self.n_regimes)
            result = model.fit()
            regimes = result.smoothed_marginal_probabilities[0].idxmax(axis=1)
            self.regimes = pd.Series(regimes.values, index=returns.index)
            self._characterize_regimes(returns)
            return self.regimes
        except Exception as e:
            # Fallback to K-means
            return self._detect_kmeans(returns)
    
    def _characterize_regimes(self, returns: pd.Series):
        """Characterize each regime."""
        for regime in range(self.n_regimes):
            regime_returns = returns[self.regimes == regime]
            if len(regime_returns) > 0:
                self.regime_characteristics[regime] = {
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'count': len(regime_returns),
                    'label': self._label_regime(regime_returns.mean(), regime_returns.std())
                }
    
    def _label_regime(self, mean_return: float, volatility: float) -> str:
        """Label regime based on characteristics."""
        if mean_return > 0.001 and volatility < 0.02:
            return "Bull Low Vol"
        elif mean_return > 0.001:
            return "Bull High Vol"
        elif mean_return < -0.001 and volatility < 0.02:
            return "Bear Low Vol"
        elif mean_return < -0.001:
            return "Bear High Vol"
        else:
            return "Sideways"
    
    def get_current_regime(self) -> Optional[int]:
        """Get current regime."""
        if self.regimes is not None and len(self.regimes) > 0:
            return self.regimes.iloc[-1]
        return None
    
    def get_regime_probabilities(self) -> Dict[int, float]:
        """Get probability of each regime."""
        if self.regimes is not None:
            probs = self.regimes.value_counts(normalize=True)
            return probs.to_dict()
        return {}


class AlternativeDataProcessor:
    """
    Process alternative data sources for trading signals.
    """
    
    @staticmethod
    def calculate_sentiment_score(text_data: List[str]) -> float:
        """
        Calculate sentiment score from text data.
        
        Args:
            text_data: List of text strings
        
        Returns:
            Sentiment score (-1 to 1)
        """
        # Simple keyword-based sentiment (can be enhanced with NLP models)
        positive_words = ['bullish', 'buy', 'growth', 'positive', 'up', 'gain', 'profit']
        negative_words = ['bearish', 'sell', 'decline', 'negative', 'down', 'loss', 'crash']
        
        score = 0
        total_words = 0
        
        for text in text_data:
            words = text.lower().split()
            total_words += len(words)
            for word in words:
                if word in positive_words:
                    score += 1
                elif word in negative_words:
                    score -= 1
        
        if total_words > 0:
            return np.clip(score / total_words * 10, -1, 1)
        return 0.0
    
    @staticmethod
    def calculate_volume_profile(prices: pd.Series, volumes: pd.Series, 
                                 bins: int = 20) -> Dict[str, float]:
        """
        Calculate volume profile (price levels with most volume).
        
        Args:
            prices: Price series
            volumes: Volume series
            bins: Number of price bins
        
        Returns:
            Dictionary with price levels and volume
        """
        # Create price bins
        price_bins = pd.cut(prices, bins=bins)
        
        # Aggregate volume by price level
        volume_profile = volumes.groupby(price_bins).sum()
        
        # Find high volume nodes
        high_volume_nodes = volume_profile.nlargest(3)
        
        return {
            'high_volume_levels': high_volume_nodes.to_dict(),
            'total_volume': volumes.sum(),
            'average_price': prices.mean()
        }
    
    @staticmethod
    def detect_anomalies(returns: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect anomalous returns using z-score.
        
        Args:
            returns: Returns series
            threshold: Z-score threshold
        
        Returns:
            Boolean series indicating anomalies
        """
        z_scores = (returns - returns.mean()) / (returns.std() + 1e-6)
        return np.abs(z_scores) > threshold


class PortfolioOptimizerAdvanced:
    """
    Advanced portfolio optimization with constraints and risk models.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        self.risk_free_rate = risk_free_rate
    
    def optimize_max_sharpe(self, returns: pd.DataFrame, 
                           constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Optimize portfolio for maximum Sharpe ratio.
        
        Args:
            returns: Asset returns DataFrame
            constraints: Optional constraints (min_weight, max_weight, etc.)
        
        Returns:
            Optimal weights dictionary
        """
        try:
            from scipy.optimize import minimize
            
            n_assets = len(returns.columns)
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Default constraints
            if constraints is None:
                constraints = {
                    'min_weight': 0.0,
                    'max_weight': 1.0,
                    'sum_to_one': True
                }
            
            def negative_sharpe(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (portfolio_return - self.risk_free_rate/252) / portfolio_std
                return -sharpe
            
            # Constraints
            cons = []
            if constraints['sum_to_one']:
                cons.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            
            bounds = [(constraints['min_weight'], constraints['max_weight']) 
                     for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(negative_sharpe, x0, method='SLSQP', 
                           bounds=bounds, constraints=cons)
            
            if result.success:
                weights = result.x
                return dict(zip(returns.columns, weights))
            else:
                # Fallback to equal weights
                return {col: 1/n_assets for col in returns.columns}
        
        except Exception as e:
            # Fallback to equal weights
            n_assets = len(returns.columns)
            return {col: 1/n_assets for col in returns.columns}
    
    def optimize_risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Risk parity optimization (equal risk contribution).
        
        Args:
            returns: Asset returns DataFrame
        
        Returns:
            Optimal weights dictionary
        """
        try:
            from scipy.optimize import minimize
            
            cov_matrix = returns.cov().values
            n_assets = len(returns.columns)
            
            def risk_parity_objective(weights):
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                risk_contributions = weights * np.dot(cov_matrix, weights) / portfolio_std
                target_risk = 1.0 / n_assets
                return np.sum((risk_contributions - target_risk) ** 2)
            
            # Constraints
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            x0 = np.ones(n_assets) / n_assets
            
            result = minimize(risk_parity_objective, x0, method='SLSQP',
                            bounds=bounds, constraints=cons)
            
            if result.success:
                weights = result.x
                return dict(zip(returns.columns, weights))
            else:
                return {col: 1/n_assets for col in returns.columns}
        
        except Exception as e:
            n_assets = len(returns.columns)
            return {col: 1/n_assets for col in returns.columns}
