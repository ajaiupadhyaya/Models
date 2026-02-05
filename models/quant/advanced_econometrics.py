"""
Advanced Econometric Models
Institutional-grade time series and econometric methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize
from scipy.stats import norm, t
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.regime_switching import MarkovRegression
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import arch
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


class VectorAutoregression:
    """
    Vector Autoregression (VAR) Model.
    Models multiple time series simultaneously.
    """
    
    def __init__(self, maxlags: int = 4):
        """
        Initialize VAR model.
        
        Args:
            maxlags: Maximum number of lags
        """
        self.maxlags = maxlags
        self.model = None
        self.fitted_model = None
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit VAR model.
        
        Args:
            data: DataFrame with multiple time series
        
        Returns:
            Model fit statistics
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required for VAR")
        
        try:
            self.model = VAR(data)
            self.fitted_model = self.model.fit(maxlags=self.maxlags, ic='aic')
            
            return {
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'fpe': float(self.fitted_model.fpe),
                'hqic': float(self.fitted_model.hqic),
                'n_lags': int(self.fitted_model.k_ar),
                'n_obs': int(self.fitted_model.nobs)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def forecast(self, steps: int = 1) -> pd.DataFrame:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps ahead
        
        Returns:
            Forecast DataFrame
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.forecast(self.fitted_model.y, steps=steps)


class ARIMAGARCH:
    """
    ARIMA-GARCH Model.
    Combines ARIMA for mean and GARCH for volatility.
    """
    
    def __init__(self, ar_order: Tuple[int, int, int] = (1, 0, 1),
                 garch_order: Tuple[int, int] = (1, 1)):
        """
        Initialize ARIMA-GARCH model.
        
        Args:
            ar_order: ARIMA order (p, d, q)
            garch_order: GARCH order (p, q)
        """
        self.ar_order = ar_order
        self.garch_order = garch_order
        self.model = None
    
    def fit(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit ARIMA-GARCH model.
        
        Args:
            returns: Returns series
        
        Returns:
            Model fit statistics
        """
        if not HAS_STATSMODELS or not HAS_ARCH:
            raise ImportError("statsmodels and arch required for ARIMA-GARCH")
        
        try:
            # Fit ARIMA for mean
            from statsmodels.tsa.arima.model import ARIMA
            arima_model = ARIMA(returns, order=self.ar_order)
            arima_fitted = arima_model.fit()
            
            # Extract residuals for GARCH
            residuals = arima_fitted.resid
            
            # Fit GARCH for volatility
            garch_model = arch.arch_model(
                residuals * 100,
                vol='GARCH',
                p=self.garch_order[0],
                q=self.garch_order[1]
            )
            garch_fitted = garch_model.fit(disp='off', show_warning=False)
            
            return {
                'arima_aic': float(arima_fitted.aic),
                'arima_bic': float(arima_fitted.bic),
                'garch_aic': float(garch_fitted.aic),
                'garch_bic': float(garch_fitted.bic),
                'arima_params': {k: float(v) for k, v in arima_fitted.params.items()},
                'garch_params': {k: float(v) for k, v in garch_fitted.params.items()}
            }
        except Exception as e:
            return {'error': str(e)}


class RegimeSwitchingModel:
    """
    Markov Regime-Switching Model.
    Models time series with different regimes.
    """
    
    def __init__(self, n_regimes: int = 2):
        """
        Initialize regime-switching model.
        
        Args:
            n_regimes: Number of regimes
        """
        self.n_regimes = n_regimes
        self.model = None
    
    def fit(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit regime-switching model.
        
        Args:
            returns: Returns series
        
        Returns:
            Model fit statistics
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required for regime-switching")
        
        try:
            self.model = MarkovRegression(returns.values, k_regimes=self.n_regimes)
            fitted = self.model.fit()
            
            # Get regime probabilities
            smoothed_probs = fitted.smoothed_marginal_probabilities[0]
            
            return {
                'aic': float(fitted.aic),
                'bic': float(fitted.bic),
                'log_likelihood': float(fitted.loglikelihood),
                'regime_probabilities': smoothed_probs.values.tolist(),
                'transition_matrix': fitted.transition_matrix.tolist()
            }
        except Exception as e:
            return {'error': str(e)}


class CointegrationAnalysis:
    """
    Cointegration Analysis for Pairs Trading.
    """
    
    @staticmethod
    def engle_granger_test(series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """
        Engle-Granger cointegration test.
        
        Args:
            series1: First series
            series2: Second series
        
        Returns:
            Test results
        """
        if not HAS_STATSMODELS:
            return {'error': 'statsmodels required'}
        
        try:
            from statsmodels.tsa.stattools import coint
            
            score, pvalue, critical_values = coint(series1, series2)
            
            return {
                'cointegration_statistic': float(score),
                'p_value': float(pvalue),
                'critical_values': {k: float(v) for k, v in critical_values.items()},
                'is_cointegrated': pvalue < 0.05
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def johansen_test(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Johansen cointegration test for multiple series.
        
        Args:
            data: DataFrame with multiple series
        
        Returns:
            Test results
        """
        if not HAS_STATSMODELS:
            return {'error': 'statsmodels required'}
        
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            
            result = coint_johansen(data.values, det_order=0, k_ar_diff=1)
            
            return {
                'trace_statistics': result.lr1.tolist(),
                'max_eigen_statistics': result.lr2.tolist(),
                'critical_values_trace': result.cvt.tolist(),
                'critical_values_eigen': result.cvm.tolist(),
                'cointegrating_relations': int(result.r)
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def calculate_hedge_ratio(series1: pd.Series, series2: pd.Series) -> float:
        """
        Calculate optimal hedge ratio using OLS.
        
        Args:
            series1: First series
            series2: Second series
        
        Returns:
            Hedge ratio
        """
        if not HAS_STATSMODELS:
            return 1.0
        
        try:
            from statsmodels.regression.linear_model import OLS
            
            aligned = pd.concat([series1, series2], axis=1).dropna()
            model = OLS(aligned.iloc[:, 0], aligned.iloc[:, 1]).fit()
            
            return float(model.params.iloc[0])
        except:
            return 1.0


class KalmanFilter:
    """
    Kalman Filter for state-space modeling.
    """
    
    def __init__(self, n_states: int = 1, n_obs: int = 1):
        """
        Initialize Kalman filter.
        
        Args:
            n_states: Number of state variables
            n_obs: Number of observed variables
        """
        self.n_states = n_states
        self.n_obs = n_obs
        self.state = np.zeros(n_states)
        self.covariance = np.eye(n_states)
    
    def predict(self, F: np.ndarray, Q: np.ndarray):
        """
        Prediction step.
        
        Args:
            F: State transition matrix
            Q: Process noise covariance
        """
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
    
    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        Update step.
        
        Args:
            z: Measurement
            H: Observation matrix
            R: Measurement noise covariance
        """
        # Innovation
        y = z - H @ self.state
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain
        K = self.covariance @ H.T @ inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        self.covariance = (np.eye(self.n_states) - K @ H) @ self.covariance
    
    def filter(self, observations: np.ndarray,
               F: np.ndarray, H: np.ndarray,
               Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter on observations.
        
        Args:
            observations: Observation array
            F: State transition matrix
            H: Observation matrix
            Q: Process noise covariance
            R: Measurement noise covariance
        
        Returns:
            Filtered states and covariances
        """
        filtered_states = []
        filtered_covariances = []
        
        for z in observations:
            # Predict
            self.predict(F, Q)
            
            # Update
            self.update(z, H, R)
            
            filtered_states.append(self.state.copy())
            filtered_covariances.append(self.covariance.copy())
        
        return np.array(filtered_states), np.array(filtered_covariances)
