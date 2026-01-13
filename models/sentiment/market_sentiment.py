"""
Market Sentiment Indicators
VIX, put/call ratios, breadth indicators, fear/greed metrics
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MarketSentimentIndicators:
    """
    Calculate various market sentiment indicators.
    """
    
    def __init__(self):
        """Initialize market sentiment analyzer."""
        pass
    
    def get_vix_data(self, period: str = '1y') -> pd.DataFrame:
        """
        Get VIX (volatility index) data.
        
        Args:
            period: Time period
        
        Returns:
            DataFrame with VIX data
        """
        try:
            vix = yf.Ticker("^VIX")
            df = vix.history(period=period)
            return df
        except Exception as e:
            print(f"Error fetching VIX: {e}")
            return pd.DataFrame()
    
    def calculate_put_call_ratio(self, 
                                 ticker: str,
                                 period: str = '1mo') -> Dict:
        """
        Calculate put/call ratio (requires option data).
        
        Args:
            ticker: Stock ticker
            period: Time period
        
        Returns:
            Dictionary with put/call metrics
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = stock.options
            
            if not expirations:
                return {'error': 'No option data available'}
            
            # Use nearest expiration
            exp_date = expirations[0]
            
            # Get option chain
            opt_chain = stock.option_chain(exp_date)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Calculate volumes
            call_volume = calls['volume'].sum() if 'volume' in calls else 0
            put_volume = puts['volume'].sum() if 'volume' in puts else 0
            
            # Calculate open interest
            call_oi = calls['openInterest'].sum() if 'openInterest' in calls else 0
            put_oi = puts['openInterest'].sum() if 'openInterest' in puts else 0
            
            # Calculate ratios
            pcr_volume = put_volume / call_volume if call_volume > 0 else np.nan
            pcr_oi = put_oi / call_oi if call_oi > 0 else np.nan
            
            return {
                'expiration': exp_date,
                'call_volume': call_volume,
                'put_volume': put_volume,
                'call_open_interest': call_oi,
                'put_open_interest': put_oi,
                'pcr_volume': pcr_volume,
                'pcr_open_interest': pcr_oi,
                'interpretation': self._interpret_pcr(pcr_volume)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _interpret_pcr(self, pcr: float) -> str:
        """Interpret put/call ratio."""
        if np.isnan(pcr):
            return 'Unknown'
        elif pcr > 1.0:
            return 'Bearish (more puts than calls)'
        elif pcr > 0.7:
            return 'Cautious'
        elif pcr < 0.5:
            return 'Bullish (more calls than puts)'
        else:
            return 'Neutral'
    
    def advance_decline_line(self, 
                           tickers: List[str],
                           period: str = '6mo') -> pd.Series:
        """
        Calculate advance-decline line for group of stocks.
        
        Args:
            tickers: List of stock tickers
            period: Time period
        
        Returns:
            Advance-decline line series
        """
        advances = []
        declines = []
        dates = None
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    if dates is None:
                        dates = hist.index
                    
                    # Calculate daily changes
                    changes = hist['Close'].pct_change()
                    
                    # Count advances and declines
                    advance = (changes > 0).astype(int)
                    decline = (changes < 0).astype(int)
                    
                    advances.append(advance)
                    declines.append(decline)
                    
            except Exception as e:
                continue
        
        if not advances:
            return pd.Series()
        
        # Sum across all stocks
        total_advances = pd.concat(advances, axis=1).sum(axis=1)
        total_declines = pd.concat(declines, axis=1).sum(axis=1)
        
        # Calculate net advances
        net_advances = total_advances - total_declines
        
        # Cumulative sum to create AD line
        ad_line = net_advances.cumsum()
        
        return ad_line
    
    def breadth_thrust(self, 
                      tickers: List[str],
                      period: str = '3mo',
                      threshold: float = 0.615) -> Dict:
        """
        Calculate breadth thrust indicator.
        
        Args:
            tickers: List of stock tickers
            period: Time period
            threshold: Threshold for thrust signal (Zweig: 0.615)
        
        Returns:
            Dictionary with breadth thrust metrics
        """
        ad_line = self.advance_decline_line(tickers, period)
        
        if ad_line.empty:
            return {'error': 'Could not calculate breadth thrust'}
        
        # Calculate 10-day EMA of advances / (advances + declines)
        # Simplified: use rolling average of AD line slope
        breadth_ratio = ad_line.diff(10).rolling(10).mean()
        
        # Normalize to 0-1 range
        breadth_normalized = (breadth_ratio - breadth_ratio.min()) / (breadth_ratio.max() - breadth_ratio.min())
        
        # Identify thrust signals (crosses above threshold)
        thrust_signals = (breadth_normalized > threshold) & (breadth_normalized.shift(1) <= threshold)
        
        signal_dates = breadth_normalized[thrust_signals].index.tolist()
        
        return {
            'current_breadth': breadth_normalized.iloc[-1] if len(breadth_normalized) > 0 else np.nan,
            'threshold': threshold,
            'thrust_signals': len(signal_dates),
            'signal_dates': signal_dates,
            'interpretation': 'Bullish thrust' if breadth_normalized.iloc[-1] > threshold else 'No thrust'
        }
    
    def calculate_mcclellan_oscillator(self, 
                                      tickers: List[str],
                                      period: str = '6mo') -> pd.Series:
        """
        Calculate McClellan Oscillator.
        
        Args:
            tickers: List of stock tickers
            period: Time period
        
        Returns:
            McClellan Oscillator series
        """
        # Get advance-decline data
        advances = []
        declines = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    changes = hist['Close'].pct_change()
                    advances.append((changes > 0).astype(int))
                    declines.append((changes < 0).astype(int))
            except:
                continue
        
        if not advances:
            return pd.Series()
        
        # Calculate net advances
        total_advances = pd.concat(advances, axis=1).sum(axis=1)
        total_declines = pd.concat(declines, axis=1).sum(axis=1)
        net_advances = total_advances - total_declines
        
        # Calculate EMAs (19-day and 39-day)
        ema_19 = net_advances.ewm(span=19, adjust=False).mean()
        ema_39 = net_advances.ewm(span=39, adjust=False).mean()
        
        # McClellan Oscillator = EMA(19) - EMA(39)
        oscillator = ema_19 - ema_39
        
        return oscillator
    
    def high_low_index(self, 
                      tickers: List[str],
                      period: str = '6mo',
                      lookback: int = 52) -> pd.Series:
        """
        Calculate High-Low Index.
        
        Args:
            tickers: List of stock tickers
            period: Time period
            lookback: Number of periods for high/low
        
        Returns:
            High-Low Index series
        """
        new_highs = []
        new_lows = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if len(hist) < lookback:
                    continue
                
                # Calculate rolling highs and lows
                rolling_high = hist['High'].rolling(lookback).max()
                rolling_low = hist['Low'].rolling(lookback).min()
                
                # Identify new highs and lows
                is_new_high = (hist['High'] >= rolling_high).astype(int)
                is_new_low = (hist['Low'] <= rolling_low).astype(int)
                
                new_highs.append(is_new_high)
                new_lows.append(is_new_low)
                
            except:
                continue
        
        if not new_highs:
            return pd.Series()
        
        # Sum across stocks
        total_new_highs = pd.concat(new_highs, axis=1).sum(axis=1)
        total_new_lows = pd.concat(new_lows, axis=1).sum(axis=1)
        
        # High-Low Index = (New Highs) / (New Highs + New Lows)
        hl_index = total_new_highs / (total_new_highs + total_new_lows)
        hl_index = hl_index.fillna(0.5)  # Neutral if no new highs or lows
        
        return hl_index


class FearGreedIndex:
    """
    Calculate custom Fear & Greed Index.
    """
    
    def __init__(self):
        """Initialize Fear & Greed calculator."""
        self.indicators = MarketSentimentIndicators()
    
    def calculate_index(self, 
                       spy_period: str = '3mo',
                       vix_period: str = '3mo') -> Dict:
        """
        Calculate Fear & Greed Index from multiple indicators.
        
        Args:
            spy_period: Period for SPY data
            vix_period: Period for VIX data
        
        Returns:
            Dictionary with Fear & Greed metrics
        """
        scores = []
        weights = []
        
        # 1. Price Momentum (SPY)
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period=spy_period)
            
            if not spy_hist.empty:
                returns_125 = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100
                momentum_score = self._normalize_momentum(returns_125)
                scores.append(momentum_score)
                weights.append(0.25)
        except:
            pass
        
        # 2. VIX (Volatility)
        try:
            vix_data = self.indicators.get_vix_data(period=vix_period)
            
            if not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                avg_vix = vix_data['Close'].mean()
                vix_score = self._normalize_vix(current_vix, avg_vix)
                scores.append(vix_score)
                weights.append(0.25)
        except:
            pass
        
        # 3. Market Breadth (advance-decline)
        try:
            # Use S&P 100 subset as proxy
            sp100_sample = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
                           'META', 'TSLA', 'BRK.B', 'JPM', 'V']
            
            ad_line = self.indicators.advance_decline_line(sp100_sample, period='1mo')
            
            if not ad_line.empty:
                # Breadth improving if recent AD line trending up
                recent_trend = ad_line.iloc[-5:].diff().mean()
                breadth_score = self._normalize_breadth(recent_trend)
                scores.append(breadth_score)
                weights.append(0.20)
        except:
            pass
        
        # 4. Safe Haven Demand (TLT vs SPY)
        try:
            tlt = yf.Ticker("TLT")
            tlt_hist = tlt.history(period='1mo')
            spy_hist = spy.history(period='1mo')
            
            if not tlt_hist.empty and not spy_hist.empty:
                tlt_return = (tlt_hist['Close'].iloc[-1] / tlt_hist['Close'].iloc[0] - 1)
                spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1)
                
                # If bonds outperforming, that's fear
                safe_haven_score = self._normalize_safe_haven(tlt_return - spy_return)
                scores.append(safe_haven_score)
                weights.append(0.15)
        except:
            pass
        
        # 5. High Yield Spreads (HYG vs TLT)
        try:
            hyg = yf.Ticker("HYG")
            hyg_hist = hyg.history(period='1mo')
            
            if not hyg_hist.empty and not tlt_hist.empty:
                hyg_return = (hyg_hist['Close'].iloc[-1] / hyg_hist['Close'].iloc[0] - 1)
                tlt_return = (tlt_hist['Close'].iloc[-1] / tlt_hist['Close'].iloc[0] - 1)
                
                # High yield outperforming treasuries = greed
                hy_score = self._normalize_high_yield(hyg_return - tlt_return)
                scores.append(hy_score)
                weights.append(0.15)
        except:
            pass
        
        # Calculate weighted average
        if scores:
            weights_array = np.array(weights)
            weights_normalized = weights_array / weights_array.sum()
            
            fear_greed_score = np.average(scores, weights=weights_normalized)
        else:
            fear_greed_score = 50  # Neutral
        
        # Interpret score
        interpretation = self._interpret_fear_greed(fear_greed_score)
        
        return {
            'score': fear_greed_score,
            'interpretation': interpretation,
            'components': {
                'momentum': scores[0] if len(scores) > 0 else None,
                'volatility': scores[1] if len(scores) > 1 else None,
                'breadth': scores[2] if len(scores) > 2 else None,
                'safe_haven': scores[3] if len(scores) > 3 else None,
                'high_yield': scores[4] if len(scores) > 4 else None
            }
        }
    
    def _normalize_momentum(self, returns: float) -> float:
        """Normalize momentum to 0-100 scale."""
        # Transform returns to 0-100 scale
        # Assume -20% to +20% maps to 0-100
        normalized = 50 + (returns / 0.4) * 50
        return np.clip(normalized, 0, 100)
    
    def _normalize_vix(self, current_vix: float, avg_vix: float) -> float:
        """Normalize VIX to 0-100 scale (inverted - low VIX = greed)."""
        # VIX below average = greed (higher score)
        ratio = current_vix / avg_vix
        normalized = 100 - (ratio - 0.5) * 100
        return np.clip(normalized, 0, 100)
    
    def _normalize_breadth(self, trend: float) -> float:
        """Normalize breadth trend to 0-100 scale."""
        # Positive trend = greed
        normalized = 50 + trend * 50
        return np.clip(normalized, 0, 100)
    
    def _normalize_safe_haven(self, spread: float) -> float:
        """Normalize safe haven demand to 0-100 scale."""
        # Bonds outperforming = fear (lower score)
        normalized = 50 - spread * 500
        return np.clip(normalized, 0, 100)
    
    def _normalize_high_yield(self, spread: float) -> float:
        """Normalize high yield performance to 0-100 scale."""
        # HY outperforming = greed (higher score)
        normalized = 50 + spread * 500
        return np.clip(normalized, 0, 100)
    
    def _interpret_fear_greed(self, score: float) -> str:
        """Interpret Fear & Greed score."""
        if score >= 75:
            return 'Extreme Greed'
        elif score >= 55:
            return 'Greed'
        elif score >= 45:
            return 'Neutral'
        elif score >= 25:
            return 'Fear'
        else:
            return 'Extreme Fear'
    
    def fear_greed_history(self, 
                          lookback_days: int = 90) -> pd.DataFrame:
        """
        Calculate historical Fear & Greed Index.
        
        Args:
            lookback_days: Number of days to look back
        
        Returns:
            DataFrame with daily Fear & Greed scores
        """
        # This would require calculating the index for each historical day
        # Simplified version just returns current
        current = self.calculate_index()
        
        return pd.DataFrame([{
            'date': datetime.now(),
            'score': current['score'],
            'interpretation': current['interpretation']
        }])
