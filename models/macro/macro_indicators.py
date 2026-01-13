"""
Macro Economic Indicators and Cycle Analysis
Real-time economic data from FRED API
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import fredapi
except ImportError:
    fredapi = None


class MacroIndicators:
    """
    Retrieve and analyze macroeconomic indicators from FRED.
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize macro indicators analyzer.
        
        Args:
            fred_api_key: FRED API key for data access
        """
        self.fred_api_key = fred_api_key
        self.fred = None
        
        if fred_api_key and fredapi:
            try:
                self.fred = fredapi.Fred(api_key=fred_api_key)
            except Exception as e:
                print(f"Error initializing FRED API: {e}")
    
    # Key FRED Series IDs
    FRED_SERIES = {
        'gdp': 'GDP',  # Gross Domestic Product
        'gdp_growth': 'A191RA1Q225SBEA',  # Real GDP Growth Rate
        'unemployment': 'UNRATE',  # Unemployment Rate
        'inflation_cpi': 'CPIAUCSL',  # Consumer Price Index
        'inflation_pce': 'PCEPI',  # PCE Price Index
        'fed_funds_rate': 'FEDFUNDS',  # Federal Funds Rate
        'treasury_3m': 'DGS3MO',  # 3-Month Treasury
        'treasury_10y': 'DGS10',  # 10-Year Treasury
        'yield_curve': 'T10Y2Y',  # 10Y-2Y Spread
        'ism_manufacturing': 'MMNRNJ',  # ISM Manufacturing PMI
        'ism_services': 'ISMRSL',  # ISM Services PMI
        'unemployment_duration': 'UEMPMIS',  # Median Unemployment Duration
        'jobless_claims': 'ICSA',  # Initial Claims
        'consumer_sentiment': 'UMCSENT',  # University of Michigan Sentiment
        'pce_spending': 'PCE',  # Personal Consumption Expenditures
        'housing_starts': 'HOUST',  # Housing Starts
        'industrial_production': 'INDPRO',  # Industrial Production
        'corp_profit_margin': 'CPROF',  # Corporate Profit Margin
        'consumer_credit': 'REVOLSL',  # Revolving Consumer Credit
        'durable_goods': 'DGOMNSA',  # Durable Goods New Orders
    }
    
    def get_indicator(self, 
                     indicator_key: str,
                     observation_start: Optional[str] = None,
                     observation_end: Optional[str] = None) -> pd.Series:
        """
        Get economic indicator from FRED.
        
        Args:
            indicator_key: Key from FRED_SERIES dictionary
            observation_start: Start date (YYYY-MM-DD)
            observation_end: End date (YYYY-MM-DD)
        
        Returns:
            Pandas Series with indicator data
        """
        if not self.fred:
            print("FRED API not available")
            return pd.Series()
        
        try:
            series_id = self.FRED_SERIES.get(indicator_key)
            if not series_id:
                print(f"Unknown indicator: {indicator_key}")
                return pd.Series()
            
            data = self.fred.get_series(
                series_id,
                observation_start=observation_start,
                observation_end=observation_end
            )
            
            return data
            
        except Exception as e:
            print(f"Error fetching {indicator_key}: {e}")
            return pd.Series()
    
    def get_dashboard(self, lookback_months: int = 24) -> Dict:
        """
        Get comprehensive macro dashboard.
        
        Args:
            lookback_months: Number of months to look back
        
        Returns:
            Dictionary with key macro indicators
        """
        if not self.fred:
            return {'error': 'FRED API not available'}
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_months*30)).strftime('%Y-%m-%d')
        
        dashboard = {}
        
        # Growth Indicators
        try:
            gdp = self.get_indicator('gdp_growth', start_date, end_date)
            if not gdp.empty:
                dashboard['gdp_growth_rate'] = {
                    'current': gdp.iloc[-1],
                    'previous': gdp.iloc[-2] if len(gdp) > 1 else None,
                    'change': gdp.iloc[-1] - gdp.iloc[-2] if len(gdp) > 1 else None,
                    'date': gdp.index[-1]
                }
        except:
            pass
        
        # Employment Indicators
        try:
            unemployment = self.get_indicator('unemployment', start_date, end_date)
            if not unemployment.empty:
                dashboard['unemployment_rate'] = {
                    'current': unemployment.iloc[-1],
                    'previous': unemployment.iloc[-2] if len(unemployment) > 1 else None,
                    'change': unemployment.iloc[-1] - unemployment.iloc[-2] if len(unemployment) > 1 else None,
                    'date': unemployment.index[-1]
                }
        except:
            pass
        
        # Inflation Indicators
        try:
            cpi = self.get_indicator('inflation_cpi', start_date, end_date)
            if not cpi.empty:
                cpi_yoy = cpi.pct_change(12) * 100
                dashboard['inflation_cpi_yoy'] = {
                    'current': cpi_yoy.iloc[-1],
                    'previous': cpi_yoy.iloc[-13] if len(cpi_yoy) > 13 else None,
                    'date': cpi.index[-1]
                }
        except:
            pass
        
        # Monetary Policy
        try:
            fed_funds = self.get_indicator('fed_funds_rate', start_date, end_date)
            if not fed_funds.empty:
                dashboard['fed_funds_rate'] = {
                    'current': fed_funds.iloc[-1],
                    'previous': fed_funds.iloc[-2] if len(fed_funds) > 1 else None,
                    'date': fed_funds.index[-1]
                }
        except:
            pass
        
        # Yield Curve
        try:
            yield_curve = self.get_indicator('yield_curve', start_date, end_date)
            if not yield_curve.empty:
                dashboard['yield_curve_10y2y'] = {
                    'current': yield_curve.iloc[-1],
                    'previous': yield_curve.iloc[-2] if len(yield_curve) > 1 else None,
                    'inverted': yield_curve.iloc[-1] < 0,
                    'date': yield_curve.index[-1]
                }
        except:
            pass
        
        # PMI Indicators
        try:
            ism_mfg = self.get_indicator('ism_manufacturing', start_date, end_date)
            if not ism_mfg.empty:
                dashboard['ism_manufacturing_pmi'] = {
                    'current': ism_mfg.iloc[-1],
                    'previous': ism_mfg.iloc[-2] if len(ism_mfg) > 1 else None,
                    'above_50': ism_mfg.iloc[-1] > 50,
                    'date': ism_mfg.index[-1]
                }
        except:
            pass
        
        # Consumer Sentiment
        try:
            sentiment = self.get_indicator('consumer_sentiment', start_date, end_date)
            if not sentiment.empty:
                dashboard['consumer_sentiment'] = {
                    'current': sentiment.iloc[-1],
                    'previous': sentiment.iloc[-2] if len(sentiment) > 1 else None,
                    'change': sentiment.iloc[-1] - sentiment.iloc[-2] if len(sentiment) > 1 else None,
                    'date': sentiment.index[-1]
                }
        except:
            pass
        
        return dashboard
    
    def calculate_macro_trend(self, 
                            indicator_key: str,
                            lookback_periods: int = 12) -> Dict:
        """
        Calculate trend for macro indicator.
        
        Args:
            indicator_key: Key from FRED_SERIES
            lookback_periods: Number of periods to analyze
        
        Returns:
            Dictionary with trend analysis
        """
        data = self.get_indicator(indicator_key)
        
        if data.empty or len(data) < lookback_periods:
            return {'error': 'Insufficient data'}
        
        recent = data.iloc[-lookback_periods:]
        
        # Simple trend: calculate slope
        x = np.arange(len(recent))
        y = recent.values
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Trend direction
        if slope > 0:
            trend = 'Improving'
        elif slope < 0:
            trend = 'Deteriorating'
        else:
            trend = 'Flat'
        
        return {
            'indicator': indicator_key,
            'trend': trend,
            'slope': slope,
            'current_value': recent.iloc[-1],
            'avg_value': recent.mean(),
            'min_value': recent.min(),
            'max_value': recent.max(),
            'volatility': recent.std()
        }


class EconomicCycleForecast:
    """
    Forecast economic cycle phase using leading indicators.
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize economic cycle forecaster.
        
        Args:
            fred_api_key: FRED API key
        """
        self.macro = MacroIndicators(fred_api_key)
    
    # Leading indicators
    LEADING_INDICATORS = [
        'yield_curve',  # 10Y-2Y spread
        'ism_manufacturing',  # PMI > 50 = expansion
        'durable_goods',  # Forward-looking
        'consumer_sentiment',  # Consumer expectations
        'jobless_claims',  # Labor market strength
    ]
    
    # Coincident indicators
    COINCIDENT_INDICATORS = [
        'gdp_growth',
        'unemployment',
        'industrial_production',
        'pce_spending',
    ]
    
    # Lagging indicators
    LAGGING_INDICATORS = [
        'unemployment_duration',
        'corp_profit_margin',
        'inflation_cpi',
    ]
    
    def forecast_cycle_phase(self) -> Dict:
        """
        Forecast current economic cycle phase.
        
        Returns:
            Dictionary with cycle forecast
        """
        scores = []
        indicators_status = {}
        
        # Score leading indicators
        for indicator in self.LEADING_INDICATORS:
            try:
                trend = self.macro.calculate_macro_trend(indicator, lookback_periods=6)
                if 'trend' in trend:
                    score = 1 if trend['trend'] == 'Improving' else -1 if trend['trend'] == 'Deteriorating' else 0
                    scores.append(score)
                    indicators_status[indicator] = {
                        'trend': trend['trend'],
                        'value': trend.get('current_value'),
                        'weight': 0.4
                    }
            except:
                continue
        
        # Score coincident indicators
        for indicator in self.COINCIDENT_INDICATORS:
            try:
                trend = self.macro.calculate_macro_trend(indicator, lookback_periods=6)
                if 'trend' in trend:
                    score = 1 if trend['trend'] == 'Improving' else -1 if trend['trend'] == 'Deteriorating' else 0
                    scores.append(score * 0.7)  # Lower weight
                    indicators_status[indicator] = {
                        'trend': trend['trend'],
                        'value': trend.get('current_value'),
                        'weight': 0.35
                    }
            except:
                continue
        
        # Calculate overall score
        if scores:
            overall_score = np.mean(scores)
        else:
            overall_score = 0
        
        # Determine phase
        if overall_score > 0.3:
            phase = 'Expansion'
        elif overall_score > -0.3:
            phase = 'Transition'
        else:
            phase = 'Contraction'
        
        return {
            'phase': phase,
            'score': overall_score,
            'confidence': len(scores) / (len(self.LEADING_INDICATORS) + len(self.COINCIDENT_INDICATORS)),
            'indicators': indicators_status
        }
    
    def recession_probability(self) -> Dict:
        """
        Estimate recession probability using leading indicators.
        
        Returns:
            Dictionary with recession metrics
        """
        recession_signals = 0
        total_signals = 0
        
        # Yield curve inversion is strong recession signal
        try:
            yield_curve = self.macro.get_indicator('yield_curve')
            if not yield_curve.empty:
                if yield_curve.iloc[-1] < 0:
                    recession_signals += 2  # Strong signal
                total_signals += 2
        except:
            pass
        
        # ISM PMI below 50 signals contraction
        try:
            ism = self.macro.get_indicator('ism_manufacturing')
            if not ism.empty:
                if ism.iloc[-1] < 50:
                    recession_signals += 1
                total_signals += 1
        except:
            pass
        
        # Unemployment rising (higher unemployment rate)
        try:
            unemployment = self.macro.get_indicator('unemployment')
            if not unemployment.empty and len(unemployment) > 3:
                if unemployment.iloc[-1] > unemployment.iloc[-3]:
                    recession_signals += 1
                total_signals += 1
        except:
            pass
        
        # Initial claims rising
        try:
            claims = self.macro.get_indicator('jobless_claims')
            if not claims.empty and len(claims) > 4:
                if claims.iloc[-1] > claims.iloc[-4]:
                    recession_signals += 1
                total_signals += 1
        except:
            pass
        
        # Calculate probability
        if total_signals > 0:
            recession_prob = recession_signals / total_signals
        else:
            recession_prob = 0.5
        
        return {
            'recession_probability': recession_prob,
            'probability_pct': recession_prob * 100,
            'assessment': 'High Risk' if recession_prob > 0.6 else 'Moderate Risk' if recession_prob > 0.4 else 'Low Risk',
            'signals_triggered': recession_signals,
            'total_signals': total_signals
        }
    
    def growth_expectations(self) -> Dict:
        """
        Estimate near-term growth expectations.
        
        Returns:
            Dictionary with growth metrics
        """
        growth_indicators = []
        
        # GDP growth momentum
        try:
            gdp = self.macro.get_indicator('gdp_growth')
            if not gdp.empty:
                recent_growth = gdp.iloc[-1]
                growth_indicators.append(('GDP Growth', recent_growth, 0.3))
        except:
            pass
        
        # PMI (proxy for near-term growth)
        try:
            ism = self.macro.get_indicator('ism_manufacturing')
            if not ism.empty:
                # Normalize PMI to growth proxy (50 = 0%, 60 = 2%, etc)
                growth_proxy = (ism.iloc[-1] - 50) * 0.2
                growth_indicators.append(('ISM PMI Growth Proxy', growth_proxy, 0.2))
        except:
            pass
        
        # Consumer spending (PCE)
        try:
            pce = self.macro.get_indicator('pce_spending')
            if not pce.empty:
                pce_growth = pce.pct_change(12).iloc[-1] * 100
                growth_indicators.append(('PCE Growth', pce_growth, 0.25))
        except:
            pass
        
        # Industrial production
        try:
            ip = self.macro.get_indicator('industrial_production')
            if not ip.empty:
                ip_growth = ip.pct_change(12).iloc[-1] * 100
                growth_indicators.append(('IP Growth', ip_growth, 0.25))
        except:
            pass
        
        # Calculate weighted average
        if growth_indicators:
            weighted_growth = sum(value * weight for _, value, weight in growth_indicators) / sum(weight for _, _, weight in growth_indicators)
        else:
            weighted_growth = 0
        
        return {
            'expected_growth': weighted_growth,
            'growth_pct': weighted_growth,
            'assessment': 'Above Potential' if weighted_growth > 2.5 else 'Below Potential' if weighted_growth < 1.5 else 'Moderate',
            'components': dict((name, value) for name, value, _ in growth_indicators)
        }
