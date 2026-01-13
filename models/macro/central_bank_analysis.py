"""
Central Bank and Monetary Policy Analysis
Fed, ECB, BOJ, BOE policy tracking and analysis
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


class CentralBankTracker:
    """
    Track central bank policies and decisions.
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize central bank tracker.
        
        Args:
            fred_api_key: FRED API key for US data
        """
        self.fred_api_key = fred_api_key
        self.fred = None
        
        if fred_api_key and fredapi:
            try:
                self.fred = fredapi.Fred(api_key=fred_api_key)
            except:
                pass
        
        self._setup_cb_data()
    
    def _setup_cb_data(self):
        """Setup central bank reference data."""
        self.central_banks = {
            'fed': {
                'name': 'Federal Reserve (US)',
                'fred_rate': 'FEDFUNDS',
                'next_meeting': None,
                'current_stance': 'Neutral',
                'inflation_target': 2.0
            },
            'ecb': {
                'name': 'European Central Bank',
                'fred_rate': 'IRLTLT01EZM156N',
                'next_meeting': None,
                'current_stance': 'Neutral',
                'inflation_target': 2.0
            },
            'boj': {
                'name': 'Bank of Japan',
                'current_stance': 'Easing',
                'inflation_target': 2.0
            },
            'boe': {
                'name': 'Bank of England',
                'current_stance': 'Neutral',
                'inflation_target': 2.0
            }
        }
    
    def get_fed_funds_rate(self) -> Dict:
        """
        Get current Federal Funds Rate.
        
        Returns:
            Dictionary with Fed Funds data
        """
        if not self.fred:
            return {'error': 'FRED API not available'}
        
        try:
            ffr = self.fred.get_series('FEDFUNDS')
            
            current = ffr.iloc[-1]
            previous = ffr.iloc[-2] if len(ffr) > 1 else None
            
            return {
                'rate': current,
                'previous': previous,
                'change': current - previous if previous else None,
                'date': ffr.index[-1],
                'data_points': len(ffr)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_fed_communications(self) -> Dict:
        """
        Analyze Fed's policy guidance and communications.
        
        Returns:
            Dictionary with Fed outlook
        """
        # In production, would parse FOMC statements, minutes, speeches
        # For now, return framework
        
        return {
            'fomc_statement_date': datetime.now().strftime('%Y-%m-%d'),
            'projected_rates': {
                'current': 4.33,  # Placeholder
                'end_of_year': 3.75,
                'next_year': 3.50
            },
            'dot_plot_summary': 'Market expects 2 rate cuts in 2025',
            'inflation_projections': {
                'current': 2.8,
                'end_of_year': 2.5
            },
            'unemployment_projections': {
                'current': 4.2,
                'end_of_year': 4.0
            },
            'gdp_projections': {
                'current': 2.1,
                'end_of_year': 2.0
            },
            'policy_stance': 'Restrictive - but data dependent',
            'next_meeting': '2026-01-28'
        }
    
    def rate_expectations(self) -> Dict:
        """
        Analyze market expectations for rate changes.
        
        Returns:
            Dictionary with rate expectations
        """
        # Would integrate with Fed Funds futures data
        
        return {
            'current_rate': 4.33,
            'market_expectations': {
                'next_meeting': {
                    'hold': 0.75,
                    'cut_25bp': 0.15,
                    'cut_50bp': 0.05,
                    'hike_25bp': 0.05
                },
                'end_of_2026': 3.75,
                'end_of_2027': 3.50
            },
            'probability_distribution': {
                'cuts': 0.65,
                'hold': 0.20,
                'hikes': 0.15
            },
            'implied_path': 'Gradual easing throughout 2025-2026'
        }
    
    def yield_curve_implications(self) -> Dict:
        """
        Analyze yield curve implications for policy.
        
        Returns:
            Dictionary with yield curve analysis
        """
        if not self.fred:
            return {'error': 'FRED API not available'}
        
        try:
            # Get key yield curve points
            y2y = self.fred.get_series('DGS2')
            y5y = self.fred.get_series('DGS5')
            y10y = self.fred.get_series('DGS10')
            
            # Calculate spreads
            y5y2y = y5y.iloc[-1] - y2y.iloc[-1]
            y10y2y = y10y.iloc[-1] - y2y.iloc[-1]
            y10y5y = y10y.iloc[-1] - y5y.iloc[-1]
            
            # Determine curve shape
            if y10y2y < 0:
                shape = 'Inverted'
                implication = 'Recession risk in 12-18 months'
            elif y10y2y < 0.25:
                shape = 'Flat'
                implication = 'Transition period, uncertainty elevated'
            else:
                shape = 'Steep'
                implication = 'Normal expansion, growth expected'
            
            return {
                'current_rates': {
                    '2Y': y2y.iloc[-1],
                    '5Y': y5y.iloc[-1],
                    '10Y': y10y.iloc[-1]
                },
                'spreads': {
                    '5Y-2Y': y5y2y,
                    '10Y-2Y': y10y2y,
                    '10Y-5Y': y10y5y
                },
                'curve_shape': shape,
                'policy_implication': implication
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def compare_central_banks(self) -> Dict:
        """
        Compare policy stances across major central banks.
        
        Returns:
            Dictionary with CB comparison
        """
        comparison = {
            'fed': {
                'rate': 4.33,
                'trend': 'Holding/Easing',
                'inflation': 2.8,
                'growth': 2.1,
                'next_action': 'Likely hold',
                'timeline': 'Next cut possible mid-2025'
            },
            'ecb': {
                'rate': 3.50,
                'trend': 'Easing',
                'inflation': 2.1,
                'growth': 1.2,
                'next_action': 'Further cuts likely',
                'timeline': 'Ongoing easing cycle'
            },
            'boj': {
                'rate': 0.50,
                'trend': 'Tightening from ultra-low',
                'inflation': 2.2,
                'growth': 1.2,
                'next_action': 'Further normalization',
                'timeline': 'Gradual increases'
            },
            'boe': {
                'rate': 4.75,
                'trend': 'Steady',
                'inflation': 2.2,
                'growth': 0.9,
                'next_action': 'Cuts possible',
                'timeline': 'Depends on inflation data'
            }
        }
        
        return {
            'comparison_date': datetime.now().strftime('%Y-%m-%d'),
            'central_banks': comparison,
            'consensus': 'Global trend toward easing in 2025-2026',
            'divergence_risks': [
                'Fed might hold longer if inflation sticky',
                'ECB may accelerate cuts if growth slows',
                'BOJ normalization could be faster than expected'
            ]
        }


class PolicyAnalysis:
    """
    Analyze policy transmission to markets and economy.
    """
    
    def __init__(self):
        """Initialize policy analysis."""
        pass
    
    def transmission_mechanisms(self, 
                               policy_change: str,
                               magnitude: str = 'standard') -> Dict:
        """
        Analyze policy transmission to markets.
        
        Args:
            policy_change: Type of change ('rate_cut', 'rate_hike', 'qe', 'qt')
            magnitude: Magnitude of change ('large', 'standard', 'small')
        
        Returns:
            Dictionary with transmission analysis
        """
        transmission_paths = {
            'rate_cut': {
                'immediate': {
                    'short_term_rates': 'Down',
                    'credit_spreads': 'Tighter',
                    'dollar': 'Weaker'
                },
                'financial_conditions': {
                    'stocks': 'Up',
                    'bonds': 'Up (duration benefit)',
                    'credit': 'Tighter spreads'
                },
                'real_economy_lag': {
                    'timeline': '6-12 months',
                    'housing': 'Improving demand',
                    'credit_growth': 'Accelerating'
                }
            },
            'rate_hike': {
                'immediate': {
                    'short_term_rates': 'Up',
                    'credit_spreads': 'Wider',
                    'dollar': 'Stronger'
                },
                'financial_conditions': {
                    'stocks': 'Down (short-term)',
                    'bonds': 'Down (duration loss)',
                    'credit': 'Wider spreads'
                },
                'real_economy_lag': {
                    'timeline': '12-18 months',
                    'housing': 'Declining demand',
                    'credit_growth': 'Slowing'
                }
            },
            'qe': {
                'immediate': {
                    'long_term_rates': 'Down',
                    'credit_spreads': 'Tighter',
                    'asset_prices': 'Up'
                },
                'financial_conditions': {
                    'stocks': 'Up (portfolio rebalancing)',
                    'bonds': 'Up',
                    'commodities': 'Up (weaker currency)'
                },
                'real_economy_lag': {
                    'timeline': '6-9 months',
                    'wealth_effect': 'Positive',
                    'credit_conditions': 'Improving'
                }
            },
            'qt': {
                'immediate': {
                    'long_term_rates': 'Up',
                    'credit_spreads': 'Wider',
                    'asset_prices': 'Down'
                },
                'financial_conditions': {
                    'stocks': 'Down',
                    'bonds': 'Down',
                    'commodities': 'Down'
                },
                'real_economy_lag': {
                    'timeline': '9-12 months',
                    'wealth_effect': 'Negative',
                    'credit_conditions': 'Tightening'
                }
            }
        }
        
        if policy_change not in transmission_paths:
            return {'error': f'Unknown policy change: {policy_change}'}
        
        paths = transmission_paths[policy_change]
        
        # Adjust for magnitude
        magnitude_multipliers = {
            'large': 1.5,
            'standard': 1.0,
            'small': 0.5
        }
        multiplier = magnitude_multipliers.get(magnitude, 1.0)
        
        return {
            'policy_change': policy_change,
            'magnitude': magnitude,
            'transmission_paths': paths,
            'impact_multiplier': multiplier,
            'key_market_impacts': self._summarize_impacts(paths),
            'timeline_to_full_effect': paths.get('real_economy_lag', {}).get('timeline', 'Unknown')
        }
    
    def _summarize_impacts(self, transmission_paths: Dict) -> Dict:
        """Summarize key market impacts."""
        return {
            'financial_markets': transmission_paths.get('financial_conditions', {}),
            'real_economy': transmission_paths.get('real_economy_lag', {})
        }
    
    def policy_reaction_function(self,
                                economic_data: Dict) -> Dict:
        """
        Estimate central bank reaction function.
        
        Args:
            economic_data: Dictionary with current economic data
        
        Returns:
            Dictionary with estimated policy reaction
        """
        # Simplified Taylor Rule approximation
        inflation_gap = economic_data.get('inflation', 2.8) - 2.0
        output_gap = economic_data.get('gdp_growth', 2.1) - 2.5
        
        # Taylor Rule: r* + inflation + 0.5(inflation_gap) + 0.5(output_gap)
        # Where r* = 2.5 (natural rate)
        r_star = 2.5
        inflation = economic_data.get('inflation', 2.8)
        
        implied_rate = r_star + inflation + 0.5 * inflation_gap + 0.5 * output_gap
        
        # Current rate
        current_rate = economic_data.get('current_rate', 4.33)
        
        # Policy gap
        policy_gap = current_rate - implied_rate
        
        # Determine implied action
        if policy_gap > 0.5:
            implied_action = 'Rate cuts likely (too restrictive)'
        elif policy_gap > 0.25:
            implied_action = 'Rate cuts possible'
        elif policy_gap < -0.5:
            implied_action = 'Rate hikes possible (too loose)'
        else:
            implied_action = 'Current stance appropriate'
        
        return {
            'natural_rate': r_star,
            'inflation_target': 2.0,
            'implied_rate_taylor_rule': implied_rate,
            'current_rate': current_rate,
            'policy_gap': policy_gap,
            'implied_action': implied_action,
            'inflation_gap': inflation_gap,
            'output_gap': output_gap
        }
    
    def monetary_conditions_index(self,
                                 rate_change: float,
                                 currency_change: float,
                                 credit_spread_change: float) -> Dict:
        """
        Calculate Monetary Conditions Index.
        
        Args:
            rate_change: Change in short rates (basis points)
            currency_change: Change in currency index (%)
            credit_spread_change: Change in credit spreads (basis points)
        
        Returns:
            Dictionary with MCI
        """
        # Weighted combination of monetary indicators
        # Weights are approximate
        
        # Rate component (negative rate changes = easing)
        rate_component = -rate_change / 50  # Normalize: 50bp = -1
        
        # Currency component (stronger currency = tightening)
        currency_component = currency_change / 5  # Normalize: 5% = 1
        
        # Credit component (wider spreads = tightening)
        credit_component = credit_spread_change / 100  # Normalize: 100bp = 1
        
        # Calculate MCI (weighted average)
        mci = 0.5 * rate_component + 0.25 * currency_component + 0.25 * credit_component
        
        # Interpret
        if mci > 0.5:
            interpretation = 'Restrictive conditions'
        elif mci > 0.1:
            interpretation = 'Modestly restrictive'
        elif mci < -0.5:
            interpretation = 'Accommodative conditions'
        elif mci < -0.1:
            interpretation = 'Modestly accommodative'
        else:
            interpretation = 'Neutral conditions'
        
        return {
            'monetary_conditions_index': mci,
            'interpretation': interpretation,
            'components': {
                'rate_component': rate_component,
                'currency_component': currency_component,
                'credit_component': credit_component
            },
            'recent_trend': 'Easing' if mci < 0 else 'Tightening' if mci > 0 else 'Stable'
        }
