"""
Stress Testing Framework
Historical scenario replay and hypothetical stress testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class StressScenario:
    """
    Represents a stress scenario for portfolio testing.
    """
    
    def __init__(self,
                 scenario_id: str,
                 name: str,
                 description: str,
                 date: Optional[datetime] = None):
        """
        Initialize stress scenario.
        
        Args:
            scenario_id: Unique scenario identifier
            name: Scenario name
            description: Scenario description
            date: Date scenario occurred
        """
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.date = date or datetime.now()
        self.shocks = {}  # {asset: return_shock}
    
    def add_shock(self, asset: str, shock: float):
        """
        Add asset shock to scenario.
        
        Args:
            asset: Asset symbol
            shock: Return shock (e.g., -0.20 for -20%)
        """
        self.shocks[asset] = shock
    
    def to_dict(self) -> Dict:
        """Convert scenario to dictionary."""
        return {
            'scenario_id': self.scenario_id,
            'name': self.name,
            'description': self.description,
            'date': self.date,
            'shocks': self.shocks
        }


class HistoricalScenarioAnalyzer:
    """
    Replay historical market scenarios.
    """
    
    # Historical scenarios
    SCENARIOS = {
        'black_monday_1987': {
            'name': 'Black Monday 1987',
            'description': 'October 19, 1987 crash',
            'shocks': {
                'SPY': -0.2206,
                'QQQ': -0.1900,
                'EEM': -0.2100,
                'TLT': 0.0200,
                'GLD': 0.0150,
                'DXY': 0.0100
            }
        },
        'asian_crisis_1997': {
            'name': 'Asian Financial Crisis 1997',
            'description': 'July 1997 Thai Baht collapse',
            'shocks': {
                'SPY': -0.1900,
                'EEM': -0.3500,
                'QQQ': -0.1200,
                'TLT': 0.0500,
                'GLD': 0.0800,
                'DXY': 0.0300
            }
        },
        'dot_com_bubble_2000': {
            'name': 'Dot-com Bubble 2000',
            'description': 'March 2000 tech crash',
            'shocks': {
                'QQQ': -0.3900,
                'SPY': -0.0900,
                'IWM': -0.1500,
                'TLT': 0.1200,
                'GLD': 0.0500,
                'DXY': -0.0200
            }
        },
        'sept_11_2001': {
            'name': '9/11 Attacks 2001',
            'description': 'September 11, 2001 terrorist attacks',
            'shocks': {
                'SPY': -0.1140,
                'QQQ': -0.1600,
                'IWM': -0.1200,
                'TLT': 0.0800,
                'GLD': 0.0300,
                'DXY': 0.0250
            }
        },
        'financial_crisis_2008': {
            'name': 'Financial Crisis 2008',
            'description': 'September 2008 Lehman collapse',
            'shocks': {
                'SPY': -0.0900,  # Single worst day
                'QQQ': -0.0950,
                'IWM': -0.0900,
                'HYG': -0.0800,
                'EEM': -0.1100,
                'TLT': 0.0300,
                'GLD': 0.0400,
                'USO': -0.1100,
                'DXY': 0.0200
            }
        },
        'flash_crash_2010': {
            'name': 'Flash Crash 2010',
            'description': 'May 6, 2010 rapid decline',
            'shocks': {
                'SPY': -0.0960,
                'QQQ': -0.0920,
                'IWM': -0.1000,
                'TLT': 0.0200,
                'GLD': 0.0050
            }
        },
        'european_crisis_2011': {
            'name': 'European Debt Crisis 2011',
            'description': 'August 2011 sovereign debt fears',
            'shocks': {
                'SPY': -0.1995,
                'EWG': -0.1600,
                'EWI': -0.1400,
                'SCHF': -0.1800,
                'TLT': 0.0600,
                'GLD': 0.1000,
                'DXY': 0.0100
            }
        },
        'taper_tantrum_2013': {
            'name': 'Taper Tantrum 2013',
            'description': 'May-June 2013 Fed taper expectations',
            'shocks': {
                'TLT': -0.0600,
                'EMB': -0.0450,
                'EEM': -0.0800,
                'QQQ': -0.0200,
                'SPY': -0.0100,
                'USO': -0.0500
            }
        },
        'vix_spike_2015': {
            'name': 'VIX Spike 2015',
            'description': 'August 2015 China devaluation',
            'shocks': {
                'SPY': -0.0364,
                'QQQ': -0.0550,
                'EEM': -0.0850,
                'EWZ': -0.1200,
                'TLT': 0.0250,
                'GLD': 0.0300,
                'USO': -0.0450
            }
        },
        'covid_crash_2020': {
            'name': 'COVID-19 Crash 2020',
            'description': 'March 2020 pandemic panic',
            'shocks': {
                'SPY': -0.1200,
                'QQQ': -0.0970,
                'IWM': -0.1800,
                'EEM': -0.1400,
                'HYG': -0.1500,
                'USO': -0.2065,
                'GLD': 0.0170,
                'TLT': 0.0280,
                'DXY': 0.0340
            }
        },
        'vix_volatility_2022': {
            'name': 'VIX Volatility 2022',
            'description': 'June 2022 Fed tightening fears',
            'shocks': {
                'SPY': -0.0849,
                'QQQ': -0.0849,
                'TLT': -0.0500,
                'VCIT': -0.0350,
                'EEM': -0.0700,
                'GLD': 0.0050,
                'USO': 0.0300
            }
        }
    }
    
    @staticmethod
    def get_scenario(scenario_key: str) -> StressScenario:
        """
        Get historical scenario.
        
        Args:
            scenario_key: Scenario key from SCENARIOS
        
        Returns:
            StressScenario instance
        """
        if scenario_key not in HistoricalScenarioAnalyzer.SCENARIOS:
            return None
        
        scenario_data = HistoricalScenarioAnalyzer.SCENARIOS[scenario_key]
        scenario = StressScenario(
            scenario_id=scenario_key,
            name=scenario_data['name'],
            description=scenario_data['description']
        )
        
        for asset, shock in scenario_data['shocks'].items():
            scenario.add_shock(asset, shock)
        
        return scenario
    
    @staticmethod
    def list_scenarios() -> List[Dict]:
        """List all available historical scenarios."""
        return [
            {
                'id': key,
                'name': data['name'],
                'description': data['description']
            }
            for key, data in HistoricalScenarioAnalyzer.SCENARIOS.items()
        ]


class HypotheticalScenarioBuilder:
    """
    Build custom hypothetical stress scenarios.
    """
    
    @staticmethod
    def rate_shock_scenario(shock_bps: float) -> StressScenario:
        """
        Create interest rate shock scenario.
        
        Args:
            shock_bps: Shock in basis points (e.g., 50 for +50bps)
        
        Returns:
            StressScenario
        """
        scenario = StressScenario(
            scenario_id=f'rate_shock_{shock_bps}bps',
            name=f'Rate Shock: +{shock_bps}bps',
            description=f'Interest rates increase by {shock_bps} basis points'
        )
        
        # Inverse duration impact
        scenario.add_shock('TLT', -shock_bps * 0.0001 * 15)  # ~15 year duration
        scenario.add_shock('IEF', -shock_bps * 0.0001 * 7)   # ~7 year duration
        scenario.add_shock('SHY', -shock_bps * 0.0001 * 1.5) # ~1.5 year duration
        
        # Equity impact (moderate negative)
        scenario.add_shock('SPY', -shock_bps * 0.0001 * 2)
        scenario.add_shock('QQQ', -shock_bps * 0.0001 * 2.5)
        
        # Flight to quality
        scenario.add_shock('GLD', shock_bps * 0.00005)
        
        return scenario
    
    @staticmethod
    def credit_spread_shock(shock_bps: float) -> StressScenario:
        """
        Create credit spread widening scenario.
        
        Args:
            shock_bps: Spread widening in basis points
        
        Returns:
            StressScenario
        """
        scenario = StressScenario(
            scenario_id=f'credit_shock_{shock_bps}bps',
            name=f'Credit Spread Widening: +{shock_bps}bps',
            description=f'Credit spreads widen by {shock_bps} basis points'
        )
        
        # Corporate bond impact
        scenario.add_shock('LQD', -shock_bps * 0.0001 * 5)   # IG bonds
        scenario.add_shock('HYG', -shock_bps * 0.0001 * 3)   # High yield
        scenario.add_shock('ANGL', -shock_bps * 0.0001 * 4)  # Fallen angels
        
        # Equity impact (worst hit on financials)
        scenario.add_shock('SPY', -shock_bps * 0.0001 * 1.5)
        scenario.add_shock('XLF', -shock_bps * 0.0001 * 2.5)
        
        # Safe havens
        scenario.add_shock('TLT', shock_bps * 0.00008)
        scenario.add_shock('GLD', shock_bps * 0.0001)
        
        return scenario
    
    @staticmethod
    def equity_crash_scenario(crash_pct: float) -> StressScenario:
        """
        Create equity crash scenario.
        
        Args:
            crash_pct: Equity crash percentage (e.g., -0.20 for -20%)
        
        Returns:
            StressScenario
        """
        scenario = StressScenario(
            scenario_id=f'equity_crash_{crash_pct}',
            name=f'Equity Crash: {crash_pct*100:.1f}%',
            description=f'All equities decline by {crash_pct*100:.1f}%'
        )
        
        # Equity shocks (smaller caps hit harder)
        scenario.add_shock('SPY', crash_pct)
        scenario.add_shock('QQQ', crash_pct * 1.2)
        scenario.add_shock('IWM', crash_pct * 1.3)
        scenario.add_shock('EEM', crash_pct * 1.2)
        
        # Sector differentiation
        scenario.add_shock('XLV', crash_pct * 0.8)  # Healthcare defensive
        scenario.add_shock('XLK', crash_pct * 1.0)  # Tech
        scenario.add_shock('XLF', crash_pct * 1.3)  # Financials
        scenario.add_shock('XLE', crash_pct * 1.2)  # Energy
        
        # Safe havens rally
        scenario.add_shock('TLT', -crash_pct * 4)   # Inverse equity correlation
        scenario.add_shock('GLD', -crash_pct * 2)
        scenario.add_shock('DXY', -crash_pct * 2)
        
        return scenario
    
    @staticmethod
    def currency_crisis_scenario(usd_strength: float) -> StressScenario:
        """
        Create currency crisis scenario.
        
        Args:
            usd_strength: USD appreciation (e.g., 0.10 for +10%)
        
        Returns:
            StressScenario
        """
        scenario = StressScenario(
            scenario_id=f'currency_crisis_{usd_strength}',
            name=f'Currency Crisis: USD +{usd_strength*100:.1f}%',
            description=f'US Dollar strengthens by {usd_strength*100:.1f}%, emerging markets pressured'
        )
        
        # Emerging market impact
        scenario.add_shock('EEM', -usd_strength * 3)
        scenario.add_shock('EWZ', -usd_strength * 3.5)  # Brazil
        scenario.add_shock('FXI', -usd_strength * 2.5)  # China
        scenario.add_shock('EWZ', -usd_strength * 3.5)  # Brazil
        
        # Commodity impact
        scenario.add_shock('DBC', -usd_strength * 2)  # Commodities
        scenario.add_shock('USO', -usd_strength * 2.5)  # Oil
        scenario.add_shock('GLD', -usd_strength * 1.5)  # Gold
        
        # Developed market benefit
        scenario.add_shock('SPY', usd_strength * 0.5)
        scenario.add_shock('EWJ', usd_strength * 2)  # Japan
        
        return scenario
    
    @staticmethod
    def volatility_spike_scenario(vix_level: float) -> StressScenario:
        """
        Create VIX spike scenario.
        
        Args:
            vix_level: Target VIX level (e.g., 40)
        
        Returns:
            StressScenario
        """
        # Assume normal VIX ~15, calculate shock
        normal_vix = 15
        vix_shock = (vix_level - normal_vix) / normal_vix
        
        scenario = StressScenario(
            scenario_id=f'vix_spike_{vix_level}',
            name=f'VIX Spike to {vix_level}',
            description=f'Volatility spike with VIX reaching {vix_level}'
        )
        
        # Equity impact correlates with VIX
        equity_shock = -vix_shock * 0.15
        scenario.add_shock('SPY', equity_shock)
        scenario.add_shock('QQQ', equity_shock * 1.2)
        scenario.add_shock('IWM', equity_shock * 1.3)
        
        # Volatility products
        scenario.add_shock('VXX', vix_shock * 2)
        
        # Safe havens
        scenario.add_shock('TLT', -equity_shock * 2)
        scenario.add_shock('GLD', -equity_shock * 1.5)
        
        return scenario


class PortfolioStressTester:
    """
    Test portfolio under stress scenarios.
    """
    
    def __init__(self, portfolio: Dict[str, float]):
        """
        Initialize stress tester.
        
        Args:
            portfolio: {asset: weight} dictionary
        """
        self.portfolio = portfolio
        self.total_value = sum(portfolio.values())
        self.normalized_weights = {
            asset: weight/self.total_value 
            for asset, weight in portfolio.items()
        }
    
    def apply_scenario(self, scenario: StressScenario) -> Dict:
        """
        Apply scenario to portfolio.
        
        Args:
            scenario: StressScenario to apply
        
        Returns:
            Results dictionary
        """
        results = {
            'scenario': scenario.name,
            'scenario_id': scenario.scenario_id,
            'portfolio_return': 0.0,
            'asset_impacts': {},
            'max_loss': float('-inf'),
            'max_gain': float('inf'),
            'worst_position': None,
            'best_position': None
        }
        
        portfolio_return = 0.0
        
        for asset, weight in self.normalized_weights.items():
            # Get shock for asset (or 0 if not in scenario)
            shock = scenario.shocks.get(asset, 0.0)
            
            # Portfolio contribution
            position_impact = shock * weight
            portfolio_return += position_impact
            
            results['asset_impacts'][asset] = {
                'weight': weight,
                'shock': shock,
                'impact': position_impact,
                'position_value': self.portfolio[asset],
                'loss': self.portfolio[asset] * shock
            }
            
            # Track extremes
            if position_impact < results['max_loss']:
                results['max_loss'] = position_impact
                results['worst_position'] = asset
            
            if position_impact > results['max_gain']:
                results['max_gain'] = position_impact
                results['best_position'] = asset
        
        results['portfolio_return'] = portfolio_return
        results['portfolio_loss'] = self.total_value * portfolio_return
        
        return results
    
    def stress_test_historical(self) -> List[Dict]:
        """
        Run portfolio against all historical scenarios.
        
        Returns:
            List of results for each scenario
        """
        results = []
        
        for scenario_key in HistoricalScenarioAnalyzer.SCENARIOS.keys():
            scenario = HistoricalScenarioAnalyzer.get_scenario(scenario_key)
            result = self.apply_scenario(scenario)
            results.append(result)
        
        return results
    
    def rank_scenarios_by_loss(self, results: List[Dict]) -> List[Dict]:
        """
        Rank scenarios by portfolio loss.
        
        Args:
            results: Results from stress_test_historical
        
        Returns:
            Ranked results
        """
        ranked = sorted(results, key=lambda x: x['portfolio_return'])
        return ranked
    
    def summary_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate summary statistics from stress tests.
        
        Args:
            results: Results from stress tests
        
        Returns:
            Summary statistics
        """
        returns = [r['portfolio_return'] for r in results]
        losses = [r['portfolio_loss'] for r in results]
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'total_scenarios': len(results),
            'scenarios_with_loss': sum(1 for r in returns if r < 0),
            'mean_loss': np.mean([l for l in losses if l < 0]) if any(l < 0 for l in losses) else 0,
            'worst_case_loss': np.min(losses),
            'best_case_gain': np.max(losses)
        }
