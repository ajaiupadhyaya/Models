"""
Institutional-Grade DCF Valuation
Enhanced with Monte Carlo simulation, scenario analysis, and proper WACC calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from models.valuation.dcf_model import DCFModel


class InstitutionalDCF(DCFModel):
    """
    Institutional-grade DCF with Monte Carlo simulation and sensitivity analysis.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize institutional DCF."""
        super().__init__(*args, **kwargs)
        self.monte_carlo_results = None
    
    def monte_carlo_valuation(self,
                             n_simulations: int = 10000,
                             wacc_std: float = 0.02,
                             growth_std: float = 0.01,
                             fcf_volatility: float = 0.15) -> Dict[str, Any]:
        """
        Monte Carlo simulation for DCF valuation.
        
        Args:
            n_simulations: Number of simulations
            wacc_std: Standard deviation of WACC
            growth_std: Standard deviation of terminal growth
            fcf_volatility: Volatility of FCF projections
        
        Returns:
            Monte Carlo results
        """
        valuations = []
        
        for _ in range(n_simulations):
            # Sample WACC
            wacc_sample = np.clip(
                np.random.normal(self.wacc, wacc_std),
                0.01, 0.30
            )
            
            # Sample terminal growth
            growth_sample = np.clip(
                np.random.normal(self.terminal_growth_rate, growth_std),
                -0.05, wacc_sample - 0.01
            )
            
            # Sample FCF with volatility
            fcf_sample = self.free_cash_flows.copy()
            for i in range(1, len(fcf_sample)):
                fcf_sample[i] = fcf_sample[i-1] * np.exp(
                    np.random.normal(0, fcf_volatility)
                )
            
            # Calculate valuation
            try:
                temp_wacc = self.wacc
                temp_growth = self.terminal_growth_rate
                temp_fcf = self.free_cash_flows
                
                self.wacc = wacc_sample
                self.terminal_growth_rate = growth_sample
                self.free_cash_flows = fcf_sample
                
                ev = self.calculate_enterprise_value()
                valuations.append(ev)
                
                # Restore
                self.wacc = temp_wacc
                self.terminal_growth_rate = temp_growth
                self.free_cash_flows = temp_fcf
            except:
                continue
        
        valuations = np.array(valuations)
        
        self.monte_carlo_results = {
            'mean_valuation': float(np.mean(valuations)),
            'median_valuation': float(np.median(valuations)),
            'std_valuation': float(np.std(valuations)),
            'percentile_5': float(np.percentile(valuations, 5)),
            'percentile_95': float(np.percentile(valuations, 95)),
            'confidence_interval_90': (
                float(np.percentile(valuations, 5)),
                float(np.percentile(valuations, 95))
            )
        }
        
        return self.monte_carlo_results
    
    def calculate_wacc(self,
                      equity_value: float,
                      debt_value: float,
                      cost_of_equity: float,
                      cost_of_debt: float,
                      tax_rate: float = 0.21) -> float:
        """
        Calculate Weighted Average Cost of Capital (WACC).
        
        Args:
            equity_value: Market value of equity
            debt_value: Market value of debt
            cost_of_equity: Cost of equity (CAPM)
            cost_of_debt: Cost of debt
            tax_rate: Corporate tax rate
        
        Returns:
            WACC
        """
        total_value = equity_value + debt_value
        
        if total_value == 0:
            return 0.10  # Default
        
        equity_weight = equity_value / total_value
        debt_weight = debt_value / total_value
        
        wacc = (equity_weight * cost_of_equity + 
                debt_weight * cost_of_debt * (1 - tax_rate))
        
        return wacc
    
    def calculate_cost_of_equity_capm(self,
                                    risk_free_rate: float,
                                    beta: float,
                                    market_risk_premium: float = 0.06) -> float:
        """
        Calculate cost of equity using CAPM.
        
        Args:
            risk_free_rate: Risk-free rate
            beta: Stock beta
            market_risk_premium: Market risk premium
        
        Returns:
            Cost of equity
        """
        return risk_free_rate + beta * market_risk_premium
    
    def scenario_analysis(self,
                         scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
        """
        Scenario analysis (base case, bull case, bear case).
        
        Args:
            scenarios: Dictionary of scenarios with parameters
        
        Returns:
            Scenario valuations
        """
        results = {}
        base_wacc = self.wacc
        base_growth = self.terminal_growth_rate
        
        for scenario_name, params in scenarios.items():
            self.wacc = params.get('wacc', base_wacc)
            self.terminal_growth_rate = params.get('growth', base_growth)
            
            if 'fcf_multiplier' in params:
                self.free_cash_flows = self.free_cash_flows * params['fcf_multiplier']
            
            try:
                ev = self.calculate_enterprise_value()
                results[scenario_name] = {
                    'enterprise_value': float(ev),
                    'wacc': float(self.wacc),
                    'terminal_growth': float(self.terminal_growth_rate)
                }
            except:
                results[scenario_name] = {'error': 'Invalid parameters'}
            
            # Restore
            self.wacc = base_wacc
            self.terminal_growth_rate = base_growth
        
        return results
