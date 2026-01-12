"""
Discounted Cash Flow (DCF) Valuation Model
Institutional-grade DCF implementation with sensitivity analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class DCFModel:
    """
    Discounted Cash Flow valuation model.
    """
    
    def __init__(self, 
                 free_cash_flows: List[float],
                 terminal_growth_rate: float = 0.03,
                 wacc: float = 0.10,
                 terminal_multiple: Optional[float] = None):
        """
        Initialize DCF model.
        
        Args:
            free_cash_flows: List of projected free cash flows
            terminal_growth_rate: Terminal growth rate (g)
            wacc: Weighted Average Cost of Capital
            terminal_multiple: Optional terminal EV/EBITDA multiple
        """
        self.free_cash_flows = np.array(free_cash_flows)
        self.terminal_growth_rate = terminal_growth_rate
        self.wacc = wacc
        self.terminal_multiple = terminal_multiple
    
    def calculate_terminal_value(self, final_fcf: float) -> float:
        """
        Calculate terminal value using perpetuity method.
        
        Args:
            final_fcf: Final year free cash flow
        
        Returns:
            Terminal value
        """
        if self.terminal_growth_rate >= self.wacc:
            raise ValueError("Terminal growth rate must be less than WACC")
        
        terminal_fcf = final_fcf * (1 + self.terminal_growth_rate)
        terminal_value = terminal_fcf / (self.wacc - self.terminal_growth_rate)
        return terminal_value
    
    def calculate_pv_cash_flows(self) -> float:
        """
        Calculate present value of projected cash flows.
        
        Returns:
            Present value of cash flows
        """
        pv = 0
        for i, fcf in enumerate(self.free_cash_flows, 1):
            pv += fcf / ((1 + self.wacc) ** i)
        return pv
    
    def calculate_enterprise_value(self) -> float:
        """
        Calculate enterprise value.
        
        Returns:
            Enterprise value
        """
        pv_cf = self.calculate_pv_cash_flows()
        terminal_value = self.calculate_terminal_value(self.free_cash_flows[-1])
        pv_terminal = terminal_value / ((1 + self.wacc) ** len(self.free_cash_flows))
        
        return pv_cf + pv_terminal
    
    def calculate_equity_value(self, 
                              enterprise_value: float,
                              cash: float = 0,
                              debt: float = 0,
                              minority_interest: float = 0,
                              preferred_stock: float = 0) -> float:
        """
        Calculate equity value from enterprise value.
        
        Args:
            enterprise_value: Enterprise value
            cash: Cash and cash equivalents
            debt: Total debt
            minority_interest: Minority interest
            preferred_stock: Preferred stock
        
        Returns:
            Equity value
        """
        equity_value = (enterprise_value 
                       + cash 
                       - debt 
                       - minority_interest 
                       - preferred_stock)
        return equity_value
    
    def calculate_share_price(self, 
                              equity_value: float,
                              shares_outstanding: float) -> float:
        """
        Calculate price per share.
        
        Args:
            equity_value: Equity value
            shares_outstanding: Shares outstanding
        
        Returns:
            Price per share
        """
        return equity_value / shares_outstanding
    
    def sensitivity_analysis(self,
                            wacc_range: List[float],
                            growth_range: List[float]) -> pd.DataFrame:
        """
        Perform sensitivity analysis on WACC and terminal growth.
        
        Args:
            wacc_range: Range of WACC values
            growth_range: Range of terminal growth rates
        
        Returns:
            DataFrame with sensitivity matrix
        """
        results = []
        
        for wacc in wacc_range:
            row = []
            for growth in growth_range:
                self.wacc = wacc
                self.terminal_growth_rate = growth
                try:
                    ev = self.calculate_enterprise_value()
                    row.append(ev)
                except:
                    row.append(np.nan)
            results.append(row)
        
        # Reset to original values
        self.wacc = wacc_range[len(wacc_range)//2] if wacc_range else 0.10
        self.terminal_growth_rate = growth_range[len(growth_range)//2] if growth_range else 0.03
        
        return pd.DataFrame(results, index=wacc_range, columns=growth_range)
    
    def get_summary(self,
                   cash: float = 0,
                   debt: float = 0,
                   shares_outstanding: float = 1) -> Dict:
        """
        Get comprehensive valuation summary.
        
        Args:
            cash: Cash and cash equivalents
            debt: Total debt
            shares_outstanding: Shares outstanding
        
        Returns:
            Dictionary with valuation metrics
        """
        ev = self.calculate_enterprise_value()
        equity_value = self.calculate_equity_value(ev, cash, debt)
        share_price = self.calculate_share_price(equity_value, shares_outstanding)
        
        return {
            'enterprise_value': ev,
            'equity_value': equity_value,
            'share_price': share_price,
            'wacc': self.wacc,
            'terminal_growth_rate': self.terminal_growth_rate,
            'pv_cash_flows': self.calculate_pv_cash_flows(),
            'terminal_value': self.calculate_terminal_value(self.free_cash_flows[-1])
        }


def create_dcf_template() -> Dict:
    """
    Create a DCF model template with example inputs.
    
    Returns:
        Dictionary with template structure
    """
    return {
        'projection_years': 5,
        'free_cash_flows': [100, 120, 140, 160, 180],  # Example projections
        'wacc': 0.10,
        'terminal_growth_rate': 0.03,
        'cash': 500,
        'debt': 1000,
        'shares_outstanding': 100,
        'minority_interest': 0,
        'preferred_stock': 0
    }
