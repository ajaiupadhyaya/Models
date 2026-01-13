"""
Financial Statement Analysis
Trend analysis, common-size statements, quality checks
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FinancialStatementAnalyzer:
    """
    Comprehensive financial statement analysis.
    """
    
    def __init__(self, 
                 income_statement: pd.DataFrame,
                 balance_sheet: pd.DataFrame,
                 cash_flow: pd.DataFrame):
        """
        Initialize financial statement analyzer.
        
        Args:
            income_statement: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
            cash_flow: Cash flow statement DataFrame
        """
        self.income_statement = income_statement
        self.balance_sheet = balance_sheet
        self.cash_flow = cash_flow
    
    def common_size_income_statement(self) -> pd.DataFrame:
        """
        Create common-size income statement (% of revenue).
        
        Returns:
            Common-size income statement DataFrame
        """
        common_size = self.income_statement.copy()
        
        for col in common_size.columns:
            revenue = common_size.loc['Total Revenue', col] if 'Total Revenue' in common_size.index else 1
            if revenue != 0:
                common_size[col] = common_size[col] / revenue * 100
        
        return common_size
    
    def common_size_balance_sheet(self) -> pd.DataFrame:
        """
        Create common-size balance sheet (% of total assets).
        
        Returns:
            Common-size balance sheet DataFrame
        """
        common_size = self.balance_sheet.copy()
        
        for col in common_size.columns:
            total_assets = common_size.loc['Total Assets', col] if 'Total Assets' in common_size.index else 1
            if total_assets != 0:
                common_size[col] = common_size[col] / total_assets * 100
        
        return common_size
    
    def trend_analysis(self, statement: str = 'income') -> pd.DataFrame:
        """
        Perform trend analysis (indexed to earliest period).
        
        Args:
            statement: Which statement to analyze ('income', 'balance', 'cash')
        
        Returns:
            Trend analysis DataFrame (indexed to 100)
        """
        if statement == 'income':
            df = self.income_statement
        elif statement == 'balance':
            df = self.balance_sheet
        elif statement == 'cash':
            df = self.cash_flow
        else:
            raise ValueError("Statement must be 'income', 'balance', or 'cash'")
        
        if len(df.columns) == 0:
            return df
        
        # Index to earliest period (rightmost column)
        base_period = df.iloc[:, -1]
        trend = df.copy()
        
        for col in trend.columns:
            trend[col] = (trend[col] / base_period) * 100
        
        return trend
    
    def growth_rates(self, statement: str = 'income', periods: int = 3) -> pd.DataFrame:
        """
        Calculate year-over-year growth rates.
        
        Args:
            statement: Which statement to analyze
            periods: Number of periods to analyze
        
        Returns:
            DataFrame with growth rates
        """
        if statement == 'income':
            df = self.income_statement
        elif statement == 'balance':
            df = self.balance_sheet
        elif statement == 'cash':
            df = self.cash_flow
        else:
            raise ValueError("Statement must be 'income', 'balance', or 'cash'")
        
        if len(df.columns) < 2:
            return pd.DataFrame()
        
        growth = pd.DataFrame(index=df.index)
        
        for i in range(min(len(df.columns) - 1, periods)):
            col_name = f"YoY_Growth_{i+1}"
            current = df.iloc[:, i]
            previous = df.iloc[:, i + 1]
            
            growth[col_name] = ((current - previous) / previous.abs()) * 100
            growth[col_name] = growth[col_name].replace([np.inf, -np.inf], np.nan)
        
        return growth
    
    def cagr_analysis(self, line_items: List[str], periods: int = 3) -> Dict:
        """
        Calculate CAGR for specific line items.
        
        Args:
            line_items: List of line items to analyze
            periods: Number of years
        
        Returns:
            Dictionary with CAGR for each line item
        """
        cagrs = {}
        
        for item in line_items:
            if item in self.income_statement.index:
                values = self.income_statement.loc[item].values[:periods]
                
                if len(values) >= 2 and values[-1] > 0 and values[0] > 0:
                    years = len(values) - 1
                    cagr = (values[0] / values[-1]) ** (1 / years) - 1
                    cagrs[item] = cagr
        
        return cagrs
    
    def quality_of_earnings(self) -> Dict:
        """
        Assess quality of earnings.
        
        Returns:
            Dictionary with earnings quality metrics
        """
        try:
            latest_is = self.income_statement.iloc[:, 0] if len(self.income_statement.columns) > 0 else pd.Series()
            latest_cf = self.cash_flow.iloc[:, 0] if len(self.cash_flow.columns) > 0 else pd.Series()
            latest_bs = self.balance_sheet.iloc[:, 0] if len(self.balance_sheet.columns) > 0 else pd.Series()
            
            net_income = latest_is.get('Net Income', 0)
            operating_cf = latest_cf.get('Operating Cash Flow', 0)
            
            # Accruals
            accruals = net_income - operating_cf
            accruals_ratio = accruals / abs(net_income) if net_income != 0 else None
            
            # Cash flow to net income ratio
            cf_to_ni = operating_cf / net_income if net_income > 0 else None
            
            # Calculate total assets for accruals ratio
            total_assets = latest_bs.get('Total Assets', 1)
            accruals_to_assets = accruals / total_assets if total_assets > 0 else None
            
            # Quality assessment
            if cf_to_ni and cf_to_ni > 1:
                quality = "High"
                reason = "Cash flow exceeds net income"
            elif cf_to_ni and cf_to_ni > 0.8:
                quality = "Good"
                reason = "Cash flow supports earnings"
            elif cf_to_ni and cf_to_ni > 0.5:
                quality = "Moderate"
                reason = "Significant non-cash earnings"
            else:
                quality = "Low"
                reason = "Low cash generation relative to earnings"
            
            return {
                'net_income': net_income,
                'operating_cash_flow': operating_cf,
                'accruals': accruals,
                'accruals_ratio': accruals_ratio,
                'cash_flow_to_net_income': cf_to_ni,
                'accruals_to_assets': accruals_to_assets,
                'quality_assessment': quality,
                'quality_reason': reason
            }
        except Exception as e:
            return {'error': str(e)}
    
    def working_capital_analysis(self) -> Dict:
        """
        Analyze working capital trends.
        
        Returns:
            Dictionary with working capital metrics
        """
        try:
            results = {}
            
            for i, col in enumerate(self.balance_sheet.columns[:3]):  # Last 3 periods
                bs = self.balance_sheet[col]
                
                current_assets = bs.get('Current Assets', 0)
                current_liabilities = bs.get('Current Liabilities', 0)
                inventory = bs.get('Inventory', 0)
                accounts_receivable = bs.get('Accounts Receivable', 0)
                accounts_payable = bs.get('Accounts Payable', 0)
                
                working_capital = current_assets - current_liabilities
                
                results[f'period_{i}'] = {
                    'working_capital': working_capital,
                    'current_assets': current_assets,
                    'current_liabilities': current_liabilities,
                    'inventory': inventory,
                    'accounts_receivable': accounts_receivable,
                    'accounts_payable': accounts_payable,
                    'working_capital_ratio': working_capital / current_assets if current_assets > 0 else None
                }
            
            # Calculate changes
            if len(results) >= 2:
                wc_change = results['period_0']['working_capital'] - results['period_1']['working_capital']
                results['working_capital_change'] = wc_change
            
            return results
        except Exception as e:
            return {'error': str(e)}
    
    def free_cash_flow_analysis(self) -> Dict:
        """
        Analyze free cash flow.
        
        Returns:
            Dictionary with FCF metrics
        """
        try:
            fcf_history = []
            
            for col in self.cash_flow.columns:
                cf = self.cash_flow[col]
                
                operating_cf = cf.get('Operating Cash Flow', 0)
                capex = abs(cf.get('Capital Expenditure', 0))
                
                fcf = operating_cf - capex
                fcf_history.append(fcf)
            
            # Get revenue for FCF margin
            revenue_history = []
            for col in self.income_statement.columns[:len(fcf_history)]:
                revenue = self.income_statement.loc['Total Revenue', col] if 'Total Revenue' in self.income_statement.index else 0
                revenue_history.append(revenue)
            
            fcf_margins = [(fcf / rev * 100) if rev > 0 else None 
                          for fcf, rev in zip(fcf_history, revenue_history)]
            
            # Calculate average and trend
            avg_fcf = np.mean([f for f in fcf_history if f > 0]) if any(f > 0 for f in fcf_history) else 0
            
            return {
                'latest_fcf': fcf_history[0] if len(fcf_history) > 0 else None,
                'fcf_history': fcf_history,
                'fcf_margins': fcf_margins,
                'average_fcf': avg_fcf,
                'fcf_trend': 'improving' if len(fcf_history) >= 2 and fcf_history[0] > fcf_history[1] else 'declining',
                'fcf_consistency': np.std(fcf_history) / avg_fcf if avg_fcf > 0 else None
            }
        except Exception as e:
            return {'error': str(e)}
    
    def capital_structure_evolution(self) -> pd.DataFrame:
        """
        Analyze evolution of capital structure.
        
        Returns:
            DataFrame with capital structure over time
        """
        capital_structure = pd.DataFrame()
        
        for col in self.balance_sheet.columns:
            bs = self.balance_sheet[col]
            
            total_debt = bs.get('Total Debt', 0)
            total_equity = bs.get('Stockholders Equity', 0)
            total_capital = total_debt + total_equity
            
            if total_capital > 0:
                capital_structure[col] = pd.Series({
                    'debt': total_debt,
                    'equity': total_equity,
                    'total_capital': total_capital,
                    'debt_pct': (total_debt / total_capital) * 100,
                    'equity_pct': (total_equity / total_capital) * 100
                })
        
        return capital_structure
    
    def revenue_breakdown(self) -> Dict:
        """
        Analyze revenue composition if segment data available.
        
        Returns:
            Dictionary with revenue analysis
        """
        try:
            latest_is = self.income_statement.iloc[:, 0] if len(self.income_statement.columns) > 0 else pd.Series()
            
            total_revenue = latest_is.get('Total Revenue', 0)
            cost_of_revenue = latest_is.get('Cost Of Revenue', 0)
            gross_profit = latest_is.get('Gross Profit', 0)
            
            return {
                'total_revenue': total_revenue,
                'cost_of_revenue': cost_of_revenue,
                'gross_profit': gross_profit,
                'gross_margin': (gross_profit / total_revenue * 100) if total_revenue > 0 else None
            }
        except Exception as e:
            return {'error': str(e)}
    
    def comprehensive_analysis(self) -> Dict:
        """
        Get comprehensive financial statement analysis.
        
        Returns:
            Dictionary with all analyses
        """
        return {
            'common_size_income': self.common_size_income_statement(),
            'common_size_balance': self.common_size_balance_sheet(),
            'income_trends': self.trend_analysis('income'),
            'income_growth': self.growth_rates('income'),
            'quality_of_earnings': self.quality_of_earnings(),
            'working_capital': self.working_capital_analysis(),
            'free_cash_flow': self.free_cash_flow_analysis(),
            'capital_structure': self.capital_structure_evolution(),
            'revenue_analysis': self.revenue_breakdown()
        }


class StatementReconciliation:
    """
    Reconcile financial statements and check for consistency.
    """
    
    @staticmethod
    def reconcile_cash_flow(balance_sheet: pd.DataFrame,
                           cash_flow: pd.DataFrame) -> Dict:
        """
        Reconcile cash flow statement with balance sheet cash changes.
        
        Args:
            balance_sheet: Balance sheet DataFrame
            cash_flow: Cash flow statement DataFrame
        
        Returns:
            Dictionary with reconciliation
        """
        try:
            if len(balance_sheet.columns) < 2 or len(cash_flow.columns) < 1:
                return {'error': 'Insufficient data for reconciliation'}
            
            # Get cash from balance sheet
            current_cash = balance_sheet.iloc[:, 0].get('Cash And Cash Equivalents', 0)
            previous_cash = balance_sheet.iloc[:, 1].get('Cash And Cash Equivalents', 0)
            bs_cash_change = current_cash - previous_cash
            
            # Get cash flows
            latest_cf = cash_flow.iloc[:, 0]
            operating_cf = latest_cf.get('Operating Cash Flow', 0)
            investing_cf = latest_cf.get('Investing Cash Flow', 0)
            financing_cf = latest_cf.get('Financing Cash Flow', 0)
            
            cf_total_change = operating_cf + investing_cf + financing_cf
            
            # Reconciliation difference
            difference = bs_cash_change - cf_total_change
            
            return {
                'balance_sheet_cash_change': bs_cash_change,
                'cash_flow_total_change': cf_total_change,
                'difference': difference,
                'reconciles': abs(difference) < 1000,  # Within rounding error
                'operating_cf': operating_cf,
                'investing_cf': investing_cf,
                'financing_cf': financing_cf
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def reconcile_retained_earnings(balance_sheet: pd.DataFrame,
                                   income_statement: pd.DataFrame,
                                   dividends_paid: float = 0) -> Dict:
        """
        Reconcile retained earnings.
        
        Ending RE = Beginning RE + Net Income - Dividends
        
        Args:
            balance_sheet: Balance sheet DataFrame
            income_statement: Income statement DataFrame
            dividends_paid: Dividends paid during period
        
        Returns:
            Dictionary with reconciliation
        """
        try:
            if len(balance_sheet.columns) < 2:
                return {'error': 'Insufficient data'}
            
            current_re = balance_sheet.iloc[:, 0].get('Retained Earnings', 0)
            previous_re = balance_sheet.iloc[:, 1].get('Retained Earnings', 0)
            
            net_income = income_statement.iloc[:, 0].get('Net Income', 0) if len(income_statement.columns) > 0 else 0
            
            expected_re = previous_re + net_income - dividends_paid
            difference = current_re - expected_re
            
            return {
                'beginning_retained_earnings': previous_re,
                'net_income': net_income,
                'dividends_paid': dividends_paid,
                'expected_ending_retained_earnings': expected_re,
                'actual_ending_retained_earnings': current_re,
                'difference': difference,
                'reconciles': abs(difference) < 1000
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def balance_sheet_equation_check(balance_sheet: pd.DataFrame) -> Dict:
        """
        Verify balance sheet equation: Assets = Liabilities + Equity
        
        Args:
            balance_sheet: Balance sheet DataFrame
        
        Returns:
            Dictionary with verification results
        """
        results = {}
        
        for col in balance_sheet.columns:
            bs = balance_sheet[col]
            
            total_assets = bs.get('Total Assets', 0)
            total_liabilities = bs.get('Total Liabilities Net Minority Interest', 0)
            stockholders_equity = bs.get('Stockholders Equity', 0)
            
            liabilities_plus_equity = total_liabilities + stockholders_equity
            difference = total_assets - liabilities_plus_equity
            
            results[col] = {
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'stockholders_equity': stockholders_equity,
                'liabilities_plus_equity': liabilities_plus_equity,
                'difference': difference,
                'balances': abs(difference) < 1000
            }
        
        return results
