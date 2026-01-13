"""
Financial Ratios and Metrics Calculator
Comprehensive ratio analysis for fundamental research
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class FinancialRatios:
    """
    Comprehensive financial ratio calculations.
    """
    
    @staticmethod
    def liquidity_ratios(balance_sheet: pd.DataFrame) -> Dict:
        """
        Calculate liquidity ratios.
        
        Args:
            balance_sheet: Balance sheet DataFrame
        
        Returns:
            Dictionary with liquidity ratios
        """
        latest = balance_sheet.iloc[:, 0] if len(balance_sheet.columns) > 0 else pd.Series()
        
        current_assets = latest.get('Current Assets', 0)
        current_liabilities = latest.get('Current Liabilities', 0)
        inventory = latest.get('Inventory', 0)
        cash = latest.get('Cash And Cash Equivalents', 0)
        
        return {
            'current_ratio': current_assets / current_liabilities if current_liabilities > 0 else None,
            'quick_ratio': (current_assets - inventory) / current_liabilities if current_liabilities > 0 else None,
            'cash_ratio': cash / current_liabilities if current_liabilities > 0 else None,
            'working_capital': current_assets - current_liabilities,
            'working_capital_ratio': (current_assets - current_liabilities) / current_assets if current_assets > 0 else None
        }
    
    @staticmethod
    def leverage_ratios(balance_sheet: pd.DataFrame, income_statement: pd.DataFrame) -> Dict:
        """
        Calculate leverage/solvency ratios.
        
        Args:
            balance_sheet: Balance sheet DataFrame
            income_statement: Income statement DataFrame
        
        Returns:
            Dictionary with leverage ratios
        """
        latest_bs = balance_sheet.iloc[:, 0] if len(balance_sheet.columns) > 0 else pd.Series()
        latest_is = income_statement.iloc[:, 0] if len(income_statement.columns) > 0 else pd.Series()
        
        total_debt = latest_bs.get('Total Debt', 0)
        total_assets = latest_bs.get('Total Assets', 0)
        total_equity = latest_bs.get('Stockholders Equity', 0)
        ebit = latest_is.get('EBIT', 0)
        ebitda = latest_is.get('EBITDA', 0)
        interest_expense = abs(latest_is.get('Interest Expense', 0))
        
        return {
            'debt_to_equity': total_debt / total_equity if total_equity > 0 else None,
            'debt_to_assets': total_debt / total_assets if total_assets > 0 else None,
            'equity_multiplier': total_assets / total_equity if total_equity > 0 else None,
            'debt_ratio': total_debt / total_assets if total_assets > 0 else None,
            'interest_coverage': ebit / interest_expense if interest_expense > 0 else None,
            'debt_service_coverage': ebitda / interest_expense if interest_expense > 0 else None,
            'equity_ratio': total_equity / total_assets if total_assets > 0 else None
        }
    
    @staticmethod
    def profitability_ratios(income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> Dict:
        """
        Calculate profitability ratios.
        
        Args:
            income_statement: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
        
        Returns:
            Dictionary with profitability ratios
        """
        latest_is = income_statement.iloc[:, 0] if len(income_statement.columns) > 0 else pd.Series()
        latest_bs = balance_sheet.iloc[:, 0] if len(balance_sheet.columns) > 0 else pd.Series()
        
        revenue = latest_is.get('Total Revenue', 0)
        gross_profit = latest_is.get('Gross Profit', 0)
        operating_income = latest_is.get('Operating Income', 0)
        ebit = latest_is.get('EBIT', 0)
        ebitda = latest_is.get('EBITDA', 0)
        net_income = latest_is.get('Net Income', 0)
        
        total_assets = latest_bs.get('Total Assets', 0)
        total_equity = latest_bs.get('Stockholders Equity', 0)
        
        return {
            'gross_margin': gross_profit / revenue if revenue > 0 else None,
            'operating_margin': operating_income / revenue if revenue > 0 else None,
            'ebit_margin': ebit / revenue if revenue > 0 else None,
            'ebitda_margin': ebitda / revenue if revenue > 0 else None,
            'net_margin': net_income / revenue if revenue > 0 else None,
            'roa': net_income / total_assets if total_assets > 0 else None,
            'roe': net_income / total_equity if total_equity > 0 else None,
            'roic': FinancialRatios._calculate_roic(latest_is, latest_bs)
        }
    
    @staticmethod
    def efficiency_ratios(income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> Dict:
        """
        Calculate efficiency/activity ratios.
        
        Args:
            income_statement: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
        
        Returns:
            Dictionary with efficiency ratios
        """
        latest_is = income_statement.iloc[:, 0] if len(income_statement.columns) > 0 else pd.Series()
        latest_bs = balance_sheet.iloc[:, 0] if len(balance_sheet.columns) > 0 else pd.Series()
        
        revenue = latest_is.get('Total Revenue', 0)
        cogs = latest_is.get('Cost Of Revenue', 0)
        
        total_assets = latest_bs.get('Total Assets', 0)
        inventory = latest_bs.get('Inventory', 0)
        accounts_receivable = latest_bs.get('Accounts Receivable', 0)
        accounts_payable = latest_bs.get('Accounts Payable', 0)
        total_equity = latest_bs.get('Stockholders Equity', 0)
        
        inventory_turnover = cogs / inventory if inventory > 0 else None
        receivables_turnover = revenue / accounts_receivable if accounts_receivable > 0 else None
        payables_turnover = cogs / accounts_payable if accounts_payable > 0 else None
        
        return {
            'asset_turnover': revenue / total_assets if total_assets > 0 else None,
            'inventory_turnover': inventory_turnover,
            'receivables_turnover': receivables_turnover,
            'payables_turnover': payables_turnover,
            'equity_turnover': revenue / total_equity if total_equity > 0 else None,
            'days_inventory': 365 / inventory_turnover if inventory_turnover else None,
            'days_receivables': 365 / receivables_turnover if receivables_turnover else None,
            'days_payables': 365 / payables_turnover if payables_turnover else None,
            'cash_conversion_cycle': FinancialRatios._calculate_ccc(inventory_turnover, receivables_turnover, payables_turnover)
        }
    
    @staticmethod
    def market_ratios(income_statement: pd.DataFrame, market_data: Dict) -> Dict:
        """
        Calculate market/valuation ratios.
        
        Args:
            income_statement: Income statement DataFrame
            market_data: Dictionary with market data (price, shares, market_cap)
        
        Returns:
            Dictionary with market ratios
        """
        latest_is = income_statement.iloc[:, 0] if len(income_statement.columns) > 0 else pd.Series()
        
        net_income = latest_is.get('Net Income', 0)
        revenue = latest_is.get('Total Revenue', 0)
        ebitda = latest_is.get('EBITDA', 0)
        
        price = market_data.get('price', 0)
        shares = market_data.get('shares_outstanding', 0)
        market_cap = market_data.get('market_cap', price * shares)
        enterprise_value = market_data.get('enterprise_value', market_cap)
        book_value = market_data.get('book_value', 0)
        
        eps = net_income / shares if shares > 0 else 0
        
        return {
            'pe_ratio': price / eps if eps > 0 else None,
            'price_to_book': market_cap / book_value if book_value > 0 else None,
            'price_to_sales': market_cap / revenue if revenue > 0 else None,
            'ev_to_sales': enterprise_value / revenue if revenue > 0 else None,
            'ev_to_ebitda': enterprise_value / ebitda if ebitda > 0 else None,
            'earnings_yield': eps / price if price > 0 else None,
            'market_to_book': market_cap / book_value if book_value > 0 else None
        }
    
    @staticmethod
    def dupont_analysis(income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> Dict:
        """
        Perform DuPont analysis (ROE decomposition).
        
        ROE = Net Margin × Asset Turnover × Equity Multiplier
        
        Args:
            income_statement: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
        
        Returns:
            Dictionary with DuPont components
        """
        latest_is = income_statement.iloc[:, 0] if len(income_statement.columns) > 0 else pd.Series()
        latest_bs = balance_sheet.iloc[:, 0] if len(balance_sheet.columns) > 0 else pd.Series()
        
        net_income = latest_is.get('Net Income', 0)
        revenue = latest_is.get('Total Revenue', 0)
        total_assets = latest_bs.get('Total Assets', 0)
        total_equity = latest_bs.get('Stockholders Equity', 0)
        
        net_margin = net_income / revenue if revenue > 0 else 0
        asset_turnover = revenue / total_assets if total_assets > 0 else 0
        equity_multiplier = total_assets / total_equity if total_equity > 0 else 0
        
        roe = net_margin * asset_turnover * equity_multiplier
        
        return {
            'roe': roe,
            'net_margin': net_margin,
            'asset_turnover': asset_turnover,
            'equity_multiplier': equity_multiplier,
            'leverage_contribution': (equity_multiplier - 1) / equity_multiplier if equity_multiplier > 0 else 0
        }
    
    @staticmethod
    def cash_flow_ratios(cash_flow: pd.DataFrame, income_statement: pd.DataFrame, balance_sheet: pd.DataFrame) -> Dict:
        """
        Calculate cash flow ratios.
        
        Args:
            cash_flow: Cash flow statement DataFrame
            income_statement: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
        
        Returns:
            Dictionary with cash flow ratios
        """
        latest_cf = cash_flow.iloc[:, 0] if len(cash_flow.columns) > 0 else pd.Series()
        latest_is = income_statement.iloc[:, 0] if len(income_statement.columns) > 0 else pd.Series()
        latest_bs = balance_sheet.iloc[:, 0] if len(balance_sheet.columns) > 0 else pd.Series()
        
        operating_cf = latest_cf.get('Operating Cash Flow', 0)
        capex = abs(latest_cf.get('Capital Expenditure', 0))
        free_cash_flow = operating_cf - capex
        
        net_income = latest_is.get('Net Income', 0)
        revenue = latest_is.get('Total Revenue', 0)
        current_liabilities = latest_bs.get('Current Liabilities', 0)
        total_debt = latest_bs.get('Total Debt', 0)
        
        return {
            'operating_cash_flow': operating_cf,
            'free_cash_flow': free_cash_flow,
            'fcf_margin': free_cash_flow / revenue if revenue > 0 else None,
            'ocf_ratio': operating_cf / current_liabilities if current_liabilities > 0 else None,
            'cash_flow_to_debt': operating_cf / total_debt if total_debt > 0 else None,
            'ocf_to_net_income': operating_cf / net_income if net_income > 0 else None,
            'capex_to_operating_cf': capex / operating_cf if operating_cf > 0 else None,
            'free_cash_flow_yield': free_cash_flow / revenue if revenue > 0 else None
        }
    
    @staticmethod
    def _calculate_roic(income_statement: pd.Series, balance_sheet: pd.Series) -> Optional[float]:
        """Calculate Return on Invested Capital."""
        try:
            operating_income = income_statement.get('Operating Income', 0)
            tax_rate = 0.21  # Approximate corporate tax rate
            nopat = operating_income * (1 - tax_rate)
            
            total_assets = balance_sheet.get('Total Assets', 0)
            current_liabilities = balance_sheet.get('Current Liabilities', 0)
            cash = balance_sheet.get('Cash And Cash Equivalents', 0)
            
            invested_capital = total_assets - current_liabilities - cash
            
            return nopat / invested_capital if invested_capital > 0 else None
        except:
            return None
    
    @staticmethod
    def _calculate_ccc(inventory_turnover: Optional[float], 
                      receivables_turnover: Optional[float],
                      payables_turnover: Optional[float]) -> Optional[float]:
        """Calculate Cash Conversion Cycle."""
        try:
            dio = 365 / inventory_turnover if inventory_turnover else 0
            dso = 365 / receivables_turnover if receivables_turnover else 0
            dpo = 365 / payables_turnover if payables_turnover else 0
            
            return dio + dso - dpo
        except:
            return None
    
    @staticmethod
    def comprehensive_ratios(income_statement: pd.DataFrame,
                           balance_sheet: pd.DataFrame,
                           cash_flow: pd.DataFrame,
                           market_data: Dict) -> Dict:
        """
        Calculate all ratio categories.
        
        Args:
            income_statement: Income statement DataFrame
            balance_sheet: Balance sheet DataFrame
            cash_flow: Cash flow statement DataFrame
            market_data: Market data dictionary
        
        Returns:
            Dictionary with all ratios organized by category
        """
        return {
            'liquidity': FinancialRatios.liquidity_ratios(balance_sheet),
            'leverage': FinancialRatios.leverage_ratios(balance_sheet, income_statement),
            'profitability': FinancialRatios.profitability_ratios(income_statement, balance_sheet),
            'efficiency': FinancialRatios.efficiency_ratios(income_statement, balance_sheet),
            'market': FinancialRatios.market_ratios(income_statement, market_data),
            'dupont': FinancialRatios.dupont_analysis(income_statement, balance_sheet),
            'cash_flow': FinancialRatios.cash_flow_ratios(cash_flow, income_statement, balance_sheet)
        }


class RatioAnalysis:
    """
    Ratio trend analysis and interpretation.
    """
    
    @staticmethod
    def trend_analysis(ratios_time_series: Dict[str, List[float]], periods: int = 3) -> Dict:
        """
        Analyze trends in ratios over time.
        
        Args:
            ratios_time_series: Dictionary mapping ratio name to list of values
            periods: Number of periods
        
        Returns:
            Dictionary with trend analysis
        """
        trends = {}
        
        for ratio_name, values in ratios_time_series.items():
            if len(values) >= 2:
                # Calculate trend
                x = np.arange(len(values))
                y = np.array(values)
                
                # Linear regression
                if len(x) > 1:
                    slope, intercept = np.polyfit(x, y, 1)
                    
                    # Calculate percentage change
                    pct_change = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
                    
                    trends[ratio_name] = {
                        'current': values[-1],
                        'previous': values[-2],
                        'change': values[-1] - values[-2],
                        'pct_change': pct_change,
                        'trend': 'improving' if slope > 0 else 'declining',
                        'slope': slope,
                        'volatility': np.std(values)
                    }
        
        return trends
    
    @staticmethod
    def benchmark_comparison(company_ratios: Dict, industry_averages: Dict) -> Dict:
        """
        Compare company ratios to industry benchmarks.
        
        Args:
            company_ratios: Company's ratios
            industry_averages: Industry average ratios
        
        Returns:
            Dictionary with comparisons
        """
        comparison = {}
        
        for ratio_name, company_value in company_ratios.items():
            if ratio_name in industry_averages and company_value is not None:
                industry_value = industry_averages[ratio_name]
                
                if industry_value and industry_value != 0:
                    difference = company_value - industry_value
                    pct_difference = (difference / industry_value) * 100
                    
                    comparison[ratio_name] = {
                        'company': company_value,
                        'industry': industry_value,
                        'difference': difference,
                        'pct_difference': pct_difference,
                        'relative_performance': 'above' if difference > 0 else 'below'
                    }
        
        return comparison
    
    @staticmethod
    def interpret_ratios(ratios: Dict) -> Dict[str, str]:
        """
        Provide interpretation of key ratios.
        
        Args:
            ratios: Dictionary of ratios
        
        Returns:
            Dictionary with interpretations
        """
        interpretations = {}
        
        # Current Ratio
        current_ratio = ratios.get('current_ratio')
        if current_ratio:
            if current_ratio > 2:
                interpretations['current_ratio'] = "Strong liquidity position"
            elif current_ratio > 1:
                interpretations['current_ratio'] = "Adequate liquidity"
            else:
                interpretations['current_ratio'] = "Potential liquidity concerns"
        
        # Debt to Equity
        dte = ratios.get('debt_to_equity')
        if dte:
            if dte < 0.5:
                interpretations['debt_to_equity'] = "Conservative capital structure"
            elif dte < 1.5:
                interpretations['debt_to_equity'] = "Moderate leverage"
            else:
                interpretations['debt_to_equity'] = "High leverage - monitor closely"
        
        # ROE
        roe = ratios.get('roe')
        if roe:
            if roe > 0.20:
                interpretations['roe'] = "Excellent returns on equity"
            elif roe > 0.15:
                interpretations['roe'] = "Strong returns on equity"
            elif roe > 0.10:
                interpretations['roe'] = "Moderate returns on equity"
            else:
                interpretations['roe'] = "Below-average returns on equity"
        
        # Interest Coverage
        interest_cov = ratios.get('interest_coverage')
        if interest_cov:
            if interest_cov > 5:
                interpretations['interest_coverage'] = "Strong debt service capability"
            elif interest_cov > 2.5:
                interpretations['interest_coverage'] = "Adequate debt service capability"
            else:
                interpretations['interest_coverage'] = "Potential debt service stress"
        
        return interpretations
