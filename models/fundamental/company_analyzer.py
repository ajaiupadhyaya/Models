"""
Comprehensive Company Analyzer
Fundamental analysis, financial metrics, valuation
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


class CompanyAnalyzer:
    """
    Institutional-grade company analysis and fundamental research.
    """
    
    def __init__(self, ticker: str):
        """
        Initialize company analyzer.
        
        Args:
            ticker: Stock ticker symbol
        """
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self._info = None
        self._financials = None
        self._balance_sheet = None
        self._cash_flow = None
    
    @property
    def info(self) -> Dict:
        """Get company information."""
        if self._info is None:
            self._info = self.stock.info
        return self._info
    
    @property
    def financials(self) -> pd.DataFrame:
        """Get income statement."""
        if self._financials is None:
            self._financials = self.stock.financials
        return self._financials
    
    @property
    def balance_sheet(self) -> pd.DataFrame:
        """Get balance sheet."""
        if self._balance_sheet is None:
            self._balance_sheet = self.stock.balance_sheet
        return self._balance_sheet
    
    @property
    def cash_flow(self) -> pd.DataFrame:
        """Get cash flow statement."""
        if self._cash_flow is None:
            self._cash_flow = self.stock.cashflow
        return self._cash_flow
    
    def get_company_profile(self) -> Dict:
        """
        Get comprehensive company profile.
        
        Returns:
            Dictionary with company information
        """
        info = self.info
        
        return {
            'name': info.get('longName', 'N/A'),
            'ticker': self.ticker,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', 'N/A'),
            'description': info.get('longBusinessSummary', 'N/A'),
            'employees': info.get('fullTimeEmployees', 0),
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'founded': info.get('founded', 'N/A')
        }
    
    def get_price_data(self, period: str = "1y") -> Dict:
        """
        Get current price data and statistics.
        
        Args:
            period: Time period for analysis
        
        Returns:
            Dictionary with price metrics
        """
        info = self.info
        hist = self.stock.history(period=period)
        
        current_price = hist['Close'].iloc[-1] if len(hist) > 0 else 0
        period_high = hist['High'].max() if len(hist) > 0 else 0
        period_low = hist['Low'].min() if len(hist) > 0 else 0
        
        return {
            'current_price': current_price,
            'previous_close': info.get('previousClose', 0),
            'day_high': info.get('dayHigh', 0),
            'day_low': info.get('dayLow', 0),
            'period_high': period_high,
            'period_low': period_low,
            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
            '52_week_low': info.get('fiftyTwoWeekLow', 0),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
            'beta': info.get('beta', 0)
        }
    
    def get_valuation_metrics(self) -> Dict:
        """
        Calculate comprehensive valuation metrics.
        
        Returns:
            Dictionary with valuation multiples
        """
        info = self.info
        
        return {
            # Price multiples
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'peg_ratio': info.get('pegRatio', None),
            'price_to_book': info.get('priceToBook', None),
            'price_to_sales': info.get('priceToSalesTrailing12Months', None),
            
            # Enterprise value multiples
            'ev_to_revenue': info.get('enterpriseToRevenue', None),
            'ev_to_ebitda': info.get('enterpriseToEbitda', None),
            
            # Market metrics
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'shares_outstanding': info.get('sharesOutstanding', 0),
            'float_shares': info.get('floatShares', 0),
            'shares_short': info.get('sharesShort', 0),
            'short_ratio': info.get('shortRatio', 0),
            'short_percent_float': info.get('shortPercentOfFloat', 0)
        }
    
    def get_profitability_metrics(self) -> Dict:
        """
        Calculate profitability metrics.
        
        Returns:
            Dictionary with profitability ratios
        """
        info = self.info
        
        # Get latest financial data
        try:
            if len(self.financials.columns) > 0:
                latest_financials = self.financials.iloc[:, 0]
                revenue = latest_financials.get('Total Revenue', 0)
                gross_profit = latest_financials.get('Gross Profit', 0)
                operating_income = latest_financials.get('Operating Income', 0)
                net_income = latest_financials.get('Net Income', 0)
                ebitda = latest_financials.get('EBITDA', 0)
                
                # Calculate margins
                gross_margin = gross_profit / revenue if revenue > 0 else None
                operating_margin = operating_income / revenue if revenue > 0 else None
                net_margin = net_income / revenue if revenue > 0 else None
                ebitda_margin = ebitda / revenue if revenue > 0 else None
            else:
                gross_margin = info.get('grossMargins', None)
                operating_margin = info.get('operatingMargins', None)
                net_margin = info.get('profitMargins', None)
                ebitda_margin = info.get('ebitdaMargins', None)
        except:
            gross_margin = info.get('grossMargins', None)
            operating_margin = info.get('operatingMargins', None)
            net_margin = info.get('profitMargins', None)
            ebitda_margin = info.get('ebitdaMargins', None)
        
        return {
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'net_profit_margin': net_margin,
            'ebitda_margin': ebitda_margin,
            'roe': info.get('returnOnEquity', None),
            'roa': info.get('returnOnAssets', None),
            'roic': self._calculate_roic()
        }
    
    def get_financial_health_metrics(self) -> Dict:
        """
        Calculate financial health and leverage metrics.
        
        Returns:
            Dictionary with financial health ratios
        """
        info = self.info
        
        try:
            if len(self.balance_sheet.columns) > 0:
                latest_bs = self.balance_sheet.iloc[:, 0]
                
                total_assets = latest_bs.get('Total Assets', 0)
                total_liabilities = latest_bs.get('Total Liabilities Net Minority Interest', 0)
                total_debt = latest_bs.get('Total Debt', 0)
                cash = latest_bs.get('Cash And Cash Equivalents', 0)
                current_assets = latest_bs.get('Current Assets', 0)
                current_liabilities = latest_bs.get('Current Liabilities', 0)
                stockholders_equity = latest_bs.get('Stockholders Equity', 0)
                
                # Calculate ratios
                debt_to_equity = total_debt / stockholders_equity if stockholders_equity > 0 else None
                debt_to_assets = total_debt / total_assets if total_assets > 0 else None
                current_ratio = current_assets / current_liabilities if current_liabilities > 0 else None
                quick_ratio = (current_assets - latest_bs.get('Inventory', 0)) / current_liabilities if current_liabilities > 0 else None
            else:
                debt_to_equity = info.get('debtToEquity', None)
                debt_to_assets = None
                current_ratio = info.get('currentRatio', None)
                quick_ratio = info.get('quickRatio', None)
                cash = info.get('totalCash', 0)
                total_debt = info.get('totalDebt', 0)
        except:
            debt_to_equity = info.get('debtToEquity', None)
            debt_to_assets = None
            current_ratio = info.get('currentRatio', None)
            quick_ratio = info.get('quickRatio', None)
            cash = info.get('totalCash', 0)
            total_debt = info.get('totalDebt', 0)
        
        return {
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'debt_to_equity': debt_to_equity,
            'debt_to_assets': debt_to_assets,
            'interest_coverage': self._calculate_interest_coverage(),
            'cash_to_debt': cash / total_debt if total_debt > 0 else None,
            'total_cash': cash,
            'total_debt': total_debt,
            'net_debt': total_debt - cash
        }
    
    def get_growth_metrics(self) -> Dict:
        """
        Calculate growth metrics.
        
        Returns:
            Dictionary with growth rates
        """
        info = self.info
        
        # Historical growth rates
        revenue_growth = self._calculate_cagr(self.financials, 'Total Revenue')
        earnings_growth = self._calculate_cagr(self.financials, 'Net Income')
        
        return {
            'revenue_growth_yoy': info.get('revenueGrowth', None),
            'revenue_growth_qoq': info.get('quarterlyRevenueGrowth', None),
            'earnings_growth_yoy': info.get('earningsGrowth', None),
            'earnings_growth_qoq': info.get('quarterlyEarningsGrowth', None),
            'revenue_cagr_3y': revenue_growth,
            'earnings_cagr_3y': earnings_growth,
            'earnings_estimate_growth': info.get('earningsQuarterlyGrowth', None)
        }
    
    def get_dividend_metrics(self) -> Dict:
        """
        Calculate dividend metrics.
        
        Returns:
            Dictionary with dividend information
        """
        info = self.info
        
        return {
            'dividend_rate': info.get('dividendRate', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'payout_ratio': info.get('payoutRatio', 0),
            'ex_dividend_date': info.get('exDividendDate', None),
            'five_year_avg_yield': info.get('fiveYearAvgDividendYield', 0)
        }
    
    def get_efficiency_metrics(self) -> Dict:
        """
        Calculate operational efficiency metrics.
        
        Returns:
            Dictionary with efficiency ratios
        """
        try:
            if len(self.financials.columns) > 0 and len(self.balance_sheet.columns) > 0:
                latest_financials = self.financials.iloc[:, 0]
                latest_bs = self.balance_sheet.iloc[:, 0]
                
                revenue = latest_financials.get('Total Revenue', 0)
                cogs = latest_financials.get('Cost Of Revenue', 0)
                
                total_assets = latest_bs.get('Total Assets', 0)
                inventory = latest_bs.get('Inventory', 0)
                accounts_receivable = latest_bs.get('Accounts Receivable', 0)
                accounts_payable = latest_bs.get('Accounts Payable', 0)
                
                # Turnover ratios
                asset_turnover = revenue / total_assets if total_assets > 0 else None
                inventory_turnover = cogs / inventory if inventory > 0 else None
                receivables_turnover = revenue / accounts_receivable if accounts_receivable > 0 else None
                payables_turnover = cogs / accounts_payable if accounts_payable > 0 else None
                
                # Days metrics
                days_inventory = 365 / inventory_turnover if inventory_turnover else None
                days_receivables = 365 / receivables_turnover if receivables_turnover else None
                days_payables = 365 / payables_turnover if payables_turnover else None
                
                # Cash conversion cycle
                ccc = (days_inventory or 0) + (days_receivables or 0) - (days_payables or 0)
            else:
                asset_turnover = None
                inventory_turnover = None
                receivables_turnover = None
                days_inventory = None
                days_receivables = None
                days_payables = None
                ccc = None
        except:
            asset_turnover = None
            inventory_turnover = None
            receivables_turnover = None
            days_inventory = None
            days_receivables = None
            days_payables = None
            ccc = None
        
        return {
            'asset_turnover': asset_turnover,
            'inventory_turnover': inventory_turnover,
            'receivables_turnover': receivables_turnover,
            'days_inventory_outstanding': days_inventory,
            'days_sales_outstanding': days_receivables,
            'days_payables_outstanding': days_payables,
            'cash_conversion_cycle': ccc
        }
    
    def _calculate_roic(self) -> Optional[float]:
        """Calculate Return on Invested Capital."""
        try:
            if len(self.financials.columns) > 0 and len(self.balance_sheet.columns) > 0:
                latest_financials = self.financials.iloc[:, 0]
                latest_bs = self.balance_sheet.iloc[:, 0]
                
                nopat = latest_financials.get('Operating Income', 0) * (1 - 0.21)  # Assuming 21% tax rate
                invested_capital = (latest_bs.get('Total Assets', 0) - 
                                  latest_bs.get('Current Liabilities', 0))
                
                return nopat / invested_capital if invested_capital > 0 else None
        except:
            return None
    
    def _calculate_interest_coverage(self) -> Optional[float]:
        """Calculate interest coverage ratio."""
        try:
            if len(self.financials.columns) > 0:
                latest_financials = self.financials.iloc[:, 0]
                
                ebit = latest_financials.get('EBIT', 0)
                interest_expense = latest_financials.get('Interest Expense', 0)
                
                return abs(ebit / interest_expense) if interest_expense != 0 else None
        except:
            return None
    
    def _calculate_cagr(self, df: pd.DataFrame, metric: str, periods: int = 3) -> Optional[float]:
        """Calculate Compound Annual Growth Rate."""
        try:
            if len(df.columns) >= periods:
                values = []
                for col in df.columns[:periods]:
                    if metric in df.index:
                        values.append(df.loc[metric, col])
                
                if len(values) >= 2:
                    start_value = values[-1]
                    end_value = values[0]
                    
                    if start_value > 0 and end_value > 0:
                        cagr = (end_value / start_value) ** (1 / (periods - 1)) - 1
                        return cagr
        except:
            pass
        
        return None
    
    def comprehensive_analysis(self) -> Dict:
        """
        Get comprehensive company analysis.
        
        Returns:
            Dictionary with all analysis sections
        """
        return {
            'profile': self.get_company_profile(),
            'price_data': self.get_price_data(),
            'valuation': self.get_valuation_metrics(),
            'profitability': self.get_profitability_metrics(),
            'financial_health': self.get_financial_health_metrics(),
            'growth': self.get_growth_metrics(),
            'dividends': self.get_dividend_metrics(),
            'efficiency': self.get_efficiency_metrics()
        }
    
    def get_analyst_recommendations(self) -> Dict:
        """
        Get analyst recommendations and estimates.
        
        Returns:
            Dictionary with analyst data
        """
        info = self.info
        
        return {
            'recommendation': info.get('recommendationKey', 'N/A'),
            'target_price': info.get('targetMeanPrice', None),
            'target_high': info.get('targetHighPrice', None),
            'target_low': info.get('targetLowPrice', None),
            'num_analysts': info.get('numberOfAnalystOpinions', 0),
            'recommendation_mean': info.get('recommendationMean', None)
        }


class FundamentalMetrics:
    """
    Advanced fundamental metrics and scoring systems.
    """
    
    @staticmethod
    def altman_z_score(financials: Dict) -> float:
        """
        Calculate Altman Z-Score for bankruptcy prediction.
        
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        
        Where:
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Sales / Total Assets
        
        Args:
            financials: Dictionary with financial data
        
        Returns:
            Z-Score (>2.99 safe, <1.81 distress)
        """
        try:
            working_capital = financials.get('working_capital', 0)
            retained_earnings = financials.get('retained_earnings', 0)
            ebit = financials.get('ebit', 0)
            market_cap = financials.get('market_cap', 0)
            total_liabilities = financials.get('total_liabilities', 1)
            sales = financials.get('revenue', 0)
            total_assets = financials.get('total_assets', 1)
            
            x1 = working_capital / total_assets
            x2 = retained_earnings / total_assets
            x3 = ebit / total_assets
            x4 = market_cap / total_liabilities
            x5 = sales / total_assets
            
            z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
            
            return z_score
        except:
            return np.nan
    
    @staticmethod
    def piotroski_f_score(financials: Dict) -> int:
        """
        Calculate Piotroski F-Score (0-9).
        
        Higher is better. Measures financial strength across 9 criteria.
        
        Args:
            financials: Dictionary with financial data
        
        Returns:
            F-Score (0-9)
        """
        score = 0
        
        try:
            # Profitability (4 points)
            if financials.get('net_income', 0) > 0:
                score += 1
            if financials.get('roa', 0) > 0:
                score += 1
            if financials.get('operating_cash_flow', 0) > 0:
                score += 1
            if financials.get('operating_cash_flow', 0) > financials.get('net_income', 0):
                score += 1
            
            # Leverage, Liquidity, Source of Funds (3 points)
            if financials.get('debt_to_equity_change', 0) < 0:
                score += 1
            if financials.get('current_ratio_change', 0) > 0:
                score += 1
            if financials.get('shares_outstanding_change', 0) <= 0:
                score += 1
            
            # Operating Efficiency (2 points)
            if financials.get('gross_margin_change', 0) > 0:
                score += 1
            if financials.get('asset_turnover_change', 0) > 0:
                score += 1
        except:
            pass
        
        return score
    
    @staticmethod
    def beneish_m_score(financials: Dict) -> float:
        """
        Calculate Beneish M-Score for earnings manipulation detection.
        
        M-Score > -2.22 suggests possible manipulation.
        
        Args:
            financials: Dictionary with financial data
        
        Returns:
            M-Score
        """
        try:
            # This is a simplified version
            # Full implementation requires multi-year data
            dsri = financials.get('dsri', 1)  # Days Sales in Receivables Index
            gmi = financials.get('gmi', 1)    # Gross Margin Index
            aqi = financials.get('aqi', 1)    # Asset Quality Index
            sgi = financials.get('sgi', 1)    # Sales Growth Index
            depi = financials.get('depi', 1)  # Depreciation Index
            sgai = financials.get('sgai', 1)  # Sales, General, Admin Expenses Index
            lvgi = financials.get('lvgi', 1)  # Leverage Index
            tata = financials.get('tata', 0)  # Total Accruals to Total Assets
            
            m_score = (-4.84 + 0.92*dsri + 0.528*gmi + 0.404*aqi + 
                      0.892*sgi + 0.115*depi - 0.172*sgai + 
                      4.679*tata - 0.327*lvgi)
            
            return m_score
        except:
            return np.nan
