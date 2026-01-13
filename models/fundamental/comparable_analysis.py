"""
Comparable Company Analysis
Peer group analysis, relative valuation, industry benchmarking
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ComparableCompanies:
    """
    Comparable company analysis and peer benchmarking.
    """
    
    def __init__(self, target_ticker: str, peer_tickers: List[str]):
        """
        Initialize comparable company analysis.
        
        Args:
            target_ticker: Target company ticker
            peer_tickers: List of peer company tickers
        """
        self.target_ticker = target_ticker.upper()
        self.peer_tickers = [t.upper() for t in peer_tickers]
        self.all_tickers = [self.target_ticker] + self.peer_tickers
        
        self._data = {}
        self._load_data()
    
    def _load_data(self):
        """Load data for all companies."""
        for ticker in self.all_tickers:
            try:
                stock = yf.Ticker(ticker)
                self._data[ticker] = {
                    'info': stock.info,
                    'financials': stock.financials,
                    'balance_sheet': stock.balance_sheet,
                    'cash_flow': stock.cashflow
                }
            except Exception as e:
                print(f"Error loading data for {ticker}: {e}")
                self._data[ticker] = None
    
    def get_valuation_multiples(self) -> pd.DataFrame:
        """
        Get valuation multiples for all companies.
        
        Returns:
            DataFrame with valuation multiples
        """
        multiples = []
        
        for ticker in self.all_tickers:
            if self._data.get(ticker):
                info = self._data[ticker]['info']
                
                row = {
                    'ticker': ticker,
                    'company_name': info.get('shortName', ticker),
                    'market_cap': info.get('marketCap', None),
                    'enterprise_value': info.get('enterpriseValue', None),
                    'pe_ratio': info.get('trailingPE', None),
                    'forward_pe': info.get('forwardPE', None),
                    'peg_ratio': info.get('pegRatio', None),
                    'price_to_book': info.get('priceToBook', None),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                    'ev_to_revenue': info.get('enterpriseToRevenue', None),
                    'ev_to_ebitda': info.get('enterpriseToEbitda', None),
                    'dividend_yield': info.get('dividendYield', None)
                }
                
                multiples.append(row)
        
        df = pd.DataFrame(multiples)
        if len(df) > 0:
            df = df.set_index('ticker')
        
        return df
    
    def get_profitability_metrics(self) -> pd.DataFrame:
        """
        Get profitability metrics for all companies.
        
        Returns:
            DataFrame with profitability metrics
        """
        metrics = []
        
        for ticker in self.all_tickers:
            if self._data.get(ticker):
                info = self._data[ticker]['info']
                
                row = {
                    'ticker': ticker,
                    'company_name': info.get('shortName', ticker),
                    'gross_margin': info.get('grossMargins', None),
                    'operating_margin': info.get('operatingMargins', None),
                    'profit_margin': info.get('profitMargins', None),
                    'roe': info.get('returnOnEquity', None),
                    'roa': info.get('returnOnAssets', None),
                    'roic': self._calculate_roic(ticker)
                }
                
                metrics.append(row)
        
        df = pd.DataFrame(metrics)
        if len(df) > 0:
            df = df.set_index('ticker')
        
        return df
    
    def get_growth_metrics(self) -> pd.DataFrame:
        """
        Get growth metrics for all companies.
        
        Returns:
            DataFrame with growth metrics
        """
        metrics = []
        
        for ticker in self.all_tickers:
            if self._data.get(ticker):
                info = self._data[ticker]['info']
                
                row = {
                    'ticker': ticker,
                    'company_name': info.get('shortName', ticker),
                    'revenue_growth': info.get('revenueGrowth', None),
                    'earnings_growth': info.get('earningsGrowth', None),
                    'revenue_per_share': info.get('revenuePerShare', None),
                    'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', None)
                }
                
                metrics.append(row)
        
        df = pd.DataFrame(metrics)
        if len(df) > 0:
            df = df.set_index('ticker')
        
        return df
    
    def get_financial_health(self) -> pd.DataFrame:
        """
        Get financial health metrics for all companies.
        
        Returns:
            DataFrame with financial health metrics
        """
        metrics = []
        
        for ticker in self.all_tickers:
            if self._data.get(ticker):
                info = self._data[ticker]['info']
                
                row = {
                    'ticker': ticker,
                    'company_name': info.get('shortName', ticker),
                    'current_ratio': info.get('currentRatio', None),
                    'quick_ratio': info.get('quickRatio', None),
                    'debt_to_equity': info.get('debtToEquity', None),
                    'total_cash': info.get('totalCash', None),
                    'total_debt': info.get('totalDebt', None),
                    'free_cash_flow': info.get('freeCashflow', None)
                }
                
                metrics.append(row)
        
        df = pd.DataFrame(metrics)
        if len(df) > 0:
            df = df.set_index('ticker')
        
        return df
    
    def get_peer_statistics(self) -> Dict:
        """
        Calculate peer group statistics.
        
        Returns:
            Dictionary with peer statistics (mean, median, min, max)
        """
        multiples_df = self.get_valuation_multiples()
        
        # Exclude target company for peer statistics
        peer_df = multiples_df.drop(self.target_ticker, errors='ignore')
        
        stats = {}
        
        for column in peer_df.select_dtypes(include=[np.number]).columns:
            stats[column] = {
                'mean': peer_df[column].mean(),
                'median': peer_df[column].median(),
                'min': peer_df[column].min(),
                'max': peer_df[column].max(),
                'std': peer_df[column].std()
            }
        
        return stats
    
    def relative_valuation(self, method: str = 'pe_ratio') -> Dict:
        """
        Perform relative valuation based on peer multiples.
        
        Args:
            method: Valuation method (pe_ratio, ev_to_ebitda, price_to_sales, etc.)
        
        Returns:
            Dictionary with valuation results
        """
        multiples_df = self.get_valuation_multiples()
        
        # Get target and peer multiples
        target_multiple = multiples_df.loc[self.target_ticker, method] if self.target_ticker in multiples_df.index else None
        peer_multiples = multiples_df.drop(self.target_ticker, errors='ignore')[method]
        
        # Calculate peer statistics
        peer_mean = peer_multiples.mean()
        peer_median = peer_multiples.median()
        
        # Get target metrics
        if self._data.get(self.target_ticker):
            info = self._data[self.target_ticker]['info']
            current_price = info.get('currentPrice', 0)
            
            # Calculate implied valuations
            if method == 'pe_ratio':
                eps = info.get('trailingEps', 0)
                implied_price_mean = peer_mean * eps
                implied_price_median = peer_median * eps
            elif method == 'price_to_book':
                book_value_per_share = info.get('bookValue', 0)
                implied_price_mean = peer_mean * book_value_per_share
                implied_price_median = peer_median * book_value_per_share
            elif method == 'price_to_sales':
                revenue_per_share = info.get('revenuePerShare', 0)
                implied_price_mean = peer_mean * revenue_per_share
                implied_price_median = peer_median * revenue_per_share
            else:
                implied_price_mean = None
                implied_price_median = None
            
            upside_mean = (implied_price_mean - current_price) / current_price if implied_price_mean and current_price else None
            upside_median = (implied_price_median - current_price) / current_price if implied_price_median and current_price else None
        else:
            implied_price_mean = None
            implied_price_median = None
            upside_mean = None
            upside_median = None
            current_price = None
        
        return {
            'method': method,
            'target_ticker': self.target_ticker,
            'current_price': current_price,
            'target_multiple': target_multiple,
            'peer_mean_multiple': peer_mean,
            'peer_median_multiple': peer_median,
            'implied_price_mean': implied_price_mean,
            'implied_price_median': implied_price_median,
            'upside_downside_mean': upside_mean,
            'upside_downside_median': upside_median,
            'premium_discount_to_peers': (target_multiple - peer_mean) / peer_mean if target_multiple and peer_mean else None
        }
    
    def comprehensive_comparison(self) -> Dict:
        """
        Get comprehensive comparison across all metrics.
        
        Returns:
            Dictionary with all comparison data
        """
        return {
            'valuation_multiples': self.get_valuation_multiples(),
            'profitability': self.get_profitability_metrics(),
            'growth': self.get_growth_metrics(),
            'financial_health': self.get_financial_health(),
            'peer_statistics': self.get_peer_statistics(),
            'relative_valuation_pe': self.relative_valuation('pe_ratio'),
            'relative_valuation_ev_ebitda': self.relative_valuation('ev_to_ebitda'),
            'relative_valuation_ps': self.relative_valuation('price_to_sales')
        }
    
    def _calculate_roic(self, ticker: str) -> Optional[float]:
        """Calculate ROIC for a company."""
        try:
            data = self._data.get(ticker)
            if not data or len(data['financials'].columns) == 0:
                return None
            
            latest_financials = data['financials'].iloc[:, 0]
            latest_bs = data['balance_sheet'].iloc[:, 0]
            
            operating_income = latest_financials.get('Operating Income', 0)
            nopat = operating_income * (1 - 0.21)  # After-tax
            
            total_assets = latest_bs.get('Total Assets', 0)
            current_liabilities = latest_bs.get('Current Liabilities', 0)
            cash = latest_bs.get('Cash And Cash Equivalents', 0)
            
            invested_capital = total_assets - current_liabilities - cash
            
            return nopat / invested_capital if invested_capital > 0 else None
        except:
            return None


class ValuationMultiples:
    """
    Advanced valuation multiple calculations and analysis.
    """
    
    @staticmethod
    def calculate_ev(market_cap: float,
                    total_debt: float,
                    cash: float,
                    minority_interest: float = 0,
                    preferred_stock: float = 0) -> float:
        """
        Calculate Enterprise Value.
        
        EV = Market Cap + Debt - Cash + Minority Interest + Preferred Stock
        
        Args:
            market_cap: Market capitalization
            total_debt: Total debt
            cash: Cash and equivalents
            minority_interest: Minority interest
            preferred_stock: Preferred stock
        
        Returns:
            Enterprise value
        """
        return market_cap + total_debt - cash + minority_interest + preferred_stock
    
    @staticmethod
    def forward_multiples(trailing_multiple: float,
                        growth_rate: float,
                        periods: int = 1) -> float:
        """
        Calculate forward multiple from trailing multiple.
        
        Args:
            trailing_multiple: Trailing multiple
            growth_rate: Expected growth rate
            periods: Number of periods forward
        
        Returns:
            Forward multiple
        """
        return trailing_multiple / ((1 + growth_rate) ** periods)
    
    @staticmethod
    def normalized_multiples(multiples: List[float],
                           method: str = 'winsorize') -> List[float]:
        """
        Normalize multiples by removing outliers.
        
        Args:
            multiples: List of multiples
            method: Normalization method ('winsorize', 'clip', 'zscore')
        
        Returns:
            Normalized multiples
        """
        multiples_array = np.array(multiples)
        
        if method == 'winsorize':
            # Winsorize at 5th and 95th percentiles
            p5, p95 = np.percentile(multiples_array, [5, 95])
            return np.clip(multiples_array, p5, p95).tolist()
        
        elif method == 'clip':
            # Remove values outside 1.5 * IQR
            q1, q3 = np.percentile(multiples_array, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return np.clip(multiples_array, lower, upper).tolist()
        
        elif method == 'zscore':
            # Remove values with |z-score| > 3
            mean = multiples_array.mean()
            std = multiples_array.std()
            z_scores = np.abs((multiples_array - mean) / std)
            return multiples_array[z_scores < 3].tolist()
        
        return multiples
    
    @staticmethod
    def harmonic_mean_multiple(multiples: List[float]) -> float:
        """
        Calculate harmonic mean of multiples (better for ratios).
        
        Args:
            multiples: List of multiples
        
        Returns:
            Harmonic mean
        """
        multiples_array = np.array([m for m in multiples if m > 0])
        if len(multiples_array) == 0:
            return 0
        
        return len(multiples_array) / np.sum(1 / multiples_array)
    
    @staticmethod
    def weighted_average_multiple(multiples: List[float],
                                 weights: List[float]) -> float:
        """
        Calculate weighted average multiple.
        
        Args:
            multiples: List of multiples
            weights: List of weights (e.g., market caps)
        
        Returns:
            Weighted average multiple
        """
        total_weight = sum(weights)
        if total_weight == 0:
            return 0
        
        return sum(m * w for m, w in zip(multiples, weights)) / total_weight
