"""
Company Search Module

Search and select companies for analysis with smart matching and caching.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import yfinance as yf
from fuzzywuzzy import fuzz
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CompanySearch:
    """
    Smart company search with fuzzy matching, caching, and comprehensive filtering.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize company search.
        
        Args:
            cache_dir: Directory for caching search data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "company_database.json"
        self.cache_ttl = timedelta(days=7)  # Refresh weekly
        self._company_db = None
    
    @property
    def company_db(self) -> Dict:
        """Lazy load company database with caching."""
        if self._company_db is None:
            self._company_db = self._load_or_build_database()
        return self._company_db
    
    def _load_or_build_database(self) -> Dict:
        """Load database from cache or build new one."""
        # Check if cache exists and is fresh
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
                if datetime.now() - cache_time < self.cache_ttl:
                    logger.info(f"Loaded {len(cached_data['companies'])} companies from cache")
                    return cached_data['companies']
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        
        # Build new database
        return self._build_database()
    
    def _build_database(self) -> Dict:
        """Build comprehensive company database from multiple sources."""
        logger.info("Building company database...")
        
        companies = {}
        
        # Major indices components
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000'
        }
        
        # Get tickers from major ETFs as proxy for indices
        etfs = ['SPY', 'QQQ', 'DIA', 'IWM']
        
        for etf in etfs:
            try:
                ticker = yf.Ticker(etf)
                holdings = ticker.get_holdings()
                if holdings is not None and not holdings.empty:
                    for idx, row in holdings.iterrows():
                        symbol = row.get('symbol', '')
                        if symbol:
                            companies[symbol] = {
                                'ticker': symbol,
                                'name': row.get('name', ''),
                                'sector': row.get('sector', ''),
                                'weight': row.get('weight', 0)
                            }
            except Exception as e:
                logger.warning(f"Could not fetch {etf} holdings: {e}")
        
        # Add common large-cap stocks manually (fallback)
        manual_companies = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'BRK.B': 'Berkshire Hathaway Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.',
            'JNJ': 'Johnson & Johnson',
            'WMT': 'Walmart Inc.',
            'PG': 'Procter & Gamble Co.',
            'MA': 'Mastercard Inc.',
            'UNH': 'UnitedHealth Group Inc.',
            'DIS': 'The Walt Disney Company',
            'HD': 'The Home Depot Inc.',
            'PYPL': 'PayPal Holdings Inc.',
            'BAC': 'Bank of America Corp',
            'NFLX': 'Netflix Inc.',
            'ADBE': 'Adobe Inc.',
            'CRM': 'Salesforce Inc.',
            'CMCSA': 'Comcast Corporation',
            'PFE': 'Pfizer Inc.',
            'XOM': 'Exxon Mobil Corporation',
            'KO': 'The Coca-Cola Company',
            'PEP': 'PepsiCo Inc.',
            'INTC': 'Intel Corporation',
            'AMD': 'Advanced Micro Devices Inc.',
            'NKE': 'NIKE Inc.',
            'TMO': 'Thermo Fisher Scientific Inc.',
            'ABT': 'Abbott Laboratories',
            'COST': 'Costco Wholesale Corporation',
            'AVGO': 'Broadcom Inc.',
            'CVX': 'Chevron Corporation',
            'ACN': 'Accenture plc',
            'MCD': 'McDonald\'s Corporation',
            'MDT': 'Medtronic plc',
            'TXN': 'Texas Instruments Inc.',
            'UNP': 'Union Pacific Corporation',
            'HON': 'Honeywell International Inc.',
            'QCOM': 'QUALCOMM Inc.',
            'NEE': 'NextEra Energy Inc.',
            'LIN': 'Linde plc',
            'UPS': 'United Parcel Service Inc.',
            'LOW': 'Lowe\'s Companies Inc.',
            'BMY': 'Bristol-Myers Squibb Co.',
            'IBM': 'International Business Machines',
            'BA': 'The Boeing Company',
            'SBUX': 'Starbucks Corporation',
            'GE': 'General Electric Company',
            'F': 'Ford Motor Company',
            'GM': 'General Motors Company',
            'UBER': 'Uber Technologies Inc.',
            'LYFT': 'Lyft Inc.',
            'SPOT': 'Spotify Technology S.A.',
            'SNAP': 'Snap Inc.',
            'SQ': 'Block Inc.',
            'COIN': 'Coinbase Global Inc.',
            'RBLX': 'Roblox Corporation',
            'ABNB': 'Airbnb Inc.',
            'RIVN': 'Rivian Automotive Inc.',
            'LCID': 'Lucid Group Inc.',
        }
        
        for ticker, name in manual_companies.items():
            if ticker not in companies:
                companies[ticker] = {
                    'ticker': ticker,
                    'name': name,
                    'sector': '',
                    'weight': 0
                }
        
        # Enrich with additional data
        for ticker in list(companies.keys())[:100]:  # Limit to avoid rate limits
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                companies[ticker].update({
                    'name': info.get('longName', companies[ticker].get('name', '')),
                    'sector': info.get('sector', companies[ticker].get('sector', '')),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'country': info.get('country', ''),
                })
            except Exception as e:
                logger.warning(f"Could not enrich {ticker}: {e}")
        
        # Save to cache
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'companies': companies
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cached {len(companies)} companies")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
        
        return companies
    
    def search(self,
               query: str,
               limit: int = 10,
               min_score: int = 60) -> List[Dict]:
        """
        Search for companies with fuzzy matching.
        
        Args:
            query: Search query (company name or ticker)
            limit: Maximum number of results
            min_score: Minimum fuzzy match score (0-100)
        
        Returns:
            List of matching companies with scores
        """
        query = query.upper().strip()
        results = []
        
        for ticker, data in self.company_db.items():
            # Exact ticker match
            if query == ticker:
                results.append({
                    **data,
                    'match_score': 100,
                    'match_type': 'ticker_exact'
                })
                continue
            
            # Fuzzy ticker match
            ticker_score = fuzz.ratio(query, ticker)
            if ticker_score >= min_score:
                results.append({
                    **data,
                    'match_score': ticker_score,
                    'match_type': 'ticker_fuzzy'
                })
                continue
            
            # Fuzzy name match
            name = data.get('name', '')
            name_score = fuzz.partial_ratio(query, name.upper())
            if name_score >= min_score:
                results.append({
                    **data,
                    'match_score': name_score,
                    'match_type': 'name_fuzzy'
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results[:limit]
    
    def get_by_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Get company by exact ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Company data or None
        """
        ticker = ticker.upper()
        
        # Check database first
        if ticker in self.company_db:
            return self.company_db[ticker]
        
        # Try to fetch from yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info and 'symbol' in info:
                return {
                    'ticker': ticker,
                    'name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'country': info.get('country', ''),
                }
        except Exception as e:
            logger.warning(f"Could not fetch {ticker}: {e}")
        
        return None
    
    def filter_by_sector(self, sector: str) -> List[Dict]:
        """Filter companies by sector."""
        return [
            data for data in self.company_db.values()
            if sector.lower() in data.get('sector', '').lower()
        ]
    
    def filter_by_market_cap(self,
                            min_cap: Optional[float] = None,
                            max_cap: Optional[float] = None) -> List[Dict]:
        """Filter companies by market capitalization."""
        results = []
        for data in self.company_db.values():
            market_cap = data.get('market_cap', 0)
            if min_cap and market_cap < min_cap:
                continue
            if max_cap and market_cap > max_cap:
                continue
            results.append(data)
        return results
    
    def get_top_companies(self, n: int = 50) -> List[Dict]:
        """Get top N companies by market cap."""
        companies = [
            data for data in self.company_db.values()
            if data.get('market_cap', 0) > 0
        ]
        companies.sort(key=lambda x: x.get('market_cap', 0), reverse=True)
        return companies[:n]
    
    def validate_ticker(self, ticker: str) -> Tuple[bool, str]:
        """
        Validate if a ticker exists and is tradeable.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            
            if not info or 'symbol' not in info:
                return False, f"Ticker {ticker} not found"
            
            # Check if actively traded
            if info.get('regularMarketPrice') is None:
                return False, f"Ticker {ticker} is not actively traded"
            
            return True, f"Valid ticker: {info.get('longName', ticker)}"
        
        except Exception as e:
            return False, f"Error validating {ticker}: {str(e)}"


def search_companies(query: str, limit: int = 10) -> List[Dict]:
    """
    Convenience function for company search.
    
    Args:
        query: Search query
        limit: Maximum results
    
    Returns:
        List of matching companies
    """
    searcher = CompanySearch()
    return searcher.search(query, limit=limit)


def get_company(ticker: str) -> Optional[Dict]:
    """
    Convenience function to get company by ticker.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Company data or None
    """
    searcher = CompanySearch()
    return searcher.get_by_ticker(ticker)
