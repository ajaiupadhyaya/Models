"""
News Sentiment Analysis
Analyze sentiment from financial news using free APIs and NLP
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import requests
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class NewsAggregator:
    """
    Aggregate news from multiple free sources.
    """
    
    def __init__(self):
        """Initialize news aggregator."""
        self.sources = {
            'finnhub': 'https://finnhub.io/api/v1',
            'alpha_vantage': 'https://www.alphavantage.co/query'
        }
    
    def get_company_news(self, 
                        ticker: str,
                        days_back: int = 7,
                        api_key: Optional[str] = None) -> List[Dict]:
        """
        Get company news from free sources.
        
        Args:
            ticker: Stock ticker
            days_back: Number of days to look back
            api_key: Optional API key for premium sources
        
        Returns:
            List of news articles
        """
        news_items = []
        
        # Try Finnhub (free tier available)
        if api_key:
            try:
                from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
                
                url = f"{self.sources['finnhub']}/company-news"
                params = {
                    'symbol': ticker,
                    'from': from_date,
                    'to': to_date,
                    'token': api_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    news_items.extend(response.json())
            except Exception as e:
                print(f"Error fetching from Finnhub: {e}")
        
        return news_items
    
    def get_market_news(self, 
                       category: str = 'general',
                       limit: int = 50) -> List[Dict]:
        """
        Get general market news.
        
        Args:
            category: News category
            limit: Number of articles
        
        Returns:
            List of news articles
        """
        # This would integrate with free news APIs
        # For now, returns empty list - requires API keys
        return []


class NewsSentimentAnalyzer:
    """
    Analyze sentiment from news articles using NLP.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self._load_sentiment_lexicon()
    
    def _load_sentiment_lexicon(self):
        """Load financial sentiment lexicon."""
        # Simplified financial sentiment words
        self.positive_words = {
            'gain', 'gains', 'growth', 'profit', 'profits', 'strong', 'beat',
            'beats', 'bullish', 'upgrade', 'upgraded', 'outperform', 'buy',
            'positive', 'rise', 'rises', 'surge', 'surges', 'rally', 'rallies',
            'boost', 'boosts', 'improve', 'improves', 'improved', 'success',
            'successful', 'opportunity', 'opportunities', 'recover', 'recovery',
            'advance', 'advances', 'breakthrough', 'innovation', 'strength'
        }
        
        self.negative_words = {
            'loss', 'losses', 'decline', 'declines', 'weak', 'miss', 'misses',
            'bearish', 'downgrade', 'downgraded', 'underperform', 'sell', 'negative',
            'fall', 'falls', 'drop', 'drops', 'plunge', 'plunges', 'crash', 'risk',
            'risks', 'concern', 'concerns', 'worry', 'worried', 'threat', 'threats',
            'fail', 'failure', 'problem', 'problems', 'crisis', 'scandal', 'fraud'
        }
        
        self.intensifiers = {'very', 'extremely', 'highly', 'significantly', 'substantially'}
        self.negations = {'not', 'no', 'never', 'neither', 'nor', 'nobody', 'none'}
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {'score': 0, 'sentiment': 'neutral', 'confidence': 0}
        
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        positive_count = 0
        negative_count = 0
        total_words = len(words)
        
        # Analyze with simple lexicon approach
        for i, word in enumerate(words):
            # Check for negation
            negated = False
            if i > 0 and words[i-1] in self.negations:
                negated = True
            
            # Check for intensifier
            intensified = False
            if i > 0 and words[i-1] in self.intensifiers:
                intensified = True
            
            weight = 1.5 if intensified else 1.0
            
            if word in self.positive_words:
                if negated:
                    negative_count += weight
                else:
                    positive_count += weight
            elif word in self.negative_words:
                if negated:
                    positive_count += weight
                else:
                    negative_count += weight
        
        # Calculate sentiment score (-1 to 1)
        if positive_count + negative_count > 0:
            score = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            score = 0
        
        # Determine sentiment category
        if score > 0.2:
            sentiment = 'positive'
        elif score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Calculate confidence based on number of sentiment words
        sentiment_words = positive_count + negative_count
        confidence = min(sentiment_words / max(total_words * 0.1, 1), 1.0)
        
        return {
            'score': score,
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_words': total_words
        }
    
    def analyze_news_batch(self, news_items: List[Dict]) -> Dict:
        """
        Analyze sentiment for batch of news articles.
        
        Args:
            news_items: List of news articles with 'headline' and/or 'summary'
        
        Returns:
            Dictionary with aggregated sentiment
        """
        if not news_items:
            return {
                'overall_score': 0,
                'overall_sentiment': 'neutral',
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        scores = []
        sentiments = []
        
        for article in news_items:
            # Combine headline and summary
            text = ""
            if 'headline' in article:
                text += article['headline'] + " "
            if 'summary' in article:
                text += article['summary']
            
            if text:
                result = self.analyze_text(text)
                scores.append(result['score'])
                sentiments.append(result['sentiment'])
        
        # Calculate aggregate metrics
        overall_score = np.mean(scores) if scores else 0
        
        sentiment_counts = Counter(sentiments)
        
        if overall_score > 0.1:
            overall_sentiment = 'positive'
        elif overall_score < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_score': overall_score,
            'overall_sentiment': overall_sentiment,
            'article_count': len(news_items),
            'positive_count': sentiment_counts.get('positive', 0),
            'negative_count': sentiment_counts.get('negative', 0),
            'neutral_count': sentiment_counts.get('neutral', 0),
            'score_distribution': {
                'mean': overall_score,
                'std': np.std(scores) if scores else 0,
                'min': min(scores) if scores else 0,
                'max': max(scores) if scores else 0
            }
        }
    
    def sentiment_time_series(self, 
                             news_items: List[Dict],
                             date_field: str = 'datetime') -> pd.Series:
        """
        Create sentiment time series from news.
        
        Args:
            news_items: List of news articles with dates
            date_field: Name of date field
        
        Returns:
            Time series of sentiment scores
        """
        sentiment_data = []
        
        for article in news_items:
            if date_field in article:
                text = article.get('headline', '') + ' ' + article.get('summary', '')
                result = self.analyze_text(text)
                
                date = pd.to_datetime(article[date_field])
                sentiment_data.append({
                    'date': date,
                    'score': result['score']
                })
        
        if not sentiment_data:
            return pd.Series()
        
        df = pd.DataFrame(sentiment_data)
        df = df.set_index('date')
        
        # Aggregate by date (average if multiple articles per day)
        return df['score'].resample('D').mean()
    
    def extract_key_themes(self, news_items: List[Dict], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract key themes/words from news.
        
        Args:
            news_items: List of news articles
            top_n: Number of top themes to return
        
        Returns:
            List of (theme, count) tuples
        """
        # Combine all text
        all_text = ""
        for article in news_items:
            all_text += article.get('headline', '') + ' '
            all_text += article.get('summary', '') + ' '
        
        # Simple word frequency (excluding common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                     'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
                     'these', 'those', 'it', 'its', 'they', 'their', 'them', 'we', 'our'}
        
        words = all_text.lower().split()
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        word_counts = Counter(filtered_words)
        
        return word_counts.most_common(top_n)


class SentimentSignals:
    """
    Generate trading signals from sentiment.
    """
    
    @staticmethod
    def sentiment_momentum(sentiment_series: pd.Series, window: int = 5) -> pd.Series:
        """
        Calculate sentiment momentum.
        
        Args:
            sentiment_series: Time series of sentiment scores
            window: Rolling window size
        
        Returns:
            Sentiment momentum series
        """
        return sentiment_series.rolling(window=window).mean()
    
    @staticmethod
    def sentiment_divergence(sentiment_series: pd.Series, 
                            price_series: pd.Series) -> pd.Series:
        """
        Calculate divergence between sentiment and price.
        
        Args:
            sentiment_series: Sentiment time series
            price_series: Price time series
        
        Returns:
            Divergence series
        """
        # Normalize both series
        sentiment_norm = (sentiment_series - sentiment_series.mean()) / sentiment_series.std()
        price_returns = price_series.pct_change()
        price_norm = (price_returns - price_returns.mean()) / price_returns.std()
        
        # Calculate divergence
        divergence = sentiment_norm - price_norm
        
        return divergence
    
    @staticmethod
    def sentiment_extremes(sentiment_series: pd.Series,
                          threshold: float = 2.0) -> Dict:
        """
        Identify extreme sentiment periods.
        
        Args:
            sentiment_series: Sentiment time series
            threshold: Number of std deviations for extreme
        
        Returns:
            Dictionary with extreme periods
        """
        mean = sentiment_series.mean()
        std = sentiment_series.std()
        
        extreme_positive = sentiment_series[sentiment_series > mean + threshold * std]
        extreme_negative = sentiment_series[sentiment_series < mean - threshold * std]
        
        return {
            'extreme_positive_dates': extreme_positive.index.tolist(),
            'extreme_negative_dates': extreme_negative.index.tolist(),
            'extreme_positive_count': len(extreme_positive),
            'extreme_negative_count': len(extreme_negative)
        }
