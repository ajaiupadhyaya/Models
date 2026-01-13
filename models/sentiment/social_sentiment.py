"""
Social Media Sentiment Analysis
Reddit, Twitter/X sentiment (placeholder for when APIs available)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SocialMediaSentiment:
    """
    Social media sentiment analysis (requires API keys).
    """
    
    def __init__(self, 
                 reddit_client_id: Optional[str] = None,
                 reddit_client_secret: Optional[str] = None,
                 twitter_bearer_token: Optional[str] = None):
        """
        Initialize social media sentiment analyzer.
        
        Args:
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            twitter_bearer_token: Twitter API bearer token
        """
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret
        self.twitter_bearer_token = twitter_bearer_token
        
        self._setup_apis()
    
    def _setup_apis(self):
        """Setup API connections."""
        self.reddit_available = False
        self.twitter_available = False
        
        # Reddit setup (requires praw)
        if self.reddit_client_id and self.reddit_client_secret:
            try:
                import praw
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent='financial_models_app'
                )
                self.reddit_available = True
            except ImportError:
                print("praw not installed. Install with: pip install praw")
            except Exception as e:
                print(f"Reddit API setup failed: {e}")
        
        # Twitter setup (requires tweepy)
        if self.twitter_bearer_token:
            try:
                import tweepy
                self.twitter = tweepy.Client(bearer_token=self.twitter_bearer_token)
                self.twitter_available = True
            except ImportError:
                print("tweepy not installed. Install with: pip install tweepy")
            except Exception as e:
                print(f"Twitter API setup failed: {e}")
    
    def get_reddit_sentiment(self, 
                           ticker: str,
                           subreddit: str = 'wallstreetbets',
                           limit: int = 100) -> Dict:
        """
        Get sentiment from Reddit.
        
        Args:
            ticker: Stock ticker
            subreddit: Subreddit to search
            limit: Number of posts to analyze
        
        Returns:
            Dictionary with sentiment metrics
        """
        if not self.reddit_available:
            return {'error': 'Reddit API not available'}
        
        try:
            # Search for ticker mentions
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            posts = []
            for submission in subreddit_obj.search(ticker, limit=limit, sort='new'):
                posts.append({
                    'title': submission.title,
                    'text': submission.selftext,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'created_utc': datetime.fromtimestamp(submission.created_utc),
                    'upvote_ratio': submission.upvote_ratio
                })
            
            if not posts:
                return {'error': f'No posts found for {ticker}'}
            
            # Analyze sentiment (simple approach based on upvote ratio and score)
            df = pd.DataFrame(posts)
            
            avg_sentiment = df['upvote_ratio'].mean()
            total_engagement = df['score'].sum() + df['num_comments'].sum()
            mention_count = len(posts)
            
            return {
                'ticker': ticker,
                'subreddit': subreddit,
                'mention_count': mention_count,
                'avg_upvote_ratio': avg_sentiment,
                'total_engagement': total_engagement,
                'avg_score': df['score'].mean(),
                'avg_comments': df['num_comments'].mean(),
                'sentiment_interpretation': 'Bullish' if avg_sentiment > 0.7 else 'Bearish' if avg_sentiment < 0.5 else 'Neutral',
                'posts': posts[:10]  # Return top 10 posts
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_twitter_sentiment(self, 
                            ticker: str,
                            max_results: int = 100) -> Dict:
        """
        Get sentiment from Twitter/X.
        
        Args:
            ticker: Stock ticker (will search for $TICKER format)
            max_results: Maximum tweets to fetch
        
        Returns:
            Dictionary with sentiment metrics
        """
        if not self.twitter_available:
            return {'error': 'Twitter API not available'}
        
        try:
            # Search for ticker (cashtag format)
            query = f'${ticker} -is:retweet'
            
            tweets = self.twitter.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # API limit
                tweet_fields=['created_at', 'public_metrics', 'text']
            )
            
            if not tweets.data:
                return {'error': f'No tweets found for ${ticker}'}
            
            tweet_data = []
            for tweet in tweets.data:
                tweet_data.append({
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count']
                })
            
            df = pd.DataFrame(tweet_data)
            
            # Simple engagement-based sentiment
            total_engagement = df['likes'].sum() + df['retweets'].sum() + df['replies'].sum()
            avg_engagement = total_engagement / len(df)
            
            return {
                'ticker': ticker,
                'tweet_count': len(tweet_data),
                'total_engagement': total_engagement,
                'avg_engagement': avg_engagement,
                'avg_likes': df['likes'].mean(),
                'avg_retweets': df['retweets'].mean(),
                'recent_tweets': tweet_data[:10]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def reddit_trending_tickers(self, 
                               subreddit: str = 'wallstreetbets',
                               limit: int = 100,
                               min_mentions: int = 3) -> List[Dict]:
        """
        Find trending tickers on Reddit.
        
        Args:
            subreddit: Subreddit to analyze
            limit: Number of posts to analyze
            min_mentions: Minimum mentions to be included
        
        Returns:
            List of trending tickers with metrics
        """
        if not self.reddit_available:
            return [{'error': 'Reddit API not available'}]
        
        try:
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            # Common tickers to look for
            common_tickers = [
                'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'NVDA', 'META', 'AMD', 'GME', 'AMC', 'PLTR', 'COIN'
            ]
            
            ticker_mentions = {ticker: 0 for ticker in common_tickers}
            ticker_sentiment = {ticker: [] for ticker in common_tickers}
            
            for submission in subreddit_obj.hot(limit=limit):
                text = (submission.title + ' ' + submission.selftext).upper()
                
                for ticker in common_tickers:
                    if f'${ticker}' in text or f' {ticker} ' in text:
                        ticker_mentions[ticker] += 1
                        ticker_sentiment[ticker].append(submission.upvote_ratio)
            
            # Filter and format results
            trending = []
            for ticker, mentions in ticker_mentions.items():
                if mentions >= min_mentions:
                    avg_sentiment = np.mean(ticker_sentiment[ticker]) if ticker_sentiment[ticker] else 0
                    
                    trending.append({
                        'ticker': ticker,
                        'mentions': mentions,
                        'avg_sentiment': avg_sentiment,
                        'sentiment_interpretation': 'Bullish' if avg_sentiment > 0.7 else 'Bearish' if avg_sentiment < 0.5 else 'Neutral'
                    })
            
            # Sort by mentions
            trending.sort(key=lambda x: x['mentions'], reverse=True)
            
            return trending
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def social_volume_indicator(self, 
                               ticker: str,
                               days: int = 7) -> Dict:
        """
        Track social media volume over time (requires multiple calls).
        
        Args:
            ticker: Stock ticker
            days: Number of days to track
        
        Returns:
            Dictionary with volume metrics
        """
        # This would require storing historical data
        # For now, return current metrics
        reddit_data = self.get_reddit_sentiment(ticker)
        twitter_data = self.get_twitter_sentiment(ticker)
        
        return {
            'ticker': ticker,
            'reddit_mentions': reddit_data.get('mention_count', 0),
            'twitter_mentions': twitter_data.get('tweet_count', 0),
            'total_social_mentions': reddit_data.get('mention_count', 0) + twitter_data.get('tweet_count', 0)
        }


class SentimentAggregator:
    """
    Aggregate sentiment from multiple sources.
    """
    
    def __init__(self):
        """Initialize sentiment aggregator."""
        pass
    
    def aggregate_sentiment(self, 
                          news_sentiment: Optional[Dict] = None,
                          reddit_sentiment: Optional[Dict] = None,
                          twitter_sentiment: Optional[Dict] = None,
                          market_sentiment: Optional[Dict] = None) -> Dict:
        """
        Aggregate sentiment from multiple sources.
        
        Args:
            news_sentiment: News sentiment data
            reddit_sentiment: Reddit sentiment data
            twitter_sentiment: Twitter sentiment data
            market_sentiment: Market sentiment indicators
        
        Returns:
            Aggregated sentiment dictionary
        """
        scores = []
        sources = []
        
        # News sentiment (if available)
        if news_sentiment and 'overall_score' in news_sentiment:
            # Convert -1 to 1 scale to 0 to 100
            news_score = (news_sentiment['overall_score'] + 1) * 50
            scores.append(news_score)
            sources.append('news')
        
        # Reddit sentiment
        if reddit_sentiment and 'avg_upvote_ratio' in reddit_sentiment:
            reddit_score = reddit_sentiment['avg_upvote_ratio'] * 100
            scores.append(reddit_score)
            sources.append('reddit')
        
        # Twitter sentiment (use engagement as proxy)
        if twitter_sentiment and 'avg_engagement' in twitter_sentiment:
            # Normalize engagement to 0-100 (arbitrary scale)
            twitter_score = min(twitter_sentiment['avg_engagement'] / 10, 100)
            scores.append(twitter_score)
            sources.append('twitter')
        
        # Market sentiment (Fear & Greed)
        if market_sentiment and 'score' in market_sentiment:
            scores.append(market_sentiment['score'])
            sources.append('market')
        
        if not scores:
            return {
                'error': 'No sentiment data available',
                'overall_score': 50,
                'interpretation': 'Neutral'
            }
        
        # Calculate weighted average (equal weights)
        overall_score = np.mean(scores)
        
        # Interpret overall sentiment
        if overall_score >= 70:
            interpretation = 'Very Bullish'
        elif overall_score >= 55:
            interpretation = 'Bullish'
        elif overall_score >= 45:
            interpretation = 'Neutral'
        elif overall_score >= 30:
            interpretation = 'Bearish'
        else:
            interpretation = 'Very Bearish'
        
        return {
            'overall_score': overall_score,
            'interpretation': interpretation,
            'sources': sources,
            'component_scores': dict(zip(sources, scores)),
            'confidence': len(sources) / 4.0  # Confidence based on number of sources
        }
