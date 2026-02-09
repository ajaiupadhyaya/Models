"""
Sentiment Analysis Engine

Provides NLP-based sentiment analysis for financial news and social media:
- News article sentiment scoring
- Entity extraction (companies, people, locations)
- Topic modeling and classification
- Sentiment aggregation over time
- Multi-source sentiment fusion

Usage:
    from core.sentiment_analysis import SentimentAnalyzer, SentimentScore
    
    analyzer = SentimentAnalyzer()
    score = analyzer.analyze_text("Apple announces record quarterly earnings")
    print(f"Sentiment: {score.label} (confidence: {score.confidence:.2f})")
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment analysis result."""
    
    text: str
    label: str  # "positive", "negative", "neutral"
    confidence: float  # 0.0 to 1.0
    polarity: float  # -1.0 (negative) to +1.0 (positive)
    subjectivity: float  # 0.0 (objective) to 1.0 (subjective)
    entities: List[str]  # Extracted entities
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "text": self.text[:200],  # Truncate for storage
            "label": self.label,
            "confidence": self.confidence,
            "polarity": self.polarity,
            "subjectivity": self.subjectivity,
            "entities": self.entities,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment over multiple texts."""
    
    symbol: str
    num_articles: int
    avg_polarity: float
    avg_confidence: float
    sentiment_distribution: Dict[str, int]  # {"positive": 10, "negative": 2, "neutral": 5}
    trending_sentiment: str  # Overall trend
    start_date: datetime
    end_date: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "num_articles": self.num_articles,
            "avg_polarity": self.avg_polarity,
            "avg_confidence": self.avg_confidence,
            "sentiment_distribution": self.sentiment_distribution,
            "trending_sentiment": self.trending_sentiment,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }


class SentimentAnalyzer:
    """
    Core sentiment analysis engine using rule-based and lexicon approaches.
    
    Provides:
    - Text preprocessing and cleaning
    - Sentiment classification (positive/negative/neutral)
    - Entity extraction (companies, tickers, people)
    - Subjectivity analysis
    - Financial keyword detection
    """
    
    # Financial sentiment lexicons
    POSITIVE_WORDS = {
        "profit", "gain", "rise", "surge", "rally", "growth", "strong", "bullish",
        "upgrade", "beat", "outperform", "record", "high", "soar", "boom", "success",
        "earnings", "revenue", "innovation", "breakthrough", "expansion", "recovery",
        "positive", "optimistic", "confident", "momentum", "exceed", "robust", "stellar",
        "excellent", "great", "amazing", "impressive", "good", "improving"
    }
    
    NEGATIVE_WORDS = {
        "loss", "losses", "decline", "declining", "fall", "plunge", "crash", "weak", "bearish", "downgrade",
        "miss", "underperform", "low", "slump", "recession", "bankruptcy", "debt",
        "lawsuit", "scandal", "fraud", "investigation", "warning", "concern", "risk",
        "negative", "pessimistic", "volatile", "uncertainty", "disappointing", "struggle",
        "bad", "poor", "faces", "allegations", "amid"
    }
    
    # Intensifiers and negations
    INTENSIFIERS = {"very", "extremely", "highly", "significantly", "substantially"}
    NEGATIONS = {"not", "no", "never", "neither", "nobody", "nothing", "nowhere", "n't"}
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.positive_words = self.POSITIVE_WORDS
        self.negative_words = self.NEGATIVE_WORDS
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """
        Extract potential entities (company names, tickers).
        
        Args:
            text: Input text
            
        Returns:
            List of entities
        """
        entities = []
        
        # Extract stock tickers ($XXX pattern)
        tickers = re.findall(r'\$[A-Z]{1,5}\b', text)
        entities.extend(tickers)
        
        # Extract uppercase words (potential company names)
        # Look for sequences of capitalized words
        company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        companies = re.findall(company_pattern, text)
        entities.extend([c for c in companies if len(c) > 3])
        
        return list(set(entities))
    
    def calculate_polarity(self, text: str) -> Tuple[float, float]:
        """
        Calculate sentiment polarity and confidence.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Tuple of (polarity, confidence)
        """
        words = text.split()
        
        positive_count = 0.0
        negative_count = 0.0
        
        # Check each word with context
        for i, word in enumerate(words):
            # Check for negation in previous 1-3 words
            is_negated = False
            for j in range(max(0, i-3), i):
                if words[j] in self.NEGATIONS:
                    is_negated = True
                    break
            
            # Check for intensifier
            intensity = 1.0
            if i > 0 and words[i-1] in self.INTENSIFIERS:
                intensity = 1.5
            
            # Count sentiment
            if word in self.positive_words:
                if is_negated:
                    negative_count += intensity
                else:
                    positive_count += intensity
            elif word in self.negative_words:
                if is_negated:
                    positive_count += intensity
                else:
                    negative_count += intensity
        
        # Calculate polarity (-1 to +1)
        total = positive_count + negative_count
        if total == 0:
            return 0.0, 0.0  # Neutral with no confidence
        
        polarity = (positive_count - negative_count) / total
        
        # Confidence based on number of sentiment words found
        # More sentiment words = higher confidence
        confidence = min(total / 5.0, 1.0)  # Max confidence at 5 words (was 10)
        
        return polarity, confidence
    
    def calculate_subjectivity(self, text: str) -> float:
        """
        Calculate text subjectivity (0=objective, 1=subjective).
        
        Args:
            text: Preprocessed text
            
        Returns:
            Subjectivity score
        """
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        # Count subjective indicators
        sentiment_word_count = sum(
            1 for word in words 
            if word in self.positive_words or word in self.negative_words
        )
        
        intensifier_count = sum(1 for word in words if word in self.INTENSIFIERS)
        
        # Subjectivity based on proportion of opinionated words
        subjectivity = (sentiment_word_count + intensifier_count) / len(words)
        
        return min(subjectivity * 2, 1.0)  # Scale and cap at 1.0
    
    def classify_sentiment(self, polarity: float, confidence: float) -> str:
        """
        Classify sentiment label from polarity.
        
        Args:
            polarity: Polarity score (-1 to +1)
            confidence: Confidence in score
            
        Returns:
            Sentiment label
        """
        # Require minimum confidence for non-neutral classification
        if confidence < 0.2:
            return "neutral"
        
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def analyze_text(
        self,
        text: str,
        timestamp: Optional[datetime] = None,
    ) -> SentimentScore:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            SentimentScore object
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract entities before preprocessing (to preserve capitalization)
        entities = self.extract_entities(text)
        
        # Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Calculate metrics
        polarity, confidence = self.calculate_polarity(cleaned_text)
        subjectivity = self.calculate_subjectivity(cleaned_text)
        label = self.classify_sentiment(polarity, confidence)
        
        return SentimentScore(
            text=text,
            label=label,
            confidence=confidence,
            polarity=polarity,
            subjectivity=subjectivity,
            entities=entities,
            timestamp=timestamp,
        )
    
    def analyze_batch(
        self,
        texts: List[str],
        timestamps: Optional[List[datetime]] = None,
    ) -> List[SentimentScore]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts
            timestamps: Optional list of timestamps
            
        Returns:
            List of SentimentScore objects
        """
        if timestamps is None:
            timestamps = [datetime.now() for _ in texts]
        
        return [
            self.analyze_text(text, timestamp)
            for text, timestamp in zip(texts, timestamps)
        ]
    
    def aggregate_sentiment(
        self,
        symbol: str,
        scores: List[SentimentScore],
    ) -> AggregatedSentiment:
        """
        Aggregate sentiment scores for a symbol.
        
        Args:
            symbol: Stock symbol
            scores: List of sentiment scores
            
        Returns:
            AggregatedSentiment object
        """
        if not scores:
            return AggregatedSentiment(
                symbol=symbol,
                num_articles=0,
                avg_polarity=0.0,
                avg_confidence=0.0,
                sentiment_distribution={"positive": 0, "negative": 0, "neutral": 0},
                trending_sentiment="neutral",
                start_date=datetime.now(),
                end_date=datetime.now(),
            )
        
        # Calculate averages
        avg_polarity = np.mean([s.polarity for s in scores])
        avg_confidence = np.mean([s.confidence for s in scores])
        
        # Distribution
        distribution = {
            "positive": sum(1 for s in scores if s.label == "positive"),
            "negative": sum(1 for s in scores if s.label == "negative"),
            "neutral": sum(1 for s in scores if s.label == "neutral"),
        }
        
        # Determine trending sentiment
        if avg_polarity > 0.15:
            trending = "bullish"
        elif avg_polarity < -0.15:
            trending = "bearish"
        else:
            trending = "neutral"
        
        # Date range
        timestamps = [s.timestamp for s in scores]
        
        return AggregatedSentiment(
            symbol=symbol,
            num_articles=len(scores),
            avg_polarity=float(avg_polarity),
            avg_confidence=float(avg_confidence),
            sentiment_distribution=distribution,
            trending_sentiment=trending,
            start_date=min(timestamps),
            end_date=max(timestamps),
        )


class SentimentTimeSeries:
    """
    Time series analysis of sentiment data.
    
    Tracks sentiment changes over time and detects trends.
    """
    
    @staticmethod
    def create_time_series(
        scores: List[SentimentScore],
        frequency: str = "D",
    ) -> pd.DataFrame:
        """
        Create time series DataFrame from sentiment scores.
        
        Args:
            scores: List of sentiment scores
            frequency: Resampling frequency ('D'=daily, 'H'=hourly)
            
        Returns:
            DataFrame with time-indexed sentiment metrics
        """
        if not scores:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = {
            "timestamp": [s.timestamp for s in scores],
            "polarity": [s.polarity for s in scores],
            "confidence": [s.confidence for s in scores],
            "subjectivity": [s.subjectivity for s in scores],
        }
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        # Resample to specified frequency
        resampled = df.resample(frequency).agg({
            "polarity": "mean",
            "confidence": "mean",
            "subjectivity": "mean",
        })
        
        return resampled
    
    @staticmethod
    def detect_sentiment_shift(
        time_series: pd.DataFrame,
        window: int = 7,
    ) -> List[Dict]:
        """
        Detect significant sentiment shifts.
        
        Args:
            time_series: Time series DataFrame
            window: Rolling window size
            
        Returns:
            List of detected shifts with timestamps
        """
        if len(time_series) < window * 2:
            return []
        
        shifts = []
        
        # Calculate rolling mean
        rolling_mean = time_series["polarity"].rolling(window=window).mean()
        
        # Detect significant changes (> 0.3 difference)
        for i in range(window, len(rolling_mean) - window):
            current_mean = rolling_mean.iloc[i]
            previous_mean = rolling_mean.iloc[i - window]
            
            change = current_mean - previous_mean
            
            if abs(change) > 0.3:
                shifts.append({
                    "timestamp": time_series.index[i],
                    "change": float(change),
                    "direction": "positive" if change > 0 else "negative",
                    "magnitude": abs(float(change)),
                })
        
        return shifts
    
    @staticmethod
    def calculate_sentiment_momentum(
        time_series: pd.DataFrame,
        short_window: int = 3,
        long_window: int = 10,
    ) -> pd.Series:
        """
        Calculate sentiment momentum (similar to price momentum).
        
        Args:
            time_series: Time series DataFrame
            short_window: Short-term window
            long_window: Long-term window
            
        Returns:
            Momentum series
        """
        if len(time_series) < long_window:
            return pd.Series(dtype=float)
        
        short_ma = time_series["polarity"].rolling(window=short_window).mean()
        long_ma = time_series["polarity"].rolling(window=long_window).mean()
        
        momentum = short_ma - long_ma
        
        return momentum


# Convenience functions
def quick_sentiment(text: str) -> str:
    """
    Quick sentiment classification (positive/negative/neutral).
    
    Args:
        text: Input text
        
    Returns:
        Sentiment label
    """
    analyzer = SentimentAnalyzer()
    score = analyzer.analyze_text(text)
    return score.label


def batch_sentiment(texts: List[str]) -> List[Dict]:
    """
    Batch sentiment analysis returning dictionaries.
    
    Args:
        texts: List of texts
        
    Returns:
        List of sentiment dictionaries
    """
    analyzer = SentimentAnalyzer()
    scores = analyzer.analyze_batch(texts)
    return [s.to_dict() for s in scores]
