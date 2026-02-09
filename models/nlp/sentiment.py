"""
Financial Sentiment Analysis using FinBERT and Transformers.
Phase 2 - Awesome Quant Integration
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


class FinBERTSentiment:
    """
    Financial sentiment analysis using FinBERT from ProsusAI.
    Pre-trained on financial texts for accurate sentiment classification.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: Optional[str] = None):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cpu' or 'cuda' (auto-detects if None)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load FinBERT model: {e}")
    
    def analyze(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for financial texts.
        
        Args:
            texts: List of financial texts (news headlines, reports, etc.)
        
        Returns:
            DataFrame with sentiment scores and labels
        """
        if not texts:
            return pd.DataFrame()
        
        results = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # Predict
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                
                # FinBERT labels: 0=negative, 1=neutral, 2=positive
                sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                predicted_class = int(probs.argmax())
                sentiment = sentiment_map[predicted_class]
                confidence = float(probs[predicted_class])
                
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'negative_score': float(probs[0]),
                    'neutral_score': float(probs[1]),
                    'positive_score': float(probs[2])
                })
        
        return pd.DataFrame(results)
    
    def get_aggregate_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get aggregated sentiment across multiple texts.
        
        Args:
            texts: List of financial texts
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not texts:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'avg_confidence': 0.0,
                'num_texts': 0
            }
        
        df = self.analyze(texts)
        
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        
        # Calculate sentiment score: positive=+1, neutral=0, negative=-1
        sentiment_values = df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
        sentiment_score = float(sentiment_values.mean())
        
        # Determine overall sentiment
        max_sentiment = sentiment_counts.idxmax() if not sentiment_counts.empty else 'neutral'
        
        return {
            'overall_sentiment': max_sentiment,
            'sentiment_score': sentiment_score,
            'positive_ratio': float(sentiment_counts.get('positive', 0) / total),
            'negative_ratio': float(sentiment_counts.get('negative', 0) / total),
            'neutral_ratio': float(sentiment_counts.get('neutral', 0) / total),
            'avg_confidence': float(df['confidence'].mean()),
            'num_texts': total
        }


class SimpleSentiment:
    """
    Simple rule-based sentiment as fallback when FinBERT unavailable.
    Uses keyword matching for basic sentiment classification.
    """
    
    POSITIVE_WORDS = {
        'profit', 'growth', 'gain', 'increase', 'rose', 'surge', 'beat',
        'outperform', 'strong', 'bullish', 'upgrade', 'positive', 'success',
        'record', 'high', 'rally', 'boom', 'soar', 'jump'
    }
    
    NEGATIVE_WORDS = {
        'loss', 'decline', 'fall', 'decrease', 'drop', 'miss', 'weak',
        'underperform', 'bearish', 'downgrade', 'negative', 'failure',
        'low', 'crash', 'plunge', 'tumble', 'slump', 'recession', 'risk'
    }
    
    @classmethod
    def analyze(cls, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment using keyword matching.
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            DataFrame with sentiment scores
        """
        results = []
        
        for text in texts:
            text_lower = text.lower()
            words = set(text_lower.split())
            
            positive_count = sum(1 for word in cls.POSITIVE_WORDS if word in words)
            negative_count = sum(1 for word in cls.NEGATIVE_WORDS if word in words)
            
            total = positive_count + negative_count
            
            if total == 0:
                sentiment = 'neutral'
                positive_score = 0.33
                negative_score = 0.33
                neutral_score = 0.34
                confidence = 0.5
            else:
                positive_score = positive_count / total
                negative_score = negative_count / total
                neutral_score = 0.0
                
                if positive_score > negative_score:
                    sentiment = 'positive'
                    confidence = positive_score
                elif negative_score > positive_score:
                    sentiment = 'negative'
                    confidence = negative_score
                else:
                    sentiment = 'neutral'
                    confidence = 0.5
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'negative_score': negative_score,
                'neutral_score': neutral_score,
                'positive_score': positive_score
            })
        
        return pd.DataFrame(results)


class SentimentDrivenStrategy:
    """
    Trading strategy that incorporates sentiment signals.
    Combines price action with sentiment for signal generation.
    """
    
    def __init__(self, sentiment_analyzer: Optional[FinBERTSentiment] = None):
        """
        Initialize sentiment-driven strategy.
        
        Args:
            sentiment_analyzer: FinBERTSentiment instance (uses SimpleSentiment if None)
        """
        self.sentiment_analyzer = sentiment_analyzer or SimpleSentiment()
    
    def generate_signals(
        self,
        news_texts: List[str],
        price_signal: float = 0.0,
        sentiment_weight: float = 0.3
    ) -> float:
        """
        Generate trading signal combining sentiment and price action.
        
        Args:
            news_texts: Recent news headlines/texts
            price_signal: Technical signal from price action (-1 to 1)
            sentiment_weight: Weight for sentiment (0-1), rest for price
        
        Returns:
            Combined signal (-1 to 1): 1=strong buy, -1=strong sell, 0=neutral
        """
        if not news_texts:
            return price_signal
        
        # Get sentiment
        if isinstance(self.sentiment_analyzer, FinBERTSentiment):
            sentiment_data = self.sentiment_analyzer.get_aggregate_sentiment(news_texts)
            sentiment_score = sentiment_data['sentiment_score']
        else:
            df = self.sentiment_analyzer.analyze(news_texts)
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            sentiment_score = df['sentiment'].map(sentiment_map).mean()
        
        # Combine signals
        combined_signal = (
            sentiment_weight * sentiment_score +
            (1 - sentiment_weight) * price_signal
        )
        
        return float(np.clip(combined_signal, -1, 1))
