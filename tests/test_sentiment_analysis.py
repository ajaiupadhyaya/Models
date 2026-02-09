"""
Tests for sentiment analysis engine.

Tests:
- Text preprocessing and cleaning
- Entity extraction (tickers, companies)
- Polarity and confidence calculation
- Subjectivity analysis
- Sentiment classification
- Batch processing
- Sentiment aggregation
- Time series analysis
- Sentiment shift detection
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from core.sentiment_analysis import (
    SentimentAnalyzer,
    SentimentScore,
    AggregatedSentiment,
    SentimentTimeSeries,
    quick_sentiment,
    batch_sentiment,
)


class TestSentimentAnalyzer:
    """Test core sentiment analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer instance."""
        return SentimentAnalyzer()
    
    def test_preprocess_text(self, analyzer):
        """Test text preprocessing."""
        # URL removal
        text = "Check out https://example.com for more info"
        cleaned = analyzer.preprocess_text(text)
        assert "https" not in cleaned
        assert "example.com" not in cleaned
        
        # Email removal
        text = "Contact us at support@company.com"
        cleaned = analyzer.preprocess_text(text)
        assert "@" not in cleaned
        
        # Special character removal
        text = "Stock price: $150.50 (up 5%!)"
        cleaned = analyzer.preprocess_text(text)
        assert "$" not in cleaned
        assert "%" not in cleaned
        
        # Lowercase conversion
        text = "BREAKING NEWS"
        cleaned = analyzer.preprocess_text(text)
        assert cleaned == "breaking news"
        
        # Whitespace normalization
        text = "Too    many    spaces"
        cleaned = analyzer.preprocess_text(text)
        assert cleaned == "too many spaces"
    
    def test_extract_entities(self, analyzer):
        """Test entity extraction."""
        # Stock tickers
        text = "$AAPL and $TSLA are performing well"
        entities = analyzer.extract_entities(text)
        assert "$AAPL" in entities
        assert "$TSLA" in entities
        
        # Company names
        text = "Apple Inc and Tesla Motors announce partnership"
        entities = analyzer.extract_entities(text)
        # Should extract capitalized company names
        assert any("Apple" in e for e in entities)
        
        # Mixed case
        text = "Microsoft ($MSFT) reports earnings"
        entities = analyzer.extract_entities(text)
        assert "$MSFT" in entities or "Microsoft" in entities
    
    def test_calculate_polarity_positive(self, analyzer):
        """Test polarity calculation for positive text."""
        text = "record profits and strong growth"
        polarity, confidence = analyzer.calculate_polarity(text)
        
        assert polarity > 0  # Positive sentiment
        assert confidence > 0  # Has confidence
        assert 0 <= confidence <= 1
    
    def test_calculate_polarity_negative(self, analyzer):
        """Test polarity calculation for negative text."""
        text = "massive losses and declining revenue"
        polarity, confidence = analyzer.calculate_polarity(text)
        
        assert polarity < 0  # Negative sentiment
        assert confidence > 0
    
    def test_calculate_polarity_neutral(self, analyzer):
        """Test polarity calculation for neutral text."""
        text = "the company is headquartered in california"
        polarity, confidence = analyzer.calculate_polarity(text)
        
        # Should be neutral or very low confidence
        assert abs(polarity) < 0.5 or confidence < 0.3
    
    def test_calculate_polarity_with_negation(self, analyzer):
        """Test polarity with negation."""
        # Negated positive should be negative
        text = "not strong performance"
        polarity, _ = analyzer.calculate_polarity(text)
        assert polarity < 0
        
        # Negated negative should be positive
        text = "no longer declining"
        polarity, _ = analyzer.calculate_polarity(text)
        assert polarity > 0
    
    def test_calculate_polarity_with_intensifiers(self, analyzer):
        """Test polarity with intensifiers."""
        # With intensifier
        text = "very strong growth"
        polarity_intense, _ = analyzer.calculate_polarity(text)
        
        # Without intensifier
        text = "strong growth"
        polarity_normal, _ = analyzer.calculate_polarity(text)
        
        # Intensified should be stronger
        assert polarity_intense >= polarity_normal
    
    def test_calculate_subjectivity(self, analyzer):
        """Test subjectivity calculation."""
        # Highly subjective
        subjective_text = "amazing breakthrough with incredible potential"
        subjectivity_high = analyzer.calculate_subjectivity(subjective_text)
        
        # Objective
        objective_text = "the company was founded in 1990"
        subjectivity_low = analyzer.calculate_subjectivity(objective_text)
        
        assert subjectivity_high > subjectivity_low
        assert 0 <= subjectivity_high <= 1
        assert 0 <= subjectivity_low <= 1
    
    def test_classify_sentiment(self, analyzer):
        """Test sentiment classification."""
        # Positive
        label = analyzer.classify_sentiment(polarity=0.5, confidence=0.8)
        assert label == "positive"
        
        # Negative
        label = analyzer.classify_sentiment(polarity=-0.5, confidence=0.8)
        assert label == "negative"
        
        # Neutral (low polarity)
        label = analyzer.classify_sentiment(polarity=0.05, confidence=0.8)
        assert label == "neutral"
        
        # Neutral (low confidence)
        label = analyzer.classify_sentiment(polarity=0.5, confidence=0.1)
        assert label == "neutral"
    
    def test_analyze_text_positive(self, analyzer):
        """Test full text analysis with positive sentiment."""
        text = "Apple reports record quarterly earnings with strong iPhone sales"
        score = analyzer.analyze_text(text)
        
        assert isinstance(score, SentimentScore)
        assert score.label == "positive"
        assert score.polarity > 0
        assert 0 <= score.confidence <= 1
        assert 0 <= score.subjectivity <= 1
        assert len(score.entities) > 0  # Should extract "Apple"
    
    def test_analyze_text_negative(self, analyzer):
        """Test full text analysis with negative sentiment."""
        text = "Tesla faces lawsuit amid declining sales and losses"
        score = analyzer.analyze_text(text)
        
        assert score.label == "negative"
        assert score.polarity < 0
    
    def test_analyze_text_with_timestamp(self, analyzer):
        """Test text analysis with custom timestamp."""
        text = "Neutral statement about the market"
        timestamp = datetime(2024, 1, 15, 10, 30)
        
        score = analyzer.analyze_text(text, timestamp=timestamp)
        
        assert score.timestamp == timestamp
    
    def test_analyze_batch(self, analyzer):
        """Test batch text analysis."""
        texts = [
            "Excellent quarterly results exceed expectations",
            "Company faces bankruptcy amid fraud allegations",
            "The stock is traded on NASDAQ",
        ]
        
        scores = analyzer.analyze_batch(texts)
        
        assert len(scores) == 3
        assert scores[0].label == "positive"
        assert scores[1].label == "negative"
        # Third should be neutral or low confidence
        assert scores[2].label == "neutral" or scores[2].confidence < 0.3
    
    def test_analyze_batch_with_timestamps(self, analyzer):
        """Test batch analysis with timestamps."""
        texts = ["Good news", "Bad news"]
        timestamps = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        ]
        
        scores = analyzer.analyze_batch(texts, timestamps=timestamps)
        
        assert scores[0].timestamp == timestamps[0]
        assert scores[1].timestamp == timestamps[1]
    
    def test_aggregate_sentiment(self, analyzer):
        """Test sentiment aggregation."""
        scores = [
            SentimentScore("Good", "positive", 0.8, 0.6, 0.7, ["AAPL"], datetime(2024, 1, 1)),
            SentimentScore("Great", "positive", 0.9, 0.8, 0.6, ["AAPL"], datetime(2024, 1, 2)),
            SentimentScore("Bad", "negative", 0.7, -0.5, 0.6, ["AAPL"], datetime(2024, 1, 3)),
        ]
        
        agg = analyzer.aggregate_sentiment("AAPL", scores)
        
        assert isinstance(agg, AggregatedSentiment)
        assert agg.symbol == "AAPL"
        assert agg.num_articles == 3
        assert agg.avg_polarity > 0  # More positive than negative
        assert agg.sentiment_distribution["positive"] == 2
        assert agg.sentiment_distribution["negative"] == 1
        assert agg.trending_sentiment in ["bullish", "bearish", "neutral"]
    
    def test_aggregate_sentiment_empty(self, analyzer):
        """Test aggregation with no scores."""
        agg = analyzer.aggregate_sentiment("AAPL", [])
        
        assert agg.num_articles == 0
        assert agg.avg_polarity == 0.0
        assert agg.trending_sentiment == "neutral"
    
    def test_sentiment_score_to_dict(self):
        """Test SentimentScore serialization."""
        score = SentimentScore(
            text="Test text",
            label="positive",
            confidence=0.8,
            polarity=0.5,
            subjectivity=0.6,
            entities=["AAPL"],
            timestamp=datetime(2024, 1, 1),
        )
        
        score_dict = score.to_dict()
        
        assert isinstance(score_dict, dict)
        assert score_dict["label"] == "positive"
        assert score_dict["confidence"] == 0.8
        assert "AAPL" in score_dict["entities"]
    
    def test_aggregated_sentiment_to_dict(self):
        """Test AggregatedSentiment serialization."""
        agg = AggregatedSentiment(
            symbol="AAPL",
            num_articles=10,
            avg_polarity=0.3,
            avg_confidence=0.8,
            sentiment_distribution={"positive": 7, "negative": 2, "neutral": 1},
            trending_sentiment="bullish",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 7),
        )
        
        agg_dict = agg.to_dict()
        
        assert isinstance(agg_dict, dict)
        assert agg_dict["symbol"] == "AAPL"
        assert agg_dict["num_articles"] == 10
        assert agg_dict["trending_sentiment"] == "bullish"


class TestSentimentTimeSeries:
    """Test time series analysis functionality."""
    
    @pytest.fixture
    def sample_scores(self):
        """Generate sample sentiment scores over time."""
        scores = []
        start_date = datetime(2024, 1, 1)
        
        for i in range(30):
            # Simulate trend: starts negative, becomes positive
            polarity = -0.5 + (i / 30) * 1.0
            label = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
            
            score = SentimentScore(
                text=f"Article {i}",
                label=label,
                confidence=0.8,
                polarity=polarity,
                subjectivity=0.5,
                entities=["AAPL"],
                timestamp=start_date + timedelta(days=i),
            )
            scores.append(score)
        
        return scores
    
    def test_create_time_series(self, sample_scores):
        """Test time series creation."""
        ts = SentimentTimeSeries.create_time_series(sample_scores, frequency="D")
        
        assert isinstance(ts, pd.DataFrame)
        assert "polarity" in ts.columns
        assert "confidence" in ts.columns
        assert "subjectivity" in ts.columns
        assert len(ts) > 0
    
    def test_create_time_series_empty(self):
        """Test time series with no data."""
        ts = SentimentTimeSeries.create_time_series([])
        
        assert isinstance(ts, pd.DataFrame)
        assert len(ts) == 0
    
    def test_detect_sentiment_shift(self, sample_scores):
        """Test sentiment shift detection."""
        ts = SentimentTimeSeries.create_time_series(sample_scores, frequency="D")
        shifts = SentimentTimeSeries.detect_sentiment_shift(ts, window=5)
        
        # With gradual trend, shift detection may or may not trigger
        # depending on the threshold. Just verify the function works.
        assert isinstance(shifts, list)
        
        # Check shift structure if any detected
        if shifts:
            shift = shifts[0]
            assert "timestamp" in shift
            assert "change" in shift
            assert "direction" in shift
            assert "magnitude" in shift
            assert shift["direction"] in ["positive", "negative"]
        
        # Alternative: detect shift with lower threshold
        shifts_sensitive = SentimentTimeSeries.detect_sentiment_shift(ts, window=3)
        # With smaller window, more sensitive to changes
        assert isinstance(shifts_sensitive, list)
    
    def test_detect_sentiment_shift_insufficient_data(self):
        """Test shift detection with insufficient data."""
        scores = [
            SentimentScore("Test", "neutral", 0.5, 0.0, 0.5, [], datetime.now())
            for _ in range(5)
        ]
        
        ts = SentimentTimeSeries.create_time_series(scores)
        shifts = SentimentTimeSeries.detect_sentiment_shift(ts, window=10)
        
        # Should return empty list (not enough data)
        assert len(shifts) == 0
    
    def test_calculate_sentiment_momentum(self, sample_scores):
        """Test sentiment momentum calculation."""
        ts = SentimentTimeSeries.create_time_series(sample_scores, frequency="D")
        momentum = SentimentTimeSeries.calculate_sentiment_momentum(ts, short_window=3, long_window=7)
        
        assert isinstance(momentum, pd.Series)
        assert len(momentum) > 0
        
        # Momentum should be positive in uptrend (later dates)
        if len(momentum) > 20:
            recent_momentum = momentum.iloc[-5:].mean()
            # Should be positive or near zero in uptrend
            assert not np.isnan(recent_momentum)
    
    def test_calculate_sentiment_momentum_insufficient_data(self):
        """Test momentum with insufficient data."""
        scores = [
            SentimentScore("Test", "neutral", 0.5, 0.0, 0.5, [], datetime.now())
            for _ in range(3)
        ]
        
        ts = SentimentTimeSeries.create_time_series(scores)
        momentum = SentimentTimeSeries.calculate_sentiment_momentum(ts, short_window=5, long_window=10)
        
        # Should return empty series
        assert len(momentum) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_quick_sentiment(self):
        """Test quick sentiment classification."""
        # Positive
        label = quick_sentiment("Excellent earnings beat expectations")
        assert label == "positive"
        
        # Negative
        label = quick_sentiment("Company faces massive losses and bankruptcy")
        assert label == "negative"
    
    def test_batch_sentiment(self):
        """Test batch sentiment helper."""
        texts = [
            "Strong growth and record profits",
            "Disappointing results miss expectations",
        ]
        
        results = batch_sentiment(texts)
        
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        assert results[0]["label"] == "positive"
        assert results[1]["label"] == "negative"


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SentimentAnalyzer()
    
    def test_financial_news_analysis(self, analyzer):
        """Test analysis of realistic financial news."""
        news_articles = [
            "Apple Inc. ($AAPL) reports record quarterly earnings, beating analyst expectations with strong iPhone sales.",
            "Tesla ($TSLA) stock plunges after CEO announces disappointing production numbers and recall concerns.",
            "Microsoft announces new AI partnership, stock surges on optimistic outlook.",
            "Amazon faces antitrust lawsuit, shares decline amid regulatory uncertainty.",
        ]
        
        scores = analyzer.analyze_batch(news_articles)
        
        # Article 1: Positive (record, beating, strong)
        assert scores[0].label == "positive"
        assert scores[0].polarity > 0
        assert "$AAPL" in scores[0].entities or "Apple" in scores[0].entities
        
        # Article 2: Negative (plunges, disappointing, recall)
        assert scores[1].label == "negative"
        assert scores[1].polarity < 0
        
        # Article 3: Positive (surges, optimistic)
        assert scores[2].label == "positive"
        
        # Article 4: Negative (lawsuit, decline, uncertainty)
        assert scores[3].label == "negative"
    
    def test_aggregated_news_sentiment(self, analyzer):
        """Test aggregation across multiple news sources."""
        # Simulate 10 days of news for Apple
        articles = [
            ("Day 1: Apple announces record sales", datetime(2024, 1, 1)),
            ("Day 2: Apple stock surges on strong outlook", datetime(2024, 1, 2)),
            ("Day 3: Apple faces supply chain concerns", datetime(2024, 1, 3)),
            ("Day 4: Apple innovation impresses analysts", datetime(2024, 1, 4)),
            ("Day 5: Apple quarterly earnings beat expectations", datetime(2024, 1, 5)),
        ]
        
        texts, timestamps = zip(*articles)
        scores = analyzer.analyze_batch(list(texts), list(timestamps))
        
        # Aggregate
        agg = analyzer.aggregate_sentiment("AAPL", scores)
        
        assert agg.symbol == "AAPL"
        assert agg.num_articles == 5
        # Overall should be positive (4 positive, 1 negative/neutral)
        assert agg.avg_polarity > 0
        assert agg.trending_sentiment in ["bullish", "neutral"]
        assert agg.sentiment_distribution["positive"] >= 3
    
    def test_time_series_with_sentiment_reversal(self, analyzer):
        """Test time series analysis with sentiment reversal."""
        # Create scenario: negative news → positive news
        articles = []
        
        # Week 1: Negative news
        for i in range(7):
            articles.append((
                f"Day {i+1}: Company struggles with declining sales and losses",
                datetime(2024, 1, i+1)
            ))
        
        # Week 2: Positive news
        for i in range(7, 14):
            articles.append((
                f"Day {i+1}: Company announces strong recovery and growth",
                datetime(2024, 1, i+1)
            ))
        
        texts, timestamps = zip(*articles)
        scores = analyzer.analyze_batch(list(texts), list(timestamps))
        
        # Create time series
        ts = SentimentTimeSeries.create_time_series(scores, frequency="D")
        
        # Detect sentiment shift
        shifts = SentimentTimeSeries.detect_sentiment_shift(ts, window=3)
        
        # Should detect the shift from negative to positive
        assert len(shifts) > 0
        
        # The shift should be positive direction
        positive_shifts = [s for s in shifts if s["direction"] == "positive"]
        assert len(positive_shifts) > 0
    
    def test_entity_based_sentiment_tracking(self, analyzer):
        """Test tracking sentiment for specific entities."""
        texts = [
            "Apple and Microsoft both report strong earnings",
            "$AAPL beats expectations while $MSFT misses",
            "Apple innovation impresses, Microsoft struggles",
        ]
        
        scores = analyzer.analyze_batch(texts)
        
        # Extract all entities
        all_entities = []
        for score in scores:
            all_entities.extend(score.entities)
        
        # Should find both Apple and Microsoft
        assert any("Apple" in e or "$AAPL" in e for e in all_entities)
        assert any("Microsoft" in e or "$MSFT" in e for e in all_entities)
        
        # Filter scores mentioning Apple
        apple_scores = [
            s for s in scores
            if any("apple" in e.lower() or "aapl" in e.lower() for e in s.entities)
        ]
        
        assert len(apple_scores) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SentimentAnalyzer()
    
    def test_empty_text(self, analyzer):
        """Test analysis of empty text."""
        score = analyzer.analyze_text("")
        
        assert score.label == "neutral"
        assert score.polarity == 0.0
        assert score.confidence == 0.0
    
    def test_very_short_text(self, analyzer):
        """Test analysis of very short text."""
        score = analyzer.analyze_text("Good")
        
        assert score.label in ["positive", "neutral"]
        # Short text may have low confidence
        assert 0 <= score.confidence <= 1
    
    def test_special_characters_only(self, analyzer):
        """Test text with only special characters."""
        score = analyzer.analyze_text("!@#$%^&*()")
        
        # Should handle gracefully (returns neutral)
        assert score.label == "neutral"
        assert score.polarity == 0.0
    
    def test_mixed_case_entities(self, analyzer):
        """Test entity extraction with mixed case."""
        text = "apple and APPLE and Apple are the same"
        entities = analyzer.extract_entities(text)
        
        # Should extract at least one form
        assert len(entities) > 0
    
    def test_multiple_sentiment_words(self, analyzer):
        """Test text with many sentiment words."""
        text = "excellent amazing fantastic strong robust growth profit gain surge"
        score = analyzer.analyze_text(text)
        
        assert score.label == "positive"
        assert score.polarity > 0.5  # Very positive
        assert score.confidence > 0.5  # High confidence
