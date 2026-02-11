"""
Transformer Models for Financial Text Analysis
GPT-style models for news sentiment, earnings call analysis, and financial text processing
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForCausalLM, pipeline, TextClassificationPipeline
    )
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers library not available. Install with: pip install transformers torch")


class FinancialSentimentAnalyzer:
    """
    Financial sentiment analysis using transformer models.
    Analyzes news, earnings calls, social media for market sentiment.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize financial sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name for financial sentiment
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required. Install: pip install transformers torch")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded financial sentiment model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            # Fallback to general sentiment
            try:
                self.pipeline = pipeline("sentiment-analysis")
                logger.warning("Using general sentiment model as fallback")
            except Exception as e2:
                logger.error(f"Could not load any sentiment model: {e2}")
                raise
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of financial text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        if not self.pipeline:
            return {'error': 'Model not loaded'}
        
        try:
            result = self.pipeline(text)[0]
            return {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'label': result['label'],
                'score': result['score'],
                'sentiment': 'positive' if 'positive' in result['label'].lower() else 
                            'negative' if 'negative' in result['label'].lower() else 'neutral'
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {'error': str(e)}
    
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_news_headlines(self, headlines: List[str]) -> Dict:
        """
        Analyze sentiment of news headlines.
        
        Args:
            headlines: List of news headlines
        
        Returns:
            Dictionary with aggregated sentiment metrics
        """
        df = self.analyze_batch(headlines)
        
        if df.empty or 'error' in df.columns:
            return {'error': 'Analysis failed'}
        
        sentiment_counts = df['sentiment'].value_counts()
        avg_score = df['score'].mean()
        
        return {
            'total_headlines': len(headlines),
            'positive_count': sentiment_counts.get('positive', 0),
            'negative_count': sentiment_counts.get('negative', 0),
            'neutral_count': sentiment_counts.get('neutral', 0),
            'sentiment_ratio': sentiment_counts.get('positive', 0) / max(sentiment_counts.get('negative', 1), 1),
            'average_confidence': avg_score,
            'overall_sentiment': 'bullish' if sentiment_counts.get('positive', 0) > sentiment_counts.get('negative', 0) else 'bearish',
            'details': df.to_dict('records')
        }


class EarningsCallAnalyzer:
    """
    Analyze earnings call transcripts using transformer models.
    Extracts key insights, sentiment, and forward-looking statements.
    """
    
    def __init__(self):
        """Initialize earnings call analyzer."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required")
        
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        
        # Keywords for important sections
        self.key_sections = {
            'forward_guidance': ['guidance', 'outlook', 'forecast', 'expect', 'anticipate'],
            'revenue': ['revenue', 'sales', 'top line', 'gross revenue'],
            'profitability': ['profit', 'margin', 'earnings', 'bottom line', 'net income'],
            'risks': ['risk', 'challenge', 'uncertainty', 'concern', 'headwind'],
            'opportunities': ['opportunity', 'growth', 'expansion', 'investment', 'initiative']
        }
    
    def extract_sections(self, transcript: str) -> Dict[str, List[str]]:
        """
        Extract key sections from earnings call transcript.
        
        Args:
            transcript: Full transcript text
        
        Returns:
            Dictionary with extracted sections
        """
        transcript_lower = transcript.lower()
        sections = {key: [] for key in self.key_sections.keys()}
        
        sentences = transcript.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for section_type, keywords in self.key_sections.items():
                if any(keyword in sentence_lower for keyword in keywords):
                    sections[section_type].append(sentence.strip())
        
        return sections
    
    def analyze_transcript(self, transcript: str) -> Dict:
        """
        Comprehensive analysis of earnings call transcript.
        
        Args:
            transcript: Full transcript text
        
        Returns:
            Dictionary with analysis results
        """
        sections = self.extract_sections(transcript)
        
        # Analyze sentiment of each section
        section_sentiments = {}
        for section_type, sentences in sections.items():
            if sentences:
                combined_text = ' '.join(sentences[:5])  # Analyze first 5 sentences
                sentiment = self.sentiment_analyzer.analyze_text(combined_text)
                section_sentiments[section_type] = {
                    'sentence_count': len(sentences),
                    'sentiment': sentiment.get('sentiment', 'neutral'),
                    'confidence': sentiment.get('score', 0.5)
                }
        
        # Overall sentiment
        overall_sentiment = self.sentiment_analyzer.analyze_text(transcript[:1000])
        
        return {
            'overall_sentiment': overall_sentiment.get('sentiment', 'neutral'),
            'overall_confidence': overall_sentiment.get('score', 0.5),
            'section_analysis': section_sentiments,
            'key_metrics': {
                'total_sections_found': sum(1 for v in sections.values() if v),
                'forward_guidance_mentions': len(sections['forward_guidance']),
                'risk_mentions': len(sections['risks']),
                'opportunity_mentions': len(sections['opportunities'])
            }
        }


class FinancialTextGenerator:
    """
    Generate financial text using GPT-style transformer models.
    Useful for report generation, summaries, explanations.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize text generator.
        
        Args:
            model_name: HuggingFace model name (use financial fine-tuned if available)
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required")
        
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded text generation model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def generate_summary(self, 
                        prompt: str,
                        max_length: int = 200,
                        num_return_sequences: int = 1) -> List[str]:
        """
        Generate text summary from prompt.
        
        Args:
            prompt: Starting text/prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
        
        Returns:
            List of generated text sequences
        """
        try:
            results = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=0.7,
                do_sample=True
            )
            
            return [result['generated_text'] for result in results]
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return [f"Error: {str(e)}"]
    
    def generate_report_section(self, 
                                section_title: str,
                                data_summary: str,
                                max_length: int = 300) -> str:
        """
        Generate a report section based on title and data.
        
        Args:
            section_title: Title of the section
            data_summary: Summary of data/analysis
            max_length: Maximum length
        
        Returns:
            Generated report text
        """
        prompt = f"Section: {section_title}\n\nData Summary: {data_summary}\n\nAnalysis:"
        
        generated = self.generate_summary(prompt, max_length=max_length)
        return generated[0] if generated else ""


class MarketNewsAnalyzer:
    """
    Analyze market news and extract actionable insights.
    Combines sentiment analysis with entity extraction.
    """
    
    def __init__(self):
        """Initialize market news analyzer."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required")
        
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        
        try:
            from transformers import pipeline as hf_pipeline
            self.ner_pipeline = hf_pipeline("ner", aggregation_strategy="simple")
        except Exception as e:
            logger.warning(f"Could not load NER model: {e}")
            self.ner_pipeline = None
    
    def analyze_news_article(self, article_text: str) -> Dict:
        """
        Analyze a news article for sentiment and entities.
        
        Args:
            article_text: Full article text
        
        Returns:
            Dictionary with analysis results
        """
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.analyze_text(article_text)
        
        # Entity extraction
        entities = []
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(article_text[:512])  # Limit length
                entities = [{'text': e['word'], 'label': e['entity_group'], 'score': e['score']} 
                           for e in ner_results]
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")
        
        # Extract potential tickers (simple pattern matching)
        import re
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        potential_tickers = re.findall(ticker_pattern, article_text)
        potential_tickers = [t for t in potential_tickers if len(t) >= 2 and len(t) <= 5]
        
        return {
            'sentiment': sentiment.get('sentiment', 'neutral'),
            'sentiment_score': sentiment.get('score', 0.5),
            'entities': entities,
            'potential_tickers': list(set(potential_tickers)),
            'article_length': len(article_text),
            'key_insights': self._extract_insights(article_text)
        }
    
    def _extract_insights(self, text: str) -> List[str]:
        """Extract key insights from text."""
        # Simple keyword-based extraction
        insight_keywords = {
            'earnings': ['beat', 'miss', 'earnings', 'profit', 'revenue'],
            'guidance': ['raise', 'lower', 'guidance', 'forecast', 'outlook'],
            'merger': ['acquisition', 'merger', 'deal', 'buyout'],
            'regulation': ['regulation', 'approval', 'fda', 'sec', 'investigation']
        }
        
        insights = []
        text_lower = text.lower()
        
        for insight_type, keywords in insight_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                insights.append(insight_type)
        
        return insights
