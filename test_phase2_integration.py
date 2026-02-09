"""
Phase 2 Integration Tests - Awesome Quant Integration
Tests: Sentiment Analysis, Multi-Factor Models, ML Feature Engineering, Factor Analysis
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all Phase 2 modules can be imported."""
    print("\n" + "="*60)
    print("TEST 1: Validating Phase 2 Imports")
    print("="*60)
    
    try:
        from models.nlp.sentiment import FinBERTSentiment, SimpleSentiment, SentimentDrivenStrategy
        print("✓ Sentiment analysis modules imported successfully")
        
        from models.factors.multi_factor import MultiFactorModel, FactorConstructor
        print("✓ Multi-factor model modules imported successfully")
        
        from models.ml.feature_engineering import LabelGenerator, FeatureTransformer
        print("✓ ML feature engineering modules imported successfully")
        
        from models.factors.factor_analysis import FactorAnalysis, SimpleFactorAnalysis
        print("✓ Factor analysis modules imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_simple_sentiment():
    """Test SimpleSentiment analyzer."""
    print("\n" + "="*60)
    print("TEST 2: SimpleSentiment Analysis")
    print("="*60)
    
    try:
        from models.nlp.sentiment import SimpleSentiment
        
        # Sample financial headlines
        headlines = [
            "Apple beats earnings expectations with strong iPhone sales",
            "Tesla stock plunges amid weak delivery numbers",
            "Microsoft announces neutral quarterly results",
            "Amazon revenue surges on cloud growth",
            "Market crash fears grow as recession risks increase"
        ]
        
        analyzer = SimpleSentiment()
        results = analyzer.analyze(headlines)
        
        print(f"Analyzed {len(headlines)} headlines")
        print(f"Results shape: {results.shape}")
        print(f"\nSentiment breakdown:")
        print(results['sentiment'].value_counts())
        print(f"\nAverage confidence: {results['confidence'].mean():.3f}")
        
        # Check structure
        assert 'sentiment' in results.columns
        assert 'confidence' in results.columns
        assert len(results) == len(headlines)
        
        print("✓ SimpleSentiment test passed")
        return True
    except Exception as e:
        print(f"✗ SimpleSentiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_driven_strategy():
    """Test SentimentDrivenStrategy."""
    print("\n" + "="*60)
    print("TEST 3: Sentiment-Driven Strategy")
    print("="*60)
    
    try:
        from models.nlp.sentiment import SentimentDrivenStrategy, SimpleSentiment
        
        # Sample news with different sentiments
        positive_news = [
            "Company profits soar to record highs",
            "Strong growth outlook announced",
            "Stock reaches new all-time high"
        ]
        
        negative_news = [
            "Company reports massive losses",
            "CEO resigns amid scandal",
            "Stock crashes 20% on weak guidance"
        ]
        
        strategy = SentimentDrivenStrategy(SimpleSentiment())
        
        # Test with positive news
        pos_signal = strategy.generate_signals(positive_news, price_signal=0.5, sentiment_weight=0.3)
        print(f"Positive news signal: {pos_signal:.3f} (expected > 0.5)")
        
        # Test with negative news
        neg_signal = strategy.generate_signals(negative_news, price_signal=-0.5, sentiment_weight=0.3)
        print(f"Negative news signal: {neg_signal:.3f} (expected < -0.5)")
        
        # Validate ranges
        assert -1 <= pos_signal <= 1, "Signal out of range"
        assert -1 <= neg_signal <= 1, "Signal out of range"
        assert pos_signal > neg_signal, "Positive news should give higher signal"
        
        print("✓ Sentiment-driven strategy test passed")
        return True
    except Exception as e:
        print(f"✗ Sentiment strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_factor_model():
    """Test MultiFactorModel with synthetic data."""
    print("\n" + "="*60)
    print("TEST 4: Multi-Factor Model")
    print("="*60)
    
    try:
        from models.factors.multi_factor import MultiFactorModel
        
        # Generate synthetic data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Factor returns (market, size, value)
        market_factor = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates, name='Market')
        size_factor = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates, name='Size')
        value_factor = pd.Series(np.random.normal(0.0003, 0.01, 252), index=dates, name='Value')
        
        factors = pd.DataFrame({
            'Market': market_factor,
            'Size': size_factor,
            'Value': value_factor
        })
        
        # Asset returns (linear combination of factors + alpha + noise)
        alpha = 0.0002  # 5% annual alpha
        beta_market = 1.2
        beta_size = 0.3
        beta_value = -0.5
        
        asset_returns = (
            alpha +
            beta_market * market_factor +
            beta_size * size_factor +
            beta_value * value_factor +
            np.random.normal(0, 0.01, 252)
        )
        
        # Fit model
        model = MultiFactorModel(asset_returns, factors)
        results = model.fit()
        
        print(f"Model R-squared: {results.rsquared:.4f}")
        
        alpha_est, alpha_pval = model.get_alpha()
        print(f"Alpha estimate: {alpha_est:.6f} (true: {alpha:.6f})")
        print(f"Alpha p-value: {alpha_pval:.4f}")
        
        exposures = model.get_factor_exposures()
        print(f"\nFactor exposures (betas):")
        print(f"  Market: {exposures['Market']:.3f} (true: {beta_market:.3f})")
        print(f"  Size: {exposures['Size']:.3f} (true: {beta_size:.3f})")
        print(f"  Value: {exposures['Value']:.3f} (true: {beta_value:.3f})")
        
        attribution = model.factor_attribution()
        print(f"\nFactor attribution:")
        for factor, contrib in attribution.items():
            print(f"  {factor}: {contrib:.6f}")
        
        # Validate results
        assert results.rsquared > 0.3, "R-squared too low"
        assert abs(exposures['Market'] - beta_market) < 0.3, "Market beta estimate off"
        
        print("✓ Multi-factor model test passed")
        return True
    except Exception as e:
        print(f"✗ Multi-factor model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_generation():
    """Test LabelGenerator methods."""
    print("\n" + "="*60)
    print("TEST 5: ML Label Generation")
    print("="*60)
    
    try:
        from models.ml.feature_engineering import LabelGenerator
        
        # Generate synthetic price series
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 200))), index=dates)
        returns = prices.pct_change().dropna()
        
        # Test 1: Fixed horizon labels
        labels_fixed = LabelGenerator.fixed_horizon_labels(returns, horizon=5, threshold=0.01)
        print(f"Fixed horizon labels: {len(labels_fixed)} generated")
        print(f"Label distribution: {labels_fixed.value_counts().to_dict()}")
        
        # Test 2: Triple-barrier labels
        labels_triple = LabelGenerator.triple_barrier_labels(
            prices,
            target_profit=0.03,
            stop_loss=0.02,
            max_holding_period=10
        )
        print(f"\nTriple-barrier labels: {len(labels_triple)} generated")
        print(f"Label distribution: {labels_triple['label'].value_counts().to_dict()}")
        print(f"Barrier distribution: {labels_triple['barrier_touched'].value_counts().to_dict()}")
        print(f"Average holding period: {labels_triple['holding_period'].mean():.1f} days")
        
        # Test 3: Meta-labeling
        primary_preds = pd.Series(np.random.choice([-1, 1], size=len(returns)), index=returns.index)
        forward_returns = returns.shift(-5)
        meta_labels = LabelGenerator.meta_labeling(primary_preds, forward_returns, threshold=0.0)
        print(f"\nMeta-labels: {len(meta_labels)} generated")
        print(f"Accuracy: {meta_labels.mean():.2%}")
        
        # Validate
        assert len(labels_fixed) > 0, "No fixed labels generated"
        assert len(labels_triple) > 0, "No triple-barrier labels generated"
        assert 'label' in labels_triple.columns, "Missing label column"
        assert 'barrier_touched' in labels_triple.columns, "Missing barrier_touched column"
        
        print("✓ Label generation test passed")
        return True
    except Exception as e:
        print(f"✗ Label generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_transformations():
    """Test FeatureTransformer methods."""
    print("\n" + "="*60)
    print("TEST 6: Feature Transformations")
    print("="*60)
    
    try:
        from models.ml.feature_engineering import FeatureTransformer
        
        # Generate synthetic series
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        # Non-stationary random walk
        prices = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 200)), index=dates)
        
        # Test 1: Fractional differentiation
        frac_diff = FeatureTransformer.fractional_differentiation(prices, d=0.5)
        print(f"Fractional diff: {frac_diff.notna().sum()} values (input: {len(prices)})")
        print(f"Mean: {frac_diff.mean():.4f}, Std: {frac_diff.std():.4f}")
        
        # Test 2: Time decay features
        features = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        decayed = FeatureTransformer.time_decay_features(features, half_life=20)
        print(f"\nTime-decayed features shape: {decayed.shape}")
        print(f"Recent weights higher: {decayed.iloc[-1].abs().mean():.4f} > {decayed.iloc[0].abs().mean():.4f}")
        
        # Test 3: Target returns
        target_simple = FeatureTransformer.create_target_returns(prices, horizon=5, method='simple')
        target_log = FeatureTransformer.create_target_returns(prices, horizon=5, method='log')
        print(f"\nTarget returns (simple): {target_simple.notna().sum()} values")
        print(f"Target returns (log): {target_log.notna().sum()} values")
        
        # Validate
        assert frac_diff.notna().sum() > 0, "No fractionally differentiated values"
        assert decayed.shape == features.shape, "Shape mismatch in time decay"
        
        print("✓ Feature transformation test passed")
        return True
    except Exception as e:
        print(f"✗ Feature transformation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_factor_analysis():
    """Test SimpleFactorAnalysis."""
    print("\n" + "="*60)
    print("TEST 7: Simple Factor Analysis")
    print("="*60)
    
    try:
        from models.factors.factor_analysis import SimpleFactorAnalysis
        
        # Generate synthetic factor and returns
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Factor values (e.g., momentum scores)
        factor_values = pd.DataFrame(
            np.random.randn(100, 5),
            index=dates,
            columns=assets
        )
        
        # Simulate forward returns correlated with factor
        forward_returns = pd.DataFrame(
            factor_values.values * 0.01 + np.random.randn(100, 5) * 0.02,
            index=dates,
            columns=assets
        )
        
        # Test 1: Information Coefficient
        ic = SimpleFactorAnalysis.calculate_ic(factor_values, forward_returns, method='spearman')
        print(f"IC series: {len(ic)} observations")
        print(f"IC mean: {ic.mean():.4f}")
        print(f"IC std: {ic.std():.4f}")
        print(f"IC Information Ratio: {ic.mean() / ic.std():.4f}")
        
        # Test 2: Quantile analysis
        factor_series = factor_values['AAPL']
        return_series = forward_returns['AAPL']
        quantile_rets = SimpleFactorAnalysis.quantile_analysis(factor_series, return_series, n_quantiles=5)
        print(f"\nQuantile returns:")
        for q, ret in quantile_rets.items():
            print(f"  Q{q}: {ret:.4f}")
        
        # Validate
        assert len(ic) > 0, "No IC values calculated"
        assert len(quantile_rets) > 0, "No quantile returns calculated"
        
        print("✓ Simple factor analysis test passed")
        return True
    except Exception as e:
        print(f"✗ Simple factor analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 2 tests."""
    print("\n" + "="*60)
    print("PHASE 2 AWESOME QUANT INTEGRATION - TEST SUITE")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    tests = [
        ("Imports", test_imports),
        ("SimpleSentiment", test_simple_sentiment),
        ("Sentiment Strategy", test_sentiment_driven_strategy),
        ("Multi-Factor Model", test_multi_factor_model),
        ("Label Generation", test_label_generation),
        ("Feature Transformations", test_feature_transformations),
        ("Factor Analysis", test_simple_factor_analysis)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL PHASE 2 TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
