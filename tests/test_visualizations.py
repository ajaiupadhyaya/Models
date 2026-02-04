"""
Unit tests for chart builders: ChartBuilder, PublicationCharts, InteractiveCharts.
Minimal DataFrames -> assert returned object is Plotly Figure, has traces, no crash.
"""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def minimal_ohlcv():
    """Minimal OHLCV DataFrame (5 rows) for chart tests."""
    n = 5
    close = 100 + np.cumsum(np.random.RandomState(42).randn(n))
    return pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 0.5,
        "Low": close - 1.0,
        "Close": close,
        "Volume": [1_000_000] * n,
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))


@pytest.fixture
def minimal_series():
    """Minimal time series for line chart."""
    return pd.Series(
        [100, 101, 102, 101, 103],
        index=pd.date_range("2024-01-01", periods=5, freq="B"),
    )


def test_chart_builder_candlestick_returns_figure(minimal_ohlcv):
    """ChartBuilder.candlestick_chart returns Plotly Figure with at least one trace."""
    from core.visualizations import ChartBuilder
    fig = ChartBuilder.candlestick_chart(minimal_ohlcv, title="Test", show_volume=True)
    assert hasattr(fig, "data")
    assert len(fig.data) >= 1
    assert fig.layout.title.text == "Test"


def test_chart_builder_line_chart_returns_figure(minimal_series):
    """ChartBuilder.line_chart returns Plotly Figure with at least one trace."""
    from core.visualizations import ChartBuilder
    fig = ChartBuilder.line_chart(minimal_series, title="Line")
    assert len(fig.data) >= 1
    assert fig.layout.title.text == "Line"


def test_chart_builder_correlation_heatmap_returns_figure():
    """ChartBuilder.correlation_heatmap returns Plotly Figure."""
    from core.visualizations import ChartBuilder
    df = pd.DataFrame(np.random.RandomState(42).randn(5, 3), columns=["A", "B", "C"])
    fig = ChartBuilder.correlation_heatmap(df, title="Correlation")
    assert len(fig.data) >= 1
    title = getattr(fig.layout.title, "text", None) or str(fig.layout.title)
    assert "Correlation" in (title or "")


def test_publication_charts_waterfall_returns_figure():
    """PublicationCharts.waterfall_chart returns Plotly Figure with traces."""
    from core.advanced_visualizations import PublicationCharts
    data = {"Start": 100, "Add": 20, "Sub": -10, "End": 110}
    fig = PublicationCharts.waterfall_chart(data, title="Waterfall")
    assert len(fig.data) >= 1
    assert fig.layout.title.text == "Waterfall"


def test_interactive_charts_time_series_returns_figure():
    """InteractiveCharts.time_series returns Plotly Figure."""
    from core.advanced_viz.interactive_charts import InteractiveCharts
    data = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=pd.date_range("2024-01-01", periods=3, freq="D"))
    charts = InteractiveCharts()
    fig = charts.time_series(data, title="Interactive TS")
    assert len(fig.data) >= 1
    assert fig.layout.title.text == "Interactive TS"


def test_interactive_charts_candlestick_returns_figure(minimal_ohlcv):
    """InteractiveCharts.candlestick returns Plotly Figure."""
    from core.advanced_viz.interactive_charts import InteractiveCharts
    charts = InteractiveCharts()
    fig = charts.candlestick(minimal_ohlcv, title="Candles")
    assert len(fig.data) >= 1
