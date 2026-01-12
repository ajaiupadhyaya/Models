"""
Advanced visualization module with Plotly and D3.js-style charts.
Institutional-grade financial charting.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class ChartBuilder:
    """
    Professional financial charting with interactive visualizations.
    """
    
    @staticmethod
    def candlestick_chart(df: pd.DataFrame, 
                         title: str = "Stock Price",
                         show_volume: bool = True) -> go.Figure:
        """
        Create professional candlestick chart with volume.
        
        Args:
            df: DataFrame with OHLCV data
            title: Chart title
            show_volume: Whether to show volume subplot
        
        Returns:
            Plotly figure
        """
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(title, 'Volume')
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price"
                ),
                row=1, col=1
            )
            
            # Volume bars
            colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
                     else 'green' for i in range(len(df))]
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors),
                row=2, col=1
            )
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price"
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=600,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def line_chart(data: Union[pd.Series, pd.DataFrame],
                  title: str = "Time Series",
                  labels: Optional[Dict] = None) -> go.Figure:
        """
        Create line chart for time series data.
        
        Args:
            data: Series or DataFrame
            title: Chart title
            labels: Axis labels dict
        
        Returns:
            Plotly figure
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        fig = go.Figure()
        
        for col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[col],
                    mode='lines',
                    name=col,
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=500,
            xaxis_title=labels.get('x', 'Date') if labels else 'Date',
            yaxis_title=labels.get('y', 'Value') if labels else 'Value',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, 
                           title: str = "Correlation Matrix") -> go.Figure:
        """
        Create correlation heatmap.
        
        Args:
            df: DataFrame with numeric columns
            title: Chart title
        
        Returns:
            Plotly figure
        """
        corr = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmid=0,
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=600,
            width=800
        )
        
        return fig
    
    @staticmethod
    def portfolio_performance(returns: pd.Series,
                            benchmark: Optional[pd.Series] = None,
                            title: str = "Portfolio Performance") -> go.Figure:
        """
        Create cumulative returns chart with optional benchmark.
        
        Args:
            returns: Portfolio returns series
            benchmark: Benchmark returns series
            title: Chart title
        
        Returns:
            Plotly figure
        """
        cumulative = (1 + returns).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode='lines',
                name='Portfolio',
                line=dict(width=3, color='#00D4FF')
            )
        )
        
        if benchmark is not None:
            benchmark_cum = (1 + benchmark).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cum.index,
                    y=benchmark_cum.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(width=2, color='#FF6B6B', dash='dash')
                )
            )
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=500,
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def risk_return_scatter(returns_df: pd.DataFrame,
                           title: str = "Risk-Return Analysis") -> go.Figure:
        """
        Create risk-return scatter plot for multiple assets.
        
        Args:
            returns_df: DataFrame with asset returns
            title: Chart title
        
        Returns:
            Plotly figure
        """
        annual_returns = returns_df.mean() * 252
        annual_vol = returns_df.std() * np.sqrt(252)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=annual_vol,
                y=annual_returns,
                mode='markers+text',
                text=returns_df.columns,
                textposition="top center",
                marker=dict(
                    size=12,
                    color=annual_returns / annual_vol,  # Sharpe ratio coloring
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                name="Assets"
            )
        )
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=600,
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def economic_dashboard(macro_data: Dict[str, pd.Series],
                          title: str = "Macroeconomic Dashboard") -> go.Figure:
        """
        Create comprehensive macroeconomic dashboard.
        
        Args:
            macro_data: Dictionary with economic series
            title: Dashboard title
        
        Returns:
            Plotly figure with subplots
        """
        n_series = len(macro_data)
        cols = 2
        rows = (n_series + 1) // 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(macro_data.keys()),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        row, col = 1, 1
        for name, series in macro_data.items():
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='lines',
                    name=name,
                    line=dict(width=2)
                ),
                row=row, col=col
            )
            
            col += 1
            if col > cols:
                col = 1
                row += 1
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=300 * rows,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def distribution_plot(data: pd.Series,
                        title: str = "Distribution",
                        bins: int = 50) -> go.Figure:
        """
        Create histogram with KDE overlay.
        
        Args:
            data: Data series
            title: Chart title
            bins: Number of bins
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=bins,
                name="Histogram",
                opacity=0.7
            )
        )
        
        # Add KDE curve
        from scipy import stats
        kde = stats.gaussian_kde(data.dropna())
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_values = kde(x_range)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_values * len(data) * (data.max() - data.min()) / bins,
                mode='lines',
                name='KDE',
                line=dict(width=2, color='red')
            )
        )
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=400,
            xaxis_title="Value",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        return fig
    
    @staticmethod
    def efficient_frontier(expected_returns: pd.Series,
                          cov_matrix: pd.DataFrame,
                          risk_free_rate: float = 0.02,
                          title: str = "Efficient Frontier") -> go.Figure:
        """
        Plot efficient frontier for portfolio optimization.
        
        Args:
            expected_returns: Expected returns for assets
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            title: Chart title
        
        Returns:
            Plotly figure
        """
        from scipy.optimize import minimize
        
        n_assets = len(expected_returns)
        
        def portfolio_performance(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_std, portfolio_return
        
        def negative_sharpe(weights):
            std, ret = portfolio_performance(weights)
            return -(ret - risk_free_rate) / std
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Generate efficient frontier
        target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 100)
        efficient_portfolios = []
        
        for target_ret in target_returns:
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_ret}
            ]
            result = minimize(
                lambda x: portfolio_performance(x)[0],
                x0=np.array([1/n_assets] * n_assets),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list
            )
            if result.success:
                std, ret = portfolio_performance(result.x)
                efficient_portfolios.append((std, ret))
        
        efficient_portfolios = np.array(efficient_portfolios)
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(
            go.Scatter(
                x=efficient_portfolios[:, 0] * np.sqrt(252),
                y=efficient_portfolios[:, 1] * 252,
                mode='lines',
                name='Efficient Frontier',
                line=dict(width=3, color='#00D4FF')
            )
        )
        
        # Individual assets
        individual_std = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
        individual_ret = expected_returns * 252
        
        fig.add_trace(
            go.Scatter(
                x=individual_std,
                y=individual_ret,
                mode='markers+text',
                text=expected_returns.index,
                textposition="top center",
                marker=dict(size=10, color='#FF6B6B'),
                name='Assets'
            )
        )
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=600,
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            hovermode='closest'
        )
        
        return fig
