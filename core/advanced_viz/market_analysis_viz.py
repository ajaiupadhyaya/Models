"""
Market Analysis Visualizations
Technical analysis, trading signals, market structure
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MarketAnalysisViz:
    """
    Market analysis visualizations.
    """
    
    def __init__(self):
        """Initialize market analysis viz."""
        self.theme = "plotly_dark"
    
    def price_with_indicators(self,
                             price_data: pd.DataFrame,
                             indicators: Optional[Dict[str, pd.Series]] = None,
                             title: str = "Price & Indicators") -> go.Figure:
        """
        Create price chart with technical indicators.
        
        Args:
            price_data: DataFrame with OHLC data
            indicators: Dictionary of indicator series
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Determine number of subplots
        num_indicators = len(indicators) if indicators else 0
        
        fig = make_subplots(
            rows=1+num_indicators, cols=1,
            shared_xaxes=True,
            row_heights=[0.6] + [0.4/num_indicators]*num_indicators if num_indicators else [1],
            vertical_spacing=0.1
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Add indicators
        if indicators:
            for idx, (ind_name, ind_data) in enumerate(indicators.items(), 1):
                fig.add_trace(go.Scatter(
                    x=ind_data.index,
                    y=ind_data,
                    name=ind_name,
                    line=dict(width=2)
                ), row=idx+1, col=1)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600 + 200*num_indicators,
            xaxis_title="Date"
        )
        
        return fig
    
    def volatility_surface(self,
                          strikes: np.ndarray,
                          maturities: np.ndarray,
                          implied_vols: np.ndarray,
                          title: str = "Volatility Surface") -> go.Figure:
        """
        Create 3D volatility surface.
        
        Args:
            strikes: Array of strike prices
            maturities: Array of time to maturity
            implied_vols: 2D array of implied volatilities
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Surface(
            x=strikes,
            y=maturities,
            z=implied_vols,
            colorscale='Viridis'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="Time to Maturity",
                zaxis_title="Implied Volatility"
            ),
            height=700
        )
        
        return fig
    
    def market_breadth(self,
                      advancing: pd.Series,
                      declining: pd.Series,
                      title: str = "Market Breadth") -> go.Figure:
        """
        Create market breadth visualization.
        
        Args:
            advancing: Series of advancing stocks
            declining: Series of declining stocks
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Advances vs Declines", "Advance/Decline Line"),
            shared_xaxes=True
        )
        
        # Stacked bar
        fig.add_trace(go.Bar(x=advancing.index, y=advancing, name='Advancing',
                            marker_color='green'), row=1, col=1)
        fig.add_trace(go.Bar(x=declining.index, y=declining, name='Declining',
                            marker_color='red'), row=1, col=1)
        
        # A/D line
        ad_line = (advancing - declining).cumsum()
        fig.add_trace(go.Scatter(x=ad_line.index, y=ad_line, name='A/D Line',
                                fill='tozeroy'), row=2, col=1)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            barmode='stack'
        )
        
        return fig
    
    def relative_strength(self,
                         stock_returns: pd.Series,
                         benchmark_returns: pd.Series,
                         title: str = "Relative Strength") -> go.Figure:
        """
        Create relative strength visualization.
        
        Args:
            stock_returns: Stock return series
            benchmark_returns: Benchmark return series
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Calculate cumulative returns
        stock_cumulative = (1 + stock_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        # Calculate relative performance
        relative = stock_cumulative / benchmark_cumulative
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Absolute Returns", "Relative Performance"),
            shared_xaxes=True
        )
        
        # Absolute
        fig.add_trace(go.Scatter(x=stock_cumulative.index, 
                                y=stock_cumulative, name='Stock',
                                fill='tozeroy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=benchmark_cumulative.index,
                                y=benchmark_cumulative, name='Benchmark',
                                fill='tozeroy'), row=1, col=1)
        
        # Relative
        fig.add_trace(go.Scatter(x=relative.index, y=relative, name='Relative',
                                fill='tozeroy', marker_color='purple'),
                     row=2, col=1)
        fig.add_hline(y=1, line_dash="dash", row=2, col=1)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600
        )
        
        return fig
    
    def yield_curve_animation(self,
                             dates: List[str],
                             curves: List[np.ndarray],
                             maturities: np.ndarray,
                             title: str = "Yield Curve Evolution") -> go.Figure:
        """
        Create animated yield curve.
        
        Args:
            dates: List of dates
            curves: List of yield curve arrays
            maturities: Array of maturities
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add traces for each date
        for date, curve in zip(dates, curves):
            fig.add_trace(
                go.Scatter(x=maturities, y=curve, name=date, visible=False)
            )
        
        # Make first visible
        fig.data[0].visible = True
        
        # Create frames for animation
        frames = [go.Frame(data=[go.Scatter(x=maturities, y=curve)],
                          name=date)
                 for date, curve in zip(dates, curves)]
        
        fig.frames = frames
        
        # Animation controls
        fig.update_layout(
            updatemenus=[{
                'buttons': [
                    {'label': 'Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 500}}]},
                    {'label': 'Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                ]
            }],
            sliders=[{
                'active': 0,
                'steps': [{'args': [[f.name], {}], 'label': f.name, 'method': 'animate'}
                         for f in frames]
            }],
            title=title,
            xaxis_title="Maturity",
            yaxis_title="Yield",
            template=self.theme,
            height=600
        )
        
        return fig
    
    def correlation_network(self,
                           correlation_matrix: pd.DataFrame,
                           threshold: float = 0.3,
                           title: str = "Correlation Network") -> go.Figure:
        """
        Create correlation network visualization.
        
        Args:
            correlation_matrix: Correlation matrix
            threshold: Minimum correlation to display
            title: Chart title
        
        Returns:
            Plotly figure (static representation)
        """
        # This is a simplified 2D representation
        # For true network visualization, consider using plotly-dash with networkx
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=700
        )
        
        return fig
