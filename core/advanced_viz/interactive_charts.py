"""
Interactive Charts using Plotly
High-quality, publication-ready financial visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ChartTypes(Enum):
    """Available chart types."""
    LINE = "line"
    CANDLESTICK = "candlestick"
    OHLC = "ohlc"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    AREA = "area"
    HISTOGRAM = "histogram"
    BOX = "box"
    TREEMAP = "treemap"


class InteractiveCharts:
    """
    Create interactive financial visualizations.
    """
    
    def __init__(self):
        """Initialize chart creator."""
        self.theme = "plotly_dark"
        self.colors = {
            'primary': '#1f77b4',
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6'
        }
    
    def time_series(self,
                   data: pd.DataFrame,
                   title: str = "Time Series",
                   y_label: str = "Value",
                   colors: Optional[Dict] = None) -> go.Figure:
        """
        Create interactive time series plot.
        
        Args:
            data: DataFrame with DatetimeIndex and columns to plot
            title: Chart title
            y_label: Y-axis label
            colors: Optional color mapping
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add traces for each column
        for col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode='lines',
                name=col,
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label,
            template=self.theme,
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def candlestick(self,
                   data: pd.DataFrame,
                   title: str = "Price Action") -> go.Figure:
        """
        Create candlestick chart.
        
        Args:
            data: DataFrame with OHLC data
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        )])
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Date",
            yaxis_title="Price",
            height=600
        )
        
        return fig
    
    def multi_panel(self,
                   data_dict: Dict[str, pd.DataFrame],
                   titles: Optional[List[str]] = None,
                   height: int = 800) -> go.Figure:
        """
        Create multi-panel visualization.
        
        Args:
            data_dict: Dictionary of {panel_name: dataframe}
            titles: Optional list of panel titles
            height: Total figure height
        
        Returns:
            Plotly figure with subplots
        """
        num_panels = len(data_dict)
        
        fig = make_subplots(
            rows=num_panels, cols=1,
            subplot_titles=titles or list(data_dict.keys()),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        for idx, (panel_name, data) in enumerate(data_dict.items(), 1):
            for col in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data[col], name=col),
                    row=idx, col=1
                )
        
        fig.update_layout(
            template=self.theme,
            height=height,
            hovermode='x unified'
        )
        
        return fig
    
    def heatmap(self,
               data: pd.DataFrame,
               title: str = "Heatmap",
               colorscale: str = "RdBu") -> go.Figure:
        """
        Create heatmap visualization.
        
        Args:
            data: 2D DataFrame
            title: Chart title
            colorscale: Plotly colorscale name
        
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale=colorscale,
            hovertemplate='%{x}<br>%{y}<br>Value: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600
        )
        
        return fig
    
    def correlation_matrix(self,
                          data: pd.DataFrame,
                          title: str = "Correlation Matrix") -> go.Figure:
        """
        Create correlation matrix heatmap.
        
        Args:
            data: DataFrame with numeric columns
            title: Chart title
        
        Returns:
            Plotly figure
        """
        corr = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate='%{x} - %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            width=700
        )
        
        return fig
    
    def distribution(self,
                    data: pd.Series,
                    title: str = "Distribution",
                    nbins: int = 50) -> go.Figure:
        """
        Create distribution histogram.
        
        Args:
            data: Series of values
            title: Chart title
            nbins: Number of bins
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=nbins,
            name="Distribution",
            marker_color=self.colors['primary']
        ))
        
        # Add statistics
        mean = data.mean()
        std = data.std()
        
        fig.add_vline(x=mean, line_dash="dash", line_color="red",
                     annotation_text="Mean", annotation_position="top right")
        
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            template=self.theme,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def scatter_with_regression(self,
                               x: pd.Series,
                               y: pd.Series,
                               title: str = "Scatter Plot") -> go.Figure:
        """
        Create scatter plot with trend line.
        
        Args:
            x: X values
            y: Y values
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(size=8, color=self.colors['primary'], opacity=0.6),
            name='Data'
        ))
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_trend, y=p(x_trend),
            mode='lines',
            name='Trend',
            line=dict(color=self.colors['negative'], width=2)
        ))
        
        # Calculate R-squared
        y_pred = p(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        fig.update_layout(
            title=f"{title} (R² = {r_squared:.3f})",
            xaxis_title="X Value",
            yaxis_title="Y Value",
            template=self.theme,
            height=500
        )
        
        return fig
    
    def treemap(self,
               data: Dict[str, float],
               title: str = "Treemap") -> go.Figure:
        """
        Create treemap visualization.
        
        Args:
            data: Dictionary of {label: value}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        labels = list(data.keys())
        values = list(data.values())
        
        # Determine colors based on values
        colors = [self.colors['positive'] if v > 0 else self.colors['negative']
                 for v in values]
        
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=[""] * len(labels),
            values=[abs(v) for v in values],
            marker=dict(colors=colors),
            textposition="middle center"
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600
        )
        
        return fig
    
    def waterfall(self,
                 categories: List[str],
                 values: List[float],
                 title: str = "Waterfall") -> go.Figure:
        """
        Create waterfall chart.
        
        Args:
            categories: Category names
            values: Change values
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Calculate cumulative values
        cumulative = [0]
        for v in values[:-1]:
            cumulative.append(cumulative[-1] + v)
        
        fig = go.Figure(go.Waterfall(
            x=categories,
            y=values,
            base=cumulative,
            connector={"line": {"color": "rgba(63, 63, 63, 0.5)"}},
            decreasing={"marker": {"color": self.colors['negative']}},
            increasing={"marker": {"color": self.colors['positive']}}
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=500
        )
        
        return fig
    
    def compare_distributions(self,
                             data_dict: Dict[str, pd.Series],
                             title: str = "Distribution Comparison") -> go.Figure:
        """
        Compare multiple distributions.
        
        Args:
            data_dict: Dictionary of {name: series}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for name, series in data_dict.items():
            fig.add_trace(go.Box(
                y=series,
                name=name,
                boxmean='sd'
            ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=500,
            showlegend=True
        )
        
        return fig


class PublicationCharts(InteractiveCharts):
    """
    Publication-quality charts matching Bloomberg/FT/NYT styles.
    """
    
    def __init__(self, style: str = "bloomberg"):
        """
        Initialize publication chart creator.
        
        Args:
            style: Chart style ('bloomberg', 'ft', 'nyt')
        """
        super().__init__()
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup publication style."""
        if self.style == "bloomberg":
            self.theme = "plotly_dark"
            self.colors = {
                'primary': '#FFBC00',  # Bloomberg yellow
                'positive': '#00C679',
                'negative': '#FF5C00',
                'neutral': '#595959'
            }
        elif self.style == "ft":
            self.theme = "plotly"
            self.colors = {
                'primary': '#0D47A1',  # FT blue
                'positive': '#33A23E',
                'negative': '#C8102E',
                'neutral': '#9E9E9E'
            }
        elif self.style == "nyt":
            self.theme = "plotly"
            self.colors = {
                'primary': '#121212',  # NYT black
                'positive': '#6EBE2B',
                'negative': '#D0021B',
                'neutral': '#666666'
            }
