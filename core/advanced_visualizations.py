"""
Advanced, publication-quality visualizations inspired by NYT, Bloomberg, and Financial Times.
Interactive, creative, and visually stunning charts for financial analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class PublicationCharts:
    """
    Publication-quality charts with sophisticated styling and interactivity.
    """
    
    # NYT-inspired color palette
    NYT_COLORS = {
        'primary': '#121212',
        'secondary': '#666666',
        'accent': '#326891',
        'highlight': '#E3120B',
        'background': '#FFFFFF',
        'grid': '#E5E5E5'
    }
    
    # Financial Times-inspired palette
    FT_COLORS = {
        'primary': '#FFF1E5',
        'secondary': '#0D7680',
        'accent': '#CEE5F2',
        'highlight': '#F56400',
        'background': '#FFFFFF'
    }
    
    @staticmethod
    def waterfall_chart(data: Dict[str, float],
                       title: str = "Waterfall Analysis",
                       theme: str = 'nyt') -> go.Figure:
        """
        Create publication-quality waterfall chart.
        
        Args:
            data: Dictionary of {label: value} pairs
            title: Chart title
            theme: 'nyt' or 'ft'
        
        Returns:
            Plotly figure
        """
        colors = PublicationCharts.NYT_COLORS if theme == 'nyt' else PublicationCharts.FT_COLORS
        
        labels = list(data.keys())
        values = list(data.values())
        
        # Calculate cumulative values
        cumulative = np.cumsum([0] + values[:-1])
        
        fig = go.Figure()
        
        # Base bars
        fig.add_trace(go.Bar(
            name='Base',
            x=labels,
            y=cumulative,
            marker=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
        
        # Waterfall bars
        colors_list = [colors['accent'] if v >= 0 else colors['highlight'] for v in values]
        fig.add_trace(go.Bar(
            name='Change',
            x=labels,
            y=values,
            base=cumulative,
            marker=dict(color=colors_list),
            text=[f"{v:+.1f}" for v in values],
            textposition='outside',
            showlegend=False
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Georgia, serif'}
            },
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=12, family='Helvetica, Arial, sans-serif')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=colors['grid'],
                zeroline=True,
                zerolinecolor=colors['primary'],
                tickfont=dict(size=11, family='Helvetica, Arial, sans-serif')
            ),
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            height=500,
            margin=dict(l=60, r=60, t=80, b=60),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def sankey_flow(data: Dict[str, Dict[str, float]],
                   title: str = "Flow Analysis") -> go.Figure:
        """
        Create Sankey diagram for flow analysis.
        
        Args:
            data: Nested dict of {source: {target: value}}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        sources = []
        targets = []
        values = []
        labels = []
        label_map = {}
        
        # Build label map
        all_nodes = set()
        for source, targets_dict in data.items():
            all_nodes.add(source)
            all_nodes.update(targets_dict.keys())
        
        labels = list(all_nodes)
        label_map = {label: i for i, label in enumerate(labels)}
        
        # Build flow data
        for source, targets_dict in data.items():
            for target, value in targets_dict.items():
                sources.append(label_map[source])
                targets.append(label_map[target])
                values.append(value)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=PublicationCharts.NYT_COLORS['accent']
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=PublicationCharts.NYT_COLORS['secondary']
            )
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            font_size=12,
            height=600
        )
        
        return fig
    
    @staticmethod
    def small_multiples(data: Dict[str, pd.Series],
                       title: str = "Small Multiples Analysis",
                       n_cols: int = 3) -> go.Figure:
        """
        Create small multiples chart (Tufte-style).
        
        Args:
            data: Dictionary of series to plot
            title: Overall title
            n_cols: Number of columns
        
        Returns:
            Plotly figure
        """
        n_series = len(data)
        n_rows = (n_series + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(data.keys()),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        row, col = 1, 1
        for name, series in data.items():
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='lines',
                    line=dict(width=2, color=PublicationCharts.NYT_COLORS['accent']),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            col += 1
            if col > n_cols:
                col = 1
                row += 1
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Georgia, serif'}
            },
            height=200 * n_rows,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def heatmap_calendar(data: pd.Series,
                        title: str = "Calendar Heatmap") -> go.Figure:
        """
        Create calendar heatmap visualization.
        
        Args:
            data: Time series data
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Prepare data for calendar
        data_df = data.reset_index()
        data_df.columns = ['date', 'value']
        data_df['year'] = data_df['date'].dt.year
        data_df['month'] = data_df['date'].dt.month
        data_df['day'] = data_df['date'].dt.day
        data_df['weekday'] = data_df['date'].dt.dayofweek
        data_df['week'] = data_df['date'].dt.isocalendar().week
        
        fig = go.Figure(data=go.Heatmap(
            z=data_df['value'],
            x=data_df['week'],
            y=data_df['weekday'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Value")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Week of Year",
            yaxis_title="Day of Week",
            height=400
        )
        
        return fig
    
    @staticmethod
    def radar_chart(categories: List[str],
                   values: List[float],
                   title: str = "Radar Chart",
                   max_value: Optional[float] = None) -> go.Figure:
        """
        Create radar/spider chart.
        
        Args:
            categories: Category labels
            values: Values for each category
            title: Chart title
            max_value: Maximum value for scaling
        
        Returns:
            Plotly figure
        """
        if max_value is None:
            max_value = max(values) * 1.2
        
        # Close the radar chart
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Metrics',
            line=dict(color=PublicationCharts.NYT_COLORS['accent'], width=3)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_value]
                )
            ),
            title=title,
            height=500
        )
        
        return fig
    
    @staticmethod
    def treemap(data: Dict[str, float],
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
        parents = [''] * len(labels)
        
        fig = go.Figure(go.Treemap(
            labels=labels,
            values=values,
            parents=parents,
            marker=dict(
                colorscale='Viridis',
                showscale=True
            ),
            textinfo="label+value+percent parent"
        ))
        
        fig.update_layout(
            title=title,
            height=500
        )
        
        return fig
    
    @staticmethod
    def animated_timeseries(data: pd.DataFrame,
                           title: str = "Animated Time Series") -> go.Figure:
        """
        Create animated time series chart.
        
        Args:
            data: DataFrame with time index and multiple series
            title: Chart title
        
        Returns:
            Plotly figure with animation
        """
        fig = go.Figure()
        
        for col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                mode='lines',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis=dict(range=[data.index[0], data.index[-1]]),
            yaxis=dict(range=[data.min().min() * 0.9, data.max().max() * 1.1]),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None]
                }]
            }]
        )
        
        frames = []
        for i in range(len(data)):
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=data.index[:i+1],
                    y=data[col][:i+1],
                    name=col
                ) for col in data.columns]
            ))
        
        fig.frames = frames
        
        return fig
    
    @staticmethod
    def correlation_network(returns_df: pd.DataFrame,
                           threshold: float = 0.5,
                           title: str = "Correlation Network") -> go.Figure:
        """
        Create network graph of asset correlations.
        
        Args:
            returns_df: Returns DataFrame
            threshold: Minimum correlation to show edge
            title: Chart title
        
        Returns:
            Plotly figure
        """
        corr = returns_df.corr()
        
        # Build network
        nodes = list(corr.columns)
        edges = []
        edge_weights = []
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j and abs(corr.loc[node1, node2]) >= threshold:
                    edges.append((i, j))
                    edge_weights.append(abs(corr.loc[node1, node2]))
        
        # Create network layout (simplified spring layout)
        n_nodes = len(nodes)
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Node trace
        node_trace = go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            text=nodes,
            textposition="middle center",
            marker=dict(
                size=20,
                color=PublicationCharts.NYT_COLORS['accent'],
                line=dict(width=2, color=PublicationCharts.NYT_COLORS['primary'])
            ),
            showlegend=False
        )
        
        # Edge traces
        edge_traces = []
        for (i, j), weight in zip(edges, edge_weights):
            edge_traces.append(
                go.Scatter(
                    x=[x_pos[i], x_pos[j]],
                    y=[y_pos[i], y_pos[j]],
                    mode='lines',
                    line=dict(width=weight*5, color=PublicationCharts.NYT_COLORS['secondary']),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='white'
        )
        
        return fig
