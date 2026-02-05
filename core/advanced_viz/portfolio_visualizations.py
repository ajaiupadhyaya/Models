"""
Portfolio and Risk Visualizations
Visual analysis of portfolio allocation, risk metrics, and optimization results
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PortfolioVisualizations:
    """
    Create portfolio analysis visualizations.
    """
    
    def __init__(self):
        """Initialize portfolio visualization creator."""
        self.theme = "plotly_dark"
    
    def allocation_pie(self,
                      weights: Dict[str, float],
                      title: str = "Portfolio Allocation") -> go.Figure:
        """
        Create pie chart of portfolio allocation.
        
        Args:
            weights: Dictionary of {ticker: weight}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            textposition='inside',
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600
        )
        
        return fig
    
    def allocation_sunburst(self,
                           allocations: Dict[str, Dict],
                           title: str = "Hierarchical Allocation") -> go.Figure:
        """
        Create sunburst chart for hierarchical allocation.
        
        Args:
            allocations: Nested dictionary {sector: {position: weight}}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        labels = ["Portfolio"]
        parents = [""]
        values = [sum(sum(pos.values()) for pos in allocations.values())]
        colors = [0]
        
        color_index = 0
        for sector, positions in allocations.items():
            labels.append(sector)
            parents.append("Portfolio")
            values.append(sum(positions.values()))
            colors.append(color_index)
            color_index += 1
            
            for position, weight in positions.items():
                labels.append(f"{position}")
                parents.append(sector)
                values.append(weight)
                colors.append(color_index)
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors, colorscale='Viridis')
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=700
        )
        
        return fig
    
    def efficient_frontier(self,
                          returns: np.ndarray,
                          volatilities: np.ndarray,
                          sharpe_ratios: np.ndarray,
                          portfolio_return: Optional[float] = None,
                          portfolio_vol: Optional[float] = None) -> go.Figure:
        """
        Create efficient frontier visualization.
        
        Args:
            returns: Array of portfolio returns
            volatilities: Array of portfolio volatilities
            sharpe_ratios: Array of Sharpe ratios
            portfolio_return: Current portfolio return
            portfolio_vol: Current portfolio volatility
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add frontier
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers',
            marker=dict(
                size=8,
                color=sharpe_ratios,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            text=[f"Sharpe: {sr:.2f}" for sr in sharpe_ratios],
            hovertemplate='<b>Return:</b> %{y:.2%}<br><b>Vol:</b> %{x:.2%}<br>%{text}<extra></extra>',
            name='Efficient Frontier'
        ))
        
        # Add current portfolio if provided
        if portfolio_return is not None and portfolio_vol is not None:
            fig.add_trace(go.Scatter(
                x=[portfolio_vol],
                y=[portfolio_return],
                mode='markers+text',
                marker=dict(size=15, color='red', symbol='star'),
                text=['Current Portfolio'],
                textposition='top center',
                name='Current Portfolio'
            ))
        
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            template=self.theme,
            height=600
        )
        
        return fig
    
    def drawdown_analysis(self,
                         returns: pd.Series,
                         title: str = "Drawdown Analysis") -> go.Figure:
        """
        Create drawdown visualization.
        
        Args:
            returns: Series of returns
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Cumulative Return", "Drawdown"),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Cumulative return
        fig.add_trace(
            go.Scatter(x=cumulative.index, y=cumulative.values,
                      name='Cumulative Return', fill='tozeroy'),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      name='Drawdown', fill='tozeroy',
                      marker_color='red'),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=700
        )
        
        return fig
    
    def risk_metrics(self,
                    risk_data: Dict[str, float],
                    title: str = "Risk Metrics") -> go.Figure:
        """
        Create risk metrics bar chart.
        
        Args:
            risk_data: Dictionary of {metric: value}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(risk_data.keys()),
            y=list(risk_data.values()),
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Value",
            template=self.theme,
            height=500
        )
        
        return fig
    
    def factor_exposure(self,
                       factors: Dict[str, float],
                       title: str = "Factor Exposure") -> go.Figure:
        """
        Create factor exposure chart.
        
        Args:
            factors: Dictionary of {factor: exposure}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Separate positive and negative exposures
        pos_factors = {k: v for k, v in factors.items() if v > 0}
        neg_factors = {k: v for k, v in factors.items() if v < 0}
        
        if pos_factors:
            fig.add_trace(go.Bar(
                x=list(pos_factors.keys()),
                y=list(pos_factors.values()),
                name='Long',
                marker_color='green'
            ))
        
        if neg_factors:
            fig.add_trace(go.Bar(
                x=list(neg_factors.keys()),
                y=list(neg_factors.values()),
                name='Short',
                marker_color='red'
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Exposure",
            template=self.theme,
            height=500,
            barmode='group'
        )
        
        return fig
    
    def sector_performance(self,
                          performance: Dict[str, float],
                          title: str = "Sector Performance") -> go.Figure:
        """
        Create sector performance chart.
        
        Args:
            performance: Dictionary of {sector: return}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        sectors = list(performance.keys())
        returns = list(performance.values())
        
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sectors,
                y=returns,
                marker_color=colors,
                text=[f"{r:.2%}" for r in returns],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=title,
            yaxis_title="Return",
            template=self.theme,
            height=500
        )
        
        return fig


class RiskVisualizations:
    """
    Risk analysis visualizations.
    """
    
    def __init__(self):
        """Initialize risk visualization creator."""
        self.theme = "plotly_dark"
    
    def var_cvar_distribution(self,
                             returns: pd.Series,
                             var_95: float,
                             cvar_95: float,
                             title: str = "VaR/CVaR") -> go.Figure:
        """
        Create VaR/CVaR distribution visualization.
        
        Args:
            returns: Series of returns
            var_95: 95% VaR
            cvar_95: 95% CVaR
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Distribution
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Distribution',
            marker_color='lightblue'
        ))
        
        # VaR line
        fig.add_vline(x=var_95, line_dash="dash", line_color="orange",
                     annotation_text=f"VaR 95%: {var_95:.2%}",
                     annotation_position="top right")
        
        # CVaR line
        fig.add_vline(x=cvar_95, line_dash="dash", line_color="red",
                     annotation_text=f"CVaR 95%: {cvar_95:.2%}",
                     annotation_position="top left")
        
        fig.update_layout(
            title=title,
            xaxis_title="Return",
            yaxis_title="Frequency",
            template=self.theme,
            height=500
        )
        
        return fig
    
    def correlation_changes(self,
                           correlation_history: List[float],
                           title: str = "Correlation Over Time") -> go.Figure:
        """
        Create correlation changes visualization.
        
        Args:
            correlation_history: List of correlation values over time
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=correlation_history,
            mode='lines',
            fill='tozeroy',
            name='Correlation',
            marker_color='lightblue'
        ))
        
        # Add mean line
        mean_corr = np.mean(correlation_history)
        fig.add_hline(y=mean_corr, line_dash="dash",
                     annotation_text=f"Mean: {mean_corr:.2f}")
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="Correlation",
            template=self.theme,
            height=500
        )
        
        return fig
