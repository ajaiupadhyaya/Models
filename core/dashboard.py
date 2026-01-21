"""
Interactive financial dashboard using Dash.
Real-time updates, responsive design, institutional-grade.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from .data_fetcher import DataFetcher
from .visualizations import ChartBuilder
from .advanced_visualizations import PublicationCharts
import warnings
warnings.filterwarnings('ignore')


class FinancialDashboard:
    """
    Interactive financial analysis dashboard.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        self.app = dash.Dash(__name__)
        self.fetcher = DataFetcher()
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Financial Analysis Dashboard", 
                       style={'textAlign': 'center', 'color': '#121212', 
                             'fontFamily': 'Georgia, serif', 'marginBottom': '30px'}),
                html.P("Real-time market data and analysis",
                      style={'textAlign': 'center', 'color': '#666666',
                            'fontSize': '18px', 'marginBottom': '40px'})
            ]),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Select Ticker:", style={'fontWeight': 'bold'}),
                    dcc.Input(id='ticker-input', value='AAPL', type='text',
                             style={'width': '100px', 'marginLeft': '10px'})
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                
                html.Div([
                    html.Label("Time Period:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='period-dropdown',
                        options=[
                            {'label': '1 Month', 'value': '1mo'},
                            {'label': '3 Months', 'value': '3mo'},
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '1 Year', 'value': '1y'},
                            {'label': '2 Years', 'value': '2y'},
                            {'label': '5 Years', 'value': '5y'}
                        ],
                        value='1y',
                        style={'width': '150px', 'marginLeft': '10px', 'display': 'inline-block'}
                    )
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                
                html.Button('Update', id='update-button', n_clicks=0,
                           style={'padding': '10px 20px', 'fontSize': '16px',
                                 'backgroundColor': '#326891', 'color': 'white',
                                 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
            ], style={'textAlign': 'center', 'marginBottom': '30px', 'padding': '20px',
                     'backgroundColor': '#f5f5f5', 'borderRadius': '10px'}),
            
            # Main content
            html.Div([
                # Left column
                html.Div([
                    dcc.Graph(id='price-chart'),
                    dcc.Graph(id='returns-distribution')
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Right column
                html.Div([
                    html.Div(id='key-metrics', style={'marginBottom': '20px'}),
                    dcc.Graph(id='correlation-heatmap'),
                    dcc.Graph(id='risk-return-scatter')
                ], style={'width': '34%', 'display': 'inline-block', 'verticalAlign': 'top',
                         'marginLeft': '1%'})
            ]),
            
            # Bottom section
            html.Div([
                html.Div([
                    dcc.Graph(id='macro-dashboard')
                ], style={'width': '100%'})
            ], style={'marginTop': '30px'}),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ], style={'fontFamily': 'Helvetica, Arial, sans-serif', 'padding': '20px'})
    
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('returns-distribution', 'figure'),
             Output('key-metrics', 'children'),
             Output('correlation-heatmap', 'figure'),
             Output('risk-return-scatter', 'figure'),
             Output('macro-dashboard', 'figure')],
            [Input('update-button', 'n_clicks'),
             Input('interval-component', 'n_intervals')],
            [dash.dependencies.State('ticker-input', 'value'),
             dash.dependencies.State('period-dropdown', 'value')]
        )
        def update_dashboard(n_clicks, n_intervals, ticker, period):
            """Update all dashboard components."""
            try:
                # Fetch stock data
                stock_data = self.fetcher.get_stock_data(ticker, period=period)
                
                if len(stock_data) == 0:
                    return self._empty_figures()
                
                # Price chart
                price_fig = ChartBuilder.candlestick_chart(
                    stock_data, 
                    title=f"{ticker} Stock Price",
                    show_volume=True
                )
                price_fig.update_layout(template="plotly_white", height=400)
                
                # Returns distribution
                returns = stock_data['Close'].pct_change().dropna()
                dist_fig = ChartBuilder.distribution_plot(
                    returns,
                    title="Returns Distribution"
                )
                dist_fig.update_layout(template="plotly_white", height=300)
                
                # Key metrics
                metrics = self._calculate_metrics(stock_data, returns)
                metrics_html = self._create_metrics_html(metrics)
                
                # Correlation (if multiple stocks)
                if len(stock_data) > 0:
                    corr_fig = go.Figure()
                    corr_fig.add_annotation(
                        text="Single stock selected",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    corr_fig.update_layout(template="plotly_white", height=250)
                else:
                    corr_fig = go.Figure()
                
                # Risk-return scatter
                risk_return_fig = go.Figure()
                risk_return_fig.add_annotation(
                    text="Single stock selected",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                risk_return_fig.update_layout(template="plotly_white", height=250)
                
                # Macro dashboard
                try:
                    macro_data = self.fetcher.get_macro_dashboard_data()
                    macro_fig = ChartBuilder.economic_dashboard(
                        macro_data,
                        title="Macroeconomic Indicators"
                    )
                    macro_fig.update_layout(template="plotly_white", height=400)
                except:
                    macro_fig = go.Figure()
                    macro_fig.add_annotation(
                        text="Macro data unavailable",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    macro_fig.update_layout(template="plotly_white", height=400)
                
                return (price_fig, dist_fig, metrics_html, corr_fig, 
                       risk_return_fig, macro_fig)
            
            except Exception as e:
                print(f"Error updating dashboard: {e}")
                return self._empty_figures()
    
    def _calculate_metrics(self, stock_data: pd.DataFrame, 
                          returns: pd.Series) -> dict:
        """Calculate key financial metrics."""
        from .utils import calculate_sharpe_ratio, calculate_max_drawdown
        
        current_price = stock_data['Close'].iloc[-1]
        price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]
        price_change_pct = (price_change / stock_data['Close'].iloc[0]) * 100
        
        avg_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(returns)
        
        return {
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'annual_return': avg_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100
        }
    
    def _create_metrics_html(self, metrics: dict) -> html.Div:
        """Create HTML for key metrics."""
        return html.Div([
            html.H3("Key Metrics", style={'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.P("Current Price", style={'margin': '5px', 'color': '#666'}),
                    html.H4(f"${metrics['current_price']:.2f}", 
                           style={'margin': '5px', 'color': '#121212'})
                ], style={'display': 'inline-block', 'width': '30%', 'textAlign': 'center',
                         'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px',
                         'margin': '5px'}),
                
                html.Div([
                    html.P("Return", style={'margin': '5px', 'color': '#666'}),
                    html.H4(f"{metrics['price_change_pct']:+.2f}%",
                           style={'margin': '5px', 
                                 'color': '#E3120B' if metrics['price_change_pct'] < 0 else '#326891'})
                ], style={'display': 'inline-block', 'width': '30%', 'textAlign': 'center',
                         'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px',
                         'margin': '5px'}),
                
                html.Div([
                    html.P("Sharpe Ratio", style={'margin': '5px', 'color': '#666'}),
                    html.H4(f"{metrics['sharpe_ratio']:.2f}",
                           style={'margin': '5px', 'color': '#121212'})
                ], style={'display': 'inline-block', 'width': '30%', 'textAlign': 'center',
                         'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px',
                         'margin': '5px'})
            ]),
            
            html.Div([
                html.Div([
                    html.P("Annual Return", style={'margin': '5px', 'color': '#666'}),
                    html.H4(f"{metrics['annual_return']:.2f}%",
                           style={'margin': '5px', 'color': '#121212'})
                ], style={'display': 'inline-block', 'width': '30%', 'textAlign': 'center',
                         'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px',
                         'margin': '5px'}),
                
                html.Div([
                    html.P("Volatility", style={'margin': '5px', 'color': '#666'}),
                    html.H4(f"{metrics['volatility']:.2f}%",
                           style={'margin': '5px', 'color': '#121212'})
                ], style={'display': 'inline-block', 'width': '30%', 'textAlign': 'center',
                         'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px',
                         'margin': '5px'}),
                
                html.Div([
                    html.P("Max Drawdown", style={'margin': '5px', 'color': '#666'}),
                    html.H4(f"{metrics['max_drawdown']:.2f}%",
                           style={'margin': '5px', 'color': '#E3120B'})
                ], style={'display': 'inline-block', 'width': '30%', 'textAlign': 'center',
                         'padding': '10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px',
                         'margin': '5px'})
            ])
        ])
    
    def _empty_figures(self):
        """Return empty figures for error state."""
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Error loading data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return (empty_fig, empty_fig, html.Div("Error"), 
               empty_fig, empty_fig, empty_fig)
    
    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard."""
        self.app.run(debug=debug, port=port)


def create_dashboard():
    """Create and return dashboard instance."""
    return FinancialDashboard()
