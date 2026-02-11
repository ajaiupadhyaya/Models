"""
Unified Quantitative Financial Platform
A single, professional platform integrating all analysis, trading, and reporting capabilities.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, dash_table

# Use html components instead of dbc for compatibility
# dbc is optional - we'll use html.Div with inline styles
HAS_DBC = False
try:
    import dash_bootstrap_components as dbc
    HAS_DBC = True
except ImportError:
    # Create simple wrapper classes
    class dbc:
        @staticmethod
        def Container(children=None, **kwargs):
            if children is None:
                children = []
            elif not isinstance(children, list):
                children = [children]
            return html.Div(children, style=kwargs.get('style', {}), className=kwargs.get('className', ''))
        
        @staticmethod
        def Row(children=None, **kwargs):
            if children is None:
                children = []
            elif not isinstance(children, list):
                children = [children]
            div_kwargs = {'style': kwargs.get('style', {}), 'className': kwargs.get('className', '')}
            if 'id' in kwargs and kwargs['id'] is not None:
                div_kwargs['id'] = kwargs['id']
            return html.Div(children, **div_kwargs)
        
        @staticmethod
        def Col(children=None, **kwargs):
            if children is None:
                children = []
            elif not isinstance(children, list):
                children = [children]
            width = kwargs.get('width', 12)
            style = kwargs.get('style', {})
            if 'display' not in style:
                style['display'] = 'inline-block'
            if 'width' not in style and width:
                style['width'] = f'{width/12*100}%'
            div_kwargs = {'style': style, 'className': kwargs.get('className', '')}
            if 'id' in kwargs and kwargs['id'] is not None:
                div_kwargs['id'] = kwargs['id']
            return html.Div(children, **div_kwargs)
        
        @staticmethod
        def Card(children=None, **kwargs):
            if children is None:
                children = []
            elif not isinstance(children, list):
                children = [children]
            return html.Div(children, style=kwargs.get('style', {}))
        
        @staticmethod
        def CardBody(children=None, **kwargs):
            if children is None:
                children = []
            elif not isinstance(children, list):
                children = [children]
            return html.Div(children, style={'padding': '20px'})
        
        @staticmethod
        def Button(children, **kwargs):
            return html.Button(children, id=kwargs.get('id'), n_clicks=kwargs.get('n_clicks', 0),
                             style=kwargs.get('style', {}), className=kwargs.get('className', ''))
        
        @staticmethod
        def ButtonGroup(children, **kwargs):
            return html.Div(children, style={'display': 'inline-block'})
        
        class themes:
            CYBORG = None
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path
import logging

# Core imports
from core.data_fetcher import DataFetcher
from core.utils import (
    calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_var, calculate_cvar, format_currency, format_percentage,
    calculate_beta, annualize_returns, annualize_volatility
)
# Advanced visualizations - we'll use inline functions instead
# Import models with error handling
try:
    from models.portfolio.optimization import MeanVarianceOptimizer, optimize_portfolio_from_returns
except ImportError:
    MeanVarianceOptimizer = None
    optimize_portfolio_from_returns = None

try:
    from models.risk.var_cvar import VaRModel, CVaRModel
except ImportError:
    VaRModel = None
    CVaRModel = None

try:
    from models.trading.strategies import MomentumStrategy, MeanReversionStrategy
except ImportError:
    MomentumStrategy = None
    MeanReversionStrategy = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize components
data_fetcher = DataFetcher()

# Professional dark theme
THEME = {
    'bg': '#0a0e27',
    'card': '#151b3d',
    'accent': '#1e3a8a',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'text': '#e5e7eb',
    'text_muted': '#9ca3af',
    'border': '#1f2937',
    'grid': '#374151',
}

# Personalization config (load from file or env)
CONFIG_FILE = Path('config/user_config.json')
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, 'r') as f:
        USER_CONFIG = json.load(f)
else:
    USER_CONFIG = {
        'watchlist': ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
        'preferred_period': '1y',
        'risk_tolerance': 'moderate',
        'auto_refresh': True,
        'refresh_interval': 300,  # seconds
    }
    CONFIG_FILE.parent.mkdir(exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(USER_CONFIG, f, indent=2)

# Initialize Dash app
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
]
if HAS_DBC:
    external_stylesheets.insert(0, dbc.themes.CYBORG)

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    title="Quantitative Financial Platform"
)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Quantitative Financial Platform</title>
        {%favicon%}
        {%css%}
        <style>
            * { font-family: 'Inter', sans-serif; }
            body { background-color: #0a0e27; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


def create_styled_figure(fig: go.Figure, title: str, height: int = 450) -> go.Figure:
    """Apply professional styling to figures."""
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': THEME['text'], 'family': 'Inter, sans-serif', 'weight': '600'},
            'pad': {'t': 20}
        },
        plot_bgcolor=THEME['card'],
        paper_bgcolor=THEME['card'],
        font={'color': THEME['text'], 'size': 12, 'family': 'Inter, sans-serif'},
        xaxis=dict(
            gridcolor=THEME['grid'],
            showgrid=True,
            linecolor=THEME['border'],
            zeroline=False
        ),
        yaxis=dict(
            gridcolor=THEME['grid'],
            showgrid=True,
            linecolor=THEME['border'],
            zeroline=False
        ),
        height=height,
        margin=dict(l=60, r=40, t=80, b=50),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor=THEME['card'],
            font_size=12,
            font_family="Inter, sans-serif"
        ),
        legend=dict(
            bgcolor=THEME['card'],
            bordercolor=THEME['border'],
            borderwidth=1,
            font=dict(color=THEME['text'])
        ),
        template='plotly_dark',
    )
    return fig


def fetch_market_data(tickers: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
    """Fetch market data for multiple tickers dynamically."""
    data = {}
    for ticker in tickers:
        try:
            df = data_fetcher.get_stock_data(ticker.upper(), period=period)
            if not df.empty:
                data[ticker.upper()] = df
        except Exception as e:
            logger.warning(f"Error fetching {ticker}: {e}")
    return data


def calculate_comprehensive_metrics(df: pd.DataFrame, ticker: str) -> Dict:
    """Calculate comprehensive performance and risk metrics."""
    if df.empty or 'Close' not in df.columns:
        return {}
    
    returns = calculate_returns(df['Close'])
    if returns.empty:
        return {}
    
    current_price = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
    daily_change = current_price - prev_close
    daily_pct = (daily_change / prev_close) * 100 if prev_close > 0 else 0
    
    # Calculate all metrics
    total_return = float((current_price / df['Close'].iloc[0]) - 1)
    volatility = float(annualize_volatility(returns))
    sharpe = float(calculate_sharpe_ratio(returns)) if returns.std() > 0 else 0
    max_dd = float(calculate_max_drawdown(returns))
    var_95 = float(calculate_var(returns, 0.05))
    cvar_95 = float(calculate_cvar(returns, 0.05)) if returns.std() > 0 else 0
    
    # 52-week high/low
    if len(df) >= 252:
        high_52w = float(df['High'].rolling(252).max().iloc[-1])
        low_52w = float(df['Low'].rolling(252).min().iloc[-1])
    else:
        high_52w = float(df['High'].max())
        low_52w = float(df['Low'].min())
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'daily_change': daily_change,
        'daily_pct': daily_pct,
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'high_52w': high_52w,
        'low_52w': low_52w,
        'distance_from_high': float((current_price / high_52w - 1) * 100) if high_52w > 0 else 0,
        'distance_from_low': float((current_price / low_52w - 1) * 100) if low_52w > 0 else 0,
    }


def create_advanced_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create advanced price chart with multiple indicators."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Price Action', 'Volume', 'Technical Indicators')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color=THEME['success'],
            decreasing_line_color=THEME['danger'],
        ),
        row=1, col=1
    )
    
    # Moving averages
    if len(df) > 20:
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean() if len(df) > 200 else None
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA20'], name='MA20',
                      line=dict(color='#3b82f6', width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA50'], name='MA50',
                      line=dict(color='#f59e0b', width=1.5)),
            row=1, col=1
        )
        if df['MA200'] is not None:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MA200'], name='MA200',
                          line=dict(color='#8b5cf6', width=1.5)),
                row=1, col=1
            )
    
    # Volume
    colors_vol = [THEME['success'] if df['Close'].iloc[i] >= df['Open'].iloc[i]
                 else THEME['danger'] for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color=colors_vol, opacity=0.6),
        row=2, col=1
    )
    
    # RSI
    if len(df) > 14:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(
            go.Scatter(x=df.index, y=rsi, name='RSI',
                      line=dict(color='#ec4899', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color=THEME['danger'],
                      annotation_text="Overbought", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=THEME['success'],
                      annotation_text="Oversold", row=3, col=1)
    
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    
    return create_styled_figure(fig, f'{ticker} - Advanced Analysis', height=700)


def create_portfolio_heatmap(tickers: List[str], period: str) -> go.Figure:
    """Create correlation heatmap for portfolio."""
    data_dict = fetch_market_data(tickers, period)
    
    if not data_dict:
        return create_styled_figure(go.Figure(), "Portfolio Correlation")
    
    # Create returns DataFrame
    returns_data = {}
    for ticker, df in data_dict.items():
        if 'Close' in df.columns:
            returns_data[ticker] = df['Close'].pct_change()
    
    if not returns_data:
        return create_styled_figure(go.Figure(), "Portfolio Correlation")
    
    returns_df = pd.DataFrame(returns_data).dropna()
    
    if returns_df.empty:
        return create_styled_figure(go.Figure(), "Portfolio Correlation")
    
    corr_matrix = returns_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 11, "color": THEME['text']},
        colorbar=dict(title="Correlation", titlefont=dict(color=THEME['text'])),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    return create_styled_figure(fig, "Portfolio Correlation Matrix", height=500)


def create_efficient_frontier(tickers: List[str], period: str) -> go.Figure:
    """Create efficient frontier visualization."""
    data_dict = fetch_market_data(tickers, period)
    
    if len(data_dict) < 2:
        return create_styled_figure(go.Figure(), "Efficient Frontier")
    
    # Create returns DataFrame
    returns_data = {}
    for ticker, df in data_dict.items():
        if 'Close' in df.columns:
            returns_data[ticker] = df['Close'].pct_change()
    
    if not returns_data:
        return create_styled_figure(go.Figure(), "Efficient Frontier")
    
    returns_df = pd.DataFrame(returns_data).dropna()
    
    if returns_df.empty:
        return create_styled_figure(go.Figure(), "Efficient Frontier")
    
    # Calculate expected returns and covariance
    expected_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    # Optimize portfolio
    try:
        optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix)
        result = optimizer.optimize_sharpe()
        
        # Create scatter plot
        fig = go.Figure()
        
        # Individual assets
        for ticker in returns_df.columns:
            ret = expected_returns[ticker]
            vol = np.sqrt(cov_matrix.loc[ticker, ticker])
            fig.add_trace(go.Scatter(
                x=[vol], y=[ret],
                mode='markers+text',
                name=ticker,
                text=ticker,
                textposition="top center",
                marker=dict(size=12, color=THEME['accent'])
            ))
        
        # Optimal portfolio
        opt_ret = result['expected_return']
        opt_vol = result['volatility']
        fig.add_trace(go.Scatter(
            x=[opt_vol], y=[opt_ret],
            mode='markers+text',
            name='Optimal Portfolio',
            text='Optimal',
            textposition="top center",
            marker=dict(size=15, color=THEME['success'], symbol='star')
        ))
        
        fig.update_layout(
            xaxis_title="Volatility (Annualized)",
            yaxis_title="Expected Return (Annualized)"
        )
        
        return create_styled_figure(fig, "Efficient Frontier Analysis", height=500)
    except Exception as e:
        logger.error(f"Error creating efficient frontier: {e}")
        return create_styled_figure(go.Figure(), "Efficient Frontier")


def create_macro_dashboard() -> go.Figure:
    """Create comprehensive macroeconomic dashboard."""
    indicators = {
        'UNRATE': 'Unemployment Rate',
        'GDP': 'Gross Domestic Product',
        'CPIAUCSL': 'Consumer Price Index',
        'FEDFUNDS': 'Federal Funds Rate',
        'DGS10': '10-Year Treasury Rate',
    }
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    macro_data = {}
    for code, name in indicators.items():
        try:
            data = data_fetcher.get_economic_indicator(code, start_date, end_date)
            if not data.empty:
                macro_data[name] = data
        except Exception as e:
            logger.warning(f"Error fetching {code}: {e}")
    
    if not macro_data:
        return create_styled_figure(go.Figure(), "Macroeconomic Indicators")
    
    n_indicators = len(macro_data)
    fig = make_subplots(
        rows=n_indicators, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=list(macro_data.keys())
    )
    
    colors = px.colors.qualitative.Set3
    for idx, (name, data) in enumerate(macro_data.items(), 1):
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                name=name,
                line=dict(color=color, width=2.5),
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}',
                hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
            ),
            row=idx, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=n_indicators, col=1)
    
    return create_styled_figure(fig, "Macroeconomic Dashboard", height=800)


# App Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Quantitative Financial Platform", 
                       className="mb-2",
                       style={'color': THEME['text'], 'fontWeight': '700', 'fontSize': '2.5rem'}),
                html.P("Professional Analysis • Real-Time Data • Automated Insights",
                      style={'color': THEME['text_muted'], 'fontSize': '1.1rem'})
            ], className="text-center mb-4")
        ])
    ], className="mb-4 pt-4"),
    
    # Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Ticker:", style={'color': THEME['text'], 'fontWeight': '600', 'marginBottom': '5px'}),
                            dcc.Input(
                                id='ticker-input',
                                value=USER_CONFIG['watchlist'][0] if USER_CONFIG['watchlist'] else 'SPY',
                                type='text',
                                placeholder='Enter ticker...',
                                style={
                                    'width': '100%',
                                    'backgroundColor': THEME['card'],
                                    'color': THEME['text'],
                                    'border': f'1px solid {THEME["border"]}',
                                    'borderRadius': '6px',
                                    'padding': '8px'
                                }
                            )
                        ], width=2),
                        dbc.Col([
                            html.Label("Period:", style={'color': THEME['text'], 'fontWeight': '600', 'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='period-dropdown',
                                options=[
                                    {'label': '1 Month', 'value': '1mo'},
                                    {'label': '3 Months', 'value': '3mo'},
                                    {'label': '6 Months', 'value': '6mo'},
                                    {'label': '1 Year', 'value': '1y'},
                                    {'label': '2 Years', 'value': '2y'},
                                    {'label': '5 Years', 'value': '5y'},
                                    {'label': 'Max', 'value': 'max'}
                                ],
                                value=USER_CONFIG.get('preferred_period', '1y'),
                                style={'backgroundColor': THEME['card'], 'color': THEME['text']}
                            )
                        ], width=2),
                        dbc.Col([
                            html.Label("Watchlist:", style={'color': THEME['text'], 'fontWeight': '600', 'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='watchlist-dropdown',
                                options=[{'label': t, 'value': t} for t in USER_CONFIG['watchlist']],
                                value=USER_CONFIG['watchlist'][:5],
                                multi=True,
                                style={'backgroundColor': THEME['card'], 'color': THEME['text']}
                            )
                        ], width=4),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("Update", id="update-button", color="primary", 
                                         className="mt-4", n_clicks=0),
                                dbc.Button("Export", id="export-button", color="success",
                                         className="mt-4", n_clicks=0),
                                dbc.Button("Auto", id="auto-button", color="info",
                                         className="mt-4", n_clicks=0)
                            ])
                        ], width=2, className="d-flex align-items-end")
                    ])
                ])
            ], style={'backgroundColor': THEME['card'], 'border': f'1px solid {THEME["border"]}'})
        ])
    ], className="mb-4"),
    
    # Metrics Cards
    dbc.Row(id='metrics-cards', className="mb-4"),
    
    # Main Chart
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='main-chart', config={'displayModeBar': True, 'displaylogo': False})
        ], width=12, className="mb-4")
    ]),
    
    # Portfolio Analysis Row
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='portfolio-heatmap', config={'displayModeBar': True, 'displaylogo': False})
        ], width=6),
        dbc.Col([
            dcc.Graph(id='efficient-frontier', config={'displayModeBar': True, 'displaylogo': False})
        ], width=6)
    ], className="mb-4"),
    
    # Macro Dashboard
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='macro-dashboard', config={'displayModeBar': True, 'displaylogo': False})
        ], width=12)
    ]),
    
    # Data stores
    dcc.Store(id='current-data'),
    dcc.Store(id='current-metrics'),
    dcc.Store(id='portfolio-data'),
    dcc.Interval(
        id='interval-component',
        interval=USER_CONFIG.get('refresh_interval', 300) * 1000,
        n_intervals=0,
        disabled=not USER_CONFIG.get('auto_refresh', True)
    ),
    
], fluid=True, style={'backgroundColor': THEME['bg'], 'minHeight': '100vh', 'padding': '20px'})


# Callbacks
@callback(
    [Output('current-data', 'data'),
     Output('current-metrics', 'data'),
     Output('main-chart', 'figure'),
     Output('metrics-cards', 'children')],
    [Input('update-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('ticker-input', 'value'),
     State('period-dropdown', 'value')]
)
def update_main_dashboard(n_clicks, n_intervals, ticker, period):
    """Update main dashboard with ticker data."""
    try:
        if not ticker:
            ticker = 'SPY'
        
        ticker = ticker.upper().strip()
        
        # Fetch data
        df = data_fetcher.get_stock_data(ticker, period=period)
        
        if df.empty:
            empty_fig = create_styled_figure(go.Figure(), f"{ticker} - No Data Available")
            return {}, {}, empty_fig, html.Div("No data available", style={'color': THEME['text']})
        
        # Create advanced chart
        fig = create_advanced_price_chart(df, ticker)
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(df, ticker)
        
        # Create metrics cards
        if metrics:
            metrics_cards = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Price", style={'color': THEME['text_muted'], 'marginBottom': '5px'}),
                            html.H4(format_currency(metrics['current_price']),
                                   style={'color': THEME['text'], 'marginBottom': '0'})
                        ])
                    ], style={'backgroundColor': THEME['card'], 'border': f'1px solid {THEME["border"]}'})
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Daily Change", style={'color': THEME['text_muted'], 'marginBottom': '5px'}),
                            html.H4(
                                f"{'+' if metrics['daily_pct'] >= 0 else ''}{metrics['daily_pct']:.2f}%",
                                style={
                                    'color': THEME['success'] if metrics['daily_pct'] >= 0 else THEME['danger'],
                                    'marginBottom': '0'
                                }
                            )
                        ])
                    ], style={'backgroundColor': THEME['card'], 'border': f'1px solid {THEME["border"]}'})
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Total Return", style={'color': THEME['text_muted'], 'marginBottom': '5px'}),
                            html.H4(
                                format_percentage(metrics['total_return']),
                                style={
                                    'color': THEME['success'] if metrics['total_return'] >= 0 else THEME['danger'],
                                    'marginBottom': '0'
                                }
                            )
                        ])
                    ], style={'backgroundColor': THEME['card'], 'border': f'1px solid {THEME["border"]}'})
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Sharpe Ratio", style={'color': THEME['text_muted'], 'marginBottom': '5px'}),
                            html.H4(f"{metrics['sharpe_ratio']:.2f}",
                                   style={'color': THEME['text'], 'marginBottom': '0'})
                        ])
                    ], style={'backgroundColor': THEME['card'], 'border': f'1px solid {THEME["border"]}'})
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Volatility", style={'color': THEME['text_muted'], 'marginBottom': '5px'}),
                            html.H4(format_percentage(metrics['volatility']),
                                   style={'color': THEME['text'], 'marginBottom': '0'})
                        ])
                    ], style={'backgroundColor': THEME['card'], 'border': f'1px solid {THEME["border"]}'})
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Max Drawdown", style={'color': THEME['text_muted'], 'marginBottom': '5px'}),
                            html.H4(format_percentage(metrics['max_drawdown']),
                                   style={'color': THEME['danger'], 'marginBottom': '0'})
                        ])
                    ], style={'backgroundColor': THEME['card'], 'border': f'1px solid {THEME["border"]}'})
                ], width=2),
            ])
        else:
            metrics_cards = html.Div("Calculating metrics...", style={'color': THEME['text']})
        
        return df.to_dict('records'), metrics, fig, metrics_cards
        
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
        empty_fig = create_styled_figure(go.Figure(), "Error Loading Data")
        return {}, {}, empty_fig, html.Div(f"Error: {str(e)}", style={'color': THEME['danger']})


@callback(
    Output('portfolio-heatmap', 'figure'),
    [Input('watchlist-dropdown', 'value'),
     Input('period-dropdown', 'value')]
)
def update_portfolio_heatmap(watchlist, period):
    """Update portfolio correlation heatmap."""
    if not watchlist:
        return create_styled_figure(go.Figure(), "Portfolio Correlation")
    return create_portfolio_heatmap(watchlist, period)


@callback(
    Output('efficient-frontier', 'figure'),
    [Input('watchlist-dropdown', 'value'),
     Input('period-dropdown', 'value')]
)
def update_efficient_frontier(watchlist, period):
    """Update efficient frontier."""
    if not watchlist or len(watchlist) < 2:
        return create_styled_figure(go.Figure(), "Efficient Frontier")
    return create_efficient_frontier(watchlist, period)


@callback(
    Output('macro-dashboard', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_macro_dashboard(n_intervals):
    """Update macroeconomic dashboard."""
    return create_macro_dashboard()


@callback(
    Output('export-button', 'children'),
    Input('export-button', 'n_clicks'),
    [State('current-data', 'data'),
     State('current-metrics', 'data'),
     State('ticker-input', 'value')]
)
def export_report(n_clicks, data, metrics, ticker):
    """Export comprehensive report."""
    if n_clicks and data and metrics:
        try:
            reports_dir = Path('reports')
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = reports_dir / f'quant_report_{ticker}_{timestamp}.json'
            
            report_data = {
                'ticker': ticker,
                'timestamp': timestamp,
                'metrics': metrics,
                'data_points': len(data) if data else 0,
                'generated_by': 'Quantitative Financial Platform'
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            return f"✓ Exported: {report_path.name}"
        except Exception as e:
            logger.error(f"Error exporting: {e}")
            return f"✗ Error: {str(e)}"
    
    return "Export Report"


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 QUANTITATIVE FINANCIAL PLATFORM")
    print("="*80)
    print(f"📊 Dashboard: http://localhost:8050")
    print(f"⚙️  Config: {CONFIG_FILE}")
    print(f"🔄 Auto-refresh: {USER_CONFIG.get('auto_refresh', True)}")
    print("="*80 + "\n")
    
    app.run_server(debug=False, host='0.0.0.0', port=8050)
