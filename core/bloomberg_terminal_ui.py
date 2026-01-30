"""
Bloomberg Terminal-like Modern Web UI
Professional, multi-panel, real-time financial dashboard
"""

import dash
from dash import dcc, html, Input, Output, callback, clientside_callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from core.data_fetcher import DataFetcher
from core.visualizations import ChartBuilder
from core.advanced_visualizations import PublicationCharts
from core.automated_trading_orchestrator import AutomatedTradingOrchestrator
import logging

logger = logging.getLogger(__name__)

# Bloomberg Terminal color scheme
BLOOMBERG_COLORS = {
    'background': '#0d1117',
    'panel': '#161b22',
    'text': '#c9d1d9',
    'text_secondary': '#8b949e',
    'accent': '#58a6ff',
    'accent_green': '#3fb950',
    'accent_red': '#f85149',
    'border': '#30363d',
    'hover': '#21262d'
}


class BloombergTerminalUI:
    """
    Modern Bloomberg Terminal-like web interface.
    Multi-panel layout with real-time updates.
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize Bloomberg Terminal UI.
        
        Args:
            symbols: List of symbols to display
        """
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"]
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True
        )
        self.fetcher = DataFetcher()
        self.orchestrator = None
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up Bloomberg Terminal-style layout."""
        self.app.layout = dbc.Container([
            # Header Bar
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2("BLOOMBERG TERMINAL", style={
                            'color': BLOOMBERG_COLORS['accent'],
                            'fontWeight': 'bold',
                            'margin': 0,
                            'fontSize': '24px'
                        }),
                        html.Span(id='header-time', style={
                            'color': BLOOMBERG_COLORS['text_secondary'],
                            'marginLeft': '20px',
                            'fontSize': '14px'
                        })
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], width=6),
                dbc.Col([
                    html.Div([
                        dbc.Badge("LIVE", color="success", className="me-2"),
                        dbc.Badge("AUTO", color="info", className="me-2"),
                        dbc.Badge("AI/ML", color="warning", className="me-2"),
                    ], style={'textAlign': 'right'})
                ], width=6)
            ], className="mb-3", style={
                'backgroundColor': BLOOMBERG_COLORS['panel'],
                'padding': '15px',
                'borderBottom': f'2px solid {BLOOMBERG_COLORS["border"]}'
            }),
            
            # Main Content Grid
            dbc.Row([
                # Left Column - Watchlist & Charts
                dbc.Col([
                    # Watchlist Panel
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("WATCHLIST", style={'margin': 0, 'color': BLOOMBERG_COLORS['accent']})
                        ]),
                        dbc.CardBody([
                            html.Div(id='watchlist-table')
                        ])
                    ], className="mb-3", style={
                        'backgroundColor': BLOOMBERG_COLORS['panel'],
                        'border': f'1px solid {BLOOMBERG_COLORS["border"]}'
                    }),
                    
                    # Price Chart Panel
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.H5("PRICE CHART", style={'margin': 0, 'color': BLOOMBERG_COLORS['accent']}),
                                dcc.Dropdown(
                                    id='chart-symbol',
                                    options=[{'label': s, 'value': s} for s in self.symbols],
                                    value=self.symbols[0],
                                    style={'width': '150px', 'marginLeft': '20px'},
                                    className="d-inline-block"
                                )
                            ], style={'display': 'flex', 'alignItems': 'center'})
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='price-chart-main', style={'height': '400px'})
                        ])
                    ], className="mb-3", style={
                        'backgroundColor': BLOOMBERG_COLORS['panel'],
                        'border': f'1px solid {BLOOMBERG_COLORS["border"]}'
                    }),
                    
                    # Technical Indicators Panel
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("TECHNICAL INDICATORS", style={'margin': 0, 'color': BLOOMBERG_COLORS['accent']})
                        ]),
                        dbc.CardBody([
                            html.Div(id='technical-indicators')
                        ])
                    ], style={
                        'backgroundColor': BLOOMBERG_COLORS['panel'],
                        'border': f'1px solid {BLOOMBERG_COLORS["border"]}'
                    })
                ], width=8),
                
                # Right Column - AI/ML Signals & Portfolio
                dbc.Col([
                    # AI Trading Signals Panel
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("AI TRADING SIGNALS", style={'margin': 0, 'color': BLOOMBERG_COLORS['accent']})
                        ]),
                        dbc.CardBody([
                            html.Div(id='ai-signals')
                        ])
                    ], className="mb-3", style={
                        'backgroundColor': BLOOMBERG_COLORS['panel'],
                        'border': f'1px solid {BLOOMBERG_COLORS["border"]}'
                    }),
                    
                    # Model Performance Panel
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("MODEL PERFORMANCE", style={'margin': 0, 'color': BLOOMBERG_COLORS['accent']})
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='model-performance-chart', style={'height': '250px'})
                        ])
                    ], className="mb-3", style={
                        'backgroundColor': BLOOMBERG_COLORS['panel'],
                        'border': f'1px solid {BLOOMBERG_COLORS["border"]}'
                    }),
                    
                    # Portfolio Panel
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("PORTFOLIO", style={'margin': 0, 'color': BLOOMBERG_COLORS['accent']})
                        ]),
                        dbc.CardBody([
                            html.Div(id='portfolio-summary')
                        ])
                    ], className="mb-3", style={
                        'backgroundColor': BLOOMBERG_COLORS['panel'],
                        'border': f'1px solid {BLOOMBERG_COLORS["border"]}'
                    }),
                    
                    # Risk Metrics Panel
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("RISK METRICS", style={'margin': 0, 'color': BLOOMBERG_COLORS['accent']})
                        ]),
                        dbc.CardBody([
                            html.Div(id='risk-metrics')
                        ])
                    ], style={
                        'backgroundColor': BLOOMBERG_COLORS['panel'],
                        'border': f'1px solid {BLOOMBERG_COLORS["border"]}'
                    })
                ], width=4)
            ]),
            
            # Bottom Row - Market Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("MARKET OVERVIEW", style={'margin': 0, 'color': BLOOMBERG_COLORS['accent']})
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id='market-overview', style={'height': '300px'})
                        ])
                    ], style={
                        'backgroundColor': BLOOMBERG_COLORS['panel'],
                        'border': f'1px solid {BLOOMBERG_COLORS["border"]}'
                    })
                ], width=12)
            ], className="mt-3"),
            
            # Auto-refresh intervals
            dcc.Interval(
                id='interval-component',
                interval=5000,  # 5 seconds for real-time feel
                n_intervals=0
            ),
            dcc.Interval(
                id='interval-slow',
                interval=60000,  # 1 minute for slower updates
                n_intervals=0
            ),
            
            # Store for data
            dcc.Store(id='data-store'),
            dcc.Store(id='signals-store')
        ], fluid=True, style={
            'backgroundColor': BLOOMBERG_COLORS['background'],
            'minHeight': '100vh',
            'padding': '20px',
            'fontFamily': '"SF Mono", "Monaco", "Inconsolata", "Roboto Mono", monospace'
        })
    
    def setup_callbacks(self):
        """Set up all callbacks."""
        
        @self.app.callback(
            [Output('header-time', 'children'),
             Output('watchlist-table', 'children'),
             Output('data-store', 'data')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_header_and_watchlist(n):
            """Update header time and watchlist."""
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Fetch watchlist data
            watchlist_data = []
            for symbol in self.symbols:
                try:
                    df = self.fetcher.get_stock_data(symbol, period="1d")
                    if df is not None and len(df) > 0:
                        current_price = df['Close'].iloc[-1]
                        prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
                        change = current_price - prev_close
                        change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                        
                        watchlist_data.append({
                            'symbol': symbol,
                            'price': current_price,
                            'change': change,
                            'change_pct': change_pct
                        })
                except:
                    pass
            
            # Create watchlist table
            table_rows = []
            for data in watchlist_data:
                color = BLOOMBERG_COLORS['accent_green'] if data['change'] >= 0 else BLOOMBERG_COLORS['accent_red']
                table_rows.append(
                    html.Tr([
                        html.Td(data['symbol'], style={'color': BLOOMBERG_COLORS['text'], 'fontWeight': 'bold'}),
                        html.Td(f"${data['price']:.2f}", style={'color': BLOOMBERG_COLORS['text']}),
                        html.Td(
                            f"{data['change']:+.2f} ({data['change_pct']:+.2f}%)",
                            style={'color': color, 'fontWeight': 'bold'}
                        )
                    ])
                )
            
            watchlist_table = html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Symbol", style={'color': BLOOMBERG_COLORS['text_secondary']}),
                        html.Th("Price", style={'color': BLOOMBERG_COLORS['text_secondary']}),
                        html.Th("Change", style={'color': BLOOMBERG_COLORS['text_secondary']})
                    ])
                ]),
                html.Tbody(table_rows)
            ], className="table", style={'width': '100%'})
            
            return current_time, watchlist_table, json.dumps(watchlist_data)
        
        @self.app.callback(
            Output('price-chart-main', 'figure'),
            [Input('chart-symbol', 'value'),
             Input('interval-slow', 'n_intervals')]
        )
        def update_price_chart(symbol, n):
            """Update main price chart."""
            try:
                df = self.fetcher.get_stock_data(symbol, period="1mo")
                if df is None or len(df) == 0:
                    return go.Figure()
                
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=symbol,
                    increasing_line_color=BLOOMBERG_COLORS['accent_green'],
                    decreasing_line_color=BLOOMBERG_COLORS['accent_red']
                ))
                
                # Moving averages
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA50'] = df['Close'].rolling(50).mean()
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MA20'],
                    name='MA20',
                    line=dict(color=BLOOMBERG_COLORS['accent'], width=1)
                ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template='plotly_dark',
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                return fig
            except Exception as e:
                return go.Figure()
        
        @self.app.callback(
            Output('ai-signals', 'children'),
            [Input('interval-slow', 'n_intervals'),
             Input('signals-store', 'data')]
        )
        def update_ai_signals(n, signals_data):
            """Update AI trading signals."""
            signals = []
            
            # Try to get signals from orchestrator
            if self.orchestrator:
                try:
                    for symbol in self.symbols[:3]:  # Limit to 3 for display
                        try:
                            symbol_signals = self.orchestrator.generate_signals(symbol)
                            if symbol_signals:
                                signals.extend(symbol_signals[:1])  # Top signal per symbol
                        except Exception as e:
                            logger.warning(f"Signal generation failed for {symbol}: {e}")
                            continue
                except Exception as e:
                    logger.warning(f"Orchestrator signal error: {e}")
            
            if not signals:
                return html.Div([
                    html.P("No signals available", style={'color': BLOOMBERG_COLORS['text_secondary']}),
                    html.Small("Signals will appear here when models generate trading recommendations", 
                              style={'color': BLOOMBERG_COLORS['text_secondary'], 'fontSize': '11px'})
                ])
            
            signal_cards = []
            for signal in signals[:5]:  # Show top 5
                action_color = BLOOMBERG_COLORS['accent_green'] if signal.action == "BUY" else BLOOMBERG_COLORS['accent_red']
                signal_cards.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Strong(signal.symbol, style={'color': BLOOMBERG_COLORS['text']}),
                                html.Span(
                                    f" {signal.action}",
                                    style={'color': action_color, 'marginLeft': '10px', 'fontWeight': 'bold'}
                                )
                            ]),
                            html.P(
                                signal.reasoning[:100] + "..." if len(signal.reasoning) > 100 else signal.reasoning,
                                style={'color': BLOOMBERG_COLORS['text_secondary'], 'fontSize': '12px', 'marginTop': '5px'}
                            ),
                            html.Small(
                                f"Confidence: {signal.confidence*100:.0f}% | {signal.model_source.upper()}",
                                style={'color': BLOOMBERG_COLORS['text_secondary']}
                            )
                        ])
                    ], className="mb-2", style={
                        'backgroundColor': BLOOMBERG_COLORS['hover'],
                        'border': f'1px solid {BLOOMBERG_COLORS["border"]}'
                    })
                )
            
            return html.Div(signal_cards)
        
        @self.app.callback(
            Output('technical-indicators', 'children'),
            [Input('chart-symbol', 'value'),
             Input('interval-slow', 'n_intervals')]
        )
        def update_technical_indicators(symbol, n):
            """Update technical indicators."""
            try:
                df = self.fetcher.get_stock_data(symbol, period="3mo")
                if df is None or len(df) == 0:
                    return html.Div("No data", style={'color': BLOOMBERG_COLORS['text_secondary']})
                
                # Calculate indicators
                df['RSI'] = self._calculate_rsi(df['Close'])
                df['MACD'], df['MACD_signal'] = self._calculate_macd(df['Close'])
                
                current_price = df['Close'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                macd = df['MACD'].iloc[-1]
                macd_signal = df['MACD_signal'].iloc[-1]
                
                indicators = [
                    ("Price", f"${current_price:.2f}", BLOOMBERG_COLORS['text']),
                    ("RSI", f"{rsi:.2f}", BLOOMBERG_COLORS['accent'] if 30 < rsi < 70 else BLOOMBERG_COLORS['accent_red']),
                    ("MACD", f"{macd:.2f}", BLOOMBERG_COLORS['accent_green'] if macd > macd_signal else BLOOMBERG_COLORS['accent_red'])
                ]
                
                indicator_items = []
                for name, value, color in indicators:
                    indicator_items.append(
                        html.Div([
                            html.Span(name, style={'color': BLOOMBERG_COLORS['text_secondary'], 'fontSize': '12px'}),
                            html.Br(),
                            html.Strong(value, style={'color': color, 'fontSize': '18px'})
                        ], style={'display': 'inline-block', 'marginRight': '30px', 'marginTop': '10px'})
                    )
                
                return html.Div(indicator_items)
            except:
                return html.Div("Error loading indicators", style={'color': BLOOMBERG_COLORS['accent_red']})
        
        @self.app.callback(
            Output('market-overview', 'figure'),
            [Input('interval-slow', 'n_intervals')]
        )
        def update_market_overview(n):
            """Update market overview chart."""
            try:
                # Fetch multiple symbols for comparison
                data = []
                for symbol in self.symbols[:5]:
                    df = self.fetcher.get_stock_data(symbol, period="1mo")
                    if df is not None and len(df) > 0:
                        returns = df['Close'].pct_change().fillna(0)
                        cumulative = (1 + returns).cumprod()
                        data.append(go.Scatter(
                            x=df.index,
                            y=cumulative,
                            name=symbol,
                            line=dict(width=2)
                        ))
                
                fig = go.Figure(data=data)
                fig.update_layout(
                    title="Market Overview - Cumulative Returns",
                    template='plotly_dark',
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=True
                )
                
                return fig
            except:
                return go.Figure()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def run(self, debug: bool = False, port: int = 8050):
        """Run the dashboard."""
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')


def create_bloomberg_terminal(symbols: List[str] = None) -> BloombergTerminalUI:
    """Create Bloomberg Terminal UI instance."""
    return BloombergTerminalUI(symbols=symbols)
