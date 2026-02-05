"""
Portfolio Management Module
Portfolio optimization, risk management, and allocation strategies.
"""

try:
    from .optimization import PortfolioOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    PortfolioOptimizer = None
    OPTIMIZATION_AVAILABLE = False

__all__ = [
    'PortfolioOptimizer',
    'OPTIMIZATION_AVAILABLE',
]
