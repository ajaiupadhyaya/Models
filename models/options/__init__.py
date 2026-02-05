"""
Options Pricing Module
Black-Scholes, Heston, and other option pricing models.
"""

try:
    from .black_scholes import BlackScholes
    BLACK_SCHOLES_AVAILABLE = True
except ImportError:
    BlackScholes = None
    BLACK_SCHOLES_AVAILABLE = False

try:
    from .advanced_pricing import HestonModel, SABRModel
    ADVANCED_PRICING_AVAILABLE = True
except ImportError:
    HestonModel = None
    SABRModel = None
    ADVANCED_PRICING_AVAILABLE = False

__all__ = [
    'BlackScholes',
    'HestonModel',
    'SABRModel',
    'BLACK_SCHOLES_AVAILABLE',
    'ADVANCED_PRICING_AVAILABLE',
]
