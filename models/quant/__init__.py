"""
Quantitative Finance Models Module
Advanced risk metrics, statistical validation, and factor models.
"""

try:
    from .institutional_grade import (
        AdvancedRiskMetrics,
        StatisticalValidation,
        FamaFrenchFactorModel,
    )
    QUANT_AVAILABLE = True
except ImportError:
    AdvancedRiskMetrics = None
    StatisticalValidation = None
    FamaFrenchFactorModel = None
    QUANT_AVAILABLE = False

__all__ = [
    'AdvancedRiskMetrics',
    'StatisticalValidation',
    'FamaFrenchFactorModel',
    'QUANT_AVAILABLE',
]
