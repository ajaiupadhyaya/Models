"""
Risk Analysis Module
"""

try:
    from .var_cvar import VaRModel, CVaRModel
except ImportError:
    VaRModel = None
    CVaRModel = None

try:
    from .stress_testing import (
        StressScenario,
        HistoricalScenarioAnalyzer,
        HypotheticalScenarioBuilder,
        PortfolioStressTester
    )
except ImportError:
    StressScenario = None
    HistoricalScenarioAnalyzer = None
    HypotheticalScenarioBuilder = None
    PortfolioStressTester = None

try:
    from .scenario_analysis import (
        ScenarioAnalysisFull,
        SystemicRiskMeasures,
        CorrelationDynamics
    )
except ImportError:
    ScenarioAnalysisFull = None
    SystemicRiskMeasures = None
    CorrelationDynamics = None

__all__ = [
    'VaRModel',
    'CVaRModel',
    'StressScenario',
    'HistoricalScenarioAnalyzer',
    'HypotheticalScenarioBuilder',
    'PortfolioStressTester',
    'ScenarioAnalysisFull',
    'SystemicRiskMeasures',
    'CorrelationDynamics'
]

