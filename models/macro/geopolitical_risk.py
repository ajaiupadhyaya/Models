"""
Geopolitical Risk Analysis
Geopolitical risk assessment, policy impact, supply chain resilience
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class GeopoliticalRiskAnalyzer:
    """
    Analyze geopolitical risk factors.
    """
    
    def __init__(self):
        """Initialize geopolitical risk analyzer."""
        self._setup_risk_factors()
    
    def _setup_risk_factors(self):
        """Setup known geopolitical risk factors."""
        self.risk_categories = {
            'trade_tensions': {
                'description': 'Trade disputes and tariff conflicts',
                'asset_impact': {'stocks': -0.02, 'bonds': +0.01, 'commodities': +0.02},
                'sectors_affected': ['technology', 'retail', 'manufacturing']
            },
            'sanctions': {
                'description': 'International sanctions and embargoes',
                'asset_impact': {'stocks': -0.015, 'bonds': +0.005, 'commodities': +0.025},
                'sectors_affected': ['energy', 'materials', 'defense']
            },
            'elections': {
                'description': 'Major political elections',
                'asset_impact': {'stocks': -0.01, 'bonds': +0.005, 'commodities': 0},
                'sectors_affected': ['energy', 'financials', 'healthcare']
            },
            'military_conflicts': {
                'description': 'Military tensions and conflicts',
                'asset_impact': {'stocks': -0.025, 'bonds': +0.015, 'commodities': +0.03},
                'sectors_affected': ['energy', 'transportation', 'agriculture']
            },
            'supply_chain_disruption': {
                'description': 'Global supply chain disruptions',
                'asset_impact': {'stocks': -0.02, 'bonds': +0.01, 'commodities': +0.025},
                'sectors_affected': ['manufacturing', 'retail', 'semiconductors']
            },
            'currency_crisis': {
                'description': 'Currency or debt crises',
                'asset_impact': {'stocks': -0.03, 'bonds': +0.02, 'commodities': -0.01},
                'sectors_affected': ['financials', 'export-dependent']
            }
        }
    
    def assess_risk_level(self, 
                         risk_factor: str,
                         severity: float = 0.5) -> Dict:
        """
        Assess geopolitical risk level for a factor.
        
        Args:
            risk_factor: Type of risk (from risk_categories keys)
            severity: Risk severity (0-1 scale)
        
        Returns:
            Dictionary with risk assessment
        """
        if risk_factor not in self.risk_categories:
            return {'error': f'Unknown risk factor: {risk_factor}'}
        
        factor_info = self.risk_categories[risk_factor]
        severity = np.clip(severity, 0, 1)
        
        # Determine risk level
        if severity > 0.7:
            risk_level = 'Critical'
            color = 'red'
        elif severity > 0.5:
            risk_level = 'High'
            color = 'orange'
        elif severity > 0.3:
            risk_level = 'Medium'
            color = 'yellow'
        else:
            risk_level = 'Low'
            color = 'green'
        
        # Calculate market impact
        market_impact = {
            asset_class: impact * severity 
            for asset_class, impact in factor_info['asset_impact'].items()
        }
        
        return {
            'risk_factor': risk_factor,
            'description': factor_info['description'],
            'severity': severity,
            'risk_level': risk_level,
            'color': color,
            'affected_sectors': factor_info['sectors_affected'],
            'estimated_market_impact': market_impact,
            'recommendation': self._get_recommendation(risk_level)
        }
    
    def _get_recommendation(self, risk_level: str) -> str:
        """Get investment recommendation based on risk level."""
        recommendations = {
            'Critical': 'Consider defensive positions and reduce equity exposure',
            'High': 'Increase hedges and diversify away from affected sectors',
            'Medium': 'Monitor situation closely, limited action needed',
            'Low': 'Maintain normal positioning'
        }
        return recommendations.get(risk_level, 'Unknown')
    
    def track_key_risks(self) -> Dict:
        """
        Track key geopolitical risks.
        
        Returns:
            Dictionary with current risk assessment
        """
        # In production, this would integrate with news APIs, sentiment analysis, etc.
        # For now, returns a template
        
        current_risks = {
            'trade_tensions': 0.4,  # Current severity (would be dynamic)
            'sanctions': 0.3,
            'elections': 0.2,
            'military_conflicts': 0.5,
            'supply_chain_disruption': 0.35,
            'currency_crisis': 0.15
        }
        
        assessments = {}
        portfolio_impact = {'stocks': 0, 'bonds': 0, 'commodities': 0}
        
        for risk_factor, severity in current_risks.items():
            assessment = self.assess_risk_level(risk_factor, severity)
            assessments[risk_factor] = assessment
            
            # Aggregate portfolio impact
            for asset_class, impact in assessment['estimated_market_impact'].items():
                portfolio_impact[asset_class] += impact
        
        # Average impact by number of risks
        num_risks = len(current_risks)
        portfolio_impact = {k: v/num_risks for k, v in portfolio_impact.items()}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_assessments': assessments,
            'portfolio_impact': portfolio_impact,
            'highest_risk': max(assessments.items(), 
                               key=lambda x: x[1]['severity'])[0],
            'overall_geopolitical_score': np.mean([s for _, s in current_risks.items()])
        }


class PolicyImpactAssessor:
    """
    Assess impact of government policies on markets and companies.
    """
    
    def __init__(self):
        """Initialize policy impact assessor."""
        self._setup_policy_types()
    
    def _setup_policy_types(self):
        """Setup known policy types and impacts."""
        self.policy_types = {
            'monetary_policy': {
                'description': 'Central bank policy (rates, QE, etc)',
                'sectors_affected': ['financials', 'technology', 'real estate'],
                'asset_sensitivity': {
                    'tightening': {'stocks': -0.02, 'bonds': -0.03, 'gold': +0.01},
                    'easing': {'stocks': +0.025, 'bonds': +0.015, 'gold': -0.01}
                }
            },
            'fiscal_policy': {
                'description': 'Government spending and taxation',
                'sectors_affected': ['healthcare', 'infrastructure', 'defense', 'education'],
                'asset_sensitivity': {
                    'stimulus': {'stocks': +0.02, 'bonds': -0.02, 'commodities': +0.02},
                    'austerity': {'stocks': -0.025, 'bonds': +0.015, 'commodities': -0.015}
                }
            },
            'regulatory_policy': {
                'description': 'Regulation and compliance requirements',
                'sectors_affected': ['technology', 'healthcare', 'financials', 'energy'],
                'asset_sensitivity': {
                    'tightening': {'stocks': -0.015, 'bonds': +0.005},
                    'loosening': {'stocks': +0.02, 'bonds': -0.005}
                }
            },
            'trade_policy': {
                'description': 'Tariffs and trade agreements',
                'sectors_affected': ['technology', 'automotive', 'retail', 'agriculture'],
                'asset_sensitivity': {
                    'protectionist': {'stocks': -0.02, 'commodities': +0.02},
                    'free_trade': {'stocks': +0.015, 'commodities': -0.01}
                }
            },
            'tax_policy': {
                'description': 'Corporate and personal tax changes',
                'sectors_affected': ['financials', 'technology', 'real estate'],
                'asset_sensitivity': {
                    'increase': {'stocks': -0.03, 'bonds': +0.01},
                    'decrease': {'stocks': +0.03, 'bonds': -0.01}
                }
            },
            'environmental_policy': {
                'description': 'Environmental and climate regulations',
                'sectors_affected': ['energy', 'utilities', 'transportation', 'agriculture'],
                'asset_sensitivity': {
                    'stringent': {'energy': -0.05, 'renewables': +0.04, 'coal': -0.08},
                    'lenient': {'energy': +0.04, 'renewables': -0.03, 'coal': +0.06}
                }
            }
        }
    
    def assess_policy_impact(self,
                            policy_type: str,
                            direction: str,
                            implementation_timeline: str = 'medium_term') -> Dict:
        """
        Assess impact of a specific policy.
        
        Args:
            policy_type: Type of policy
            direction: Direction of policy (e.g., 'tightening', 'stimulus')
            implementation_timeline: 'immediate', 'medium_term', 'long_term'
        
        Returns:
            Dictionary with policy impact assessment
        """
        if policy_type not in self.policy_types:
            return {'error': f'Unknown policy type: {policy_type}'}
        
        policy_info = self.policy_types[policy_type]
        
        # Get sensitivity for direction
        sensitivities = policy_info['asset_sensitivity'].get(direction, {})
        
        # Adjust for timeline
        timeline_multipliers = {
            'immediate': 1.5,
            'medium_term': 1.0,
            'long_term': 0.6
        }
        multiplier = timeline_multipliers.get(implementation_timeline, 1.0)
        
        # Adjust impacts
        adjusted_impacts = {
            asset: impact * multiplier 
            for asset, impact in sensitivities.items()
        }
        
        return {
            'policy_type': policy_type,
            'direction': direction,
            'timeline': implementation_timeline,
            'description': policy_info['description'],
            'affected_sectors': policy_info['sectors_affected'],
            'estimated_impacts': adjusted_impacts,
            'magnitude': 'High' if abs(max(adjusted_impacts.values(), default=0)) > 0.03 else 'Moderate' if abs(max(adjusted_impacts.values(), default=0)) > 0.01 else 'Low'
        }
    
    def portfolio_policy_sensitivity(self,
                                    portfolio: Dict) -> Dict:
        """
        Assess portfolio sensitivity to various policies.
        
        Args:
            portfolio: Dictionary with sector allocations
        
        Returns:
            Dictionary with sensitivity metrics
        """
        sensitivities = {}
        
        for policy_type in self.policy_types.keys():
            # Test both directions
            impacts = {}
            
            for direction in ['tightening', 'easing'] if 'tightening' in self.policy_types[policy_type]['asset_sensitivity'] else ['stimulus', 'austerity'] if 'stimulus' in self.policy_types[policy_type]['asset_sensitivity'] else ['protectionist', 'free_trade']:
                
                assessment = self.assess_policy_impact(policy_type, direction)
                
                # Calculate portfolio impact (simplified)
                portfolio_impact = 0
                for asset, impact in assessment.get('estimated_impacts', {}).items():
                    if asset in portfolio:
                        portfolio_impact += impact * portfolio[asset]
                
                impacts[direction] = portfolio_impact
            
            sensitivities[policy_type] = impacts
        
        return {
            'portfolio_policy_sensitivities': sensitivities,
            'most_sensitive_policy': max(sensitivities.items(), 
                                        key=lambda x: max(abs(v) for v in x[1].values()))[0]
        }
