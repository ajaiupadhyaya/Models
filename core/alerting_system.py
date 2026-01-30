"""
Comprehensive Alerting System
Monitors trading signals, risk thresholds, and anomalies
"""

import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert message."""
    id: str
    type: str
    severity: AlertSeverity
    title: str
    message: str
    symbol: Optional[str]
    data: Dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            **asdict(self),
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat()
        }


class AlertingSystem:
    """
    Comprehensive alerting system.
    Monitors various conditions and sends alerts.
    """
    
    def __init__(self):
        """Initialize alerting system."""
        self.alerts: List[Alert] = []
        self.subscribers: List[Callable[[Alert], None]] = []
        self.alert_rules: Dict[str, Dict] = {}
        self.alert_count = 0
        
        # Default thresholds
        self.default_thresholds = {
            'max_drawdown': 0.10,  # 10%
            'position_size': 0.20,  # 20% of capital
            'daily_loss': 0.05,  # 5% daily loss
            'volatility_spike': 2.0,  # 2x average volatility
            'signal_confidence_min': 0.7,  # Minimum confidence for alerts
            'model_performance_drop': 0.15  # 15% performance drop
        }
    
    def subscribe(self, callback: Callable[[Alert], None]):
        """
        Subscribe to alerts.
        
        Args:
            callback: Function to call when alert is triggered
        """
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from alerts."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self, alert: Alert):
        """Notify all subscribers."""
        for callback in self.subscribers:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        symbol: Optional[str] = None,
        data: Optional[Dict] = None
    ) -> Alert:
        """
        Create and trigger an alert.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            symbol: Related symbol
            data: Additional data
        
        Returns:
            Created alert
        """
        self.alert_count += 1
        alert = Alert(
            id=f"alert_{self.alert_count}_{datetime.now().timestamp()}",
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            symbol=symbol,
            data=data or {},
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Notify subscribers
        self._notify_subscribers(alert)
        
        logger.info(f"Alert created: {alert_type} - {title}")
        
        return alert
    
    def check_risk_thresholds(
        self,
        portfolio_value: float,
        daily_pnl: float,
        max_drawdown: float,
        positions: Dict[str, float]
    ):
        """
        Check risk thresholds and create alerts if exceeded.
        
        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily profit/loss
            max_drawdown: Maximum drawdown
            positions: Current positions
        """
        # Check daily loss
        if daily_pnl < -self.default_thresholds['daily_loss'] * portfolio_value:
            self.create_alert(
                alert_type="risk",
                severity=AlertSeverity.CRITICAL,
                title="Daily Loss Threshold Exceeded",
                message=f"Daily loss of ${abs(daily_pnl):,.2f} exceeds threshold",
                data={"daily_pnl": daily_pnl, "threshold": self.default_thresholds['daily_loss']}
            )
        
        # Check max drawdown
        if abs(max_drawdown) > self.default_thresholds['max_drawdown']:
            self.create_alert(
                alert_type="risk",
                severity=AlertSeverity.WARNING,
                title="Maximum Drawdown Alert",
                message=f"Max drawdown of {abs(max_drawdown)*100:.2f}% exceeds threshold",
                data={"max_drawdown": max_drawdown}
            )
        
        # Check position sizes
        for symbol, position_value in positions.items():
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            if position_pct > self.default_thresholds['position_size']:
                self.create_alert(
                    alert_type="risk",
                    severity=AlertSeverity.WARNING,
                    title="Large Position Alert",
                    message=f"Position in {symbol} is {position_pct*100:.1f}% of portfolio",
                    symbol=symbol,
                    data={"position_value": position_value, "position_pct": position_pct}
                )
    
    def check_trading_signal(
        self,
        symbol: str,
        action: str,
        confidence: float,
        price: float,
        model_source: str
    ):
        """
        Check trading signal and create alert if significant.
        
        Args:
            symbol: Stock symbol
            action: Trading action (BUY/SELL)
            confidence: Signal confidence
            price: Current price
            model_source: Model that generated signal
        """
        if confidence >= self.default_thresholds['signal_confidence_min']:
            severity = AlertSeverity.CRITICAL if confidence > 0.9 else AlertSeverity.WARNING
            
            self.create_alert(
                alert_type="trading_signal",
                severity=severity,
                title=f"High Confidence {action} Signal",
                message=f"{model_source.upper()} model suggests {action} {symbol} with {confidence*100:.0f}% confidence at ${price:.2f}",
                symbol=symbol,
                data={
                    "action": action,
                    "confidence": confidence,
                    "price": price,
                    "model_source": model_source
                }
            )
    
    def check_model_performance(
        self,
        model_name: str,
        symbol: str,
        current_performance: float,
        baseline_performance: float
    ):
        """
        Check model performance and alert if degraded.
        
        Args:
            model_name: Model name
            symbol: Stock symbol
            current_performance: Current performance metric
            baseline_performance: Baseline performance metric
        """
        performance_drop = (baseline_performance - current_performance) / baseline_performance
        
        if performance_drop > self.default_thresholds['model_performance_drop']:
            self.create_alert(
                alert_type="model_performance",
                severity=AlertSeverity.WARNING,
                title="Model Performance Degradation",
                message=f"{model_name} for {symbol} performance dropped by {performance_drop*100:.1f}%",
                symbol=symbol,
                data={
                    "model_name": model_name,
                    "current_performance": current_performance,
                    "baseline_performance": baseline_performance,
                    "performance_drop": performance_drop
                }
            )
    
    def check_anomaly(
        self,
        symbol: str,
        anomaly_type: str,
        description: str,
        data: Optional[Dict] = None
    ):
        """
        Create anomaly alert.
        
        Args:
            symbol: Stock symbol
            anomaly_type: Type of anomaly
            description: Description
            data: Additional data
        """
        self.create_alert(
            alert_type="anomaly",
            severity=AlertSeverity.WARNING,
            title=f"Anomaly Detected: {anomaly_type}",
            message=f"{symbol}: {description}",
            symbol=symbol,
            data=data or {}
        )
    
    def get_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
        unacknowledged_only: bool = False
    ) -> List[Alert]:
        """
        Get alerts matching criteria.
        
        Args:
            alert_type: Filter by type
            severity: Filter by severity
            symbol: Filter by symbol
            limit: Maximum number of alerts
            unacknowledged_only: Only unacknowledged alerts
        
        Returns:
            List of alerts
        """
        alerts = self.alerts
        
        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        
        # Sort by timestamp (newest first)
        alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)
        
        return alerts[:limit]
    
    def acknowledge_alert(self, alert_id: str):
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return
        
        logger.warning(f"Alert not found: {alert_id}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get alerting system summary."""
        total_alerts = len(self.alerts)
        unacknowledged = len([a for a in self.alerts if not a.acknowledged])
        
        by_severity = {
            'info': len([a for a in self.alerts if a.severity == AlertSeverity.INFO]),
            'warning': len([a for a in self.alerts if a.severity == AlertSeverity.WARNING]),
            'critical': len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL])
        }
        
        by_type = {}
        for alert in self.alerts:
            by_type[alert.type] = by_type.get(alert.type, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "unacknowledged": unacknowledged,
            "by_severity": by_severity,
            "by_type": by_type,
            "subscribers": len(self.subscribers)
        }
