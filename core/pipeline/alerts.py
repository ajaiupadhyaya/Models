"""
Alert System
Generate alerts based on market conditions and data rules
"""

import pandas as pd
import numpy as np
from typing import Callable, List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """Alert condition types."""
    # Price alerts
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_CHANGE_PCT = "price_change_pct"
    
    # Technical alerts
    MOVING_AVERAGE_CROSS = "moving_average_cross"
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"
    
    # Fundamental alerts
    PE_ABOVE = "pe_above"
    PE_BELOW = "pe_below"
    DIVIDEND_YIELD = "dividend_yield"
    EARNINGS_MISS = "earnings_miss"
    
    # Market alerts
    VIX_SPIKE = "vix_spike"
    YIELD_CURVE_INVERSION = "yield_curve_inversion"
    VOLATILITY_THRESHOLD = "volatility_threshold"
    
    # Custom alerts
    CUSTOM_FUNCTION = "custom_function"


class Alert:
    """
    Represents an alert notification.
    """
    
    def __init__(self,
                 alert_id: str,
                 message: str,
                 severity: AlertSeverity,
                 asset: Optional[str] = None,
                 condition: Optional[AlertCondition] = None,
                 triggered_at: Optional[datetime] = None):
        """
        Initialize alert.
        
        Args:
            alert_id: Unique alert identifier
            message: Alert message
            severity: Alert severity level
            asset: Asset related to alert
            condition: Condition that triggered alert
            triggered_at: When alert was triggered
        """
        self.alert_id = alert_id
        self.message = message
        self.severity = severity
        self.asset = asset
        self.condition = condition
        self.triggered_at = triggered_at or datetime.now()
        self.acknowledged = False
        self.acknowledged_at = None
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'message': self.message,
            'severity': self.severity.value,
            'asset': self.asset,
            'condition': self.condition.value if self.condition else None,
            'triggered_at': self.triggered_at,
            'acknowledged': self.acknowledged
        }
    
    def acknowledge(self):
        """Mark alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_at = datetime.now()


class AlertRule:
    """
    Represents an alert rule that monitors a condition.
    """
    
    def __init__(self,
                 rule_id: str,
                 name: str,
                 asset: str,
                 condition: AlertCondition,
                 threshold: float,
                 severity: AlertSeverity = AlertSeverity.WARNING,
                 check_function: Optional[Callable] = None):
        """
        Initialize alert rule.
        
        Args:
            rule_id: Unique rule identifier
            name: Human-readable rule name
            asset: Asset to monitor
            condition: Condition type
            threshold: Threshold value
            severity: Alert severity
            check_function: Custom check function
        """
        self.rule_id = rule_id
        self.name = name
        self.asset = asset
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.check_function = check_function
        
        self.is_active = True
        self.triggered_count = 0
        self.last_triggered = None
    
    def evaluate(self, current_value: float) -> Optional[Alert]:
        """
        Evaluate if alert should be triggered.
        
        Args:
            current_value: Current value to check
        
        Returns:
            Alert if triggered, None otherwise
        """
        if not self.is_active:
            return None
        
        # Built-in condition checks
        triggered = False
        
        if self.condition == AlertCondition.PRICE_ABOVE:
            triggered = current_value > self.threshold
        
        elif self.condition == AlertCondition.PRICE_BELOW:
            triggered = current_value < self.threshold
        
        elif self.condition == AlertCondition.PRICE_CHANGE_PCT:
            # Assume threshold is percentage change
            triggered = abs(current_value) > self.threshold
        
        elif self.condition == AlertCondition.CUSTOM_FUNCTION:
            if self.check_function:
                triggered = self.check_function(current_value, self.threshold)
        
        if triggered:
            self.triggered_count += 1
            self.last_triggered = datetime.now()
            
            message = f"Alert: {self.name} - {self.asset} triggered (value: {current_value:.2f}, threshold: {self.threshold:.2f})"
            
            return Alert(
                alert_id=f"{self.rule_id}_{self.triggered_count}",
                message=message,
                severity=self.severity,
                asset=self.asset,
                condition=self.condition
            )
        
        return None


class AlertSystem:
    """
    Manages alert rules and notifications.
    """
    
    def __init__(self, email_config: Optional[Dict] = None):
        """
        Initialize alert system.
        
        Args:
            email_config: Optional email configuration for notifications
        """
        self.rules = {}
        self.alerts = []
        self.email_config = email_config
        
        self.logger = logging.getLogger("AlertSystem")
    
    def add_rule(self, rule: AlertRule) -> bool:
        """
        Add an alert rule.
        
        Args:
            rule: AlertRule instance
        
        Returns:
            True if added successfully
        """
        try:
            self.rules[rule.rule_id] = rule
            self.logger.info(f"Added alert rule: {rule.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add rule: {str(e)}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_id: ID of rule to remove
        
        Returns:
            True if removed
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False
    
    def evaluate_all(self, data: Dict[str, float]) -> List[Alert]:
        """
        Evaluate all rules against current data.
        
        Args:
            data: Dictionary of {asset: value}
        
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule in self.rules.values():
            if rule.asset in data:
                alert = rule.evaluate(data[rule.asset])
                if alert:
                    triggered_alerts.append(alert)
                    self.alerts.append(alert)
        
        if triggered_alerts:
            self.logger.info(f"Triggered {len(triggered_alerts)} alerts")
        
        return triggered_alerts
    
    def create_price_alert(self,
                          asset: str,
                          alert_type: str,
                          threshold: float,
                          severity: AlertSeverity = AlertSeverity.WARNING) -> AlertRule:
        """
        Create a price alert.
        
        Args:
            asset: Asset to monitor
            alert_type: 'above' or 'below'
            threshold: Price threshold
            severity: Alert severity
        
        Returns:
            AlertRule instance
        """
        rule_id = f"price_{asset}_{alert_type}_{threshold}"
        condition = AlertCondition.PRICE_ABOVE if alert_type == 'above' else AlertCondition.PRICE_BELOW
        
        rule = AlertRule(
            rule_id=rule_id,
            name=f"{asset} price {alert_type} {threshold}",
            asset=asset,
            condition=condition,
            threshold=threshold,
            severity=severity
        )
        
        self.add_rule(rule)
        return rule
    
    def create_technical_alert(self,
                              asset: str,
                              technical_signal: str,
                              severity: AlertSeverity = AlertSeverity.WARNING) -> AlertRule:
        """
        Create a technical analysis alert.
        
        Args:
            asset: Asset to monitor
            technical_signal: 'rsi_overbought', 'rsi_oversold', etc.
            severity: Alert severity
        
        Returns:
            AlertRule instance
        """
        signal_map = {
            'rsi_overbought': (AlertCondition.RSI_OVERBOUGHT, 70),
            'rsi_oversold': (AlertCondition.RSI_OVERSOLD, 30),
            'vix_spike': (AlertCondition.VIX_SPIKE, 25)
        }
        
        if technical_signal not in signal_map:
            return None
        
        condition, threshold = signal_map[technical_signal]
        
        rule_id = f"technical_{asset}_{technical_signal}"
        rule = AlertRule(
            rule_id=rule_id,
            name=f"{asset} {technical_signal}",
            asset=asset,
            condition=condition,
            threshold=threshold,
            severity=severity
        )
        
        self.add_rule(rule)
        return rule
    
    def send_alert(self, alert: Alert, method: str = 'log'):
        """
        Send alert notification.
        
        Args:
            alert: Alert to send
            method: 'log', 'email', or 'sms'
        """
        if method == 'log':
            level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }
            self.logger.log(level.get(alert.severity, logging.INFO), alert.message)
        
        elif method == 'email' and self.email_config:
            self._send_email_alert(alert)
        
        elif method == 'sms' and self.email_config:
            self._send_sms_alert(alert)
    
    def _send_email_alert(self, alert: Alert):
        """Send alert via email."""
        try:
            if not self.email_config:
                return
            
            sender = self.email_config.get('sender')
            password = self.email_config.get('password')
            recipients = self.email_config.get('recipients', [])
            smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.email_config.get('smtp_port', 587)
            
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.message[:50]}"
            
            body = f"""
Alert ID: {alert.alert_id}
Severity: {alert.severity.value}
Asset: {alert.asset}
Message: {alert.message}
Time: {alert.triggered_at}
"""
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
    
    def _send_sms_alert(self, alert: Alert):
        """Send alert via SMS (requires Twilio or similar)."""
        # This would integrate with Twilio API
        self.logger.warning("SMS alerts not yet implemented")
    
    def get_active_alerts(self) -> List[Dict]:
        """
        Get all active (unacknowledged) alerts.
        
        Returns:
            List of active alert dictionaries
        """
        return [a.to_dict() for a in self.alerts if not a.acknowledged]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """
        Get alert history.
        
        Args:
            hours: Number of hours to look back
        
        Returns:
            List of alerts from specified period
        """
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        return [a.to_dict() for a in self.alerts if a.triggered_at > cutoff]
    
    def acknowledge_alert(self, alert_id: str):
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of alert to acknowledge
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledge()
                break
    
    def clear_alerts(self, severity: Optional[AlertSeverity] = None):
        """
        Clear alerts.
        
        Args:
            severity: Optional severity to filter by
        """
        if severity:
            self.alerts = [a for a in self.alerts if a.severity != severity]
        else:
            self.alerts = []
