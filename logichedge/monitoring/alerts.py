"""
Monitoring and alerting system for orchestrator
"""

import asyncio
from typing import Dict, List
import smtplib
from email.mime.text import MIMEText
import requests

class AlertSystem:
    """Real-time alerting system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alerts_sent = set()
        
    async def monitor_and_alert(self, orchestrator):
        """Monitor orchestrator and send alerts"""
        while True:
            try:
                # Check service health
                for service_name, metrics in orchestrator.service_metrics.items():
                    if metrics.status.value == 'error':
                        await self._send_alert(
                            f"Service {service_name} is OFFLINE",
                            f"Service {service_name} has been offline since {metrics.last_update}"
                        )
                        
                # Check risk limits
                if orchestrator.risk_exposures.get('var_95', 0) > 10000:
                    await self._send_alert(
                        "Risk limit exceeded",
                        f"VaR 95%: ${orchestrator.risk_exposures['var_95']:,.2f}"
                    )
                    
                # Check P&L
                perf = await orchestrator.get_performance_metrics()
                if perf.get('total_pnl', 0) < -5000:
                    await self._send_alert(
                        "Large loss detected",
                        f"Total P&L: ${perf['total_pnl']:,.2f}"
                    )
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Alert system error: {e}")
                await asyncio.sleep(300)
                
    async def _send_alert(self, title: str, message: str):
        """Send alert via multiple channels"""
        alert_id = f"{title}:{message}"
        
        if alert_id in self.alerts_sent:
            return  # Don't send duplicate alerts
            
        # Email
        if self.config.get('email_alerts'):
            await self._send_email(title, message)
            
        # Telegram
        if self.config.get('telegram_alerts'):
            await self._send_telegram(title, message)
            
        # Slack
        if self.config.get('slack_alerts'):
            await self._send_slack(title, message)
            
        self.alerts_sent.add(alert_id)
        
    async def _send_email(self, subject: str, body: str):
        """Send email alert"""
        msg = MIMEText(body)
        msg['Subject'] = f"[LogicHedge Alert] {subject}"
        msg['From'] = self.config['email_from']
        msg['To'] = self.config['email_to']
        
        with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
            server.starttls()
            server.login(self.config['email_user'], self.config['email_password'])
            server.send_message(msg)
            
    async def _send_telegram(self, title: str, message: str):
        """Send Telegram alert"""
        bot_token = self.config['telegram_bot_token']
        chat_id = self.config['telegram_chat_id']
        
        text = f"*{title}*\n{message}"
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        requests.post(url, json={
            'chat_id': chat_id,
            'text': text,
            'parse_mode': 'Markdown'
        })
        
    async def _send_slack(self, title: str, message: str):
        """Send Slack alert"""
        webhook_url = self.config['slack_webhook']
        
        requests.post(webhook_url, json={
            'text': f":warning: *{title}*\n{message}"
        })
