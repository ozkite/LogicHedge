"""
Web dashboard for monitoring the orchestrator
"""

from flask import Flask, render_template, jsonify
import asyncio
from datetime import datetime
import json

app = Flask(__name__)

class Dashboard:
    """Real-time dashboard for orchestrator monitoring"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        
    @app.route('/')
    def index(self):
        """Main dashboard"""
        return render_template('dashboard.html')
        
    @app.route('/api/performance')
    async def get_performance(self):
        """Get performance metrics"""
        metrics = await self.orchestrator.get_performance_metrics()
        return jsonify(metrics)
        
    @app.route('/api/services')
    async def get_services(self):
        """Get service status"""
        services = []
        for name, metrics in self.orchestrator.service_metrics.items():
            services.append({
                'name': name,
                'status': metrics.status.value,
                'latency': metrics.latency_ms,
                'success_rate': metrics.success_rate,
                'total_pnl': metrics.total_pnl
            })
        return jsonify(services)
        
    @app.route('/api/opportunities')
    async def get_opportunities(self):
        """Get current opportunities"""
        # This would scan opportunities in real-time
        return jsonify([])
        
    @app.route('/api/risk')
    async def get_risk(self):
        """Get risk metrics"""
        return jsonify(self.orchestrator.risk_exposures)
        
    @app.route('/api/capital')
    async def get_capital(self):
        """Get capital allocation"""
        capital = {}
        for name, alloc in self.orchestrator.capital_allocations.items():
            capital[name] = {
                'allocated': alloc.allocated_usd,
                'utilized': alloc.utilized_usd,
                'available': alloc.available_usd
            }
        return jsonify(capital)
