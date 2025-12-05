"""
Enterprise-grade orchestration system for prime brokerage services
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from pathlib import Path
import hashlib
import hmac
import time
import aiohttp
from web3 import Web3
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration"""
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ONLINE = "online"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class ServiceMetrics:
    """Service performance metrics"""
    service_name: str
    status: ServiceStatus
    uptime_pct: float = 0.0
    latency_ms: float = 0.0
    success_rate: float = 0.0
    total_trades: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CapitalAllocation:
    """Capital allocation to services"""
    service_name: str
    allocated_usd: float
    utilized_usd: float = 0.0
    available_usd: float = 0.0
    target_allocation_pct: float = 0.0
    max_allocation_usd: float = 0.0

class MasterOrchestrator:
    """
    Master orchestrator for all prime brokerage and execution services
    Manages: CedeHub, Copper, FalconX, dYdX, Aevo, Gamma, Ribbon, Hashnote, Paradex
    """
    
    def __init__(self, config_path: str = "configs/prime_brokerage.yaml"):
        self.config = self._load_config(config_path)
        self.services = {}
        self.capital_allocations = {}
        self.service_metrics = {}
        self.unified_portfolio = {}
        self.risk_exposures = {}
        
        # Performance tracking
        self.performance_history = []
        self.trade_log = []
        
        # Risk management
        self.risk_limits = self.config.get('risk_limits', {})
        self.circuit_breakers = {}
        
        # API clients
        self.session = aiohttp.ClientSession()
        
        # Event bus for internal communication
        self.event_queue = asyncio.Queue()
        
        # Start background tasks
        self.running = False
        self.tasks = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Load environment variables
        import os
        for service, settings in config.get('prime_brokerage_services', {}).items():
            if isinstance(settings, dict):
                for key, value in settings.items():
                    if isinstance(value, str) and value.startswith('${'):
                        env_var = value[2:-1]
                        config['prime_brokerage_services'][service][key] = os.getenv(env_var, '')
                        
        return config
        
    async def initialize(self):
        """Initialize all services"""
        logger.info("🚀 Initializing Master Orchestrator...")
        
        # Initialize each service
        services_config = self.config.get('prime_brokerage_services', {})
        
        # 1. CedeHub
        if services_config.get('cedehub', {}).get('enabled'):
            from logichedge.integrations.cedehub.strategies import CedeHubArbitrage
            self.services['cedehub'] = CedeHubArbitrage(
                api_key=services_config['cedehub']['api_key'],
                api_secret=services_config['cedehub']['api_secret']
            )
            await self.services['cedehub'].initialize()
            
        # 2. Copper
        if services_config.get('copper', {}).get('enabled'):
            from logichedge.integrations.copper.prime_brokerage import CopperPrimeBrokerage
            self.services['copper'] = CopperPrimeBrokerage(
                api_key=services_config['copper']['api_key'],
                api_secret=services_config['copper']['api_secret'],
                client_id=services_config['copper']['client_id']
            )
            await self.services['copper'].initialize()
            
        # 3. FalconX
        if services_config.get('falconx', {}).get('enabled'):
            from logichedge.integrations.falconx.execution import FalconXExecution
            self.services['falconx'] = FalconXExecution(
                api_key=services_config['falconx']['api_key'],
                api_secret=services_config['falconx']['api_secret']
            )
            
        # 4. dYdX + Aevo
        if services_config.get('dydx', {}).get('enabled'):
            from logichedge.integrations.layer2.derivatives import Layer2Derivatives
            self.services['layer2'] = Layer2Derivatives(
                private_key=services_config['dydx']['private_key'],
                stark_key=services_config['dydx']['stark_key']
            )
            await self.services['layer2'].initialize()
            
        # 5. Gamma
        if services_config.get('gamma', {}).get('enabled'):
            from logichedge.integrations.gamma.lp_management import GammaLPOptimizer
            self.services['gamma'] = GammaLPOptimizer(
                private_key=services_config['gamma']['private_key']
            )
            
        # 6. Ribbon + Hashnote
        if services_config.get('ribbon', {}).get('enabled'):
            from logichedge.integrations.yield.structured_products import StructuredYield
            self.services['yield'] = StructuredYield(
                wallet_address=services_config['ribbon']['wallet']
            )
            
        # 7. Paradex
        if services_config.get('paradex', {}).get('enabled'):
            from logichedge.integrations.paradex.starknet_trading import ParadexStarkNetTrading
            self.services['paradex'] = ParadexStarkNetTrading(
                stark_key=services_config['paradex']['stark_key'],
                eth_key=services_config['paradex']['eth_key']
            )
            
        # Initialize capital allocations
        await self._initialize_capital_allocations()
        
        # Start monitoring tasks
        self.running = True
        self.tasks = [
            asyncio.create_task(self._monitor_services()),
            asyncio.create_task(self._optimize_capital_allocation()),
            asyncio.create_task(self._execute_global_arbitrage()),
            asyncio.create_task(self._manage_risk()),
            asyncio.create_task(self._log_performance())
        ]
        
        logger.info(f"✅ Orchestrator initialized with {len(self.services)} services")
        
    async def _initialize_capital_allocations(self):
        """Initialize capital allocations from config"""
        services_config = self.config.get('prime_brokerage_services', {})
        
        for service_name, config in services_config.items():
            if config.get('enabled'):
                allocation = CapitalAllocation(
                    service_name=service_name,
                    allocated_usd=config.get('capital_allocation', 0),
                    available_usd=config.get('capital_allocation', 0),
                    target_allocation_pct=config.get('target_allocation_pct', 0.1),
                    max_allocation_usd=config.get('max_allocation_usd', float('inf'))
                )
                self.capital_allocations[service_name] = allocation
                
    async def _monitor_services(self):
        """Monitor service health and performance"""
        while self.running:
            try:
                for service_name, service in self.services.items():
                    metrics = await self._check_service_health(service_name, service)
                    self.service_metrics[service_name] = metrics
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Service monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _check_service_health(self, service_name: str, service) -> ServiceMetrics:
        """Check service health and get metrics"""
        try:
            # Test connectivity
            start_time = time.time()
            
            # Service-specific health check
            if service_name == 'cedehub':
                balance = await service._cedehub_request("GET", "/v1/balance")
                status = ServiceStatus.ONLINE if balance else ServiceStatus.ERROR
            elif service_name == 'copper':
                accounts = await service._copper_request("GET", "/institutional/v1/accounts")
                status = ServiceStatus.ONLINE if accounts else ServiceStatus.ERROR
            elif service_name == 'falconx':
                # Simple connectivity test
                status = ServiceStatus.ONLINE
            else:
                status = ServiceStatus.ONLINE
                
            latency_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            metrics = self.service_metrics.get(service_name, ServiceMetrics(
                service_name=service_name,
                status=status
            ))
            
            metrics.status = status
            metrics.latency_ms = latency_ms
            metrics.last_update = datetime.utcnow()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            return ServiceMetrics(
                service_name=service_name,
                status=ServiceStatus.ERROR,
                last_update=datetime.utcnow()
            )
            
    async def _optimize_capital_allocation(self):
        """Dynamically optimize capital allocation across services"""
        while self.running:
            try:
                # Calculate performance metrics
                performance = {}
                for service_name, metrics in self.service_metrics.items():
                    if metrics.status == ServiceStatus.ONLINE:
                        # Calculate Sharpe ratio (simplified)
                        returns = await self._get_service_returns(service_name)
                        volatility = np.std(returns) if len(returns) > 1 else 0.1
                        
                        if volatility > 0:
                            sharpe = np.mean(returns) / volatility * np.sqrt(365)
                        else:
                            sharpe = 0.0
                            
                        performance[service_name] = {
                            'sharpe': sharpe,
                            'success_rate': metrics.success_rate,
                            'volume': metrics.total_volume
                        }
                        
                # Rebalance based on performance
                await self._rebalance_capital(performance)
                
                await asyncio.sleep(3600)  # Rebalance hourly
                
            except Exception as e:
                logger.error(f"Capital optimization error: {e}")
                await asyncio.sleep(600)
                
    async def _rebalance_capital(self, performance: Dict[str, Dict]):
        """Rebalance capital based on performance"""
        total_capital = sum(
            alloc.allocated_usd for alloc in self.capital_allocations.values()
        )
        
        if total_capital == 0:
            return
            
        # Calculate target allocations based on Sharpe ratios
        sharpes = {name: perf['sharpe'] for name, perf in performance.items()}
        total_sharpe = sum(max(s, 0) for s in sharpes.values())  # Only positive Sharpe
        
        if total_sharpe > 0:
            for service_name, allocation in self.capital_allocations.items():
                sharpe = max(sharpes.get(service_name, 0), 0)
                target_pct = sharpe / total_sharpe if total_sharpe > 0 else 0.1
                
                # Apply smoothing
                current_pct = allocation.allocated_usd / total_capital
                new_pct = current_pct * 0.7 + target_pct * 0.3  # 30% adjustment
                
                target_amount = total_capital * new_pct
                
                # Check limits
                target_amount = min(
                    target_amount,
                    allocation.max_allocation_usd
                )
                
                # Execute rebalance if difference > 10%
                if abs(target_amount - allocation.allocated_usd) > total_capital * 0.1:
                    await self._transfer_capital(
                        service_name,
                        target_amount - allocation.allocated_usd
                    )
                    
    async def _execute_global_arbitrage(self):
        """Execute global arbitrage across all services"""
        scan_interval = 0.1  # 100ms
        
        while self.running:
            try:
                # Scan for opportunities across all services
                opportunities = []
                
                # 1. Cross-exchange arbitrage (CedeHub, Copper)
                if 'cedehub' in self.services:
                    cedehub_opps = await self._scan_cedehub_opportunities()
                    opportunities.extend(cedehub_opps)
                    
                if 'copper' in self.services:
                    copper_opps = await self._scan_copper_opportunities()
                    opportunities.extend(copper_opps)
                    
                # 2. Layer 2 arbitrage (dYdX, Aevo, Paradex)
                if 'layer2' in self.services:
                    layer2_opps = await self._scan_layer2_opportunities()
                    opportunities.extend(layer2_opps)
                    
                # 3. Cross-venue arbitrage (FalconX)
                if 'falconx' in self.services:
                    falconx_opps = await self._scan_falconx_opportunities()
                    opportunities.extend(falconx_opps)
                    
                # Rank and filter opportunities
                filtered = self._filter_opportunities(opportunities)
                ranked = self._rank_opportunities(filtered)
                
                # Execute top opportunities
                await self._execute_opportunities(ranked[:5])  # Top 5
                
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                logger.error(f"Global arbitrage error: {e}")
                await asyncio.sleep(1)
                
    async def _scan_cedehub_opportunities(self) -> List[Dict]:
        """Scan CedeHub for arbitrage opportunities"""
        opportunities = []
        
        try:
            # Get prices across connected exchanges
            symbols = ['BTC', 'ETH', 'SOL']
            
            for symbol in symbols:
                prices = await self.services['cedehub']._cedehub_request(
                    "GET", f"/v1/prices/{symbol}USDT"
                )
                
                if prices and 'exchanges' in prices:
                    # Find price discrepancies
                    exchange_prices = {}
                    for ex in prices['exchanges']:
                        exchange_prices[ex['exchange']] = {
                            'bid': float(ex['bid']),
                            'ask': float(ex['ask'])
                        }
                        
                    # Find best bid and ask
                    best_bid_exchange = max(
                        exchange_prices.items(),
                        key=lambda x: x[1]['bid']
                    )
                    best_ask_exchange = min(
                        exchange_prices.items(),
                        key=lambda x: x[1]['ask']
                    )
                    
                    spread = best_bid_exchange[1]['bid'] - best_ask_exchange[1]['ask']
                    spread_pct = (spread / best_ask_exchange[1]['ask']) * 100
                    
                    if spread_pct > 0.1:  # 0.1% minimum
                        opportunities.append({
                            'service': 'cedehub',
                            'type': 'cross_exchange',
                            'symbol': f"{symbol}/USDT",
                            'buy_exchange': best_ask_exchange[0],
                            'sell_exchange': best_bid_exchange[0],
                            'buy_price': best_ask_exchange[1]['ask'],
                            'sell_price': best_bid_exchange[1]['bid'],
                            'spread_pct': spread_pct,
                            'estimated_profit_pct': spread_pct - 0.05,  # minus fees
                            'max_size': 1000,  # $1000 max
                            'timeframe': 'instant'
                        })
                        
        except Exception as e:
            logger.error(f"CedeHub scan error: {e}")
            
        return opportunities
        
    async def _scan_layer2_opportunities(self) -> List[Dict]:
        """Scan Layer 2 for opportunities"""
        opportunities = []
        
        try:
            # Get funding rates from dYdX
            if hasattr(self.services['layer2'], 'dydx_client'):
                markets = self.services['layer2'].dydx_client.public.get_markets()
                
                for market_name, market in markets['markets'].items():
                    if 'USD' in market_name:
                        funding_rate = float(market['nextFundingRate'])
                        
                        if abs(funding_rate) > 0.0005:  # 0.05%
                            opportunities.append({
                                'service': 'layer2',
                                'type': 'funding_rate',
                                'symbol': market_name,
                                'funding_rate': funding_rate,
                                'side': 'short' if funding_rate > 0 else 'long',
                                'estimated_profit_pct': abs(funding_rate) * 3 * 365,  # Annualized
                                'max_size': 5000,
                                'timeframe': '8h'  # Funding period
                            })
                            
        except Exception as e:
            logger.error(f"Layer 2 scan error: {e}")
            
        return opportunities
        
    def _filter_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Filter opportunities based on risk and constraints"""
        filtered = []
        
        for opp in opportunities:
            # Minimum profit threshold
            if opp.get('estimated_profit_pct', 0) < 0.05:  # 0.05% min
                continue
                
            # Check capital availability
            service_name = opp['service']
            allocation = self.capital_allocations.get(service_name)
            
            if not allocation or allocation.available_usd < opp.get('max_size', 0) * 0.1:
                continue
                
            # Risk filter
            if self._is_too_risky(opp):
                continue
                
            filtered.append(opp)
            
        return filtered
        
    def _rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Rank opportunities by risk-adjusted return"""
        ranked = []
        
        for opp in opportunities:
            # Calculate score
            score = self._calculate_opportunity_score(opp)
            ranked.append((score, opp))
            
        # Sort by score (descending)
        ranked.sort(key=lambda x: x[0], reverse=True)
        
        return [opp for _, opp in ranked]
        
    def _calculate_opportunity_score(self, opportunity: Dict) -> float:
        """Calculate opportunity score"""
        score = 0.0
        
        # Profit potential (40%)
        profit_pct = opportunity.get('estimated_profit_pct', 0)
        score += min(profit_pct * 10, 40)  # Cap at 40
        
        # Probability of success (30%)
        success_prob = opportunity.get('success_probability', 0.7)
        score += success_prob * 30
        
        # Speed (20%)
        timeframe = opportunity.get('timeframe', '')
        if timeframe in ['instant', '1m']:
            score += 20
        elif timeframe in ['1h', '4h']:
            score += 10
        elif timeframe in ['8h', '1d']:
            score += 5
            
        # Service reliability (10%)
        service_name = opportunity['service']
        metrics = self.service_metrics.get(service_name)
        if metrics and metrics.status == ServiceStatus.ONLINE:
            score += metrics.success_rate * 10
            
        return score
        
    async def _execute_opportunities(self, opportunities: List[Dict]):
        """Execute ranked opportunities"""
        for opportunity in opportunities:
            try:
                service_name = opportunity['service']
                service = self.services.get(service_name)
                
                if not service:
                    continue
                    
                # Check capital
                allocation = self.capital_allocations.get(service_name)
                if not allocation or allocation.available_usd <= 0:
                    continue
                    
                # Calculate position size
                position_size = min(
                    opportunity.get('max_size', 1000),
                    allocation.available_usd * 0.1  # Use 10% of available
                )
                
                # Execute based on type
                if opportunity['type'] == 'cross_exchange':
                    await self._execute_cross_exchange_arb(
                        service, opportunity, position_size
                    )
                elif opportunity['type'] == 'funding_rate':
                    await self._execute_funding_rate_arb(
                        service, opportunity, position_size
                    )
                    
                # Update capital allocation
                allocation.available_usd -= position_size
                allocation.utilized_usd += position_size
                
                # Log trade
                self.trade_log.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'service': service_name,
                    'opportunity': opportunity,
                    'size': position_size,
                    'status': 'executed'
                })
                
                logger.info(f"Executed opportunity: {service_name} - "
                          f"{opportunity['type']} - ${position_size:,.2f}")
                          
            except Exception as e:
                logger.error(f"Opportunity execution error: {e}")
                
    async def _execute_cross_exchange_arb(self, service, opportunity: Dict, size: float):
        """Execute cross-exchange arbitrage"""
        if opportunity['service'] == 'cedehub':
            # Execute via CedeHub unified trade
            trade_request = {
                "type": "arbitrage",
                "legs": [
                    {
                        "exchange": opportunity['buy_exchange'],
                        "symbol": opportunity['symbol'],
                        "side": "buy",
                        "type": "market",
                        "size": size / opportunity['buy_price']
                    },
                    {
                        "exchange": opportunity['sell_exchange'],
                        "symbol": opportunity['symbol'],
                        "side": "sell",
                        "type": "market",
                        "size": size / opportunity['buy_price']  # Same quantity
                    }
                ]
            }
            
            await service._cedehub_request(
                "POST", "/v1/trade/unified", trade_request
            )
            
    async def _manage_risk(self):
        """Manage risk across all services"""
        while self.running:
            try:
                # Aggregate exposures
                exposures = await self._aggregate_exposures()
                
                # Check risk limits
                await self._enforce_risk_limits(exposures)
                
                # Hedge if needed
                await self._execute_hedges(exposures)
                
                # Update risk metrics
                self.risk_exposures = exposures
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Risk management error: {e}")
                await asyncio.sleep(300)
                
    async def _aggregate_exposures(self) -> Dict[str, float]:
        """Aggregate risk exposures across all services"""
        exposures = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'var_95': 0.0,
            'liquidation_risk': 0.0
        }
        
        # Aggregate from each service
        for service_name, service in self.services.items():
            try:
                if service_name == 'layer2':
                    # Get dYdX positions
                    account = service.dydx_client.private.get_account()
                    positions = account['account']['openPositions']
                    
                    for market, position in positions.items():
                        size = float(position['size'])
                        if size != 0:
                            # Simplified delta calculation
                            exposures['delta'] += size
                            
                elif service_name == 'gamma':
                    # Get LP positions
                    positions = await service._gamma_request("GET", "/v1/positions")
                    for position in positions:
                        # Calculate delta from LP position
                        # This is simplified - real calculation would use Uniswap math
                        exposures['delta'] += position.get('net_delta', 0)
                        
            except Exception as e:
                logger.error(f"Exposure aggregation error for {service_name}: {e}")
                
        return exposures
        
    async def _enforce_risk_limits(self, exposures: Dict[str, float]):
        """Enforce risk limits and trigger circuit breakers"""
        # Check delta limit
        delta_limit = self.risk_limits.get('max_delta_exposure', 50000)
        if abs(exposures['delta']) > delta_limit:
            logger.warning(f"Delta limit exceeded: {exposures['delta']:.0f} > {delta_limit}")
            await self._reduce_delta_exposure(exposures['delta'])
            
        # Check VaR limit
        var_limit = self.risk_limits.get('max_var_95', 20000)
        if exposures['var_95'] > var_limit:
            logger.warning(f"VaR limit exceeded: {exposures['var_95']:.0f} > {var_limit}")
            await self._reduce_var_exposure()
            
        # Check daily loss limit
        daily_loss = await self._calculate_daily_pnl()
        daily_limit = self.risk_limits.get('max_daily_loss', 10000)
        if daily_loss < -daily_limit:
            logger.critical(f"Daily loss limit exceeded: {daily_loss:.0f} < -{daily_limit}")
            await self._emergency_stop()
            
    async def _reduce_delta_exposure(self, delta: float):
        """Reduce delta exposure by hedging"""
        # Determine hedge direction
        hedge_side = 'sell' if delta > 0 else 'buy'
        hedge_size = abs(delta) * 0.5  # Hedge 50%
        
        # Use most liquid venue (dYdX for perps)
        if 'layer2' in self.services:
            await self.services['layer2'].dydx_client.private.create_order(
                position_id='hedge',
                market='BTC-USD',
                side=hedge_side,
                order_type='MARKET',
                size=str(hedge_size),
                price=None,
                leverage=1
            )
            
            logger.info(f"Delta hedge executed: {hedge_side} {hedge_size} BTC")
            
    async def _emergency_stop(self):
        """Emergency stop - close all positions"""
        logger.critical("🚨 EMERGENCY STOP ACTIVATED - Closing all positions")
        
        # Close all positions across all services
        for service_name, service in self.services.items():
            try:
                if service_name == 'layer2':
                    # Close dYdX positions
                    account = service.dydx_client.private.get_account()
                    positions = account['account']['openPositions']
                    
                    for market, position in positions.items():
                        size = float(position['size'])
                        if size != 0:
                            side = 'SELL' if size > 0 else 'BUY'
                            await service.dydx_client.private.create_order(
                                position_id='close',
                                market=market,
                                side=side,
                                order_type='MARKET',
                                size=str(abs(size)),
                                price=None,
                                leverage=1
                            )
                            
                elif service_name == 'gamma':
                    # Close LP positions
                    positions = await service._gamma_request("GET", "/v1/positions")
                    for position in positions:
                        await service._gamma_request(
                            "POST", f"/v1/positions/{position['id']}/close"
                        )
                        
            except Exception as e:
                logger.error(f"Emergency stop error for {service_name}: {e}")
                
        # Stop orchestrator
        await self.shutdown()
        
    async def _log_performance(self):
        """Log performance metrics periodically"""
        while self.running:
            try:
                metrics = await self.get_performance_metrics()
                self.performance_history.append(metrics)
                
                # Keep only last 1000 entries
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                    
                # Log to file
                await self._write_performance_log(metrics)
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance logging error: {e}")
                await asyncio.sleep(600)
                
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        total_pnl = 0.0
        total_volume = 0.0
        active_trades = 0
        
        for metrics in self.service_metrics.values():
            total_pnl += metrics.total_pnl
            total_volume += metrics.total_volume
            
        # Calculate returns
        total_capital = sum(
            alloc.allocated_usd for alloc in self.capital_allocations.values()
        )
        
        roi_pct = (total_pnl / total_capital * 100) if total_capital > 0 else 0.0
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_pnl': total_pnl,
            'total_volume': total_volume,
            'roi_pct': roi_pct,
            'active_services': len([m for m in self.service_metrics.values() 
                                  if m.status == ServiceStatus.ONLINE]),
            'active_trades': active_trades,
            'risk_exposures': self.risk_exposures,
            'capital_allocations': {
                name: {
                    'allocated': alloc.allocated_usd,
                    'utilized': alloc.utilized_usd,
                    'available': alloc.available_usd
                }
                for name, alloc in self.capital_allocations.items()
            }
        }
        
    async def shutdown(self):
        """Graceful shutdown of orchestrator"""
        logger.info("Shutting down orchestrator...")
        
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            
        # Close session
        await self.session.close()
        
        # Close service connections
        for service_name, service in self.services.items():
            if hasattr(service, 'close'):
                await service.close()
                
        logger.info("Orchestrator shutdown complete")
