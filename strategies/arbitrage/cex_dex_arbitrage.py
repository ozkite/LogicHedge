"""
High-frequency CEX-DEX arbitrage engine.
Executes simultaneous orders across centralized and decentralized exchanges.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field
import uuid

from logichedge.core.strategy import BaseStrategy, TradeSignal
from logichedge.core.event_bus import Event

logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity between CEX and DEX"""
    id: str
    symbol: str
    cex_venue: str
    dex_venue: str
    cex_price: float
    dex_price: float
    price_difference: float
    price_difference_pct: float
    spread_bps: int  # Basis points
    estimated_profit: float
    estimated_profit_pct: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = "detected"  # detected, executing, completed, failed
    execution_id: Optional[str] = None
    
@dataclass
class ArbitrageExecution:
    """Arbitrage execution record"""
    id: str
    opportunity_id: str
    symbol: str
    side_cex: str  # "buy" or "sell" on CEX
    side_dex: str  # opposite side on DEX
    quantity: float
    cex_price: float
    dex_price: float
    cex_order_id: Optional[str] = None
    dex_order_id: Optional[str] = None
    cex_filled: bool = False
    dex_filled: bool = False
    profit_realized: float = 0.0
    fees_paid: float = 0.0
    net_profit: float = 0.0
    execution_time_ms: Optional[float] = None
    status: str = "pending"  # pending, executing, partial, complete, failed
    timestamp: datetime = field(default_factory=datetime.utcnow)

class CEXDEXArbitrage(BaseStrategy):
    """
    High-frequency CEX-DEX arbitrage strategy.
    Detects and executes arbitrage opportunities simultaneously.
    """
    
    def __init__(self, name: str, event_bus, config: Dict[str, Any]):
        super().__init__(name, event_bus, config)
        
        # Configuration
        self.min_profit_bps = config.get("min_profit_bps", 5)  # 0.05%
        self.min_profit_usd = config.get("min_profit_usd", 10.0)
        self.max_position_usd = config.get("max_position_usd", 5000.0)
        self.max_slippage_bps = config.get("max_slippage_bps", 10)  # 0.1%
        
        # Execution parameters
        self.simultaneous_execution = config.get("simultaneous_execution", True)
        self.max_execution_time_ms = config.get("max_execution_time_ms", 5000)
        self.hedge_ratio = config.get("hedge_ratio", 1.0)
        
        # State tracking
        self.opportunities: Dict[str, ArbitrageOpportunity] = {}
        self.executions: Dict[str, ArbitrageExecution] = {}
        self.active_symbols = set()
        
        # Exchange connectors
        self.cex_connectors = {}
        self.dex_connectors = {}
        
        # Performance metrics
        self.metrics = {
            "opportunities_detected": 0,
            "opportunities_executed": 0,
            "successful_arbitrages": 0,
            "failed_arbitrages": 0,
            "total_profit": 0.0,
            "total_volume": 0.0,
            "avg_profit_bps": 0.0,
            "success_rate": 0.0,
            "fastest_execution_ms": float('inf'),
            "slowest_execution_ms": 0.0
        }
        
        # Price feeds
        self.cex_prices = {}
        self.dex_prices = {}
        self.order_books = {}
        
        # Start arbitrage scanner
        asyncio.create_task(self._scan_arbitrage_opportunities())
        
    async def calculate_signals(self, market_data: Dict[str, Any]) -> List[TradeSignal]:
        """Calculate arbitrage signals"""
        # This strategy is event-driven, not signal-based
        return []
        
    async def _scan_arbitrage_opportunities(self):
        """Continuously scan for arbitrage opportunities"""
        scan_interval = self.config.get("scan_interval", 0.1)  # 100ms
        
        while self.running:
            try:
                # Update prices from all exchanges
                await self._update_prices()
                
                # Find arbitrage opportunities
                opportunities = await self._find_arbitrage_opportunities()
                
                # Execute profitable opportunities
                for opportunity in opportunities:
                    if opportunity.estimated_profit_pct >= (self.min_profit_bps / 10000):
                        await self._execute_arbitrage(opportunity)
                        
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                logger.error(f"Arbitrage scan error: {e}")
                await asyncio.sleep(1)
                
    async def _update_prices(self):
        """Update prices from all connected exchanges"""
        symbols = self.config.get("symbols", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        
        for symbol in symbols:
            try:
                # Get CEX price (e.g., Binance)
                cex_price = await self._get_cex_price(symbol)
                if cex_price:
                    self.cex_prices[symbol] = cex_price
                    
                # Get DEX price (e.g., Hyperliquid)
                dex_price = await self._get_dex_price(symbol)
                if dex_price:
                    self.dex_prices[symbol] = dex_price
                    
            except Exception as e:
                logger.error(f"Price update error for {symbol}: {e}")
                
    async def _get_cex_price(self, symbol: str) -> Optional[float]:
        """Get price from CEX"""
        # Implementation depends on your CEX connector
        # This is a simplified version
        for connector in self.cex_connectors.values():
            try:
                ticker = await connector.get_ticker(symbol)
                return (float(ticker["bid"]) + float(ticker["ask"])) / 2
            except:
                continue
        return None
        
    async def _get_dex_price(self, symbol: str) -> Optional[float]:
        """Get price from DEX"""
        # Implementation depends on your DEX connector
        for connector in self.dex_connectors.values():
            try:
                ticker = await connector.get_ticker(symbol)
                return (float(ticker["bid"]) + float(ticker["ask"])) / 2
            except:
                continue
        return None
        
    async def _find_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities between CEX and DEX"""
        opportunities = []
        
        for symbol in self.cex_prices.keys() & self.dex_prices.keys():
            cex_price = self.cex_prices[symbol]
            dex_price = self.dex_prices[symbol]
            
            if not cex_price or not dex_price:
                continue
                
            # Calculate price difference
            price_diff = abs(cex_price - dex_price)
            price_diff_pct = (price_diff / min(cex_price, dex_price)) * 100
            
            # Convert to basis points
            spread_bps = int(price_diff_pct * 100)  # 1% = 100 bps
            
            # Determine direction (where to buy, where to sell)
            if cex_price < dex_price:
                # Buy on CEX, sell on DEX
                buy_venue = "cex"
                sell_venue = "dex"
                buy_price = cex_price
                sell_price = dex_price
                estimated_profit_pct = ((sell_price - buy_price) / buy_price) * 100
            else:
                # Buy on DEX, sell on CEX
                buy_venue = "dex"
                sell_venue = "cex"
                buy_price = dex_price
                sell_price = cex_price
                estimated_profit_pct = ((sell_price - buy_price) / buy_price) * 100
                
            # Calculate estimated profit
            position_size = min(
                self.max_position_usd,
                await self._calculate_optimal_size(symbol, estimated_profit_pct)
            )
            
            estimated_profit = position_size * (estimated_profit_pct / 100)
            
            # Subtract estimated fees
            fees = await self._estimate_fees(symbol, position_size, buy_venue, sell_venue)
            estimated_profit_after_fees = estimated_profit - fees
            
            # Check if profitable
            if (estimated_profit_after_fees >= self.min_profit_usd and 
                spread_bps >= self.min_profit_bps):
                
                opportunity = ArbitrageOpportunity(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    cex_venue="binance",  # Example
                    dex_venue="hyperliquid",  # Example
                    cex_price=cex_price,
                    dex_price=dex_price,
                    price_difference=price_diff,
                    price_difference_pct=price_diff_pct,
                    spread_bps=spread_bps,
                    estimated_profit=estimated_profit_after_fees,
                    estimated_profit_pct=estimated_profit_pct
                )
                
                opportunities.append(opportunity)
                self.opportunities[opportunity.id] = opportunity
                self.metrics["opportunities_detected"] += 1
                
                logger.info(f"Arbitrage opportunity detected: {symbol} "
                          f"CEX: ${cex_price:.2f} DEX: ${dex_price:.2f} "
                          f"Spread: {spread_bps}bps Profit: ${estimated_profit_after_fees:.2f}")
                
        return opportunities
        
    async def _calculate_optimal_size(self, symbol: str, profit_pct: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        # Simplified Kelly Criterion
        win_probability = 0.7  # Estimate based on historical success
        loss_probability = 0.3
        win_amount = profit_pct / 100
        loss_amount = self.max_slippage_bps / 10000  # Convert bps to decimal
        
        # Kelly formula: f* = (p*b - q)/b
        if win_amount > 0:
            kelly_fraction = (win_probability * win_amount - loss_probability * loss_amount) / win_amount
            kelly_fraction = max(0.1, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.1
            
        # Apply to available capital
        available_capital = self.max_position_usd
        return available_capital * kelly_fraction
        
    async def _estimate_fees(self, symbol: str, size: float, 
                            buy_venue: str, sell_venue: str) -> float:
        """Estimate trading fees"""
        # CEX fees (maker/taker)
        cex_fee_rate = 0.001  # 0.1%
        dex_fee_rate = 0.003  # 0.3% (including gas)
        
        if buy_venue == "cex":
            buy_fee = size * cex_fee_rate
            sell_fee = size * dex_fee_rate
        else:
            buy_fee = size * dex_fee_rate
            sell_fee = size * cex_fee_rate
            
        return buy_fee + sell_fee
        
    async def _execute_arbitrage(self, opportunity: ArbitrageOpportunity):
        """Execute arbitrage opportunity"""
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Determine execution direction
        if opportunity.cex_price < opportunity.dex_price:
            # Buy on CEX, sell on DEX
            side_cex = "buy"
            side_dex = "sell"
            buy_price = opportunity.cex_price
            sell_price = opportunity.dex_price
        else:
            # Buy on DEX, sell on CEX
            side_cex = "sell"
            side_dex = "buy"
            buy_price = opportunity.dex_price
            sell_price = opportunity.cex_price
            
        # Calculate quantity
        quantity = await self._calculate_execution_quantity(opportunity)
        
        # Create execution record
        execution = ArbitrageExecution(
            id=execution_id,
            opportunity_id=opportunity.id,
            symbol=opportunity.symbol,
            side_cex=side_cex,
            side_dex=side_dex,
            quantity=quantity,
            cex_price=opportunity.cex_price,
            dex_price=opportunity.dex_price,
            status="executing"
        )
        
        self.executions[execution_id] = execution
        opportunity.status = "executing"
        opportunity.execution_id = execution_id
        
        try:
            # Execute simultaneously
            if self.simultaneous_execution:
                await self._execute_simultaneous(execution)
            else:
                await self._execute_sequential(execution)
                
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            execution.execution_time_ms = execution_time
            
            # Update metrics
            self.metrics["opportunities_executed"] += 1
            self.metrics["fastest_execution_ms"] = min(
                self.metrics["fastest_execution_ms"], 
                execution_time
            )
            self.metrics["slowest_execution_ms"] = max(
                self.metrics["slowest_execution_ms"], 
                execution_time
            )
            
            if execution.status == "complete":
                self.metrics["successful_arbitrages"] += 1
                self.metrics["total_profit"] += execution.net_profit
                self.metrics["total_volume"] += quantity * buy_price * 2
                
                # Update average profit
                total_executed = self.metrics["successful_arbitrages"]
                self.metrics["avg_profit_bps"] = (
                    (self.metrics["avg_profit_bps"] * (total_executed - 1) + 
                     (execution.net_profit / (quantity * buy_price) * 10000)) / total_executed
                )
                
                # Update success rate
                total = self.metrics["successful_arbitrages"] + self.metrics["failed_arbitrages"]
                if total > 0:
                    self.metrics["success_rate"] = (
                        self.metrics["successful_arbitrages"] / total * 100
                    )
                    
                logger.info(f"Arbitrage completed: {execution_id} "
                          f"Profit: ${execution.net_profit:.2f} "
                          f"Time: {execution_time:.0f}ms")
                          
            else:
                self.metrics["failed_arbitrages"] += 1
                logger.warning(f"Arbitrage failed: {execution_id}")
                
        except Exception as e:
            execution.status = "failed"
            self.metrics["failed_arbitrages"] += 1
            logger.error(f"Arbitrage execution error: {e}")
            
    async def _calculate_execution_quantity(self, opportunity: ArbitrageOpportunity) -> float:
        """Calculate execution quantity with slippage consideration"""
        # Get available liquidity
        cex_liquidity = await self._get_cex_liquidity(opportunity.symbol)
        dex_liquidity = await self._get_dex_liquidity(opportunity.symbol)
        
        # Use the minimum of available liquidity and max position
        max_by_liquidity = min(cex_liquidity, dex_liquidity)
        max_by_capital = self.max_position_usd / opportunity.cex_price
        
        return min(max_by_liquidity, max_by_capital)
        
    async def _execute_simultaneous(self, execution: ArbitrageExecution):
        """Execute both legs simultaneously"""
        # Create tasks for both orders
        cex_task = asyncio.create_task(
            self._place_cex_order(execution)
        )
        dex_task = asyncio.create_task(
            self._place_dex_order(execution)
        )
        
        # Wait for both to complete
        results = await asyncio.gather(
            cex_task, dex_task,
            return_exceptions=True
        )
        
        # Check results
        cex_result, dex_result = results
        
        if isinstance(cex_result, Exception):
            raise cex_result
        if isinstance(dex_result, Exception):
            raise dex_result
            
        # Update execution record
        execution.cex_order_id = cex_result.get("order_id")
        execution.dex_order_id = dex_result.get("order_id")
        execution.cex_filled = cex_result.get("filled", False)
        execution.dex_filled = dex_result.get("filled", False)
        
        # Calculate profit
        if execution.cex_filled and execution.dex_filled:
            execution.status = "complete"
            await self._calculate_profit(execution)
        else:
            execution.status = "partial"
            # Handle partial execution (cancel other leg)
            await self._handle_partial_execution(execution)
            
    async def _execute_sequential(self, execution: ArbitrageExecution):
        """Execute sequentially (riskier but sometimes necessary)"""
        # First leg
        if execution.side_cex == "buy":
            # Buy on CEX first, then sell on DEX
            cex_result = await self._place_cex_order(execution)
            execution.cex_order_id = cex_result.get("order_id")
            execution.cex_filled = cex_result.get("filled", False)
            
            if execution.cex_filled:
                dex_result = await self._place_dex_order(execution)
                execution.dex_order_id = dex_result.get("order_id")
                execution.dex_filled = dex_result.get("filled", False)
        else:
            # Sell on CEX first, then buy on DEX
            cex_result = await self._place_cex_order(execution)
            execution.cex_order_id = cex_result.get("order_id")
            execution.cex_filled = cex_result.get("filled", False)
            
            if execution.cex_filled:
                dex_result = await self._place_dex_order(execution)
                execution.dex_order_id = dex_result.get("order_id")
                execution.dex_filled = dex_result.get("filled", False)
                
        # Update status
        if execution.cex_filled and execution.dex_filled:
            execution.status = "complete"
            await self._calculate_profit(execution)
        else:
            execution.status = "partial"
            await self._handle_partial_execution(execution)
            
    async def _place_cex_order(self, execution: ArbitrageExecution) -> Dict[str, Any]:
        """Place order on CEX"""
        # Implementation depends on your CEX connector
        # This is a simplified version
        connector = self.cex_connectors.get("binance")
        if not connector:
            raise Exception("CEX connector not available")
            
        order = {
            "symbol": execution.symbol.replace("/", ""),
            "side": execution.side_cex,
            "type": "market",  # Use market for speed
            "quantity": execution.quantity,
            "timeInForce": "IOC"  # Immediate or cancel
        }
        
        result = await connector.place_order(order)
        return {
            "order_id": result.get("orderId"),
            "filled": result.get("status") == "FILLED",
            "avg_price": float(result.get("avgPrice", 0))
        }
        
    async def _place_dex_order(self, execution: ArbitrageExecution) -> Dict[str, Any]:
        """Place order on DEX"""
        # Implementation depends on your DEX connector
        connector = self.dex_connectors.get("hyperliquid")
        if not connector:
            raise Exception("DEX connector not available")
            
        order = {
            "symbol": execution.symbol,
            "side": execution.side_dex,
            "type": "market",
            "quantity": execution.quantity
        }
        
        result = await connector.place_order(order)
        return {
            "order_id": result.get("order_id"),
            "filled": result.get("filled", False),
            "avg_price": float(result.get("avg_price", 0))
        }
        
    async def _calculate_profit(self, execution: ArbitrageExecution):
        """Calculate realized profit"""
        # In production, you'd get actual fill prices
        # This is a simplified calculation
        
        if execution.side_cex == "buy":
            # Bought on CEX, sold on DEX
            cost = execution.quantity * execution.cex_price
            revenue = execution.quantity * execution.dex_price
        else:
            # Sold on CEX, bought on DEX
            revenue = execution.quantity * execution.cex_price
            cost = execution.quantity * execution.dex_price
            
        gross_profit = revenue - cost
        
        # Estimate fees
        fees = await self._estimate_fees(
            execution.symbol,
            execution.quantity,
            "cex" if execution.side_cex == "buy" else "dex",
            "dex" if execution.side_dex == "sell" else "cex"
        )
        
        execution.profit_realized = gross_profit
        execution.fees_paid = fees
        execution.net_profit = gross_profit - fees
        
    async def _handle_partial_execution(self, execution: ArbitrageExecution):
        """Handle partial execution (cancel other leg)"""
        if execution.cex_filled and not execution.dex_filled:
            # CEX filled, DEX didn't - need to close CEX position
            await self._hedge_position(execution)
        elif not execution.cex_filled and execution.dex_filled:
            # DEX filled, CEX didn't - need to close DEX position
            await self._hedge_position(execution)
            
    async def _hedge_position(self, execution: ArbitrageExecution):
        """Hedge partial position to minimize loss"""
        # Implement hedging logic
        logger.warning(f"Hedging required for partial execution: {execution.id}")
        
    async def _get_cex_liquidity(self, symbol: str) -> float:
        """Get available liquidity on CEX"""
        # Implementation depends on your CEX connector
        return 100.0  # Placeholder
        
    async def _get_dex_liquidity(self, symbol: str) -> float:
        """Get available liquidity on DEX"""
        # Implementation depends on your DEX connector
        return 100.0  # Placeholder
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get arbitrage performance metrics"""
        return {
            **self.metrics,
            "active_opportunities": len([o for o in self.opportunities.values() 
                                       if o.status == "detected"]),
            "active_executions": len([e for e in self.executions.values() 
                                    if e.status in ["executing", "partial"]]),
            "timestamp": datetime.utcnow().isoformat()
        }
