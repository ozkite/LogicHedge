"""
Market making strategy for Hyperliquid perpetuals.
"""

import asyncio
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime, timedelta

from logichedge.core.strategy import BaseStrategy, TradeSignal
from logichedge.core.event_bus import Event

class HyperliquidMarketMaker(BaseStrategy):
    """Market making strategy for Hyperliquid perpetuals"""
    
    def __init__(self, name: str, event_bus, config: Dict[str, Any]):
        super().__init__(name, event_bus, config)
        
        # Strategy parameters
        self.spread_pct = config.get("spread_pct", 0.001)  # 0.1%
        self.order_size = config.get("order_size", 100.0)  # USD
        self.inventory_target = config.get("inventory_target", 0.0)
        self.max_inventory = config.get("max_inventory", 1000.0)
        
        # State tracking
        self.bid_price = None
        self.ask_price = None
        self.mid_price = None
        self.inventory = 0.0
        self.pnl = 0.0
        self.last_order_time = {}
        
        # Metrics
        self.metrics = {
            "trades_count": 0,
            "total_volume": 0.0,
            "profit": 0.0,
            "spread_captured": 0.0
        }
        
    async def calculate_signals(self, market_data: Dict[str, Any]) -> List[TradeSignal]:
        """Calculate market making signals"""
        signals = []
        
        symbol = market_data.get("symbol")
        if not symbol:
            return signals
            
        # Update prices
        self.mid_price = market_data.get("mid_price")
        if not self.mid_price:
            return signals
            
        # Calculate bid/ask prices
        self.bid_price = self.mid_price * (1 - self.spread_pct / 2)
        self.ask_price = self.mid_price * (1 + self.spread_pct / 2)
        
        # Calculate inventory skew
        inventory_skew = self._calculate_inventory_skew()
        
        # Adjust prices based on inventory
        bid_adjustment = inventory_skew * 0.0005  # 0.05% adjustment per skew unit
        ask_adjustment = inventory_skew * 0.0005
        
        adjusted_bid = self.bid_price * (1 - bid_adjustment)
        adjusted_ask = self.ask_price * (1 + ask_adjustment)
        
        # Generate signals
        current_time = datetime.utcnow()
        last_order = self.last_order_time.get(symbol)
        
        # Check if we should place new orders
        if last_order is None or (current_time - last_order) > timedelta(seconds=30):
            # Place bid order
            signals.append(TradeSignal(
                symbol=symbol,
                side="BUY",
                quantity=self._calculate_order_size(),
                order_type="LIMIT",
                price=adjusted_bid,
                metadata={
                    "strategy": self.name,
                    "order_type": "maker_bid",
                    "inventory": self.inventory
                }
            ))
            
            # Place ask order
            signals.append(TradeSignal(
                symbol=symbol,
                side="SELL",
                quantity=self._calculate_order_size(),
                order_type="LIMIT",
                price=adjusted_ask,
                metadata={
                    "strategy": self.name,
                    "order_type": "maker_ask",
                    "inventory": self.inventory
                }
            ))
            
            self.last_order_time[symbol] = current_time
            
        return signals
        
    def _calculate_inventory_skew(self) -> float:
        """Calculate inventory position skew"""
        if self.max_inventory == 0:
            return 0.0
        return self.inventory / self.max_inventory
        
    def _calculate_order_size(self) -> float:
        """Calculate dynamic order size based on inventory"""
        base_size = self.order_size
        
        # Reduce size if inventory is high
        inventory_ratio = abs(self.inventory) / self.max_inventory
        if inventory_ratio > 0.5:
            size_multiplier = 1 - inventory_ratio
            base_size *= max(0.1, size_multiplier)
            
        return base_size
        
    async def on_order_filled(self, event: Event):
        """Handle filled orders"""
        await super().on_order_filled(event)
        
        order_data = event.data
        if order_data.get("strategy") != self.name:
            return
            
        # Update inventory
        side = order_data.get("side")
        quantity = order_data.get("quantity", 0)
        price = order_data.get("price", 0)
        
        if side == "BUY":
            self.inventory += quantity
        elif side == "SELL":
            self.inventory -= quantity
            
        # Update metrics
        self.metrics["trades_count"] += 1
        self.metrics["total_volume"] += quantity * price
        
        self.logger.info(f"Inventory updated: {self.inventory:.4f}")
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get market making performance metrics"""
        base_metrics = super().get_performance_metrics()
        
        metrics = {
            **base_metrics,
            "inventory": self.inventory,
            "mid_price": self.mid_price,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "current_spread": (self.ask_price - self.bid_price) / self.mid_price if self.mid_price else 0,
            "target_spread": self.spread_pct,
            "pnl": self.pnl
        }
        
        metrics.update(self.metrics)
        return metrics
