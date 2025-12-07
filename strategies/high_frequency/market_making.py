"""
Market Making Strategy
High Risk - Provides liquidity on both sides of the order book
Source: Inspired by QuantConnect/Lean and awesome-systematic-trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]
    spread: float
    mid_price: float

@dataclass
class MarketMakingOrder:
    side: str  # 'bid' or 'ask'
    price: float
    quantity: float
    order_id: str
    timestamp: datetime

class MarketMakingStrategy:
    def __init__(self,
                 symbol: str,
                 base_quantity: float,
                 spread_target: float = 0.001,  # 0.1%
                 inventory_target: float = 0.0,
                 max_inventory: float = 100.0,
                 skew_factor: float = 0.1):
        """
        Simple Market Making Strategy
        
        Args:
            symbol: Trading pair symbol
            base_quantity: Base order quantity
            spread_target: Target bid-ask spread as percentage
            inventory_target: Target inventory position
            max_inventory: Maximum inventory allowed
            skew_factor: How much to skew prices based on inventory
        """
        self.symbol = symbol
        self.base_qty = base_quantity
        self.spread_target = spread_target
        self.inventory_target = inventory_target
        self.max_inventory = max_inventory
        self.skew_factor = skew_factor
        
        self.current_inventory = 0.0
        self.active_orders: List[MarketMakingOrder] = []
        self.pnl = 0.0
        self.trade_count = 0
        
    def calculate_quotes(self, 
                        orderbook: OrderBookSnapshot,
                        current_inventory: float = None) -> Dict[str, MarketMakingOrder]:
        """
        Calculate bid and ask quotes based on current market conditions
        
        Returns:
            Dictionary with 'bid' and 'ask' orders
        """
        if current_inventory is not None:
            self.current_inventory = current_inventory
        
        # Calculate mid price from orderbook
        mid_price = orderbook.mid_price
        
        # Adjust spread based on volatility (simplified)
        realized_spread = self._calculate_realized_spread()
        if realized_spread > self.spread_target * 1.5:
            # Widening spreads due to high volatility
            current_spread = self.spread_target * 1.2
        elif realized_spread < self.spread_target * 0.5:
            # Tightening spreads due to low volatility
            current_spread = self.spread_target * 0.8
        else:
            current_spread = self.spread_target
        
        # Calculate inventory skew
        inventory_skew = self._calculate_inventory_skew()
        
        # Calculate bid and ask prices with skew
        bid_price = mid_price * (1 - current_spread / 2) * (1 - inventory_skew)
        ask_price = mid_price * (1 + current_spread / 2) * (1 + inventory_skew)
        
        # Adjust quantities based on inventory
        bid_qty = self._adjust_quantity(self.base_qty, 'bid')
        ask_qty = self._adjust_quantity(self.base_qty, 'ask')
        
        # Create order objects
        bid_order = MarketMakingOrder(
            side='bid',
            price=bid_price,
            quantity=bid_qty,
            order_id=f"bid_{datetime.now().timestamp()}",
            timestamp=datetime.now()
        )
        
        ask_order = MarketMakingOrder(
            side='ask',
            price=ask_price,
            quantity=ask_qty,
            order_id=f"ask_{datetime.now().timestamp()}",
            timestamp=datetime.now()
        )
        
        return {'bid': bid_order, 'ask': ask_order}
    
    def _calculate_inventory_skew(self) -> float:
        """Calculate price skew based on current inventory"""
        inventory_ratio = self.current_inventory / self.max_inventory
        skew = self.skew_factor * inventory_ratio
        
        # Cap skew at 2x the factor
        return max(min(skew, 2 * self.skew_factor), -2 * self.skew_factor)
    
    def _adjust_quantity(self, base_qty: float, side: str) -> float:
        """Adjust order quantity based on inventory and side"""
        if side == 'bid':
            # If we have too much inventory, reduce bid quantity
            if self.current_inventory > self.max_inventory * 0.7:
                return base_qty * 0.5
            elif self.current_inventory > self.max_inventory * 0.3:
                return base_qty * 0.8
            else:
                return base_qty
        else:  # ask
            # If we have too little inventory, reduce ask quantity
            if self.current_inventory < -self.max_inventory * 0.7:
                return base_qty * 0.5
            elif self.current_inventory < -self.max_inventory * 0.3:
                return base_qty * 0.8
            else:
                return base_qty
    
    def _calculate_realized_spread(self) -> float:
        """Calculate realized spread from recent trades"""
        # Simplified - in production, calculate from actual fills
        return self.spread_target
    
    def update_inventory(self, fill_side: str, fill_qty: float, fill_price: float):
        """Update inventory after order fill"""
        if fill_side == 'bid':
            self.current_inventory += fill_qty
            self.pnl -= fill_qty * fill_price  # Cost of buying
        else:  # ask
            self.current_inventory -= fill_qty
            self.pnl += fill_qty * fill_price  # Revenue from selling
        
        self.trade_count += 1
        
        logger.info(f"Inventory update: {fill_side} {fill_qty} @ {fill_price}")
        logger.info(f"Current inventory: {self.current_inventory}, PnL: {self.pnl}")
    
    def risk_management_check(self) -> Tuple[bool, str]:
        """
        Check if risk limits are breached
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Inventory risk check
        if abs(self.current_inventory) > self.max_inventory:
            return True, f"Inventory limit exceeded: {self.current_inventory}"
        
        # Drawdown check (simplified)
        if self.pnl < -self.max_inventory * 0.1:  # 10% drawdown
            return True, f"Drawdown limit reached: {self.pnl}"
        
        # Maximum trades per period (simplified)
        if self.trade_count > 1000:  # Example limit
            return True, f"Trade count limit reached: {self.trade_count}"
        
        return False, ""
