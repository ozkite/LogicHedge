"""
Advanced order manager inspired by NoFx trading system.
Features:
- Order lifecycle management
- Smart order routing
- Slippage control
- Order recycling
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    LIVE = "live"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(Enum):
    """Order type enumeration"""
    LIMIT = "limit"
    MARKET = "market"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"

@dataclass
class AdvancedOrder:
    """Advanced order with NoFx-inspired features"""
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_distance: Optional[float] = None
    
    # Order parameters
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    post_only: bool = False
    reduce_only: bool = False
    iceberg: bool = False
    display_quantity: Optional[float] = None
    
    # State
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    remaining_quantity: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    
    # Routing
    venue: str = "auto"  # auto, hyperliquid, binance, etc.
    routing_score: float = 1.0
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity
        
    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        active_statuses = {OrderStatus.PENDING, OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED}
        return self.status in active_statuses
        
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage"""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100
        
    def update_fill(self, fill_quantity: float, fill_price: float):
        """Update order with new fill"""
        self.filled_quantity += fill_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # Update average price
        total_value = self.average_price * (self.filled_quantity - fill_quantity)
        total_value += fill_price * fill_quantity
        self.average_price = total_value / self.filled_quantity
        
        # Update status
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
            
        self.updated_at = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "remaining_quantity": self.remaining_quantity,
            "fill_percentage": self.fill_percentage,
            "venue": self.venue,
            "strategy_id": self.strategy_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class SmartOrderRouter:
    """
    Smart order router inspired by NoFx execution system
    """
    
    def __init__(self, venue_connectors: Dict[str, Any]):
        self.venue_connectors = venue_connectors
        self.venue_metrics = {}  # Track venue performance
        self.slippage_models = {}  # Slippage prediction models
        
    async def route_order(self, order: AdvancedOrder) -> str:
        """
        Route order to best venue based on:
        - Liquidity
        - Fees
        - Latency
        - Slippage prediction
        - Historical fill rates
        """
        if order.venue != "auto":
            return order.venue
            
        # Calculate scores for each venue
        venue_scores = {}
        
        for venue_name, connector in self.venue_connectors.items():
            if not connector.is_available():
                continue
                
            # Calculate composite score
            score = await self._calculate_venue_score(venue_name, order)
            venue_scores[venue_name] = score
            
        # Select best venue
        if not venue_scores:
            raise Exception("No available venues for routing")
            
        best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
        order.routing_score = venue_scores[best_venue]
        
        return best_venue
        
    async def _calculate_venue_score(self, venue_name: str, order: AdvancedOrder) -> float:
        """Calculate score for a venue"""
        score = 1.0
        
        # 1. Liquidity score (0-1)
        liquidity_score = await self._get_liquidity_score(venue_name, order.symbol)
        score *= liquidity_score
        
        # 2. Fee score (lower fees = higher score)
        fee_score = await self._get_fee_score(venue_name, order)
        score *= fee_score
        
        # 3. Latency score (lower latency = higher score)
        latency_score = await self._get_latency_score(venue_name)
        score *= latency_score
        
        # 4. Historical fill rate
        fill_rate_score = await self._get_fill_rate_score(venue_name, order)
        score *= fill_rate_score
        
        # 5. Slippage prediction
        slippage_score = await self._get_slippage_score(venue_name, order)
        score *= slippage_score
        
        return score
        
    async def _get_liquidity_score(self, venue_name: str, symbol: str) -> float:
        """Get liquidity score for venue/symbol"""
        # Implement actual liquidity check
        return 0.9  # Placeholder
        
    async def _get_fee_score(self, venue_name: str, order: AdvancedOrder) -> float:
        """Get fee score"""
        # Implement fee calculation
        base_fee = 0.001  # 0.1%
        
        # Maker/taker differentiation
        if order.post_only:
            fee_multiplier = 0.8  # Maker fee discount
        else:
            fee_multiplier = 1.2  # Taker fee premium
            
        fee_score = 1.0 / (1.0 + base_fee * fee_multiplier)
        return fee_score
        
    async def _get_latency_score(self, venue_name: str) -> float:
        """Get latency score"""
        # Implement latency measurement
        avg_latency = 50  # ms
        latency_score = 1.0 / (1.0 + avg_latency / 1000)
        return latency_score

class AdvancedOrderManager:
    """
    Advanced order manager with NoFx-inspired features
    """
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.orders: Dict[str, AdvancedOrder] = {}
        self.order_router = SmartOrderRouter({})
        self.active_orders = set()
        
        # Performance tracking
        self.metrics = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "total_volume": 0.0,
            "avg_fill_time": 0.0,
            "success_rate": 0.0
        }
        
        # Start maintenance tasks
        asyncio.create_task(self._order_maintenance())
        
    async def create_order(self, order_data: Dict[str, Any]) -> AdvancedOrder:
        """Create a new advanced order"""
        order_id = str(uuid.uuid4())
        
        order = AdvancedOrder(
            order_id=order_id,
            client_order_id=order_data.get("client_order_id", f"order_{order_id[:8]}"),
            symbol=order_data["symbol"],
            side=order_data["side"],
            order_type=OrderType(order_data.get("order_type", "limit")),
            quantity=order_data["quantity"],
            price=order_data.get("price"),
            stop_price=order_data.get("stop_price"),
            trailing_distance=order_data.get("trailing_distance"),
            time_in_force=order_data.get("time_in_force", "GTC"),
            post_only=order_data.get("post_only", False),
            reduce_only=order_data.get("reduce_only", False),
            strategy_id=order_data.get("strategy_id"),
            venue=order_data.get("venue", "auto")
        )
        
        self.orders[order_id] = order
        self.active_orders.add(order_id)
        self.metrics["total_orders"] += 1
        
        # Route order
        order.venue = await self.order_router.route_order(order)
        
        # Send order creation event
        await self.event_bus.publish({
            "type": "order_created",
            "order": order.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Created order {order_id} for {order.symbol}")
        return order
        
    async def execute_order(self, order_id: str, connector):
        """Execute order through connector"""
        order = self.orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
            
        try:
            # Convert to exchange-specific format
            exchange_order = self._convert_to_exchange_format(order)
            
            # Execute on exchange
            result = await connector.place_order(exchange_order)
            
            # Update order status
            order.status = OrderStatus.LIVE
            order.updated_at = datetime.utcnow()
            
            # Start monitoring
            asyncio.create_task(self._monitor_order(order_id, connector))
            
            return result
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.utcnow()
            logger.error(f"Order execution failed: {e}")
            raise
            
    def _convert_to_exchange_format(self, order: AdvancedOrder) -> Dict[str, Any]:
        """Convert advanced order to exchange format"""
        exchange_order = {
            "symbol": order.symbol,
            "side": order.side,
            "type": order.order_type.value,
            "quantity": order.quantity,
            "clientOrderId": order.client_order_id
        }
        
        if order.price:
            exchange_order["price"] = order.price
            
        if order.time_in_force:
            exchange_order["timeInForce"] = order.time_in_force
            
        if order.post_only:
            exchange_order["postOnly"] = True
            
        if order.reduce_only:
            exchange_order["reduceOnly"] = True
            
        return exchange_order
        
    async def _monitor_order(self, order_id: str, connector, interval: int = 5):
        """Monitor order status and updates"""
        order = self.orders.get(order_id)
        if not order:
            return
            
        while order.is_active:
            try:
                # Check order status
                status = await connector.get_order_status(order.client_order_id)
                
                # Update order with fill information
                if status.get("filled_qty", 0) > order.filled_quantity:
                    fill_qty = status["filled_qty"] - order.filled_quantity
                    fill_price = status.get("avg_price", order.price or 0)
                    
                    order.update_fill(fill_qty, fill_price)
                    
                    # Publish fill event
                    await self.event_bus.publish({
                        "type": "order_filled",
                        "order_id": order_id,
                        "fill_quantity": fill_qty,
                        "fill_price": fill_price,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Update metrics
                    self.metrics["total_volume"] += fill_qty * fill_price
                    
                # Check if order is complete
                if order.status == OrderStatus.FILLED:
                    self.active_orders.remove(order_id)
                    self.metrics["filled_orders"] += 1
                    break
                    
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Order monitoring error: {e}")
                await asyncio.sleep(interval * 2)
                
    async def cancel_order(self, order_id: str, connector):
        """Cancel an active order"""
        order = self.orders.get(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
            
        if not order.is_active:
            logger.warning(f"Order {order_id} is not active")
            return
            
        try:
            await connector.cancel_order(order.client_order_id, order.symbol)
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            
            if order_id in self.active_orders:
                self.active_orders.remove(order_id)
                
            self.metrics["cancelled_orders"] += 1
            
            await self.event_bus.publish({
                "type": "order_cancelled",
                "order_id": order_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            raise
            
    async def create_iceberg_order(self, symbol: str, side: str, total_quantity: float,
                                  price: float, slice_size: float, delay: int = 10) -> str:
        """
        Create iceberg order (split into smaller hidden orders)
        Inspired by NoFx large order handling
        """
        parent_order_id = str(uuid.uuid4())
        slices = int(total_quantity / slice_size)
        
        for i in range(slices):
            slice_id = f"{parent_order_id}_slice_{i}"
            
            order_data = {
                "symbol": symbol,
                "side": side,
                "order_type": "limit",
                "quantity": slice_size,
                "price": price,
                "client_order_id": slice_id,
                "strategy_id": parent_order_id,
                "iceberg": True,
                "display_quantity": slice_size * 0.1  # Show only 10%
            }
            
            # Create slice with delay
            if i > 0:
                await asyncio.sleep(delay)
                
            slice_order = await self.create_order(order_data)
            
            # Link to parent
            parent_order = self.orders.get(parent_order_id, None)
            if parent_order:
                parent_order.child_order_ids.append(slice_order.order_id)
                
        return parent_order_id
        
    async def _order_maintenance(self):
        """Perform order maintenance tasks"""
        while True:
            try:
                # Clean up expired orders
                await self._cleanup_expired_orders()
                
                # Update performance metrics
                await self._update_metrics()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Order maintenance error: {e}")
                await asyncio.sleep(300)
                
    async def _cleanup_expired_orders(self):
        """Clean up expired orders"""
        now = datetime.utcnow()
        expired_orders = []
        
        for order_id, order in self.orders.items():
            if order.expires_at and order.expires_at < now and order.is_active:
                expired_orders.append(order_id)
                
        for order_id in expired_orders:
            order = self.orders[order_id]
            order.status = OrderStatus.EXPIRED
            order.updated_at = now
            
            if order_id in self.active_orders:
                self.active_orders.remove(order_id)
                
            logger.info(f"Order {order_id} expired")
            
    async def _update_metrics(self):
        """Update performance metrics"""
        total = self.metrics["total_orders"]
        filled = self.metrics["filled_orders"]
        
        if total > 0:
            self.metrics["success_rate"] = (filled / total) * 100
            
    def get_order(self, order_id: str) -> Optional[AdvancedOrder]:
        """Get order by ID"""
        return self.orders.get(order_id)
        
    def get_active_orders(self) -> List[AdvancedOrder]:
        """Get all active orders"""
        return [self.orders[oid] for oid in self.active_orders]
        
    def get_orders_by_strategy(self, strategy_id: str) -> List[AdvancedOrder]:
        """Get all orders for a strategy"""
        return [order for order in self.orders.values() 
                if order.strategy_id == strategy_id]
                
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get order manager performance metrics"""
        return {
            **self.metrics,
            "active_orders": len(self.active_orders),
            "total_order_value": self.metrics["total_volume"],
            "timestamp": datetime.utcnow().isoformat()
        }
