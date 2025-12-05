"""
Advanced position manager inspired by Perp-Dex-Trading-Bot.
Features:
- Multi-exchange position tracking
- Cross-venue hedging
- Risk-adjusted position sizing
- P&L attribution
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    HEDGED = "hedged"
    LIQUIDATING = "liquidating"

@dataclass
class Position:
    """Advanced position tracking"""
    position_id: str
    symbol: str
    venue: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    current_price: float
    leverage: float = 1.0
    
    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    
    # Risk metrics
    liquidation_price: Optional[float] = None
    margin_used: float = 0.0
    margin_ratio: float = 0.0
    
    # State
    status: PositionStatus = PositionStatus.OPEN
    entry_time: datetime = field(default_factory=datetime.utcnow)
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    strategy_id: Optional[str] = None
    parent_position_id: Optional[str] = None
    hedge_positions: List[str] = field(default_factory=list)
    
    # Order references
    entry_order_ids: List[str] = field(default_factory=list)
    exit_order_ids: List[str] = field(default_factory=list)
    
    def update_price(self, new_price: float):
        """Update position with new price"""
        self.current_price = new_price
        self.last_update = datetime.utcnow()
        
        # Calculate unrealized P&L
        if self.side == "long":
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        else:  # short
            self.unrealized_pnl = (self.entry_price - new_price) * self.quantity
            
    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
        
    @property
    def pnl_percentage(self) -> float:
        """Get P&L as percentage of position value"""
        position_value = self.entry_price * self.quantity
        if position_value == 0:
            return 0.0
        return (self.total_pnl / position_value) * 100
        
    @property
    def duration(self) -> timedelta:
        """Get position duration"""
        return datetime.utcnow() - self.entry_time
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "venue": self.venue,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "pnl_percentage": self.pnl_percentage,
            "leverage": self.leverage,
            "liquidation_price": self.liquidation_price,
            "margin_used": self.margin_used,
            "status": self.status.value,
            "strategy_id": self.strategy_id,
            "entry_time": self.entry_time.isoformat(),
            "duration_hours": self.duration.total_seconds() / 3600
        }

class AdvancedPositionManager:
    """
    Advanced position manager with features from Perp-Dex-Trading-Bot
    """
    
    def __init__(self, event_bus, risk_manager=None):
        self.event_bus = event_bus
        self.risk_manager = risk_manager
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.symbol_positions: Dict[str, List[Position]] = {}
        self.venue_positions: Dict[str, List[Position]] = {}
        
        # Metrics
        self.metrics = {
            "total_positions": 0,
            "open_positions": 0,
            "closed_positions": 0,
            "total_pnl": 0.0,
            "winning_positions": 0,
            "losing_positions": 0,
            "largest_win": 0.0,
            "largest_loss": 0.0
        }
        
        # Price feeds
        self.price_feeds = {}
        
        # Start maintenance tasks
        asyncio.create_task(self._position_maintenance())
        
    async def open_position(self, position_data: Dict[str, Any]) -> Position:
        """Open a new position"""
        position_id = f"pos_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self.positions)}"
        
        position = Position(
            position_id=position_id,
            symbol=position_data["symbol"],
            venue=position_data["venue"],
            side=position_data["side"],
            quantity=position_data["quantity"],
            entry_price=position_data["entry_price"],
            current_price=position_data["entry_price"],
            leverage=position_data.get("leverage", 1.0),
            strategy_id=position_data.get("strategy_id"),
            entry_order_ids=position_data.get("entry_order_ids", [])
        )
        
        # Calculate liquidation price (simplified)
        if position.leverage > 1:
            position.liquidation_price = await self._calculate_liquidation_price
