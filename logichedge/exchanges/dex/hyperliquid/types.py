from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum

class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill

@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    client_order_id: Optional[str] = None

@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    liquidation_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    leverage: float

@dataclass
class Balance:
    asset: str
    total: float
    available: float
    locked: float

@dataclass
class Ticker:
    symbol: str
    bid_price: float
    ask_price: float
    last_price: float
    volume_24h: float
    price_change_24h: float
