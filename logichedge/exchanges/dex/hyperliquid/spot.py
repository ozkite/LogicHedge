from .client import HyperliquidClient
from .types import Order, OrderSide, OrderType

class HyperliquidSpot(HyperliquidClient):
    """Hyperliquid spot trading specialization"""
    
    def create_market_order(self, symbol: str, side: OrderSide, quantity: float) -> Dict[str, Any]:
        """Create a market order for spot trading"""
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        return self.place_order(order)
    
    def create_limit_order(self, symbol: str, side: OrderSide, quantity: float, 
                          price: float) -> Dict[str, Any]:
        """Create a limit order for spot trading"""
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price
        )
        return self.place_order(order)
