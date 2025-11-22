import logging
from typing import Dict, List, Optional, Any
from .client import HyperliquidClient
from .types import Order, OrderSide, OrderType
from .utils import HyperliquidAPIError, handle_response

class HyperliquidPerpetuals(HyperliquidClient):
    """Hyperliquid perpetuals trading specialization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
    
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for a perpetual symbol"""
        if not self.api_key:
            raise HyperliquidAPIError("API key required for leverage settings")
            
        endpoint = f"{self.base_url}/account/leverage"
        headers = self._get_headers(requires_auth=True)
        
        leverage_data = {
            'symbol': symbol,
            'leverage': leverage
        }
        
        if self.private_key:
            signature = self._sign_message(leverage_data)
            leverage_data['signature'] = signature
        
        response = self.session.post(endpoint, json=leverage_data, headers=headers)
        return handle_response(response)
    
    def create_market_order(self, symbol: str, side: OrderSide, quantity: float, 
                          reduce_only: bool = False) -> Dict[str, Any]:
        """Create a market order for perpetuals"""
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=reduce_only
        )
        return self.place_order(order)
    
    def create_limit_order(self, symbol: str, side: OrderSide, quantity: float, 
                          price: float, reduce_only: bool = False) -> Dict[str, Any]:
        """Create a limit order for perpetuals"""
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            reduce_only=reduce_only
        )
        return self.place_order(order)
    
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close a position by creating an opposite market order"""
        positions = self.get_positions()
        position = next((p for p in positions if p.symbol == symbol), None)
        
        if not position or position.size == 0:
            raise HyperliquidAPIError(f"No open position found for {symbol}")
        
        # Determine side to close position (opposite of current position)
        close_side = OrderSide.SELL if position.size > 0 else OrderSide.BUY
        close_quantity = abs(position.size)
        
        return self.create_market_order(symbol, close_side, close_quantity, reduce_only=True)
    
    def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for a perpetual"""
        endpoint = f"{self.base_url}/funding/rate"
        params = {'symbol': symbol}
        response = self.session.get(endpoint, params=params, headers=self._get_headers())
        data = handle_response(response)
        return float(data.get('fundingRate', 0))
    
    def get_funding_history(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get funding rate history"""
        endpoint = f"{self.base_url}/funding/history"
        params = {'symbol': symbol, 'limit': limit}
        response = self.session.get(endpoint, params=params, headers=self._get_headers())
        data = handle_response(response)
        return data.get('history', [])
